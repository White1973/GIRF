import os
import time
import torch
import uvicorn
import uuid
import traceback
import argparse
import sys
import math
import re
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from typing import List, Optional, Any
from contextlib import asynccontextmanager, nullcontext
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from qwenvl.models import Qwen25_SigLIPForConditionalGeneration
from qwenvl.models import Qwen25_SigLIPConfig
from qwenvl.data.multimodal_rope import normalize_position_embedding, resolve_position_options
from fastapi import FastAPI, HTTPException
from qwen_vl_utils import process_vision_info


MODEL_PATH = None
POSITION_EMBEDDING = None
CROSS_MODAL_POSITION = None
ATTN_IMPLEMENTATION = "sdpa"
GIRF_RESOLUTION_GATE = False
GIRF_ORDINAL_ENABLED = None
GIRF_PI_MASK_ENABLED = None
GIRF_PI_MASK_COMPONENTS = None
GIRF_SOFT_VV_PENALTY = None
DISABLE_KV_CACHE = False
model = None
tokenizer = None
processor = None

VIDEO_CONTEXT_RE = re.compile(
    r"\[DIPE_VIDEO_CONTEXT_BUDGET=(\d+);MIN_VISUAL_TOKENS_PER_FRAME=(\d+)\]"
)
VIDEO_GENERATION_RESERVE = 32


def _extract_video_context_budget(messages):
    """Remove VLMEvalKit's internal control marker before model inference."""
    budget = None
    min_tokens = None
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") != "text":
                continue
            match = VIDEO_CONTEXT_RE.search(item.get("text", ""))
            if match is not None:
                budget = int(match.group(1))
                min_tokens = int(match.group(2))
                item["text"] = VIDEO_CONTEXT_RE.sub("", item["text"]).lstrip()
    return budget, min_tokens


def _uniformly_limit_images(messages, max_images):
    image_items = [
        item
        for message in messages
        for item in message.get("content", [])
        if item.get("type") == "image"
    ]
    if len(image_items) <= max_images:
        return messages

    keep_indices = {
        round(i * (len(image_items) - 1) / (max_images - 1))
        for i in range(max_images)
    } if max_images > 1 else {len(image_items) // 2}
    image_index = 0
    for message in messages:
        retained = []
        for item in message.get("content", []):
            if item.get("type") != "image":
                retained.append(item)
            else:
                if image_index in keep_indices:
                    retained.append(item)
                image_index += 1
        message["content"] = retained
    return messages


def _prepare_budgeted_inputs(messages, context_budget, min_tokens_per_frame):
    """Fit a VideoRoPE-style multi-frame request to an exact token budget."""
    image_count = sum(
        item.get("type") == "image"
        for message in messages
        for item in message.get("content", [])
    )
    if image_count == 0:
        raise ValueError("A budgeted Video-MME request contains no image frames")

    # Qwen image tokens are (grid_h * grid_w) / merge_size**2.  Allocate the
    # remaining context approximately evenly, then verify the real input_ids.
    # The source frames have already been uniformly limited by VLMEvalKit.
    max_tokens_per_frame = 1280
    for attempt in range(8):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        text_tokens = tokenizer(text, add_special_tokens=False).input_ids
        base_tokens = len(text_tokens)
        available = context_budget - VIDEO_GENERATION_RESERVE - base_tokens + image_count
        target_tokens = max(
            min_tokens_per_frame,
            min(max_tokens_per_frame, available // image_count),
        )

        patch_size = int(getattr(processor.image_processor, "patch_size", 16))
        merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        target_pixels = target_tokens * patch_size * patch_size * merge_size * merge_size
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            images_kwargs={"min_pixels": target_pixels, "max_pixels": target_pixels},
        )
        actual_tokens = int(inputs.input_ids.shape[-1])
        print(
            f"Video context fitting attempt {attempt + 1}: target={context_budget}, "
            f"actual={actual_tokens}, frames={image_count}, requested_visual_tokens={target_tokens}"
        )
        if actual_tokens <= context_budget - VIDEO_GENERATION_RESERVE:
            print(
                f"Video context budget: target={context_budget}, actual_input={actual_tokens}, "
                f"frames={image_count}, visual_tokens_per_frame≈{target_tokens}"
            )
            return inputs

        overflow_per_frame = math.ceil(
            (actual_tokens - (context_budget - VIDEO_GENERATION_RESERVE)) / image_count
        )
        max_tokens_per_frame = target_tokens - overflow_per_frame
        if max_tokens_per_frame < min_tokens_per_frame:
            theoretical_max_images = max(
                1,
                (context_budget - VIDEO_GENERATION_RESERVE - base_tokens)
                // min_tokens_per_frame,
            )
            if image_count <= 1:
                raise ValueError(
                    f"Cannot fit request into {context_budget} tokens while preserving "
                    f"{min_tokens_per_frame} visual tokens per frame; text/template tokens="
                    f"{base_tokens}, actual input tokens={actual_tokens}"
                )
            # Processor grid rounding can make the real visual-token count
            # slightly larger than target_tokens.  In that case the
            # theoretical formula may claim that every frame still fits even
            # though the verified input is over budget.  Always remove at
            # least one frame after a measured overflow so fitting makes
            # progress instead of returning HTTP 500 at this boundary.
            max_images = min(image_count - 1, theoretical_max_images)
            print(
                f"Video context overflow at minimum frame resolution: "
                f"actual={actual_tokens}, limit={context_budget - VIDEO_GENERATION_RESERVE}; "
                f"uniformly reducing frames from {image_count} to {max_images}"
            )
            messages = _uniformly_limit_images(messages, max_images)
            image_count = max_images
            max_tokens_per_frame = 1280

    raise ValueError(f"Failed to fit Video-MME request into {context_budget} tokens")


class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str = "qwen-transformers"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 50
    # Match the paper's deterministic evaluation protocol. Values above 1.0
    # also penalize option letters already present in an MCQ prompt, which can
    # spuriously favor nonexistent choices in 2/3-option questions.
    repetition_penalty: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, processor
    print(f"Loading: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = "left"
        processor = AutoProcessor.from_pretrained(MODEL_PATH, tokenizer=tokenizer, trust_remote_code=True)
        
        # model = AutoModelForImageTextToText.from_pretrained(
        #     MODEL_PATH,
        #     dtype="auto",
        #     attn_implementation="flash_attention_2",
        #     device_map="auto"
        # ).eval()

        config = Qwen25_SigLIPConfig.from_pretrained(MODEL_PATH)
        if POSITION_EMBEDDING is not None:
            position_name = normalize_position_embedding(POSITION_EMBEDDING)
            checkpoint_options = (
                config.position_embedding_options
                if config.position_embedding_type == position_name
                else None
            )
            position_options = resolve_position_options(position_name, checkpoint_options)
            config.position_embedding_type = position_name
            config.position_embedding_options = position_options
            config.text_config.position_embedding_type = position_name
            config.text_config.position_embedding_options = position_options
        if CROSS_MODAL_POSITION is not None:
            cross_modal_name = CROSS_MODAL_POSITION.lower()
            if cross_modal_name not in {
                "dipe", "native", "universal-relation", "bounded-dipe", "girf",
                "entity-aware-girf", "entity-tagged-girf",
            }:
                raise ValueError(
                    "--cross-modal-position must be dipe, native, universal-relation, "
                    "bounded-dipe, girf, entity-aware-girf, or entity-tagged-girf"
                )
            config.cross_modal_position_type = cross_modal_name
            config.text_config.cross_modal_position_type = cross_modal_name
        for target in (config, config.text_config):
            if GIRF_ORDINAL_ENABLED is not None:
                target.girf_ordinal_enabled = GIRF_ORDINAL_ENABLED
            if GIRF_PI_MASK_ENABLED is not None:
                target.girf_pi_mask_enabled = GIRF_PI_MASK_ENABLED
            if GIRF_PI_MASK_COMPONENTS is not None:
                target.girf_pi_mask_components = list(GIRF_PI_MASK_COMPONENTS)
            if GIRF_SOFT_VV_PENALTY is not None:
                target.girf_pi_soft_vv_penalty = GIRF_SOFT_VV_PENALTY
        if GIRF_RESOLUTION_GATE and ATTN_IMPLEMENTATION != "eager":
            raise ValueError("GIRF Resolution Gate requires --attn-implementation eager")
        model = Qwen25_SigLIPForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            config=config,
            attn_implementation=ATTN_IMPLEMENTATION,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
    
    except Exception as e:
        print(f"Loading failed: {e}")
        raise e
    
    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="Qwen Transformers API")


def create_openai_chunk(content: str, model_name: str, finish_reason: str = None):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }
        ]
    }



@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "position_embedding": model.config.position_embedding_type if model is not None else POSITION_EMBEDDING,
        "cross_modal_position": model.config.cross_modal_position_type if model is not None else CROSS_MODAL_POSITION,
        "girf_resolution_gate": GIRF_RESOLUTION_GATE,
        "girf_ordinal_enabled": GIRF_ORDINAL_ENABLED,
        "girf_pi_mask_enabled": GIRF_PI_MASK_ENABLED,
        "girf_pi_mask_components": GIRF_PI_MASK_COMPONENTS,
        "girf_soft_vv_penalty": GIRF_SOFT_VV_PENALTY,
        "disable_kv_cache": DISABLE_KV_CACHE,
    }


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "qwen-transformers"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, processor
    
    qwen_messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content
        
        new_content = []
        
        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})
            
        elif isinstance(content, list):
            for item in content:
                item_type = getattr(item, 'type', item.get('type') if isinstance(item, dict) else None)
                
                if item_type == "text":
                    text_val = getattr(item, 'text', item.get('text') if isinstance(item, dict) else "")
                    new_content.append({"type": "text", "text": text_val})
                    
                elif item_type == "image_url":
                    image_url_obj = getattr(item, 'image_url', item.get('image_url') if isinstance(item, dict) else {})
                    url = getattr(image_url_obj, 'url', image_url_obj.get('url') if isinstance(image_url_obj, dict) else "")
                    new_content.append({"type": "image", "image": url})
        
        qwen_messages.append({
            "role": role,
            "content": new_content
        })

    try:
        context_budget, min_tokens_per_frame = _extract_video_context_budget(qwen_messages)
        if context_budget is not None:
            inputs = _prepare_budgeted_inputs(
                qwen_messages, context_budget, min_tokens_per_frame)
        else:
            text = processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(qwen_messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        
        inputs = inputs.to(model.device)

        generation_kwargs = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        if request.max_tokens is not None:
            generation_kwargs["max_new_tokens"] = (
                min(request.max_tokens, VIDEO_GENERATION_RESERVE)
                if context_budget is not None else request.max_tokens
            )
        if request.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = request.repetition_penalty
        
        if generation_kwargs.get("temperature", 0) == 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True

        gate_context = nullcontext()
        if DISABLE_KV_CACHE or GIRF_RESOLUTION_GATE:
            generation_kwargs["use_cache"] = False
        if GIRF_RESOLUTION_GATE:
            from experiments.girf.entity_resolution_gate.runtime import installed

            # The current gate is a deterministic full-attention correction.
            # Cached one-token decoding does not carry the complete relation
            # metadata required by the gate, so generation recomputes the
            # full prefix at every step when the gate is enabled.
            gate_context = installed()
        with gate_context:
            generated_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}]
        }
    except Exception as e:
        print(f"error: {e}")
        error_msg = traceback.format_exc()
        print(f"error:\n{error_msg}") 
        
        print("=== Debug Info ===")
        if 'inputs' in locals():
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        print("==================")

        raise HTTPException(status_code=500, detail=f"error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--position-embedding", type=str, default=None)
    parser.add_argument("--cross-modal-position", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--girf-resolution-gate", action="store_true")
    parser.add_argument(
        "--girf-ordinal-enabled", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--girf-pi-mask-enabled", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--girf-pi-mask-components", type=str, default=None)
    parser.add_argument("--girf-soft-vv-penalty", type=float, default=None)
    parser.add_argument("--disable-kv-cache", action="store_true")
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    POSITION_EMBEDDING = args.position_embedding
    CROSS_MODAL_POSITION = args.cross_modal_position
    ATTN_IMPLEMENTATION = args.attn_implementation
    GIRF_RESOLUTION_GATE = args.girf_resolution_gate
    GIRF_ORDINAL_ENABLED = args.girf_ordinal_enabled
    GIRF_PI_MASK_ENABLED = args.girf_pi_mask_enabled
    if args.girf_pi_mask_components is not None:
        import json

        GIRF_PI_MASK_COMPONENTS = json.loads(args.girf_pi_mask_components)
        if not isinstance(GIRF_PI_MASK_COMPONENTS, list) or not all(
            isinstance(item, str) for item in GIRF_PI_MASK_COMPONENTS
        ):
            parser.error("--girf-pi-mask-components must be a JSON string list")
    GIRF_SOFT_VV_PENALTY = args.girf_soft_vv_penalty
    DISABLE_KV_CACHE = args.disable_kv_cache
    
    uvicorn.run(app, host=args.host, port=args.port)
