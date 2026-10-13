# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
import math
import gc
import logging
import pathlib
import torch
import transformers
import sys
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.models import Qwen25_SigLIPForConditionalGeneration
from qwenvl.models import Qwen2Config, Siglip2VisionConfig, Qwen25_SigLIPConfig
from safetensors.torch import load_file
from qwenvl.data.multimodal_rope import normalize_position_embedding, resolve_position_options
from qwenvl.models.girf_core import PI_GIRF_MASK_COMPONENTS

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.adaptor.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.adaptor.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad_(True)

        last_n = model_args.llm_trainable_last_n_layers
        if last_n >= 0:
            layers = model.language_model.layers
            if last_n > len(layers):
                raise ValueError(
                    f"llm_trainable_last_n_layers={last_n} exceeds the "
                    f"number of decoder layers ({len(layers)})"
                )
            # Preserve embeddings, output normalization, and lm_head. Only the
            # selected decoder blocks adapt during conservative recovery SFT.
            for parameter in model.language_model.parameters():
                parameter.requires_grad = False
            model.lm_head.requires_grad_(False)
            if last_n:
                for layer in layers[-last_n:]:
                    layer.requires_grad_(True)
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        # Setting ``module.requires_grad`` only creates a Python attribute and
        # does not freeze its parameters.  Pretraining is projector/relation
        # only, so freeze the actual lm_head weights as intended.
        model.lm_head.requires_grad_(False)

    # Universal-relation coefficients are the model-specific thin adapter and
    # must remain trainable even during projector-only multimodal pretraining.
    if model_args.cross_modal_position_type == "universal-relation":
        for name, parameter in model.language_model.named_parameters():
            if name.endswith("relation_coefficients"):
                parameter.requires_grad = True

    # The entity tag is a small multimodal adapter, not part of the frozen
    # vision tower. Keep it trainable in both projector pretraining and SFT.
    if model_args.cross_modal_position_type == "entity-tagged-girf":
        tagger = getattr(model, "visual_entity_tagger", None)
        if tagger is None and hasattr(model, "model"):
            tagger = getattr(model.model, "visual_entity_tagger", None)
        if tagger is None:
            raise ValueError("entity-tagged-girf requires visual_entity_tagger")
        tagger.requires_grad_(True)

def _checkpoint_files_for_component(folder_path, load):
    """Return only checkpoint shards that can contain the requested component."""
    folder = Path(folder_path)
    index_files = sorted(folder.glob("*.safetensors.index.json"))
    if index_files:
        weight_map = json.loads(index_files[0].read_text())["weight_map"]
        if load == "language_model":
            selected = {
                filename for key, filename in weight_map.items()
                if key.startswith("model.") and key != "model.lm_head.weight"
            }
        elif load == "lm_head":
            selected = {
                filename for key, filename in weight_map.items()
                if key == "lm_head.weight"
            }
        elif load == "visual":
            selected = {
                filename for key, filename in weight_map.items()
                if key.startswith("vision_model.")
            }
        else:
            raise ValueError(f"Unknown load type: {load}")
        if selected:
            return [str(folder / filename) for filename in sorted(selected)]
    return [
        str(path) for path in sorted(folder.iterdir())
        if path.suffix in {".safetensors", ".bin"}
    ]


def _reshape_siglip_patch_embedding(weight, target_shape):
    """Convert a SigLIP Conv2D/linear patch kernel to the local Conv3D kernel."""
    if tuple(weight.shape) == tuple(target_shape):
        return weight
    out_channels, in_channels, temporal, height, width = target_shape
    if weight.numel() != out_channels * in_channels * height * width:
        raise ValueError(
            "SigLIP patch embedding has an incompatible shape: "
            f"source={tuple(weight.shape)}, target={tuple(target_shape)}"
        )
    conv2d = weight.reshape(out_channels, height, width, in_channels)
    conv2d = conv2d.permute(0, 3, 1, 2).contiguous()
    return conv2d.unsqueeze(2).repeat(1, 1, temporal, 1, 1) / temporal


def load_weights_from_folder(model, folder_path, load):
    """
    Loads weights from a folder containing safetensors or bin files.
    :param model: The model to load weights into.
    :param folder_path: The path to the folder containing the weights.
    load to cpu to avoid deepspeed
    """
    files = _checkpoint_files_for_component(folder_path, load)
    if not files:
        raise ValueError(f"No checkpoint files found under {folder_path}")
    model_state = model.state_dict()
    model_keys = set(model_state)
    loaded_keys = set()
    for file_path in tqdm(files, desc=f"Loading {load} shards"):
        
        if file_path.endswith('.safetensors'):
            state_dict = load_file(file_path, device="cpu")
        else:
            state_dict = torch.load(file_path, map_location="cpu")

        shard_state = {}
        for key, value in state_dict.items():
            if load == 'visual':
                new_key = key.replace("vision_model.", "")
                if 'embeddings.position_embedding.weight' in new_key:
                    new_key = 'position_embedding.weight'
            elif load == 'language_model':
                new_key = key.replace("model.", "")
            elif load == 'lm_head':
                if key != 'lm_head.weight':
                    continue
                new_key = 'weight'
            else:
                raise ValueError(f'Unknown load type: {load}')

            if new_key in model_keys:
                if load == 'visual' and new_key == 'embeddings.patch_embedding.weight':
                    value = _reshape_siglip_patch_embedding(value, model_state[new_key].shape)
                shard_state[new_key] = value
        if shard_state:
            model.load_state_dict(shard_state, strict=False)
            loaded_keys.update(shard_state)
        del shard_state, state_dict
        gc.collect()

    if not loaded_keys:
        raise ValueError(f'No {load} weights from {folder_path} matched the target module')
    if load == 'language_model' and getattr(model.config, 'llm_backbone', None) == 'qwen3':
        required_qk_norms = {
            f'layers.{index}.self_attn.{name}.weight'
            for index in range(model.config.num_hidden_layers)
            for name in ('q_norm', 'k_norm')
        }
        missing_qk_norms = sorted(required_qk_norms - loaded_keys)
        if missing_qk_norms:
            preview = ', '.join(missing_qk_norms[:4])
            raise ValueError(
                'Qwen3 checkpoint is missing required Q/K-normalization weights: '
                f'{preview}' + (' ...' if len(missing_qk_norms) > 4 else '')
            )
    missing_count = len(model_keys - loaded_keys)
    print(
        f'Loaded {len(loaded_keys)} {load} tensors from {folder_path} '
        f'({missing_count} target tensors intentionally/unexpectedly absent)'
    )
    return model


@contextmanager
def default_construction_dtype(dtype):
    """Construct large scratch models directly in the requested training dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_text_config(llm_path: str, requested_backbone: str) -> Qwen2Config:
    """Normalize Qwen2/Qwen3 configs into the custom multimodal text config."""
    if requested_backbone not in {'auto', 'qwen2', 'qwen3'}:
        raise ValueError('--llm_backbone must be auto, qwen2, or qwen3')
    source = AutoConfig.from_pretrained(llm_path)
    source_type = getattr(source, 'model_type', None)
    detected = 'qwen3' if source_type == 'qwen3' else 'qwen2'
    if requested_backbone != 'auto' and requested_backbone != detected:
        raise ValueError(
            f'--llm_backbone={requested_backbone} conflicts with '
            f'{llm_path} (model_type={source_type!r})'
        )
    if detected == 'qwen3':
        payload = source.to_dict()
        text_config = Qwen2Config(**payload)
        text_config.llm_backbone = 'qwen3'
        text_config.qk_norm = True
        text_config.head_dim = int(getattr(source, 'head_dim',
                                           source.hidden_size // source.num_attention_heads))
        text_config.attention_bias = bool(getattr(source, 'attention_bias', False))
    else:
        text_config = Qwen2Config.from_pretrained(llm_path)
        text_config.llm_backbone = 'qwen2'
        text_config.qk_norm = False
    return text_config


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # LazySupervisedDataset performs percentage sampling and an initial list
    # shuffle before Trainer constructs its sampler. Propagate TrainingArguments
    # data_seed explicitly; otherwise random.Random() would use OS entropy and
    # nominally identical A/B runs would see different example orders.
    data_args.data_seed = training_args.data_seed

    attn_implementation = model_args.attn_implementation.lower()
    if attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError(
            "--attn_implementation must be eager, sdpa, or flash_attention_2"
        )
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "--attn_implementation=flash_attention_2 was requested, but "
                "flash_attn cannot be imported. Use --attn_implementation=eager "
                "or install a wheel matching this Python, Torch, CUDA, and CXX11 ABI."
            ) from exc

    position_embedding_type = normalize_position_embedding(model_args.position_embedding_type)
    try:
        raw_position_options = (
            json.loads(model_args.position_embedding_options)
            if model_args.position_embedding_options
            else {}
        )
    except json.JSONDecodeError as exc:
        raise ValueError("--position_embedding_options must be a valid JSON object") from exc
    position_embedding_options = resolve_position_options(
        position_embedding_type, raw_position_options
    )
    cross_modal_position_type = model_args.cross_modal_position_type.lower()
    if cross_modal_position_type not in {
        "dipe", "native", "universal-relation", "bounded-dipe", "girf", "entity-aware-girf",
        "entity-tagged-girf", "tv-neutral-girf", "pi-girf"
    }:
        raise ValueError(
            "--cross_modal_position_type must be dipe, native, universal-relation, "
            "bounded-dipe, girf, entity-aware-girf, entity-tagged-girf, "
            "tv-neutral-girf, or pi-girf"
        )
    try:
        girf_pi_mask_components = (
            json.loads(model_args.girf_pi_mask_components)
            if model_args.girf_pi_mask_components is not None
            else None
        )
    except json.JSONDecodeError as exc:
        raise ValueError("--girf_pi_mask_components must be a valid JSON list") from exc
    if girf_pi_mask_components is not None:
        if not isinstance(girf_pi_mask_components, list) or not all(
            isinstance(item, str) for item in girf_pi_mask_components
        ):
            raise ValueError("--girf_pi_mask_components must be a JSON list of strings")
        unknown = set(girf_pi_mask_components) - set(PI_GIRF_MASK_COMPONENTS)
        if unknown:
            raise ValueError(
                f"Unknown --girf_pi_mask_components entries: {sorted(unknown)}"
            )
    girf_pi_soft_vv_penalty = float(model_args.girf_pi_soft_vv_penalty)
    if not math.isfinite(girf_pi_soft_vv_penalty) or girf_pi_soft_vv_penalty < 0:
        raise ValueError("--girf_pi_soft_vv_penalty must be finite and non-negative")
    try:
        entity_girf_spatial_scales = json.loads(
            model_args.entity_aware_girf_spatial_scales
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Entity-aware GIRF spatial scales must be a valid JSON list") from exc
    if not isinstance(entity_girf_spatial_scales, list) or len(entity_girf_spatial_scales) != 2:
        raise ValueError("--entity_aware_girf_spatial_scales must be a JSON list of two numbers")
    data_args.position_embedding_type = position_embedding_type
    data_args.position_embedding_options = position_embedding_options

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.model_type = "qwen2.5vl"

    processor = None
    tokenizer = None
    model = None


    # TODO
    # this can be optimized
    if model_args.model_name_or_path and os.path.exists(model_args.model_name_or_path):
        print(f'the initlized model is {model_args.model_name_or_path}...')

        config = Qwen25_SigLIPConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        config.position_embedding_type = position_embedding_type
        config.position_embedding_options = position_embedding_options
        config.text_config.position_embedding_type = position_embedding_type
        config.text_config.position_embedding_options = position_embedding_options
        config.cross_modal_position_type = cross_modal_position_type
        config.text_config.cross_modal_position_type = cross_modal_position_type
        config.attention_chunk_elements = model_args.attention_chunk_elements
        config.text_config.attention_chunk_elements = model_args.attention_chunk_elements
        config.entity_tag_num_embeddings = model_args.entity_tag_num_embeddings
        config.entity_tag_gate_init = model_args.entity_tag_gate_init
        config.text_config.entity_tag_num_embeddings = model_args.entity_tag_num_embeddings
        config.text_config.entity_tag_gate_init = model_args.entity_tag_gate_init
        for target in (config, config.text_config):
            target.entity_aware_girf_version = "v3"
            target.entity_aware_girf_spatial_scales = entity_girf_spatial_scales
            target.girf_ordinal_enabled = model_args.girf_ordinal_enabled
            target.girf_pi_mask_enabled = model_args.girf_pi_mask_enabled
            target.girf_pi_mask_components = girf_pi_mask_components
            target.girf_pi_soft_vv_penalty = girf_pi_soft_vv_penalty
        model = Qwen25_SigLIPForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )

        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        except:
            print(f'{model_args.model_name_or_path} has no processor, load from {model_args.processor_path}')
            processor = None

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # truncation=True,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        print("Loading model from scratch...")
        # config
        text_config = load_text_config(model_args.llm_path, model_args.llm_backbone)
        text_config.position_embedding_type = position_embedding_type
        text_config.position_embedding_options = position_embedding_options
        text_config.cross_modal_position_type = cross_modal_position_type
        text_config.entity_aware_girf_version = "v3"
        text_config.entity_aware_girf_spatial_scales = entity_girf_spatial_scales
        text_config.entity_tag_num_embeddings = model_args.entity_tag_num_embeddings
        text_config.entity_tag_gate_init = model_args.entity_tag_gate_init
        text_config.girf_ordinal_enabled = model_args.girf_ordinal_enabled
        text_config.girf_pi_mask_enabled = model_args.girf_pi_mask_enabled
        text_config.girf_pi_mask_components = girf_pi_mask_components
        text_config.girf_pi_soft_vv_penalty = girf_pi_soft_vv_penalty
        vision_config = Siglip2VisionConfig.from_pretrained(model_args.visual_encoder_path)

        # Keep the outer, text, and vision configs on the same backend.
        text_config._attn_implementation = attn_implementation
        vision_config._attn_implementation = attn_implementation

        del text_config._name_or_path
        del vision_config._name_or_path
        config = Qwen25_SigLIPConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            position_embedding_type=position_embedding_type,
            position_embedding_options=position_embedding_options,
            cross_modal_position_type=cross_modal_position_type,
            attention_chunk_elements=model_args.attention_chunk_elements,
            entity_aware_girf_version="v3",
            entity_aware_girf_spatial_scales=entity_girf_spatial_scales,
            entity_tag_num_embeddings=model_args.entity_tag_num_embeddings,
            entity_tag_gate_init=model_args.entity_tag_gate_init,
            girf_ordinal_enabled=model_args.girf_ordinal_enabled,
            girf_pi_mask_enabled=model_args.girf_pi_mask_enabled,
            girf_pi_mask_components=girf_pi_mask_components,
            girf_pi_soft_vv_penalty=girf_pi_soft_vv_penalty,
        )
        config._attn_implementation = attn_implementation

        construction_dtype = (
            torch.bfloat16 if training_args.bf16
            else torch.float16 if training_args.fp16
            else torch.float32
        )
        rank0_print(f"Constructing the multimodal model in {construction_dtype}")
        with default_construction_dtype(construction_dtype):
            model = Qwen25_SigLIPForConditionalGeneration(config)

        load_weights_from_folder(model.language_model, model_args.llm_path, load="language_model")
        if config.text_config.tie_word_embeddings:
            # Qwen2.5-3B only saves model.embed_tokens.weight.  Once the
            # language model has been loaded, re-apply the declared weight
            # sharing instead of requiring a nonexistent lm_head.weight.
            model.tie_weights()
        else:
            load_weights_from_folder(model.lm_head, model_args.llm_path, load="lm_head")
        load_weights_from_folder(model.visual, model_args.visual_encoder_path, load="visual")

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_path,
            cache_dir=training_args.cache_dir,
            # truncation=True,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    

    if processor is None:
        processor = AutoProcessor.from_pretrained(model_args.processor_path)

    # The data pipeline tokenizes through processor.tokenizer, while Trainer
    # receives the separately loaded tokenizer above. Keep both limits aligned
    # so context-specific recipes enforce the value passed by
    # --model_max_length.
    processor.tokenizer.model_max_length = training_args.model_max_length
    data_args.model_max_length = training_args.model_max_length
    
    # check    
    assert processor is not None, "processor is None"
    assert tokenizer is not None, "tokenizer is None"
    assert model is not None, "model is None"
    rank0_print(
        f"LLM backbone={getattr(model.config.text_config, 'llm_backbone', 'qwen2')}; "
        f"Position embedding: {position_embedding_type}, options={position_embedding_options}; "
        f"cross-modal={cross_modal_position_type}; attention={attn_implementation}; "
        f"PI mask={model_args.girf_pi_mask_enabled}, components={girf_pi_mask_components}, "
        f"soft-VV penalty={girf_pi_soft_vv_penalty}"
    )
    
    # packing
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    # gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            model.model.print_trainable_parameters()
            model.visual.print_trainable_parameters()
    
    # data
    trainer_class = Trainer
    if data_args.niah_manifest:
        from qwenvl.train.niah_training import (
            PairedNIAHTrainer,
            make_paired_niah_module,
        )
        data_module = make_paired_niah_module(processor, data_args, training_args)
        trainer_class = PairedNIAHTrainer
    else:
        data_module = make_supervised_data_module(processor, data_args=data_args)

    processor.save_pretrained(training_args.output_dir)

    # trainer 
    trainer = trainer_class(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
