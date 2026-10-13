#!/usr/bin/env python3
"""Static validation for the custom Qwen3-SigLIP evaluation checkpoints."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
from safetensors import safe_open

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from qwenvl.models import (
        Qwen25_SigLIPConfig,
        Qwen25_SigLIPForConditionalGeneration,
    )
    from qwenvl.models.modeling_llm_eagar import Qwen25_SigLIPAttention


def weight_files_and_names(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        files = {model_path / filename for filename in weight_map.values()}
        return files, set(weight_map)

    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        temporary = model_path / "model.safetensors.temp"
        if temporary.exists():
            raise ValueError(
                "checkpoint is incomplete: model.safetensors is absent and "
                f"model.safetensors.temp exists ({temporary.stat().st_size} bytes)"
            )
        raise ValueError("missing model.safetensors or model.safetensors.index.json")
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        return {weight_path}, set(handle.keys())


def validate(args):
    model_path = args.model_path.resolve()
    processor_path = (args.processor_path or model_path).resolve()
    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    if missing_files:
        raise ValueError(f"missing checkpoint files: {', '.join(missing_files)}")
    processor_required = ["preprocessor_config.json", "tokenizer_config.json"]
    missing_processor = [
        name for name in processor_required if not (processor_path / name).is_file()
    ]
    if missing_processor:
        raise ValueError(
            f"missing processor files under {processor_path}: {', '.join(missing_processor)}"
        )

    weight_files, weight_names = weight_files_and_names(model_path)
    missing_shards = sorted(str(path) for path in weight_files if not path.is_file())
    if missing_shards:
        raise ValueError(f"missing safetensors shards: {', '.join(missing_shards)}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        config = Qwen25_SigLIPConfig.from_pretrained(model_path)
    text_config = config.text_config
    expected = {
        "llm_backbone": "qwen3",
        "qk_norm": True,
        "attention_bias": False,
        "position_embedding_type": args.position_embedding,
        "cross_modal_position_type": args.cross_modal_position,
    }
    for field, expected_value in expected.items():
        actual = getattr(text_config, field, None)
        if actual != expected_value:
            raise ValueError(f"text_config.{field}={actual!r}, expected {expected_value!r}")
        outer_actual = getattr(config, field, actual)
        if field in {"position_embedding_type", "cross_modal_position_type"} and outer_actual != expected_value:
            raise ValueError(f"config.{field}={outer_actual!r}, expected {expected_value!r}")

    if text_config.head_dim * text_config.num_attention_heads != text_config.hidden_size:
        raise ValueError("head_dim * num_attention_heads does not equal hidden_size")

    required_qk_norms = {
        f"model.language_model.layers.{layer}.{norm}.weight"
        for layer in range(text_config.num_hidden_layers)
        for norm in ("self_attn.q_norm", "self_attn.k_norm")
    }
    missing_qk_norms = sorted(required_qk_norms - weight_names)
    if missing_qk_norms:
        raise ValueError(
            "missing Qwen3 Q/K RMSNorm tensors: "
            + ", ".join(missing_qk_norms[:4])
            + (" ..." if len(missing_qk_norms) > 4 else "")
        )

    language_biases = sorted(
        name for name in weight_names
        if name.startswith("model.language_model.layers.")
        and name.endswith(("self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"))
    )
    if language_biases:
        raise ValueError(f"unexpected Qwen3 language attention biases: {language_biases[:4]}")

    if args.require_exact_weight_set:
        # Compare against the complete local runtime on the meta device. This
        # catches missing or stale tensors without allocating model storage.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with torch.device("meta"):
                meta_model = Qwen25_SigLIPForConditionalGeneration(config)
        expected_weight_names = set(meta_model.state_dict())
        missing_weights = sorted(expected_weight_names - weight_names)
        unexpected_weights = sorted(weight_names - expected_weight_names)
        if missing_weights or unexpected_weights:
            details = []
            if missing_weights:
                details.append(
                    "missing=" + ", ".join(missing_weights[:4])
                    + (" ..." if len(missing_weights) > 4 else "")
                )
            if unexpected_weights:
                details.append(
                    "unexpected=" + ", ".join(unexpected_weights[:4])
                    + (" ..." if len(unexpected_weights) > 4 else "")
                )
            raise ValueError("checkpoint/runtime weight set mismatch: " + "; ".join(details))

    # Instantiate one attention layer on the meta device. This checks the local
    # runtime implementation without allocating model weights or using a GPU.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with torch.device("meta"):
            attention = Qwen25_SigLIPAttention(text_config, layer_idx=0)
    if attention.q_norm is None or attention.k_norm is None:
        raise ValueError("local runtime does not construct Qwen3 Q/K RMSNorm")
    if any(proj.bias is not None for proj in (attention.q_proj, attention.k_proj, attention.v_proj)):
        raise ValueError("local runtime constructs invalid Qwen3 Q/K/V projection biases")

    print(
        f"VALID: {model_path} | layers={text_config.num_hidden_layers} "
        f"head_dim={text_config.head_dim} position={args.position_embedding} "
        f"cross_modal={args.cross_modal_position} weights={len(weight_names)}"
        + (" exact_weight_set=true" if args.require_exact_weight_set else "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, default=None)
    parser.add_argument("--position-embedding", required=True)
    parser.add_argument("--cross-modal-position", required=True)
    parser.add_argument("--require-exact-weight-set", action="store_true")
    args = parser.parse_args()
    try:
        validate(args)
    except Exception as exc:
        parser.exit(2, f"INVALID: {args.model_path}: {exc}\n")


if __name__ == "__main__":
    main()
