#!/usr/bin/env python3
"""Static validation for matched-v1 Qwen2-SigLIP evaluation checkpoints."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from qwenvl.models import Qwen25_SigLIPConfig
    from qwenvl.models.modeling_llm_eagar import Qwen25_SigLIPAttention


def weight_files_and_names(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return {model_path / name for name in weight_map.values()}, set(weight_map)

    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        raise ValueError("missing model.safetensors or model.safetensors.index.json")
    from safetensors import safe_open

    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        return {weight_path}, set(handle.keys())


def validate(args):
    model_path = args.model_path.resolve()
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if not (model_path / name).is_file():
            raise ValueError(f"missing checkpoint file: {name}")

    weight_files, weight_names = weight_files_and_names(model_path)
    missing_shards = sorted(str(path) for path in weight_files if not path.is_file())
    if missing_shards:
        raise ValueError(f"missing safetensors shards: {', '.join(missing_shards)}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        config = Qwen25_SigLIPConfig.from_pretrained(model_path)
    text_config = config.text_config
    expected = {
        "llm_backbone": "qwen2",
        "qk_norm": False,
        "attention_bias": True,
        "position_embedding_type": args.position_embedding,
        "cross_modal_position_type": args.cross_modal_position,
    }
    for field, expected_value in expected.items():
        actual = getattr(text_config, field, None)
        if actual != expected_value:
            raise ValueError(f"text_config.{field}={actual!r}, expected {expected_value!r}")
        if field in {"position_embedding_type", "cross_modal_position_type"}:
            outer_actual = getattr(config, field, actual)
            if outer_actual != expected_value:
                raise ValueError(f"config.{field}={outer_actual!r}, expected {expected_value!r}")

    if text_config.head_dim * text_config.num_attention_heads != text_config.hidden_size:
        raise ValueError("head_dim * num_attention_heads does not equal hidden_size")

    required_biases = {
        f"model.language_model.layers.{layer}.self_attn.{projection}.bias"
        for layer in range(text_config.num_hidden_layers)
        for projection in ("q_proj", "k_proj", "v_proj")
    }
    missing_biases = sorted(required_biases - weight_names)
    if missing_biases:
        raise ValueError(
            "missing Qwen2 language attention biases: "
            + ", ".join(missing_biases[:4])
            + (" ..." if len(missing_biases) > 4 else "")
        )
    unexpected_qk_norms = sorted(
        name for name in weight_names
        if name.startswith("model.language_model.layers.")
        and (name.endswith("self_attn.q_norm.weight") or name.endswith("self_attn.k_norm.weight"))
    )
    if unexpected_qk_norms:
        raise ValueError(f"unexpected Qwen3 Q/K RMSNorm tensors: {unexpected_qk_norms[:4]}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with torch.device("meta"):
            attention = Qwen25_SigLIPAttention(text_config, layer_idx=0)
    if attention.q_norm is not None or attention.k_norm is not None:
        raise ValueError("local runtime incorrectly constructs Q/K RMSNorm for Qwen2")
    if any(proj.bias is None for proj in (attention.q_proj, attention.k_proj, attention.v_proj)):
        raise ValueError("local runtime omits required Qwen2 Q/K/V projection biases")

    print(
        f"VALID: {model_path} | backbone=qwen2 layers={text_config.num_hidden_layers} "
        f"position={args.position_embedding} cross_modal={args.cross_modal_position} "
        f"weights={len(weight_names)} shards={len(weight_files)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--position-embedding", required=True)
    parser.add_argument("--cross-modal-position", required=True)
    args = parser.parse_args()
    try:
        validate(args)
    except Exception as exc:
        parser.exit(2, f"INVALID: {args.model_path}: {exc}\n")


if __name__ == "__main__":
    main()
