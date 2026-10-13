"""Backward-compatible wrapper around the unified multimodal RoPE registry."""

from .multimodal_rope import get_multimodal_position_ids


def get_rope_index_3(*args, **kwargs):
    return get_multimodal_position_ids(*args, **kwargs)
