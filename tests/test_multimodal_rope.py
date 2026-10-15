import pytest
import torch

from qwenvl.data.multimodal_rope import (
    SUPPORTED_POSITION_EMBEDDINGS,
    get_multimodal_position_ids,
    resolve_position_options,
)
from qwenvl.models.configuration_qwen25_siglip import Qwen2Config
from qwenvl.models.modeling_llm_eagar import (
    Qwen25_SigLIPRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)


@pytest.fixture
def image_inputs():
    # Two text/special tokens, four merged visual tokens, vision_end, two text tokens.
    input_ids = torch.tensor(
        [[100, 151652, 151655, 151655, 151655, 151655, 151653, 20, 21]]
    )
    image_grid_thw = torch.tensor([[1, 4, 4]])
    return input_ids, image_grid_thw


@pytest.mark.parametrize("name", SUPPORTED_POSITION_EMBEDDINGS)
def test_all_position_designs_return_dipe_positions(name, image_inputs):
    input_ids, image_grid_thw = image_inputs
    dynamic, anchored, deltas = get_multimodal_position_ids(
        spatial_merge_size=2,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        position_embedding_type=name,
        position_embedding_options=resolve_position_options(name),
    )

    assert dynamic.shape == anchored.shape == (3, 1, input_ids.shape[1])
    assert deltas.shape == (1, 1)
    assert torch.isfinite(dynamic).all()
    assert torch.isfinite(anchored).all()


@pytest.mark.parametrize("name", SUPPORTED_POSITION_EMBEDDINGS)
def test_all_frequency_allocations_rotate_expected_shape(name, image_inputs):
    input_ids, image_grid_thw = image_inputs
    options = resolve_position_options(name)
    positions, _, _ = get_multimodal_position_ids(
        spatial_merge_size=2,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        position_embedding_type=name,
        position_embedding_options=options,
    )
    config = Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_scaling={"rope_type": "default", "mrope_section": [16, 24, 24]},
        position_embedding_type=name,
        position_embedding_options=options,
    )
    rotary = Qwen25_SigLIPRotaryEmbedding(config)
    cos, sin = rotary(torch.randn(1, input_ids.shape[1], 128), positions)

    heads = 8 if name == "mhrope" else 1
    kv_heads = 8 if name == "mhrope" else 1
    query = torch.randn(1, heads, input_ids.shape[1], 128)
    output = apply_multimodal_rotary_pos_emb(
        query,
        cos,
        sin,
        options.get("mrope_section", [16, 24, 24]),
        name,
        options,
        kv_heads,
        heads,
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_vanilla_rope_uses_one_contiguous_sequence_axis(image_inputs):
    input_ids, image_grid_thw = image_inputs
    dynamic, anchored, deltas = get_multimodal_position_ids(
        spatial_merge_size=2,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        position_embedding_type="vanilla-rope",
    )

    expected = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    expected_anchors = torch.tensor(
        [[[0, 0, 2, 2, 2, 2, 6, 6, 6]]]
    ).expand(3, -1, -1)
    assert torch.equal(dynamic, expected)
    assert torch.equal(anchored, expected_anchors)
    assert torch.equal(deltas, torch.zeros_like(deltas))


def test_grape_learned_modes_are_rejected():
    with pytest.raises(ValueError, match="canonical"):
        resolve_position_options("grape", {"grape_mode": "axis"})


def test_mhrope_expands_qwen25_gqa_keys(image_inputs):
    input_ids, image_grid_thw = image_inputs
    options = resolve_position_options("mhrope")
    positions, _, _ = get_multimodal_position_ids(
        2,
        input_ids,
        image_grid_thw=image_grid_thw,
        position_embedding_type="mhrope",
        position_embedding_options=options,
    )
    config = Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
        position_embedding_type="mhrope",
        position_embedding_options=options,
    )
    cos, sin = Qwen25_SigLIPRotaryEmbedding(config)(
        torch.randn(1, input_ids.shape[1], 128), positions
    )
    key = torch.randn(1, 2, input_ids.shape[1], 8)
    # Match the rotary head dimension used by this synthetic key.
    cos, sin = cos[..., :8], sin[..., :8]
    output = apply_multimodal_rotary_pos_emb(
        key, cos, sin, [2, 3, 3], "mhrope", options, 2, 16
    )
    assert output.shape == (1, 16, input_ids.shape[1], 8)
