import unittest

import torch

from qwenvl.models.configuration_qwen25_siglip import Qwen2Config
from qwenvl.models.girf_core import (
    apply_entity_aware_girf_rotary,
    entity_graph_cross_entropy,
    entity_aware_girf_coordinates,
    entity_aware_pair_layout,
    implicit_entity_graph,
    resolve_entity_aware_axis_counts,
)
from qwenvl.models.modeling_llm_eagar import (
    Qwen25_SigLIPAttention,
    Qwen25_SigLIPTextModel,
    entity_aware_block_attention_forward,
)
from qwenvl.models.modeling_qwen25_siglip import (
    VisualEntityTagger,
    build_visual_entity_metadata,
    should_build_visual_entity_metadata,
)


def multi_image_fixture(prefix=2):
    # text, 2x2 image, text separator, identical 2x2 image, trailing text
    length = prefix + 4 + 1 + 4 + 2
    positions = torch.arange(length).view(1, 1, length).expand(3, 1, length).clone()
    visual = torch.zeros(1, length, dtype=torch.bool)
    first = torch.arange(prefix, prefix + 4)
    second = torch.arange(prefix + 5, prefix + 9)
    for indices in (first, second):
        visual[0, indices] = True
        positions[0, 0, indices] = 0
        positions[1, 0, indices] = torch.tensor([0, 0, 1, 1])
        positions[2, 0, indices] = torch.tensor([0, 1, 0, 1])
    return positions, visual


class EntityAwareGirfCoreTest(unittest.TestCase):
    def test_entity_tagger_is_exactly_zero_at_initialization(self):
        torch.manual_seed(0)
        tagger = VisualEntityTagger(hidden_size=6, num_entities=4, gate_init=0.0)
        hidden = torch.randn(1, 6, 6)
        entities = torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0, 3.0]])
        actual = tagger(hidden, entities)
        torch.testing.assert_close(actual, hidden, atol=0.0, rtol=0.0)

        # The zero gate itself must receive a learning signal. Embedding rows
        # begin receiving gradients after the gate moves away from zero.
        tagger.embedding.weight.data.fill_(1.0)
        tagger(hidden, entities).sum().backward()
        self.assertGreater(abs(tagger.gate.grad.item()), 0.0)

    def test_entity_tagger_changes_only_later_entities(self):
        tagger = VisualEntityTagger(hidden_size=3, num_entities=4, gate_init=1.0)
        tagger.embedding.weight.data.fill_(2.0)
        hidden = torch.zeros(1, 5, 3)
        entities = torch.tensor([[0.0, 1.0, 2.0, 2.0, 3.0]])
        actual = tagger(hidden, entities)
        torch.testing.assert_close(actual[:, :2], hidden[:, :2], atol=0.0, rtol=0.0)
        expected = torch.full_like(actual[:, 2:], 2.0 * torch.tanh(torch.tensor(1.0)))
        torch.testing.assert_close(actual[:, 2:], expected)

    def test_entity_tagger_rejects_unconfigured_entity_count(self):
        tagger = VisualEntityTagger(hidden_size=2, num_entities=2)
        with self.assertRaisesRegex(ValueError, "exceeds configured maximum"):
            tagger(torch.zeros(1, 1, 2), torch.tensor([[3.0]]))

    def test_cached_decode_does_not_rebuild_visual_entity_metadata(self):
        # Generation retains image_grid_thw after prefill but clears
        # pixel_values.  The model must not try to match that stale grid row
        # against a decode-only token slice with no image placeholders.
        ids = torch.tensor([[42]])
        grid = torch.tensor([[1, 4, 4]])
        self.assertFalse(
            should_build_visual_entity_metadata(ids, None, None, grid, None)
        )
        self.assertTrue(
            should_build_visual_entity_metadata(
                ids, torch.zeros(16, 8), None, grid, None
            )
        )

    def test_v3_config_is_self_describing_and_legacy_is_detected(self):
        config = Qwen2Config(cross_modal_position_type="entity-aware-girf")
        serialized = config.to_dict()
        self.assertEqual(serialized["entity_aware_girf_version"], "v3")
        for key in (
            "entity_aware_girf_spatial_scales",
            "entity_aware_girf_axis_weights",
            "entity_aware_girf_band_weights",
            "entity_aware_girf_long_range_bands",
            "entity_aware_girf_spatial_bands",
        ):
            self.assertIn(key, serialized)
        legacy = Qwen2Config(
            cross_modal_position_type="entity-aware-girf",
            entity_aware_girf_sections=None,
            entity_aware_girf_scales=[8, 4, 8, 16, 16],
        )
        self.assertEqual(legacy.entity_aware_girf_version, "legacy")

        tagged = Qwen2Config(
            cross_modal_position_type="entity-tagged-girf",
            entity_tag_num_embeddings=24,
            entity_tag_gate_init=0.0,
        )
        tagged_dict = tagged.to_dict()
        self.assertEqual(tagged_dict["entity_tag_num_embeddings"], 24)
        self.assertEqual(tagged_dict["entity_tag_gate_init"], 0.0)

    def test_processor_video_runs_share_one_explicit_entity(self):
        # One processor video with T=2, four merged tokens per temporal run.
        ids = torch.tensor([[10, 151656, 151656, 151656, 151656, 20, 21,
                             151656, 151656, 151656, 151656, 30]])
        entities, temporal = build_visual_entity_metadata(
            ids, None, torch.tensor([[2, 4, 4]]), 151655, 151656, 2
        )
        visual = ids == 151656
        self.assertEqual(entities[visual].tolist(), [1.0] * 8)
        self.assertEqual(temporal[visual].tolist(), [0.0] * 4 + [1.0] * 4)

        positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1).clone()
        coordinates = entity_aware_girf_coordinates(
            positions, visual, visual_entity_ids=entities, visual_temporal_ids=temporal
        )
        self.assertEqual(coordinates[0, visual[0], 1].unique().tolist(), [1.0])
        self.assertEqual(coordinates[0, visual[0], 2].unique().tolist(), [0.0, 1.0])

    def test_interleaved_images_and_videos_consume_media_grids_in_order(self):
        # image, two-time video, image, one-time video
        i, v, t = 151655, 151656, 99
        ids = torch.tensor([[
            i, i, i, i, t,
            v, v, v, v, t, v, v, v, v, t,
            i, i, i, i, t,
            v, v, v, v,
        ]])
        entities, temporal = build_visual_entity_metadata(
            ids,
            torch.tensor([[1, 4, 4], [1, 4, 4]]),
            torch.tensor([[2, 4, 4], [1, 4, 4]]),
            i, v, 2,
        )
        runs = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)]
        expected_entities = [1, 2, 2, 3, 4]
        expected_times = [0, 0, 1, 0, 0]
        for (start, end), entity, time in zip(runs, expected_entities, expected_times):
            self.assertTrue(entities[0, start:end].eq(entity).all())
            self.assertTrue(temporal[0, start:end].eq(time).all())

    def test_packed_media_metadata_resets_entity_ids(self):
        v, t = 151656, 17
        ids = torch.tensor([[v, v, v, v, t, v, v, v, v]])
        entities, temporal = build_visual_entity_metadata(
            ids, None, torch.tensor([[1, 4, 4], [1, 4, 4]]),
            151655, v, 2, packed_seq_lens=torch.tensor([0, 5, 9]),
        )
        self.assertEqual(entities[0, :4].unique().tolist(), [1.0])
        self.assertEqual(entities[0, 5:].unique().tolist(), [1.0])
        self.assertEqual(temporal[ids == v].unique().tolist(), [0.0])

    def test_entities_are_distinct_and_have_independent_geometry(self):
        positions, visual = multi_image_fixture()
        coordinates = entity_aware_girf_coordinates(positions, visual)
        entities = coordinates[0, visual[0], 1]
        self.assertEqual(entities.tolist(), [1.0] * 4 + [2.0] * 4)
        torch.testing.assert_close(coordinates[0, visual[0], 2:][0:4], coordinates[0, visual[0], 2:][4:8])

    def test_text_coordinate_is_causal_and_prefix_invariant(self):
        short_pos, short_visual = multi_image_fixture(prefix=2)
        long_pos, long_visual = multi_image_fixture(prefix=17)
        short = entity_aware_girf_coordinates(short_pos, short_visual)
        long = entity_aware_girf_coordinates(long_pos, long_visual)
        # Text immediately following image 2 is the same structural run.
        torch.testing.assert_close(short[0, -2:, 0], long[0, -2:, 0])
        self.assertLess(short[0, -2, 0].item(), short[0, -1, 0].item())

    def test_packed_samples_reset_entity_identity(self):
        positions = torch.arange(10).view(1, 1, 10).expand(3, 1, 10).clone()
        visual = torch.tensor([[False, True, True, False, False, False, True, True, False, False]])
        sample_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        coordinates = entity_aware_girf_coordinates(positions, visual, sample_ids=sample_ids)
        self.assertEqual(coordinates[0, 1, 1].item(), 1.0)
        self.assertEqual(coordinates[0, 6, 1].item(), 1.0)

    def test_rotation_preserves_norm_and_uses_all_pairs(self):
        torch.manual_seed(0)
        x = torch.randn(1, 2, 7, 18)
        coordinates = torch.randn(1, 7, 5)
        coordinates[..., 1].abs_()
        rotated = apply_entity_aware_girf_rotary(x, coordinates, (16.0, 16.0))
        torch.testing.assert_close(x.float().square().sum(-1), rotated.float().square().sum(-1))
        self.assertEqual(sum(resolve_entity_aware_axis_counts(9)), 9)
        axes, bands, frequencies = entity_aware_pair_layout(9)
        self.assertEqual(axes.numel(), 9)
        self.assertEqual(bands.shape, axes.shape)
        self.assertTrue(torch.all((bands >= 0) & (bands <= 2)))
        self.assertTrue(torch.all(frequencies > 0))

    def test_implicit_graph_supports_multi_target_supervision(self):
        logits = torch.tensor([[5.0, 4.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0]])
        graph, segments, entities = implicit_entity_graph(
            logits, torch.tensor([0, 0]), torch.tensor([0, 0, 1, 1])
        )
        self.assertEqual(segments.tolist(), [0])
        self.assertEqual(entities.tolist(), [0, 1])
        self.assertGreater(graph[0, 0].item(), graph[0, 1].item())
        loss = entity_graph_cross_entropy(graph, torch.tensor([[1.0, 0.0]]))
        self.assertTrue(torch.isfinite(loss))


class EntityAwareBlockAttentionTest(unittest.TestCase):
    def test_entity_tagged_girf_training_restores_padded_batch(self):
        torch.manual_seed(4)
        config = Qwen2Config(
            vocab_size=50, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128, attention_dropout=0.0,
            use_cache=False, cross_modal_position_type="entity-tagged-girf",
            rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
        )
        model = Qwen25_SigLIPTextModel(config).train()
        input_ids = torch.tensor(
            [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]]
        )
        valid = torch.tensor(
            [[True, True, True, True, True, True],
             [True, True, True, True, False, False]]
        )
        visual = torch.tensor(
            [[False, True, True, False, False, False],
             [False, True, False, False, False, False]]
        )
        text = valid & ~visual
        positions = torch.arange(6).view(1, 1, 6).expand(3, 2, 6).clone()
        positions[1, 0, 1:3] = torch.tensor([0, 1])
        positions[2, 0, 1:3] = torch.tensor([0, 1])
        positions[1:, 1, 1] = 0

        output = model(
            input_ids=input_ids,
            position_ids=positions,
            static_position_ids=positions.clone(),
            visual_mask=visual,
            text_mask=text,
            valid_mask=valid,
            use_cache=False,
        ).last_hidden_state

        self.assertEqual(tuple(output.shape), (2, 6, 32))
        output[valid].square().mean().backward()
        gradient = model.embed_tokens.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_entity_tagged_girf_preserves_old_girf_position_path(self):
        torch.manual_seed(3)
        common = dict(
            vocab_size=50, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128, attention_dropout=0.0,
            use_cache=False,
            rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
        )
        old_model = Qwen25_SigLIPTextModel(
            Qwen2Config(**common, cross_modal_position_type="girf")
        ).eval()
        tagged_model = Qwen25_SigLIPTextModel(
            Qwen2Config(**common, cross_modal_position_type="entity-tagged-girf")
        ).eval()
        tagged_model.load_state_dict(old_model.state_dict())

        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
        visual = torch.tensor([[False, True, True, False, True, False]])
        valid = torch.ones_like(visual)
        positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
        positions[1, 0, visual[0]] = torch.tensor([0, 1, 0])
        positions[2, 0, visual[0]] = torch.tensor([0, 1, 0])
        kwargs = dict(
            input_ids=input_ids,
            position_ids=positions,
            static_position_ids=positions.clone(),
            visual_mask=visual,
            text_mask=~visual,
            valid_mask=valid,
            use_cache=False,
        )
        with torch.no_grad():
            old_output = old_model(**kwargs).last_hidden_state
            tagged_output = tagged_model(**kwargs).last_hidden_state
        torch.testing.assert_close(tagged_output, old_output, atol=0.0, rtol=0.0)

    def test_block_attention_matches_dense_mixed_reference(self):
        torch.manual_seed(1)
        config = Qwen2Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            cross_modal_position_type="entity-aware-girf", attention_dropout=0.0,
        )
        module = Qwen25_SigLIPAttention(config, 0).eval()
        dynamic_q, girf_q = torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16)
        dynamic_k, girf_k = torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16)
        value = torch.randn(1, 2, 6, 16)
        visual = torch.tensor([False, True, True, False, True, False])
        pos = torch.arange(6)
        metadata = {
            "q_pos": pos, "k_pos": pos,
            "q_batch": torch.zeros(6, dtype=torch.int32),
            "k_batch": torch.zeros(6, dtype=torch.int32),
            "q_visual": visual, "k_visual": visual,
            "q_text": ~visual, "k_text": ~visual,
        }
        actual, _ = entity_aware_block_attention_forward(
            module, dynamic_q, girf_q, dynamic_k, girf_k, value,
            scaling=16 ** -0.5, relation_metadata=metadata, chunk_size=2,
        )
        same = visual[:, None] == visual[None, :]
        dynamic_logits = dynamic_q @ dynamic_k.transpose(-1, -2)
        girf_logits = girf_q @ girf_k.transpose(-1, -2)
        logits = torch.where(same[None, None], dynamic_logits, girf_logits) * (16 ** -0.5)
        logits.masked_fill_(~(pos[:, None] >= pos[None, :])[None, None], torch.finfo(logits.dtype).min)
        expected = (torch.softmax(logits, -1, dtype=torch.float32).to(value.dtype) @ value)
        expected = expected.transpose(1, 2).contiguous()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        # Scatter-based block assembly must retain gradients to all four paths.
        dynamic_q.requires_grad_()
        girf_q.requires_grad_()
        differentiable, _ = entity_aware_block_attention_forward(
            module, dynamic_q, girf_q, dynamic_k, girf_k, value,
            scaling=16 ** -0.5, relation_metadata=metadata, chunk_size=2,
        )
        differentiable.square().mean().backward()
        self.assertGreater(dynamic_q.grad.norm().item(), 0.0)
        self.assertGreater(girf_q.grad.norm().item(), 0.0)

    def test_kv_cache_matches_full_forward(self):
        torch.manual_seed(2)
        config = Qwen2Config(
            vocab_size=50, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
            cross_modal_position_type="entity-aware-girf", attention_dropout=0.0,
            use_cache=True, rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
        )
        model = Qwen25_SigLIPTextModel(config).eval()
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
        visual = torch.tensor([[False, True, True, False, False, False]])
        text, valid = ~visual, torch.ones_like(visual)
        positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
        positions[1, 0, 1:3] = torch.tensor([0, 1])
        positions[2, 0, 1:3] = torch.tensor([0, 1])
        static_positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
        with torch.no_grad():
            full = model(
                input_ids=input_ids, position_ids=positions, static_position_ids=static_positions,
                visual_mask=visual, text_mask=text, valid_mask=valid, use_cache=False,
            ).last_hidden_state
            prefill = model(
                input_ids=input_ids[:, :5], position_ids=positions[:, :, :5],
                static_position_ids=static_positions[:, :, :5], visual_mask=visual[:, :5],
                text_mask=text[:, :5], valid_mask=valid[:, :5], use_cache=True,
            )
            decoded = model(
                input_ids=input_ids[:, 5:], position_ids=positions, static_position_ids=static_positions,
                visual_mask=visual, text_mask=text, valid_mask=valid,
                past_key_values=prefill.past_key_values, cache_position=torch.tensor([5]), use_cache=True,
            ).last_hidden_state
        torch.testing.assert_close(full[:, -1], decoded[:, -1], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
