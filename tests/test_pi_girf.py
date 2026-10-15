import unittest

import torch

from qwenvl.models.configuration_qwen25_siglip import Qwen2Config
from qwenvl.models.girf_core import (
    girf_factorial_options,
    girf_token_coordinates,
    pi_girf_allowed_pairs,
    pi_girf_cross_entity_vv_pairs,
    pi_girf_mask_components,
    pi_girf_soft_vv_penalty,
    pi_girf_text_regions,
    tv_neutral_girf_coordinates,
)
from qwenvl.models.modeling_llm_eagar import Qwen25_SigLIPTextModel


def tiny_config(cross_modal_position_type):
    return Qwen2Config(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=0,
        cross_modal_position_type=cross_modal_position_type,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
    )


class PiGirfCoreTest(unittest.TestCase):
    def test_soft_vv_penalty_and_pair_selection(self):
        config = tiny_config("girf")
        config.girf_pi_soft_vv_penalty = 2.0
        self.assertEqual(pi_girf_soft_vv_penalty(config), 2.0)
        visual = torch.tensor([True, True, False])
        entities = torch.tensor([1, 2, 0])
        pairs = pi_girf_cross_entity_vv_pairs(
            visual, visual, entities, entities
        )
        expected = torch.tensor([
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ])
        torch.testing.assert_close(pairs, expected)
        config.girf_pi_soft_vv_penalty = -1.0
        with self.assertRaises(ValueError):
            pi_girf_soft_vv_penalty(config)

    def test_component_selection_defaults_to_full_mask(self):
        config = tiny_config("girf")
        self.assertEqual(
            pi_girf_mask_components(config),
            ("vv_same_entity", "visual_no_text", "nonpost_text_no_visual", "post_text_no_bridge"),
        )
        config.girf_pi_mask_components = ["post_text_no_bridge", "vv_same_entity"]
        self.assertEqual(
            pi_girf_mask_components(config),
            ("vv_same_entity", "post_text_no_bridge"),
        )

    def test_factorial_switches_are_independent_and_legacy_modes_are_stable(self):
        factorial = tiny_config("girf")
        factorial.girf_ordinal_enabled = False
        factorial.girf_pi_mask_enabled = True
        self.assertEqual(girf_factorial_options(factorial), (False, True))
        self.assertEqual(girf_factorial_options(tiny_config("tv-neutral-girf")), (False, False))
        self.assertEqual(girf_factorial_options(tiny_config("pi-girf")), (False, True))

    def test_tv_neutral_is_single_entity_checkpoint_compatible(self):
        positions = torch.arange(7).view(1, 1, 7).expand(3, 1, 7).clone()
        visual = torch.tensor([[False, True, True, True, False, False, False]])
        expected = girf_token_coordinates(positions, visual)
        actual = tv_neutral_girf_coordinates(positions, visual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tv_neutral_removes_only_ordinal_entity_axis(self):
        positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()
        visual = torch.tensor([[False, True, True, False, True, True, False, False]])
        legacy = girf_token_coordinates(positions, visual)
        neutral = tv_neutral_girf_coordinates(positions, visual)
        self.assertGreater(legacy[..., 1].abs().max().item(), 0.0)
        self.assertTrue(neutral[..., 1].eq(0).all())
        torch.testing.assert_close(neutral[..., [0, 2, 3, 4]], legacy[..., [0, 2, 3, 4]])

    def test_text_regions_support_packed_samples(self):
        visual = torch.tensor([[False, True, True, False, True, False, False,
                                False, True, True, False]])
        valid = torch.ones_like(visual)
        text = ~visual
        sample_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
        prefix, post, bridge = pi_girf_text_regions(
            visual, text, valid, sample_ids
        )
        self.assertEqual(torch.nonzero(prefix[0]).flatten().tolist(), [0, 7])
        self.assertEqual(torch.nonzero(bridge[0]).flatten().tolist(), [3])
        self.assertEqual(torch.nonzero(post[0]).flatten().tolist(), [5, 6, 10])

    def test_pair_mask_uses_entity_equality_not_entity_number(self):
        q_visual = torch.tensor([True, True, False, False])
        k_visual = q_visual.clone()
        q_text = ~q_visual
        k_text = ~k_visual
        entities = torch.tensor([1, 2, 0, 0])
        post = torch.tensor([False, False, False, True])
        bridge = torch.tensor([False, False, True, False])
        first = pi_girf_allowed_pairs(
            q_visual, k_visual, q_text, k_text,
            entities, entities, post, bridge,
        )
        relabeled = torch.tensor([19, 7, 0, 0])
        second = pi_girf_allowed_pairs(
            q_visual, k_visual, q_text, k_text,
            relabeled, relabeled, post, bridge,
        )
        torch.testing.assert_close(first, second)
        self.assertFalse(first[0, 1])  # cross-entity VV is blocked
        self.assertFalse(first[0, 3])  # visual queries cannot absorb text
        self.assertFalse(first[2, 0])  # bridge text cannot absorb vision
        self.assertTrue(first[3, 0])   # post-visual query can read every entity
        self.assertFalse(first[3, 2])  # but not order-carrying bridge text

    def test_pair_mask_clauses_can_be_ablated_independently(self):
        visual = torch.tensor([True, True, False, False])
        text = ~visual
        entities = torch.tensor([1, 2, 0, 0])
        post = torch.tensor([False, False, False, True])
        bridge = torch.tensor([False, False, True, False])
        expected_blocked = {
            "vv_same_entity": (0, 1),
            "visual_no_text": (0, 3),
            "nonpost_text_no_visual": (2, 0),
            "post_text_no_bridge": (3, 2),
        }
        for component, blocked_pair in expected_blocked.items():
            allowed = pi_girf_allowed_pairs(
                visual, visual, text, text, entities, entities, post, bridge,
                components=[component],
            )
            self.assertFalse(allowed[blocked_pair], component)
            for other, pair in expected_blocked.items():
                if other != component:
                    self.assertTrue(allowed[pair], f"{component} also blocked {other}")

    def test_vv_nonpost_combination_is_exact_union_of_its_two_clauses(self):
        visual = torch.tensor([True, True, False, False, False])
        text = ~visual
        entities = torch.tensor([1, 2, 0, 0, 0])
        post = torch.tensor([False, False, False, False, True])
        bridge = torch.tensor([False, False, False, True, False])
        args = (visual, visual, text, text, entities, entities, post, bridge)
        vv = pi_girf_allowed_pairs(*args, components=["vv_same_entity"])
        nonpost = pi_girf_allowed_pairs(
            *args, components=["nonpost_text_no_visual"]
        )
        combo = pi_girf_allowed_pairs(
            *args, components=["vv_same_entity", "nonpost_text_no_visual"]
        )
        torch.testing.assert_close(combo, vv & nonpost)
        self.assertTrue(combo[0, 4])  # visual query can still read post text
        self.assertTrue(combo[4, 3])  # post query can still read bridge text


class PiGirfModelTest(unittest.TestCase):
    def test_soft_vv_penalty_changes_frozen_model_forward(self):
        torch.manual_seed(5)
        base_config = tiny_config("girf")
        base_config.girf_pi_mask_enabled = True
        base_config.girf_pi_mask_components = ["nonpost_text_no_visual"]
        base_config.girf_pi_soft_vv_penalty = 0.0
        base = Qwen25_SigLIPTextModel(base_config).eval()
        soft_config = tiny_config("girf")
        soft_config.girf_pi_mask_enabled = True
        soft_config.girf_pi_mask_components = ["nonpost_text_no_visual"]
        soft_config.girf_pi_soft_vv_penalty = 2.0
        soft = Qwen25_SigLIPTextModel(soft_config).eval()
        soft.load_state_dict(base.state_dict(), strict=True)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        visual = torch.tensor([[False, True, True, False, True, True, False, False]])
        entities = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0]])
        positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()
        kwargs = dict(
            input_ids=ids, position_ids=positions, static_position_ids=positions,
            visual_mask=visual, text_mask=~visual, valid_mask=valid,
            visual_entity_ids=entities,
            visual_temporal_ids=torch.zeros_like(entities),
        )
        with torch.no_grad():
            original = base(**kwargs).last_hidden_state
            penalized = soft(**kwargs).last_hidden_state
        self.assertGreater((original - penalized).abs().max().item(), 1e-6)

    def test_factorial_flags_match_legacy_pi_mode(self):
        torch.manual_seed(7)
        legacy = Qwen25_SigLIPTextModel(tiny_config("pi-girf")).eval()
        config = tiny_config("girf")
        config.girf_ordinal_enabled = False
        config.girf_pi_mask_enabled = True
        factorial = Qwen25_SigLIPTextModel(config).eval()
        factorial.load_state_dict(legacy.state_dict(), strict=True)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        visual = torch.tensor([[False, True, True, False, True, True, False, False]])
        positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()
        entities = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0]])
        kwargs = dict(
            input_ids=ids, position_ids=positions, static_position_ids=positions,
            visual_mask=visual, text_mask=~visual, valid_mask=valid,
            visual_entity_ids=entities,
            visual_temporal_ids=torch.zeros_like(entities),
        )
        with torch.no_grad():
            expected = legacy(**kwargs).last_hidden_state
            actual = factorial(**kwargs).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tv_neutral_matches_legacy_girf_for_one_entity(self):
        torch.manual_seed(11)
        legacy = Qwen25_SigLIPTextModel(tiny_config("girf")).eval()
        neutral = Qwen25_SigLIPTextModel(tiny_config("tv-neutral-girf")).eval()
        neutral.load_state_dict(legacy.state_dict(), strict=True)
        ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        visual = torch.tensor([[False, True, True, False, False, False]])
        positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
        entities = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        kwargs = dict(
            input_ids=ids,
            position_ids=positions,
            static_position_ids=positions,
            visual_mask=visual,
            text_mask=~visual,
            valid_mask=valid,
            visual_entity_ids=entities,
            visual_temporal_ids=torch.zeros_like(entities),
        )
        with torch.no_grad():
            expected = legacy(**kwargs).last_hidden_state
            actual = neutral(**kwargs).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_pi_girf_is_invariant_to_equal_length_entity_swap(self):
        torch.manual_seed(17)
        model = Qwen25_SigLIPTextModel(tiny_config("pi-girf")).eval()
        length = 8
        embeddings = torch.randn(1, length, 32)
        swapped = embeddings.clone()
        swapped[:, 1:3] = embeddings[:, 4:6]
        swapped[:, 4:6] = embeddings[:, 1:3]
        valid = torch.ones(1, length, dtype=torch.bool)
        visual = torch.tensor([[False, True, True, False, True, True, False, False]])
        entities = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0]])
        positions = torch.arange(length).view(1, 1, length).expand(3, 1, length).clone()
        # Both entities live in the same local visual coordinate frame.
        for start in (1, 4):
            positions[:, 0, start : start + 2] = torch.tensor(
                [[0, 0], [0, 1], [0, 1]]
            )
        kwargs = dict(
            position_ids=positions,
            static_position_ids=positions,
            visual_mask=visual,
            text_mask=~visual,
            valid_mask=valid,
            visual_entity_ids=entities,
            visual_temporal_ids=torch.zeros_like(entities),
        )
        with torch.no_grad():
            original = model(inputs_embeds=embeddings, **kwargs).last_hidden_state
            permuted = model(inputs_embeds=swapped, **kwargs).last_hidden_state
        torch.testing.assert_close(
            original[:, -2:], permuted[:, -2:], rtol=2e-5, atol=2e-6
        )


if __name__ == "__main__":
    unittest.main()
