from dataclasses import dataclass
import math
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers import Qwen2RMSNorm
from transformers import Qwen2Config
from .modeling_utils import Qwen25_SigLIPPreTrainedModel, rotate_half
from .girf_core import (
    apply_entity_aware_girf_rotary,
    apply_girf_rotary,
    entity_aware_girf_coordinates,
    girf_factorial_options,
    girf_token_coordinates,
    pi_girf_allowed_pairs,
    pi_girf_cross_entity_vv_pairs,
    pi_girf_mask_components,
    pi_girf_soft_vv_penalty,
    pi_girf_text_regions,
    tv_neutral_girf_coordinates,
)
from qwenvl.data.multimodal_rope import normalize_position_embedding, resolve_position_options

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def normalized_visual_coordinates(position_ids, visual_mask):
    """Normalize each contiguous visual segment to a model-independent frame."""
    output = []
    for axis in (1, 2):
        values = position_ids[axis].float()
        normalized = torch.zeros_like(values)
        for batch_idx in range(values.shape[0]):
            indices = torch.nonzero(visual_mask[batch_idx], as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            boundaries = torch.nonzero(indices[1:] != indices[:-1] + 1, as_tuple=False).flatten() + 1
            for segment in torch.tensor_split(indices, boundaries.cpu().tolist()):
                selected = values[batch_idx, segment]
                span = selected.max() - selected.min()
                if float(span) > 0:
                    normalized[batch_idx, segment] = 2 * (
                        selected - selected.min()
                    ) / span - 1
        output.append(normalized)
    return output


def entity_aware_block_attention_forward(
    module: nn.Module,
    dynamic_query: torch.Tensor,
    girf_query: torch.Tensor,
    dynamic_key: torch.Tensor,
    girf_key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    relation_metadata: dict,
    dropout: float = 0.0,
    chunk_size: int = 128,
    output_attentions: bool = False,
):
    """Exact TT/VV MRoPE + TV/VT Entity-GIRF block attention.

    Every query-key pair is multiplied exactly once.  We still normalize over
    the union of both modality blocks, so this is mathematically equivalent to
    forming a single dense mixed-position logit matrix and applying one global
    softmax.  Query chunking bounds peak memory without changing the result.
    """
    if dynamic_key.shape[1] != module.num_heads:
        dynamic_key = repeat_kv(dynamic_key, module.num_key_value_groups)
        girf_key = repeat_kv(girf_key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    bsz, num_heads, q_len, _ = dynamic_query.shape
    kv_len = dynamic_key.shape[-2]
    max_score_elements = int(getattr(module.config, "attention_chunk_elements", 8_388_608))
    memory_bound = max(1, max_score_elements // max(1, bsz * num_heads * kv_len))
    chunk_size = max(1, min(chunk_size, q_len, memory_bound))

    k_visual = relation_metadata["k_visual"].bool()
    k_text = relation_metadata["k_text"].bool()
    output_chunks, attention_chunks = [], []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        length = end - start
        logits = torch.full(
            (bsz, num_heads, length, kv_len),
            torch.finfo(dynamic_query.dtype).min,
            dtype=dynamic_query.dtype,
            device=dynamic_query.device,
        )
        q_visual = relation_metadata["q_visual"][start:end].bool()
        q_text = relation_metadata["q_text"][start:end].bool()

        def fill_block(q_mask, k_mask, q_states, k_states):
            q_idx = torch.nonzero(q_mask, as_tuple=False).flatten()
            k_idx = torch.nonzero(k_mask, as_tuple=False).flatten()
            if q_idx.numel() == 0 or k_idx.numel() == 0:
                return
            scores = torch.matmul(
                q_states[:, :, start:end, :][:, :, q_idx, :],
                k_states[:, :, k_idx, :].transpose(-1, -2),
            ) * scaling
            logits[:, :, q_idx[:, None], k_idx[None, :]] = scores

        fill_block(q_text, k_text, dynamic_query, dynamic_key)       # TT
        fill_block(q_visual, k_visual, dynamic_query, dynamic_key)   # VV
        fill_block(q_text, k_visual, girf_query, girf_key)           # TV
        fill_block(q_visual, k_text, girf_query, girf_key)           # VT

        keep = (
            (relation_metadata["q_batch"][start:end, None] == relation_metadata["k_batch"][None, :])
            & (relation_metadata["q_pos"][start:end, None] >= relation_metadata["k_pos"][None, :])
        )
        logits.masked_fill_(~keep[None, None], torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(value.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        output_chunks.append(torch.matmul(weights, value))
        if output_attentions:
            attention_chunks.append(weights.mean(dim=1, keepdim=True).detach().to(torch.float16).cpu())

    output = torch.cat(output_chunks, dim=2).transpose(1, 2).contiguous()
    attention = torch.cat(attention_chunks, dim=2) if output_attentions else None
    return output, attention


def decoupled_eager_attention_forward(
    module: nn.Module,
    static_query: torch.Tensor,
    dynamic_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    inter_attention_mask: Optional[torch.Tensor],
    intra_attention_mask: Optional[torch.Tensor],
    scaling: float,
    inter_key: Optional[torch.Tensor] = None,
    content_query: Optional[torch.Tensor] = None,
    content_key: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    chunk_size: int = 128,
    output_attentions: bool = False,
    relation_metadata: Optional[dict] = None,
    **kwargs,
):
    if key.shape[1] != module.num_heads:
        key = repeat_kv(key, module.num_key_value_groups)
    if inter_key is None:
        inter_key = key
    elif inter_key.shape[1] != module.num_heads:
        inter_key = repeat_kv(inter_key, module.num_key_value_groups)
    if content_key is not None and content_key.shape[1] != module.num_heads:
        content_key = repeat_kv(content_key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    bsz, num_heads, q_len, head_dim = static_query.shape

    if inter_attention_mask is not None and inter_attention_mask.dim() == 3:
        inter_attention_mask = inter_attention_mask.unsqueeze(1)
    if intra_attention_mask is not None and intra_attention_mask.dim() == 3:
        intra_attention_mask = intra_attention_mask.unsqueeze(1)

    kv_len = key.shape[-2]
    max_score_elements = int(getattr(module.config, "attention_chunk_elements", 8_388_608))
    memory_bound = max(1, max_score_elements // max(1, bsz * num_heads * kv_len))
    chunk_size = max(1, min(chunk_size, q_len, memory_bound))

    output_chunks = []
    attn_weights_chunks = []

    for i in range(0, q_len, chunk_size):
        end = min(i + chunk_size, q_len)

        q_static_chunk = static_query[:, :, i:end, :]
        q_dynamic_chunk = dynamic_query[:, :, i:end, :]

        if inter_attention_mask is None or intra_attention_mask is None:
            q_batch = relation_metadata["q_batch"][i:end, None]
            k_batch = relation_metadata["k_batch"][None, :]
            q_pos_for_mask = relation_metadata["q_pos"][i:end, None]
            k_pos_for_mask = relation_metadata["k_pos"][None, :]
            q_visual_for_mask = relation_metadata["q_visual"][i:end, None]
            k_visual_for_mask = relation_metadata["k_visual"][None, :]
            q_text_for_mask = relation_metadata["q_text"][i:end, None]
            k_text_for_mask = relation_metadata["k_text"][None, :]
            keep = (q_batch == k_batch) & (q_pos_for_mask >= k_pos_for_mask)
            mask_intra_chunk = (
                keep
                & ((q_visual_for_mask & k_visual_for_mask) | (q_text_for_mask & k_text_for_mask))
            )[None, None]
            mask_inter_chunk = (
                keep
                & ((q_visual_for_mask & k_text_for_mask) | (q_text_for_mask & k_visual_for_mask))
            )[None, None]
        else:
            mask_inter_chunk = inter_attention_mask[:, :, i:end, :]
            mask_intra_chunk = intra_attention_mask[:, :, i:end, :]

        if relation_metadata is not None and girf_factorial_options(module.config)[1]:
            pi_allowed = pi_girf_allowed_pairs(
                relation_metadata["q_visual"][i:end],
                relation_metadata["k_visual"],
                relation_metadata["q_text"][i:end],
                relation_metadata["k_text"],
                relation_metadata["q_entity_ids"][i:end],
                relation_metadata["k_entity_ids"],
                relation_metadata["q_post_text"][i:end],
                relation_metadata["k_bridge_text"],
                components=pi_girf_mask_components(module.config),
            )[None, None]
            mask_inter_chunk = mask_inter_chunk & pi_allowed
            mask_intra_chunk = mask_intra_chunk & pi_allowed

        attn_static_chunk = torch.matmul(q_static_chunk, inter_key.transpose(-1, -2).contiguous()) * scaling
        attn_dynamic_chunk = torch.matmul(q_dynamic_chunk, key.transpose(-1, -2).contiguous()) * scaling

        soft_vv_penalty = pi_girf_soft_vv_penalty(module.config)
        if relation_metadata is not None and soft_vv_penalty:
            cross_entity_vv = pi_girf_cross_entity_vv_pairs(
                relation_metadata["q_visual"][i:end],
                relation_metadata["k_visual"],
                relation_metadata["q_entity_ids"][i:end],
                relation_metadata["k_entity_ids"],
            )
            attn_dynamic_chunk = attn_dynamic_chunk - (
                soft_vv_penalty * cross_entity_vv[None, None].to(attn_dynamic_chunk.dtype)
            )

        if (
            getattr(module, "cross_modal_position_type", "dipe") == "bounded-dipe"
            and content_query is not None
            and content_key is not None
            and relation_metadata is not None
        ):
            # Query-independent long-range correction.  Native DIPE is retained
            # below the radius; beyond it, cross-modal logits move smoothly
            # toward the zero-relative-phase content endpoint.  This changes
            # legal long-horizon inputs without modifying TT or VV attention.
            content_chunk = content_query[:, :, i:end, :]
            attn_content_chunk = torch.matmul(
                content_chunk, content_key.transpose(-1, -2).contiguous()
            ) * scaling
            q_pos = relation_metadata["q_pos"][i:end].float()[:, None]
            k_pos = relation_metadata["k_pos"].float()[None, :]
            distance = (q_pos - k_pos).abs()
            radius = float(getattr(module.config, "cross_modal_bounded_radius", 256.0))
            tau = max(float(getattr(module.config, "cross_modal_bounded_tau", 512.0)), 1e-6)
            strength = float(getattr(module.config, "cross_modal_bounded_strength", 0.5))
            gate = strength * (1.0 - torch.exp(-(distance - radius).clamp_min(0) / tau))
            attn_static_chunk = attn_static_chunk + gate[None, None].to(
                attn_static_chunk.dtype
            ) * (attn_content_chunk - attn_static_chunk)

        if relation_metadata is not None and hasattr(module, "relation_coefficients"):
            q_pos = relation_metadata["q_pos"][i:end]
            k_pos = relation_metadata["k_pos"]
            q_visual = relation_metadata["q_visual"][i:end]
            k_visual = relation_metadata["k_visual"]
            q_h = relation_metadata["q_h"][i:end]
            k_h = relation_metadata["k_h"]
            q_w = relation_metadata["q_w"][i:end]
            k_w = relation_metadata["k_w"]
            visual_h = torch.where(q_visual[:, None], q_h[:, None], k_h[None, :])
            visual_w = torch.where(q_visual[:, None], q_w[:, None], k_w[None, :])
            basis = torch.stack(
                (
                    torch.ones_like(visual_h),
                    torch.sign(q_pos[:, None] - k_pos[None, :]).to(visual_h.dtype),
                    visual_h,
                    visual_w,
                    torch.sin(math.pi * visual_h),
                    torch.cos(math.pi * visual_h),
                    torch.sin(math.pi * visual_w),
                    torch.cos(math.pi * visual_w),
                ),
                dim=0,
            )
            bias = torch.einsum(
                "hr,rqk->hqk", module.relation_coefficients.float(), basis.float()
            ).to(attn_static_chunk.dtype)
            attn_static_chunk = attn_static_chunk + bias.unsqueeze(0)

        oracle_relation = getattr(module.config, "oracle_spatial_relation", None)
        oracle_strength = float(getattr(module.config, "oracle_spatial_strength", 0.0))
        if relation_metadata is not None and oracle_relation and oracle_strength:
            visual_h = relation_metadata["k_h"][None, :]
            visual_w = relation_metadata["k_w"][None, :]
            if oracle_relation == "left":
                field = -visual_w
            elif oracle_relation == "right":
                field = visual_w
            elif oracle_relation == "top":
                field = -visual_h
            elif oracle_relation == "bottom":
                field = visual_h
            elif oracle_relation == "center":
                field = -(visual_h.square() + visual_w.square())
            elif oracle_relation == "random":
                field = 0.6 * visual_h - 0.8 * visual_w
            else:
                raise ValueError(f"Unknown oracle spatial relation: {oracle_relation}")
            q_text = relation_metadata["q_text"][i:end, None]
            k_visual = relation_metadata["k_visual"][None, :]
            oracle_mask = q_text & k_visual
            oracle_bias = oracle_strength * field.expand(end - i, -1)
            attn_static_chunk = attn_static_chunk + torch.where(
                oracle_mask, oracle_bias, torch.zeros_like(oracle_bias)
            )[None, None]

        attn_weights = torch.where(mask_intra_chunk, attn_dynamic_chunk, attn_static_chunk)

        mask_keep = mask_intra_chunk | mask_inter_chunk
        
        min_val = torch.tensor(-1e7, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(mask_keep, attn_weights, min_val)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        chunk_output = torch.matmul(attn_weights, value)
        output_chunks.append(chunk_output)
        
        # attention weights
        if output_attentions:
            with torch.no_grad():
                avg_attn = attn_weights.mean(dim=1, keepdim=True).detach().to(torch.float16).cpu()
                attn_weights_chunks.append(avg_attn)
            del attn_static_chunk, attn_dynamic_chunk
        else:
            del attn_static_chunk, attn_dynamic_chunk, attn_weights

    attn_output = torch.cat(output_chunks, dim=2)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # attention weights are averaged over heads and returned in float16 to save memory, and only if output_attentions is True
    full_attn_weights = torch.cat(attn_weights_chunks, dim=2) if output_attentions else None

    return attn_output, full_attn_weights


class Qwen25_SigLIPRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        return inv_freq, 1.0

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if self.rope_init_fn is None and self.rope_type == "default":
            # Transformers 5.x no longer exposes the default initializer in
            # ROPE_INIT_FUNCTIONS.  Keep checkpoint-compatible Qwen behavior
            # without depending on that private registry entry.
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(
                config, device
            )
        else:
            if self.rope_init_fn is None:
                raise ValueError(f"Unsupported RoPE type: {self.rope_type}")
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        # Pilot-only checkpoint-compatible Sim(2) gate.  It is disabled by
        # default and initialized at the exact Point/MRoPE endpoint.
        self.sim2_gate_logit = nn.Parameter(torch.zeros((), device=device), requires_grad=False)
        self._sim2_visual_mask = None

    def enable_sim2_pilot(self, trainable: bool = True) -> None:
        self.config.sim2_rope_enabled = True
        self.sim2_gate_logit.requires_grad_(trainable)

    def sim2_gate(self) -> torch.Tensor:
        return torch.tanh(self.sim2_gate_logit)

    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        visual_mask = self._sim2_visual_mask
        # In contrast to other models, Qwen25_SigLIP has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        name = normalize_position_embedding(getattr(self.config, "position_embedding_type", "mrope"))
        options = resolve_position_options(name, getattr(self.config, "position_embedding_options", {}))
        if name == "hope":
            position_ids_expanded = position_ids_expanded.clone()
            position_ids_expanded[0].masked_fill_(position_ids_expanded[0] > 0, 0)
        if name == "grape" and float(options.get("log_freq_scale", 1.0)) != 1.0:
            scale = float(options["log_freq_scale"])
            inv_freq_expanded = torch.exp(inv_freq_expanded.clamp_min(1e-30).log() / scale)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            if bool(getattr(self.config, "sim2_rope_enabled", False)) and visual_mask is not None:
                mask = visual_mask.bool()
                pair_count = freqs.shape[-1]
                radial_start, angular_start = pair_count // 2, 3 * pair_count // 4
                hybrid = freqs.clone()
                for batch_idx in range(position_ids.shape[1]):
                    current = mask[batch_idx]
                    if not bool(current.any()):
                        continue
                    y = position_ids[1, batch_idx, current].float()
                    z = position_ids[2, batch_idx, current].float()
                    y = y - .5 * (y.min() + y.max())
                    z = z - .5 * (z.min() + z.max())
                    rho = torch.log(torch.sqrt(y.square() + z.square()).clamp_min(1e-4))
                    phi = torch.atan2(y, z)
                    radial_count = angular_start - radial_start
                    angular_count = pair_count - angular_start
                    radial_frequency = torch.logspace(
                        math.log10(.25), math.log10(4.0), radial_count,
                        device=freqs.device, dtype=freqs.dtype,
                    )
                    angular_frequency = torch.arange(
                        1, angular_count + 1, device=freqs.device, dtype=freqs.dtype,
                    )
                    hybrid[:, batch_idx, current, radial_start:angular_start] = (
                        rho[:, None] * radial_frequency[None]
                    )[None]
                    hybrid[:, batch_idx, current, angular_start:] = (
                        phi[:, None] * angular_frequency[None]
                    )[None]
                gate = self.sim2_gate().float()
                freqs = freqs + gate * (hybrid - freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def _axis_ids_for_pairs(name, pair_count, options, device):
    if name in {"vanilla-rope", "v2pe"}:
        return torch.zeros(pair_count, dtype=torch.long, device=device)
    if name in {"mrope-interleave", "ilrope"} or (
        name == "grape" and options.get("frequency_allocation") == "interleaved"
    ):
        sections = options.get("mrope_section", [24, 20, 20])
        axis = torch.zeros(pair_count, dtype=torch.long, device=device)
        for dim, offset in ((1, 1), (2, 2)):
            end = min(pair_count, int(sections[dim]) * 3)
            axis[offset:end:3] = dim
        return axis
    if name in {"videorope", "hope"}:
        sections = options.get("mrope_section", [16, 24, 24])
        axis = torch.zeros(pair_count, dtype=torch.long, device=device)
        for dim, offset in ((1, 1), (2, 2)):
            end = min(pair_count, int(sections[dim]) * 2)
            axis[offset:end:2] = dim
        return axis
    if name == "mmrope":
        pattern = torch.tensor([0, 0, 1, 2, 1, 2, 1, 2], device=device)
        return pattern[torch.arange(pair_count, device=device) % 8]

    sections = options.get("mrope_section", [16, 24, 24])
    axis = []
    for dim, length in enumerate(sections):
        axis.extend([dim] * int(length))
    if len(axis) < pair_count:
        axis.extend([0] * (pair_count - len(axis)))
    return torch.tensor(axis[:pair_count], dtype=torch.long, device=device)


def _select_axis_frequencies(values, name, options, x_heads, num_attention_heads):
    """Convert [3, B, L, D] axis frequencies to a tensor broadcastable to x."""
    if name == "mhrope":
        head_sections = options.get("mrope_section", [2, 3, 3])
        base_head_count = int(options.get("num_key_value_heads", sum(head_sections)))
        base_axes = []
        for dim, count in enumerate(head_sections):
            base_axes.extend([dim] * int(count))
        base_axes.extend([-1] * max(0, base_head_count - len(base_axes)))
        base_axes = base_axes[:base_head_count]
        if num_attention_heads % base_head_count != 0:
            raise ValueError(
                "MHRoPE requires num_attention_heads to be divisible by its "
                f"base head allocation ({base_head_count})"
            )
        repeat = num_attention_heads // base_head_count
        head_axes = [axis for axis in base_axes for _ in range(repeat)]
        if x_heads != num_attention_heads:
            raise ValueError("MHRoPE frequency selection expects expanded attention heads")
        selected = []
        for axis in head_axes:
            if axis < 0:
                selected.append(torch.zeros_like(values[0]))
            else:
                selected.append(values[axis])
        return torch.stack(selected, dim=1)

    pair_count = values.shape[-1] // 2
    pair_axes = _axis_ids_for_pairs(name, pair_count, options, values.device)
    full_axes = torch.cat([pair_axes, pair_axes])
    gather_index = full_axes.view(1, 1, -1, 1).expand(values.shape[1], values.shape[2], -1, 1)
    selected = values.permute(1, 2, 3, 0).gather(-1, gather_index).squeeze(-1)
    return selected.unsqueeze(1)


def apply_multimodal_rotary_pos_emb(
    x, cos, sin, mrope_section, position_embedding_type="mrope",
    position_embedding_options=None, num_key_value_heads=None,
    num_attention_heads=None,
):
    name = normalize_position_embedding(position_embedding_type)
    options = resolve_position_options(name, position_embedding_options)
    options.setdefault("mrope_section", mrope_section)
    num_key_value_heads = num_key_value_heads or x.shape[1]
    num_attention_heads = num_attention_heads or x.shape[1]
    if name == "mhrope" and x.shape[1] != num_attention_heads:
        if num_attention_heads % x.shape[1] != 0:
            raise ValueError("MHRoPE cannot expand heads by a non-integer GQA factor")
        x = x.repeat_interleave(num_attention_heads // x.shape[1], dim=1)
    selected_cos = _select_axis_frequencies(cos, name, options, x.shape[1], num_attention_heads)
    selected_sin = _select_axis_frequencies(sin, name, options, x.shape[1], num_attention_heads)
    if name == "mhrope":
        # Unassigned MHRoPE heads are NoPE: cos=1 and sin=0.
        head_sections = options.get("mrope_section", [2, 3, 3])
        base_heads = int(options.get("num_key_value_heads", sum(head_sections)))
        assigned = sum(int(v) for v in head_sections)
        if assigned < base_heads:
            repeat = num_attention_heads // base_heads
            start = assigned * repeat
            selected_cos[:, start:] = 1
            selected_sin[:, start:] = 0
    return (x * selected_cos) + (rotate_half(x) * selected_sin)


class Qwen25_SigLIPAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        projection_bias = (
            bool(getattr(config, "attention_bias", False))
            if getattr(config, "llm_backbone", "qwen2") == "qwen3"
            else True
        )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=projection_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=projection_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=projection_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = (
            Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, "qk_norm", False) else None
        )
        self.k_norm = (
            Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, "qk_norm", False) else None
        )
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.rotary_emb = Qwen25_SigLIPRotaryEmbedding(config=config)
        self.cross_modal_position_type = getattr(
            config, "cross_modal_position_type", "dipe"
        )
        relation_names = getattr(config, "cross_modal_relation_names", None) or [
            "constant", "direction", "visual_h", "visual_w",
            "visual_h_sin", "visual_h_cos", "visual_w_sin", "visual_w_cos",
        ]
        if len(relation_names) != 8:
            raise ValueError("The initial universal-relation implementation requires 8 bases")
        if self.cross_modal_position_type == "universal-relation":
            self.relation_coefficients = nn.Parameter(
                torch.zeros(self.num_heads, len(relation_names), dtype=torch.float32)
            )
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

        static_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        kv_valid_mask: Optional[torch.Tensor] = None,
        q_valid_mask: Optional[torch.Tensor] = None,
        # MODIFIED: Changed from _block_mask to standard boolean mask
        inter_mask: Optional[torch.Tensor] = None,
        intra_mask: Optional[torch.Tensor] = None,
        relation_metadata: Optional[dict] = None,

        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # TODO
        # We performed a key-value update first, then a rope (this can be optimized)
        cos, sin = position_embeddings
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states_dynamic = apply_multimodal_rotary_pos_emb(
            key_states, cos, sin, self.rope_scaling["mrope_section"],
            self.config.position_embedding_type,
            self.config.position_embedding_options,
            self.num_key_value_heads,
            self.num_heads,
        )
        query_states_dynamic = apply_multimodal_rotary_pos_emb(
            query_states, cos[:, :, -query_states.shape[2]:], sin[:, :, -query_states.shape[2]:],
            self.rope_scaling["mrope_section"], self.config.position_embedding_type,
            self.config.position_embedding_options, self.num_key_value_heads,
            self.num_heads,
        )

        static_cos, static_sin = static_position_embeddings
        query_states_static = apply_multimodal_rotary_pos_emb(
            query_states, static_cos[:, :, -query_states.shape[2]:], static_sin[:, :, -query_states.shape[2]:],
            self.rope_scaling["mrope_section"], self.config.position_embedding_type,
            self.config.position_embedding_options, self.num_key_value_heads,
            self.num_heads,
        )

        if self.cross_modal_position_type == "native":
            query_states_static = query_states_dynamic
            key_states_inter = key_states_dynamic
            relation_metadata = None
        elif self.cross_modal_position_type in {
            "universal-relation", "girf", "entity-aware-girf", "entity-tagged-girf",
            "tv-neutral-girf", "pi-girf"
        }:
            query_states_static = query_states
            key_states_inter = key_states
        else:
            key_states_inter = key_states_dynamic
    
        k_dyn_bshd = key_states_dynamic.transpose(1, 2)
        k_dyn_bhsd = k_dyn_bshd[kv_valid_mask].unsqueeze(0).transpose(1, 2)

        k_inter_bshd = key_states_inter.transpose(1, 2)
        k_inter_bhsd = k_inter_bshd[kv_valid_mask].unsqueeze(0).transpose(1, 2)

        v_bshd = value_states.transpose(1, 2)
        v_bhsd = v_bshd[kv_valid_mask].unsqueeze(0).transpose(1, 2)

        q_dyn_bshd = query_states_dynamic.transpose(1, 2)
        q_static_bshd = query_states_static.transpose(1, 2)
        q_dyn_bhsd = q_dyn_bshd[q_valid_mask].unsqueeze(0).transpose(1, 2)
        q_static_bhsd = q_static_bshd[q_valid_mask].unsqueeze(0).transpose(1, 2)
        q_content_bshd = query_states.transpose(1, 2)
        q_content_bhsd = q_content_bshd[q_valid_mask].unsqueeze(0).transpose(1, 2)
        k_content_bshd = key_states.transpose(1, 2)
        k_content_bhsd = k_content_bshd[kv_valid_mask].unsqueeze(0).transpose(1, 2)

        if self.cross_modal_position_type in {
            "girf", "entity-tagged-girf", "tv-neutral-girf", "pi-girf"
        }:
            if relation_metadata is None:
                raise RuntimeError("GIRF requires relation metadata")
            q_girf = relation_metadata["q_girf"].unsqueeze(0)
            k_girf = relation_metadata["k_girf"].unsqueeze(0)
            q_static_bhsd = apply_girf_rotary(
                q_content_bhsd, q_girf, self.rotary_emb.inv_freq
            )
            k_inter_bhsd = apply_girf_rotary(
                k_content_bhsd, k_girf, self.rotary_emb.inv_freq
            )
        if self.cross_modal_position_type == "entity-aware-girf":
            if relation_metadata is None:
                raise RuntimeError("Entity-aware GIRF requires relation metadata")
            girf_version = getattr(self.config, "entity_aware_girf_version", None)
            if girf_version != "v3":
                raise RuntimeError(
                    "Entity-aware GIRF V3 code cannot evaluate a checkpoint with "
                    f"entity_aware_girf_version={girf_version!r}. Retrain/migrate with V3 config."
                )
            spatial_scales = getattr(
                self.config, "entity_aware_girf_spatial_scales", (16.0, 16.0)
            )
            girf_layout = {
                "axis_weights": self.config.entity_aware_girf_axis_weights,
                "band_weights": self.config.entity_aware_girf_band_weights,
                "long_range_bands": self.config.entity_aware_girf_long_range_bands,
                "spatial_bands": self.config.entity_aware_girf_spatial_bands,
            }
            q_girf = relation_metadata["q_entity_girf"].unsqueeze(0)
            k_girf = relation_metadata["k_entity_girf"].unsqueeze(0)
            q_entity_bhsd = apply_entity_aware_girf_rotary(
                q_content_bhsd, q_girf, spatial_scales, **girf_layout
            )
            k_entity_bhsd = apply_entity_aware_girf_rotary(
                k_content_bhsd, k_girf, spatial_scales, **girf_layout
            )
            attn_output, attn_weights = entity_aware_block_attention_forward(
                self,
                dynamic_query=q_dyn_bhsd,
                girf_query=q_entity_bhsd,
                dynamic_key=k_dyn_bhsd,
                girf_key=k_entity_bhsd,
                value=v_bhsd,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                output_attentions=output_attentions,
                relation_metadata=relation_metadata,
            )
        else:
            attn_output, attn_weights = decoupled_eager_attention_forward(
                self,
                static_query=q_static_bhsd,
                dynamic_query=q_dyn_bhsd,
                key=k_dyn_bhsd,
                inter_key=k_inter_bhsd,
                content_query=q_content_bhsd,
                content_key=k_content_bhsd,
                value=v_bhsd,
                inter_attention_mask=(
                    None if inter_mask is None else inter_mask[None, None, :, :]
                ),
                intra_attention_mask=(
                    None if intra_mask is None else intra_mask[None, None, :, :]
                ),
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                output_attentions=output_attentions,
                relation_metadata=relation_metadata,
            )

        # Attention is evaluated over valid tokens from all batch rows in one
        # compact sequence. Restore those tokens to the padded [batch, seq]
        # layout before the output projection. This is required during
        # training as well as inference when data_flatten=False.
        compact_output = attn_output.squeeze(0)
        valid_query_indices = torch.nonzero(
            q_valid_mask.reshape(-1), as_tuple=False
        ).flatten()
        if compact_output.shape[0] != valid_query_indices.numel():
            raise RuntimeError(
                "Compact attention output/query mask mismatch: "
                f"output has {compact_output.shape[0]} queries, "
                f"mask has {valid_query_indices.numel()} valid queries"
            )
        restored_output = attn_output.new_zeros(
            (bsz * q_len, self.num_heads, self.head_dim)
        )
        attn_output = restored_output.index_copy(
            0, valid_query_indices, compact_output
        ).view(bsz, q_len, self.num_heads, self.head_dim)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen25_SigLIPDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen25_SigLIPAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

        # args for DIPE
        static_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        kv_valid_mask: Optional[torch.Tensor] = None,
        q_valid_mask: Optional[torch.Tensor] = None,
        inter_mask: Optional[torch.Tensor] = None,
        intra_mask: Optional[torch.Tensor] = None,
        relation_metadata: Optional[dict] = None,

        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,

            static_position_embeddings=static_position_embeddings,
            kv_valid_mask=kv_valid_mask,
            q_valid_mask=q_valid_mask,
            inter_mask=inter_mask,
            intra_mask=intra_mask,
            relation_metadata=relation_metadata,

            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class Qwen25_SigLIPTextModel(Qwen25_SigLIPPreTrainedModel):
    config: Qwen2Config

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen25_SigLIPDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen25_SigLIPRotaryEmbedding(config=config)
        self.static_rotary_emb = Qwen25_SigLIPRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
        # args for DIPE
        static_position_ids: Optional[torch.LongTensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        packed_seq_lens: Optional[torch.Tensor] = None,
        visual_entity_ids: Optional[torch.Tensor] = None,
        visual_temporal_ids: Optional[torch.Tensor] = None,
        
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None
        
        
        # =====================================================================
        # Create standard boolean masks for eagar attention
        # =====================================================================
        B, L = visual_mask.shape
        device = visual_mask.device
        
        position_map = torch.arange(L, device=device).unsqueeze(0).expand(B, L).reshape(-1)
        batch_map = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)

        is_visual_flat = visual_mask.reshape(-1)
        is_text_flat = text_mask.reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)
        
        k_batch_ids = batch_map[valid_mask_flat].int()
        k_pos_ids   = position_map[valid_mask_flat].int()
        k_is_visual = is_visual_flat[valid_mask_flat]
        k_is_text   = is_text_flat[valid_mask_flat]

        # prefilling or training
        if inputs_embeds.shape[1] == valid_mask.shape[1]:
            q_valid_mask = valid_mask

            q_batch_ids = k_batch_ids
            q_pos_ids   = k_pos_ids
            q_is_visual = k_is_visual
            q_is_text   = k_is_text
        # decoding
        else:
            seqlens = valid_mask.sum(dim=1)
            cu_seqlens = seqlens.cumsum(dim=0).to(dtype=torch.long)
            q_indices = cu_seqlens - 1
            q_valid_mask = torch.ones((B, 1), device=device, dtype=torch.bool).expand(B, inputs_embeds.shape[1])
            
            q_batch_ids = k_batch_ids[q_indices]
            q_pos_ids   = k_pos_ids[q_indices]
            q_is_visual = k_is_visual[q_indices]
            q_is_text   = k_is_text[q_indices]

        # import pdb; pdb.set_trace()
        if self.training:
            if packed_seq_lens is not None:
                lengths = packed_seq_lens[1:] - packed_seq_lens[:-1]
                packed_batch_ids = torch.repeat_interleave(
                    torch.arange(lengths.numel(), device=device, dtype=torch.int32),
                    lengths.to(device=device, dtype=torch.long),
                )
                if packed_batch_ids.numel() != valid_mask.numel():
                    raise ValueError(
                        "Packed cumulative sequence lengths do not match the flattened input length"
                    )
            else:
                packed_batch_ids = batch_map.int()
            # During training with packing, batch_ids are derived from position_ids
            q_batch_ids = packed_batch_ids[q_valid_mask.view(-1)]
            k_batch_ids = packed_batch_ids[valid_mask.view(-1)]

        relation_metadata = None
        needs_relation_metadata = (
            getattr(self.config, "cross_modal_position_type", "dipe") in {
                "universal-relation", "bounded-dipe", "girf", "entity-aware-girf",
                "entity-tagged-girf", "tv-neutral-girf", "pi-girf"
            }
            or bool(getattr(self.config, "oracle_spatial_relation", None))
        )
        if needs_relation_metadata:
            normalized_h, normalized_w = normalized_visual_coordinates(
                position_ids, visual_mask
            )
            girf_coordinates = girf_token_coordinates(position_ids, visual_mask)
            cross_modal_type = getattr(
                self.config, "cross_modal_position_type", "dipe"
            )
            ordinal_enabled, pi_mask_enabled = girf_factorial_options(self.config)
            if not ordinal_enabled:
                girf_coordinates = tv_neutral_girf_coordinates(
                    position_ids, visual_mask
                )
            coordinate_sample_ids = (
                packed_batch_ids.reshape(B, L)
                if self.training and packed_seq_lens is not None
                else batch_map.reshape(B, L)
            )
            entity_girf_coordinates = entity_aware_girf_coordinates(
                position_ids,
                visual_mask,
                valid_mask=valid_mask,
                sample_ids=coordinate_sample_ids,
                visual_entity_ids=visual_entity_ids,
                visual_temporal_ids=visual_temporal_ids,
            )
            resolved_entity_ids = (
                visual_entity_ids
                if visual_entity_ids is not None
                else entity_girf_coordinates[..., 1] * visual_mask
            )
            if pi_mask_enabled:
                _, post_text, bridge_text = pi_girf_text_regions(
                    visual_mask,
                    text_mask,
                    valid_mask,
                    coordinate_sample_ids,
                )
            else:
                post_text = torch.zeros_like(text_mask, dtype=torch.bool)
                bridge_text = torch.zeros_like(text_mask, dtype=torch.bool)
            h_flat = normalized_h.reshape(-1)[valid_mask_flat]
            w_flat = normalized_w.reshape(-1)[valid_mask_flat]
            if inputs_embeds.shape[1] == valid_mask.shape[1]:
                q_h, q_w = h_flat, w_flat
            else:
                q_h, q_w = h_flat[q_indices], w_flat[q_indices]
            relation_metadata = {
                "q_pos": q_pos_ids,
                "k_pos": k_pos_ids,
                "q_batch": q_batch_ids,
                "k_batch": k_batch_ids,
                "q_visual": q_is_visual,
                "k_visual": k_is_visual,
                "q_text": q_is_text,
                "k_text": k_is_text,
                "q_h": q_h,
                "k_h": h_flat,
                "q_w": q_w,
                "k_w": w_flat,
                "q_girf": (
                    girf_coordinates.reshape(-1, 5)[valid_mask_flat]
                    if inputs_embeds.shape[1] == valid_mask.shape[1]
                    else girf_coordinates.reshape(-1, 5)[valid_mask_flat][q_indices]
                ),
                "k_girf": girf_coordinates.reshape(-1, 5)[valid_mask_flat],
                "q_entity_girf": (
                    entity_girf_coordinates.reshape(-1, 5)[valid_mask_flat]
                    if inputs_embeds.shape[1] == valid_mask.shape[1]
                    else entity_girf_coordinates.reshape(-1, 5)[valid_mask_flat][q_indices]
                ),
                "k_entity_girf": entity_girf_coordinates.reshape(-1, 5)[valid_mask_flat],
                "q_entity_ids": (
                    resolved_entity_ids.reshape(-1)[valid_mask_flat]
                    if inputs_embeds.shape[1] == valid_mask.shape[1]
                    else resolved_entity_ids.reshape(-1)[valid_mask_flat][q_indices]
                ),
                "k_entity_ids": resolved_entity_ids.reshape(-1)[valid_mask_flat],
                "q_post_text": (
                    post_text.reshape(-1)[valid_mask_flat]
                    if inputs_embeds.shape[1] == valid_mask.shape[1]
                    else post_text.reshape(-1)[valid_mask_flat][q_indices]
                ),
                "k_bridge_text": bridge_text.reshape(-1)[valid_mask_flat],
            }
            if getattr(self.config, "cross_modal_position_type", "dipe") in {
                "universal-relation", "bounded-dipe", "girf", "entity-aware-girf",
                "entity-tagged-girf", "tv-neutral-girf", "pi-girf"
            }:
                # Masks are constructed one query chunk at a time in attention.
                intra_mask = None
                inter_mask = None
            else:
                q_pos_ids_exp = q_pos_ids.unsqueeze(1)
                k_pos_ids_exp = k_pos_ids.unsqueeze(0)
                same_batch_mask = q_batch_ids.unsqueeze(1) == k_batch_ids.unsqueeze(0)
                causal_mask = q_pos_ids_exp >= k_pos_ids_exp
                intra_modality = (
                    (q_is_visual.unsqueeze(1) & k_is_visual.unsqueeze(0))
                    | (q_is_text.unsqueeze(1) & k_is_text.unsqueeze(0))
                )
                inter_modality = (
                    (q_is_visual.unsqueeze(1) & k_is_text.unsqueeze(0))
                    | (q_is_text.unsqueeze(1) & k_is_visual.unsqueeze(0))
                )
                intra_mask = same_batch_mask & causal_mask & intra_modality
                inter_mask = same_batch_mask & causal_mask & inter_modality
        else:
            q_pos_ids_exp = q_pos_ids.unsqueeze(1)
            k_pos_ids_exp = k_pos_ids.unsqueeze(0)
            same_batch_mask = q_batch_ids.unsqueeze(1) == k_batch_ids.unsqueeze(0)
            causal_mask = q_pos_ids_exp >= k_pos_ids_exp
            intra_modality = (
                (q_is_visual.unsqueeze(1) & k_is_visual.unsqueeze(0))
                | (q_is_text.unsqueeze(1) & k_is_text.unsqueeze(0))
            )
            inter_modality = (
                (q_is_visual.unsqueeze(1) & k_is_text.unsqueeze(0))
                | (q_is_text.unsqueeze(1) & k_is_visual.unsqueeze(0))
            )
            intra_mask = same_batch_mask & causal_mask & intra_modality
            inter_mask = same_batch_mask & causal_mask & inter_modality
        # =====================================================================

        hidden_states = inputs_embeds

        self.rotary_emb._sim2_visual_mask = visual_mask
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        self.rotary_emb._sim2_visual_mask = None
        static_position_embeddings = self.static_rotary_emb(hidden_states, static_position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None, # sequence packing is handled by masks
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,

                # args for DIPE
                static_position_embeddings=static_position_embeddings,
                kv_valid_mask=valid_mask,
                q_valid_mask=q_valid_mask,
                # MODIFIED: Pass standard boolean masks
                intra_mask=intra_mask,
                inter_mask=inter_mask,
                relation_metadata=relation_metadata,

                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
