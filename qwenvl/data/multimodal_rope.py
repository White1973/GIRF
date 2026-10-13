"""Unified multimodal position designs used by DIPE.

The implementations are adapted from the sibling Multimodal-RoPEs project,
but expose DIPE's additional anchored/static position ids as a third output.
"""

import math
import random
from typing import Any, Dict, Optional, Tuple

import torch


SUPPORTED_POSITION_EMBEDDINGS = (
    "vanilla-rope",
    "mrope",
    "mrope-interleave",
    "mhrope",
    "videorope",
    "hope",
    "circlerope",
    "v2pe",
    "ilrope",
    "omnirope",
    "mmrope",
    "grape",
)

ALIASES = {
    "vanilla": "vanilla-rope",
    "mrope-i": "mrope-interleave",
    "mrrope-i": "mrope-interleave",
}

DEFAULT_OPTIONS: Dict[str, Dict[str, Any]] = {
    "vanilla-rope": {},
    "mrope": {"mrope_section": [16, 24, 24], "temporal_stride": 2.0},
    "mrope-interleave": {
        "mrope_section": [24, 20, 20],
        "temporal_stride": 1.0,
        # MRoPE-I is defined with spatial-reset in Huang et al. (ICLR 2026).
        # Keep legacy checkpoints reproducible through their saved config, but
        # make newly-created runs faithful to the published method.
        "spatial_reset": True,
    },
    "mhrope": {
        "mrope_section": [2, 3, 3],
        "num_key_value_heads": 8,
        "temporal_stride": 1.0,
        "spatial_reset": True,
    },
    "videorope": {"mrope_section": [16, 24, 24], "temporal_stride": 2.0},
    "hope": {
        "mrope_section": [16, 24, 24],
        "temporal_stride": 2.0,
        "temporal_stride_lst": [0.5, 0.75, 1.0, 1.25, 1.5],
    },
    "circlerope": {
        "mrope_section": [16, 24, 24],
        "temporal_stride": 1.0,
        # Main Circle-RoPE setting (CIP, all layers): center the visual grid,
        # mix spatial and raster angles equally, and use the fixed radius 10.
        # AGE requires layer-wise position ids and is deliberately tracked as
        # a separate implementation scope by the failure-matrix protocol.
        "move_to_origin": True,
        "move_to_positive": False,
        "dff_rate": 0.0,
        "method": "circle",
        "radius": 10.0,
        "alpha": 0.5,
    },
    "v2pe": {"visual_stride": 16.0},
    "ilrope": {
        "mrope_section": [24, 20, 20],
        "temporal_stride": 1.0,
        "spatial_reset": True,
    },
    "omnirope": {"mrope_section": [16, 24, 24], "temporal_stride": 1.0},
    "mmrope": {
        "mrope_section": [16, 24, 24],
        "temporal_stride": 1.0,
        "position_scale": [1.0, 1.0, 1.0],
    },
    "grape": {
        "mrope_section": [16, 24, 24],
        "temporal_stride": 1.0,
        "frequency_allocation": "chunked",
        "log_freq_scale": 1.0,
        "grape_mode": "canonical",
    },
}


def normalize_position_embedding(name: str) -> str:
    name = ALIASES.get((name or "mrope").lower(), (name or "mrope").lower())
    if name not in SUPPORTED_POSITION_EMBEDDINGS:
        raise ValueError(
            f"Unsupported position embedding {name!r}. Supported values: "
            + ", ".join(SUPPORTED_POSITION_EMBEDDINGS)
        )
    return name


def resolve_position_options(name: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    name = normalize_position_embedding(name)
    resolved = dict(DEFAULT_OPTIONS[name])
    if options:
        resolved.update(options)
    if resolved.get("grape_mode", "canonical") != "canonical":
        raise ValueError("DIPE currently supports GRAPE only in canonical cos/sin mode")
    return resolved


def _anchor(chunk: torch.Tensor) -> torch.Tensor:
    if chunk.shape[1] == 0:
        return chunk.clone()
    return torch.full_like(chunk, chunk[0, 0])


def _next_global_offset(name: str, chunks) -> float:
    if not chunks:
        return 0.0
    # VideoRoPE/HoPE advance text along the temporal/body-diagonal axis.
    previous_max = chunks[-1][0].max() if name in {"videorope", "hope"} else chunks[-1].max()
    return float(previous_max.item() + 1)


def _circle_projection(w_index, h_index, t_index, opts):
    coords = torch.stack((w_index, h_index, t_index), dim=0).float()
    if opts.get("move_to_origin", False):
        centers = (coords.amax(dim=(1, 2, 3)) + coords.amin(dim=(1, 2, 3))) / 2
        coords = coords - centers.view(-1, 1, 1, 1)

    radius = opts.get("radius", -1.0)
    if "auto" in str(radius):
        scale = 1.0 if str(radius) == "auto" else float(str(radius).split("-", 1)[1])
        radius = coords.abs().max() * scale
    else:
        radius = float(radius)
    alpha = float(opts.get("alpha", -1.0))

    def project(source, method):
        if method == "no_circle":
            mapped = source
        elif method == "circle":
            x, y, z = source[0, 0], source[1, 0], source[2, 0]
            theta = torch.atan2(y, x)
            span = theta.max() - theta.min()
            norm_theta = (theta - theta.min()) / span * (2 * math.pi) if span > 0 else theta
            index_theta = torch.arange(
                x.numel(), device=x.device, dtype=torch.float32
            ).reshape_as(x) / x.numel() * (2 * math.pi)
            mixed_theta = alpha * norm_theta + (1 - alpha) * index_theta
            mapped = torch.stack(
                [radius * mixed_theta.cos(), radius * mixed_theta.sin(), z], dim=0
            ).unsqueeze(1)
        else:
            raise ValueError(f"Unknown CircleRoPE method: {method}")

        normal = torch.tensor([1.0, 1.0, 1.0], device=mapped.device)
        normal = normal / normal.norm()
        u = torch.stack([-normal[1], normal[0], torch.tensor(0.0, device=mapped.device)])
        u = u / u.norm()
        v = torch.cross(normal, u, dim=0)
        return torch.stack(
            [mapped[0] * u[j] + mapped[1] * v[j] for j in range(3)]
        )

    converted = project(coords, opts.get("method", "circle"))
    rate = float(opts.get("dff_rate", 0.0))
    if rate:
        converted = (1 - rate) * converted + rate * project(coords, "no_circle")
    move_positive = opts.get("move_to_positive", False)
    if move_positive:
        offset = 0.0 if move_positive == "auto" else float(move_positive)
        converted = converted + converted.min().abs() + offset
    return torch.stack([converted[2].flatten(), converted[1].flatten(), converted[0].flatten()])


def _visual_positions(name, t, h, w, offset, opts, device):
    visual_len = t * h * w
    if name in {"vanilla-rope", "v2pe"}:
        if name == "vanilla-rope":
            pos = torch.arange(visual_len, device=device, dtype=torch.float) + offset
        else:
            step = float(opts["visual_stride"]) / visual_len
            pos = torch.arange(1, visual_len + 1, device=device, dtype=torch.float) * step + offset - 1
        return pos.view(1, -1).expand(3, -1)

    ti = torch.arange(t, device=device, dtype=torch.float).view(-1, 1).expand(-1, h * w).flatten()
    hi = torch.arange(h, device=device, dtype=torch.float).view(1, -1, 1).expand(t, -1, w).flatten()
    wi = torch.arange(w, device=device, dtype=torch.float).view(1, 1, -1).expand(t, h, -1).flatten()

    if name in {"videorope", "hope"}:
        stride = random.choice(opts["temporal_stride_lst"]) if name == "hope" else opts["temporal_stride"]
        ti = ti * float(stride) + offset
        hi = hi - (h - 1) // 2 + ti
        wi = wi - (w - 1) // 2 + ti
        return torch.stack([ti, hi, wi])

    if name == "circlerope":
        grid_t = ti.view(-1, h, w)
        grid_h = hi.view(-1, h, w)
        grid_w = wi.view(-1, h, w)
        # The source method projects one ring and stacks rings along time.
        ring = _circle_projection(grid_w[:1], grid_h[:1], grid_t[:1], opts)
        result = ring.repeat(1, t)
        result += torch.arange(t, device=device).view(1, -1).repeat_interleave(h * w, 1) * float(opts["temporal_stride"])
        return result + offset

    scale = opts.get("position_scale", [1.0, 1.0, 1.0])
    ti = ti * float(opts.get("temporal_stride", 1.0)) * float(scale[0])
    hi = hi * float(scale[1])
    wi = wi * float(scale[2])
    positions = torch.stack([ti, hi, wi])
    if name in {"mrope-interleave", "mhrope", "ilrope"} and opts.get("spatial_reset", False):
        positions[0] += offset
        return positions
    if name in {"omnirope", "ilrope"}:
        positions[0] += offset
        return positions
    return positions + offset


def get_multimodal_position_ids(
    spatial_merge_size: int = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_embedding_type: str = "mrope",
    position_embedding_options: Optional[Dict[str, Any]] = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dynamic positions, DIPE anchored positions, and generation deltas."""
    name = normalize_position_embedding(position_embedding_type)
    opts = resolve_position_options(name, position_embedding_options)
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is None:
        raise ValueError("input_ids is required to build multimodal positions")
    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is None:
            base = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            delta = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long)
        else:
            base = attention_mask.long().cumsum(-1) - 1
            base.masked_fill_(attention_mask == 0, 1)
            base = base.unsqueeze(0).expand(3, -1, -1)
            delta = base.max(0)[0].max(-1, keepdim=True)[0] + 1 - attention_mask.shape[-1]
        return base, base.clone(), delta

    total_input_ids = input_ids
    attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    float_positions = name in {"videorope", "hope", "circlerope", "v2pe", "omnirope", "ilrope", "mmrope"}
    dtype = torch.float if float_positions else input_ids.dtype
    position_ids = torch.ones((3, *input_ids.shape), dtype=dtype, device=input_ids.device)
    static_position_ids = torch.ones_like(position_ids)
    deltas = []
    image_index = video_index = 0

    for batch_idx, sample_ids in enumerate(total_input_ids):
        valid = attention_mask[batch_idx] == 1
        tokens = sample_ids[valid]
        starts = torch.argwhere(tokens == vision_start_token_id).squeeze(1)
        next_tokens = tokens[starts + 1]
        image_count = int((next_tokens == image_token_id).sum())
        video_count = int((next_tokens == video_token_id).sum())
        token_list = tokens.tolist()
        chunks, static_chunks = [], []
        cursor = 0
        remaining_images, remaining_videos = image_count, video_count
        shift = 0.0

        for _ in range(image_count + video_count):
            image_pos = token_list.index(image_token_id, cursor) if remaining_images and image_token_id in token_list[cursor:] else len(token_list) + 1
            video_pos = token_list.index(video_token_id, cursor) if remaining_videos and video_token_id in token_list[cursor:] else len(token_list) + 1
            if image_pos < video_pos:
                grid = image_grid_thw[image_index]
                image_index += 1
                remaining_images -= 1
                media_pos = image_pos
            else:
                grid = video_grid_thw[video_index]
                video_index += 1
                remaining_videos -= 1
                media_pos = video_pos

            text_len = media_pos - cursor
            if name in {"omnirope", "ilrope", "v2pe"}:
                offset = shift
            else:
                offset = _next_global_offset(name, chunks)
            text = torch.arange(text_len, device=input_ids.device, dtype=torch.float).view(1, -1).expand(3, -1) + offset
            text = text.to(dtype)
            chunks.append(text)
            static_chunks.append(_anchor(text))

            t, h, w = int(grid[0]), int(grid[1]) // spatial_merge_size, int(grid[2]) // spatial_merge_size
            visual_offset = offset + text_len
            visual = _visual_positions(name, t, h, w, visual_offset, opts, input_ids.device).to(dtype)
            includes_vision_end = (
                name in {"mrope-interleave", "mhrope"}
                and opts.get("spatial_reset", False)
            )
            if includes_vision_end:
                local_max = max(
                    (t - 1) * float(opts.get("temporal_stride", 1.0)),
                    h - 1,
                    w - 1,
                )
                vision_end_position = torch.full(
                    (3, 1),
                    float(visual_offset + local_max + 1),
                    device=input_ids.device,
                    dtype=dtype,
                )
                visual = torch.cat([visual, vision_end_position], dim=1)
            chunks.append(visual)
            static_chunks.append(_anchor(visual))
            cursor = media_pos + t * h * w + int(includes_vision_end)

            if name == "v2pe":
                shift = math.ceil(float(visual.max().item())) + 1
            elif name in {"omnirope", "ilrope"}:
                extent = max((t - 1) * float(opts.get("temporal_stride", 1.0)) + 1, h, w)
                shift = visual_offset + extent

        if cursor < len(token_list):
            if name in {"omnirope", "ilrope", "v2pe"}:
                offset = shift
            else:
                offset = _next_global_offset(name, chunks)
            text = torch.arange(len(token_list) - cursor, device=input_ids.device, dtype=torch.float).view(1, -1).expand(3, -1) + offset
            text = text.to(dtype)
            chunks.append(text)
            static_chunks.append(_anchor(text))

        dynamic = torch.cat(chunks, dim=1)
        anchored = torch.cat(static_chunks, dim=1)
        position_ids[:, batch_idx, valid] = dynamic
        static_position_ids[:, batch_idx, valid] = anchored
        if name == "v2pe":
            delta = torch.ceil(dynamic.max()) + 1 - total_input_ids.shape[1]
        elif name in {"videorope", "hope"}:
            delta = dynamic[0].max() + 1 - total_input_ids.shape[1]
        else:
            delta = dynamic.max() + 1 - total_input_ids.shape[1]
        deltas.append(delta)

    return position_ids, static_position_ids, torch.stack(deltas).unsqueeze(1)
