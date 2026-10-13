"""Parameter-free Gauge-Invariant Relation Field primitives.

The legacy GIRF functions are intentionally kept unchanged.  Entity-aware
GIRF is exposed through separate functions so checkpoints using ``girf`` keep
their exact behaviour.
"""

import math

import torch


ENTITY_AWARE_AXES = ("text", "entity", "time", "height", "width")
ENTITY_AWARE_AXIS_WEIGHTS = (13, 12, 13, 13, 13)
ENTITY_AWARE_BAND_WEIGHTS = (
    (1, 5, 7),  # T: high, middle, low
    (4, 4, 4),  # E
    (2, 5, 6),  # Z
    (6, 5, 2),  # Y
    (6, 5, 2),  # X
)
ENTITY_AWARE_LONG_RANGE_BANDS = ((1.0, 1 / 4), (1 / 8, 1 / 64), (1 / 128, 1 / 8192))
ENTITY_AWARE_SPATIAL_BANDS = ((1.0, 1 / 4), (1 / 8, 1 / 32), (1 / 64, 1 / 512))
PI_GIRF_MASK_COMPONENTS = (
    "vv_same_entity",
    "visual_no_text",
    "nonpost_text_no_visual",
    "post_text_no_bridge",
)


def girf_factorial_options(config) -> tuple[bool, bool]:
    """Return independent ``(ordinal_phase, pi_mask)`` switches.

    The named legacy modes keep their historical coupled behavior.  New
    factorial experiments use ``cross_modal_position_type='girf'`` and set
    the two explicit flags, allowing each mechanism to be toggled without
    silently changing the other one.
    """
    mode = getattr(config, "cross_modal_position_type", "dipe")
    if mode == "tv-neutral-girf":
        return False, False
    if mode == "pi-girf":
        return False, True
    return (
        bool(getattr(config, "girf_ordinal_enabled", True)),
        bool(getattr(config, "girf_pi_mask_enabled", False)),
    )


def pi_girf_mask_components(config) -> tuple[str, ...]:
    """Return the enabled PI pair-mask clauses in canonical order.

    An absent or null component list preserves the historical strict/full PI
    behavior.  Explicit lists allow frozen single-clause causal ablations.
    """
    requested = getattr(config, "girf_pi_mask_components", None)
    if requested is None:
        return PI_GIRF_MASK_COMPONENTS
    if isinstance(requested, str):
        requested = [requested]
    unknown = set(requested) - set(PI_GIRF_MASK_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown PI-GIRF mask components: {sorted(unknown)}")
    return tuple(name for name in PI_GIRF_MASK_COMPONENTS if name in requested)


def pi_girf_soft_vv_penalty(config) -> float:
    """Return the finite post-scaling logit penalty for cross-entity VV pairs."""
    penalty = float(getattr(config, "girf_pi_soft_vv_penalty", 0.0))
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("girf_pi_soft_vv_penalty must be finite and non-negative")
    return penalty


def pi_girf_cross_entity_vv_pairs(
    q_visual: torch.Tensor,
    k_visual: torch.Tensor,
    q_entity_ids: torch.Tensor,
    k_entity_ids: torch.Tensor,
) -> torch.Tensor:
    """Identify visual-query/visual-key pairs belonging to different entities."""
    q_visual = q_visual.bool().view(-1)
    k_visual = k_visual.bool().view(-1)
    q_entity_ids = q_entity_ids.view(-1)
    k_entity_ids = k_entity_ids.view(-1)
    return (
        q_visual[:, None]
        & k_visual[None, :]
        & (q_entity_ids[:, None] != k_entity_ids[None, :])
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def girf_token_coordinates(position_ids: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:
    """Return GIRF potentials [text-role, entity, time, height, width].

    Every contiguous visual segment is normalized independently. Text-role is
    the neutral value zero until explicit semantic-segment metadata is present;
    inferring it from global token indices would break distractor invariance.
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("GIRF expects position_ids with shape [3, batch, sequence]")
    if visual_mask.shape != position_ids.shape[1:]:
        raise ValueError("visual_mask and position_ids disagree")

    batch, sequence = visual_mask.shape
    coordinates = torch.zeros(
        batch, sequence, 5, device=position_ids.device, dtype=torch.float32
    )
    for batch_idx in range(batch):
        indices = torch.nonzero(visual_mask[batch_idx], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        boundaries = torch.nonzero(
            indices[1:] != indices[:-1] + 1, as_tuple=False
        ).flatten() + 1
        segments = torch.tensor_split(indices, boundaries.cpu().tolist())
        current_entity = 0.0
        cursor = 0
        for entity_index, segment in enumerate(segments):
            # A fixed bounded injective potential avoids arbitrarily large
            # phases while preserving deterministic multi-image identity.
            entity = 2.0 / math.pi * math.atan(float(entity_index))
            coordinates[batch_idx, cursor : int(segment[0]), 1] = current_entity
            coordinates[batch_idx, segment, 1] = entity
            for relation_axis, position_axis in ((2, 0), (3, 1), (4, 2)):
                values = position_ids[position_axis, batch_idx, segment].float()
                span = values.max() - values.min()
                if float(span) > 0:
                    coordinates[batch_idx, segment, relation_axis] = (
                        2 * (values - values.min()) / span - 1
                    )
            current_entity = entity
            cursor = int(segment[-1]) + 1
        coordinates[batch_idx, cursor:, 1] = current_entity
    return coordinates


def tv_neutral_girf_coordinates(
    position_ids: torch.Tensor, visual_mask: torch.Tensor
) -> torch.Tensor:
    """Return legacy GIRF coordinates without an ordinal entity potential.

    The temporal/height/width coordinates are copied exactly from legacy GIRF,
    while the entity axis is fixed to the neutral value for every token.  This
    is parameter free and is bitwise identical to legacy GIRF for a single
    visual entity, whose legacy entity coordinate is already zero.
    """
    coordinates = girf_token_coordinates(position_ids, visual_mask)
    coordinates[..., 1] = 0.0
    return coordinates


def pi_girf_text_regions(
    visual_mask: torch.Tensor,
    text_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify text as prefix, inter-entity bridge, or post-visual query.

    ``sample_ids`` makes the classification valid for both padded batches and
    packed examples.  The result depends only on modality boundaries, never on
    question type, entity content, or a learned parameter.
    """
    expected = visual_mask.shape
    if any(mask.shape != expected for mask in (text_mask, valid_mask, sample_ids)):
        raise ValueError("PI-GIRF masks and sample_ids must have the same shape")
    prefix = torch.zeros_like(text_mask, dtype=torch.bool)
    post = torch.zeros_like(text_mask, dtype=torch.bool)
    bridge = torch.zeros_like(text_mask, dtype=torch.bool)
    flat_visual = visual_mask.reshape(-1).bool()
    flat_text = text_mask.reshape(-1).bool()
    flat_valid = valid_mask.reshape(-1).bool()
    flat_samples = sample_ids.reshape(-1)
    flat_prefix = prefix.reshape(-1)
    flat_post = post.reshape(-1)
    flat_bridge = bridge.reshape(-1)
    flat_positions = torch.arange(flat_valid.numel(), device=visual_mask.device)
    for sample_id in torch.unique(flat_samples[flat_valid]):
        sample = (flat_samples == sample_id) & flat_valid
        sample_text = sample & flat_text
        visual_indices = flat_positions[sample & flat_visual]
        if visual_indices.numel() == 0:
            flat_prefix[sample_text] = True
            continue
        first_visual = visual_indices.min()
        last_visual = visual_indices.max()
        flat_prefix[sample_text & (flat_positions < first_visual)] = True
        flat_post[sample_text & (flat_positions > last_visual)] = True
        flat_bridge[
            sample_text
            & (flat_positions > first_visual)
            & (flat_positions < last_visual)
        ] = True
    return prefix, post, bridge


def pi_girf_allowed_pairs(
    q_visual: torch.Tensor,
    k_visual: torch.Tensor,
    q_text: torch.Tensor,
    k_text: torch.Tensor,
    q_entity_ids: torch.Tensor,
    k_entity_ids: torch.Tensor,
    q_post_text: torch.Tensor,
    k_bridge_text: torch.Tensor,
    components: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Return the parameter-free strict PI-GIRF pair mask.

    Visual tokens attend only within the same entity and never absorb text.
    Only text after the final visual entity can read visual tokens.  Those
    query tokens also ignore inter-entity wrapper/bridge text, closing an
    indirect causal path by which an earlier entity could affect a later one.
    Entity ids are used only for equality, so arbitrary id permutations leave
    this mask unchanged.
    """
    enabled = PI_GIRF_MASK_COMPONENTS if components is None else tuple(components)
    unknown = set(enabled) - set(PI_GIRF_MASK_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown PI-GIRF mask components: {sorted(unknown)}")
    q_visual = q_visual.bool().view(-1)
    k_visual = k_visual.bool().view(-1)
    q_text = q_text.bool().view(-1)
    k_text = k_text.bool().view(-1)
    q_post_text = q_post_text.bool().view(-1)
    k_bridge_text = k_bridge_text.bool().view(-1)
    q_entity_ids = q_entity_ids.view(-1)
    k_entity_ids = k_entity_ids.view(-1)
    allowed = torch.ones(
        (q_visual.numel(), k_visual.numel()),
        dtype=torch.bool,
        device=q_visual.device,
    )
    if "vv_same_entity" in enabled:
        allowed &= ~pi_girf_cross_entity_vv_pairs(
            q_visual, k_visual, q_entity_ids, k_entity_ids
        )
    if "visual_no_text" in enabled:
        allowed &= ~(q_visual[:, None] & k_text[None, :])
    if "nonpost_text_no_visual" in enabled:
        allowed &= ~(
            q_text[:, None] & k_visual[None, :] & ~q_post_text[:, None]
        )
    if "post_text_no_bridge" in enabled:
        allowed &= ~(q_post_text[:, None] & k_bridge_text[None, :])
    return allowed


def apply_girf_rotary(
    x: torch.Tensor, coordinates: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    """Rotate Q/K using shared round-robin typed-relation channels."""
    if x.shape[-1] % 2:
        raise ValueError("GIRF requires an even attention head dimension")
    if coordinates.shape != (x.shape[0], x.shape[2], 5):
        raise ValueError("GIRF coordinates must have shape [batch, sequence, 5]")
    pair_count = x.shape[-1] // 2
    frequencies = inv_freq[:pair_count].to(device=x.device, dtype=torch.float32)
    if frequencies.numel() != pair_count:
        raise ValueError("RoPE frequency count does not match the GIRF head dimension")
    pair_axes = torch.arange(pair_count, device=x.device) % 5
    selected = coordinates.to(x.device).index_select(-1, pair_axes)
    angles = selected * frequencies.view(1, 1, -1)
    full_angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return (
        x * torch.cos(full_angles).to(x.dtype)
        + _rotate_half(x) * torch.sin(full_angles).to(x.dtype)
    )


def _contiguous_runs(indices: torch.Tensor) -> list[torch.Tensor]:
    if indices.numel() == 0:
        return []
    boundaries = torch.nonzero(indices[1:] != indices[:-1] + 1, as_tuple=False).flatten() + 1
    return list(torch.tensor_split(indices, boundaries.cpu().tolist()))


def entity_aware_girf_coordinates(
    position_ids: torch.Tensor,
    visual_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    sample_ids: torch.Tensor | None = None,
    visual_entity_ids: torch.Tensor | None = None,
    visual_temporal_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build ``[T, E, Z, Y, X]`` typed potentials.

    ``E`` is one-based for visual entities and zero for text.  Visual
    entities and text runs are inferred only from objective input structure:
    contiguous visual blocks and their complementary text runs.  ``sample_ids``
    prevents packed examples from sharing entity ids or text offsets.
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("Entity-aware GIRF expects position_ids [3, batch, sequence]")
    if visual_mask.shape != position_ids.shape[1:]:
        raise ValueError("visual_mask and position_ids disagree")
    batch, sequence = visual_mask.shape
    if valid_mask is None:
        valid_mask = torch.ones_like(visual_mask, dtype=torch.bool)
    if valid_mask.shape != visual_mask.shape:
        raise ValueError("valid_mask and visual_mask disagree")
    if sample_ids is None:
        sample_ids = torch.arange(batch, device=visual_mask.device)[:, None].expand(batch, sequence)
    if sample_ids.shape != visual_mask.shape:
        raise ValueError("sample_ids and visual_mask disagree")
    if (visual_entity_ids is None) != (visual_temporal_ids is None):
        raise ValueError("visual entity and temporal ids must be provided together")
    if visual_entity_ids is not None:
        if visual_entity_ids.shape != visual_mask.shape:
            raise ValueError("visual_entity_ids and visual_mask disagree")
        if visual_temporal_ids.shape != visual_mask.shape:
            raise ValueError("visual_temporal_ids and visual_mask disagree")
        if bool((visual_entity_ids[visual_mask] <= 0).any()):
            raise ValueError("visual tokens must have positive explicit entity ids")
        if bool((visual_temporal_ids[visual_mask] < 0).any()):
            raise ValueError("visual temporal ids must be non-negative")

    coordinates = torch.zeros(batch, sequence, 5, device=position_ids.device, dtype=torch.float32)
    for batch_idx in range(batch):
        valid_samples = torch.unique(sample_ids[batch_idx, valid_mask[batch_idx]], sorted=False)
        for logical_sample in valid_samples:
            member = valid_mask[batch_idx] & (sample_ids[batch_idx] == logical_sample)
            member_indices = torch.nonzero(member, as_tuple=False).flatten()
            if member_indices.numel() == 0:
                continue

            # A text segment is a maximal contiguous run between visual blocks.
            text_indices = torch.nonzero(member & ~visual_mask[batch_idx], as_tuple=False).flatten()
            for run in _contiguous_runs(text_indices):
                offset = torch.arange(run.numel(), device=run.device, dtype=torch.float32)
                coordinates[batch_idx, run, 0] = offset

            visual_indices = torch.nonzero(member & visual_mask[batch_idx], as_tuple=False).flatten()
            for entity_id, run in enumerate(_contiguous_runs(visual_indices)):
                if visual_entity_ids is None:
                    # Compatibility fallback for image-only/legacy callers.
                    coordinates[batch_idx, run, 1] = float(entity_id + 1)
                    temporal = position_ids[0, batch_idx, run].float()
                    coordinates[batch_idx, run, 2] = temporal - temporal.min()
                else:
                    coordinates[batch_idx, run, 1] = visual_entity_ids[batch_idx, run].float()
                    coordinates[batch_idx, run, 2] = visual_temporal_ids[batch_idx, run].float()

                # Y/X represent relative location inside each visual entity.
                for relation_axis, position_axis in ((3, 1), (4, 2)):
                    values = position_ids[position_axis, batch_idx, run].float()
                    low, high = values.min(), values.max()
                    span = high - low
                    if bool(span > 0):
                        coordinates[batch_idx, run, relation_axis] = 2.0 * (values - low) / span - 1.0
    return coordinates


def _largest_remainder(total: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    if total <= 0:
        raise ValueError("allocation total must be positive")
    weight_sum = sum(weights)
    raw = [total * weight / weight_sum for weight in weights]
    result = [int(value) for value in raw]
    remainder = total - sum(result)
    order = sorted(
        range(len(weights)),
        key=lambda idx: (raw[idx] - result[idx], weights[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        result[idx] += 1
    return tuple(result)


def resolve_entity_aware_axis_counts(pair_count: int) -> tuple[int, int, int, int, int]:
    """Resolve the no-NoPE T/E/Z/Y/X budget for an arbitrary head size."""
    return _largest_remainder(pair_count, ENTITY_AWARE_AXIS_WEIGHTS)


def _geometric_values(
    start: float, end: float, count: int, *, device: torch.device
) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    if count == 1:
        return torch.tensor([(start * end) ** 0.5], device=device, dtype=torch.float32)
    return torch.logspace(
        math.log10(start), math.log10(end), count, device=device, dtype=torch.float32
    )


def _axis_frequency_sequence(
    axis: int,
    count: int,
    device: torch.device,
    band_weights: tuple[tuple[int, int, int], ...],
    long_range_bands: tuple[tuple[float, float], ...],
    spatial_bands: tuple[tuple[float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one axis-local, band-interleaved frequency sequence."""
    band_counts = _largest_remainder(count, band_weights[axis])
    ranges = spatial_bands if axis in (3, 4) else long_range_bands
    bands = [
        list(_geometric_values(start, end, band_count, device=device).unbind())
        for (start, end), band_count in zip(ranges, band_counts)
    ]
    sequence = []
    band_ids = []
    offsets = [0, 0, 0]
    while len(sequence) < count:
        for band in range(3):
            if offsets[band] < len(bands[band]):
                sequence.append(bands[band][offsets[band]])
                band_ids.append(band)
                offsets[band] += 1
    return (
        torch.stack(sequence),
        torch.tensor(band_ids, device=device, dtype=torch.long),
    )


def entity_aware_pair_layout(
    pair_count: int,
    device: torch.device | str = "cpu",
    axis_weights: list[int] | tuple[int, ...] = ENTITY_AWARE_AXIS_WEIGHTS,
    band_weights: list[list[int]] | tuple[tuple[int, ...], ...] = ENTITY_AWARE_BAND_WEIGHTS,
    long_range_bands: list[list[float]] | tuple[tuple[float, ...], ...] = ENTITY_AWARE_LONG_RANGE_BANDS,
    spatial_bands: list[list[float]] | tuple[tuple[float, ...], ...] = ENTITY_AWARE_SPATIAL_BANDS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pair axis ids, H/M/L band ids, and independent frequencies."""
    device = torch.device(device)
    axis_weights = tuple(int(value) for value in axis_weights)
    band_weights = tuple(tuple(int(value) for value in row) for row in band_weights)
    long_range_bands = tuple(tuple(float(value) for value in row) for row in long_range_bands)
    spatial_bands = tuple(tuple(float(value) for value in row) for row in spatial_bands)
    if len(axis_weights) != 5 or any(value <= 0 for value in axis_weights):
        raise ValueError("entity_aware_girf_axis_weights must contain five positive integers")
    if len(band_weights) != 5 or any(len(row) != 3 or any(value < 0 for value in row) for row in band_weights):
        raise ValueError("entity_aware_girf_band_weights must be a non-negative 5x3 matrix")
    if any(sum(row) <= 0 for row in band_weights):
        raise ValueError("every GIRF axis must allocate at least one frequency band")
    for name, ranges in (("long_range_bands", long_range_bands), ("spatial_bands", spatial_bands)):
        if len(ranges) != 3 or any(len(row) != 2 or row[0] <= 0 or row[1] <= 0 for row in ranges):
            raise ValueError(f"entity_aware_girf_{name} must contain three positive [start, end] pairs")

    counts = _largest_remainder(pair_count, axis_weights)
    remaining = list(counts)
    axis_ids = []
    while len(axis_ids) < pair_count:
        for axis in range(len(remaining)):
            if remaining[axis] > 0:
                axis_ids.append(axis)
                remaining[axis] -= 1

    axis_layouts = [
        _axis_frequency_sequence(
            axis, count, device, band_weights, long_range_bands, spatial_bands
        )
        for axis, count in enumerate(counts)
    ]
    offsets = [0] * len(counts)
    frequencies = []
    pair_band_ids = []
    for axis in axis_ids:
        axis_frequencies, axis_band_ids = axis_layouts[axis]
        frequencies.append(axis_frequencies[offsets[axis]])
        pair_band_ids.append(axis_band_ids[offsets[axis]])
        offsets[axis] += 1
    return (
        torch.tensor(axis_ids, device=device, dtype=torch.long),
        torch.stack(pair_band_ids),
        torch.stack(frequencies),
    )


def apply_entity_aware_girf_rotary(
    x: torch.Tensor,
    coordinates: torch.Tensor,
    spatial_scales: list[float] | tuple[float, float] = (16.0, 16.0),
    axis_weights: list[int] | tuple[int, ...] = ENTITY_AWARE_AXIS_WEIGHTS,
    band_weights: list[list[int]] | tuple[tuple[int, ...], ...] = ENTITY_AWARE_BAND_WEIGHTS,
    long_range_bands: list[list[float]] | tuple[tuple[float, ...], ...] = ENTITY_AWARE_LONG_RANGE_BANDS,
    spatial_bands: list[list[float]] | tuple[tuple[float, ...], ...] = ENTITY_AWARE_SPATIAL_BANDS,
) -> torch.Tensor:
    """Apply simplified V3 GIRF with interleaved axes and independent spectra."""
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("x must be [batch, heads, sequence, even_head_dim]")
    if coordinates.shape != (x.shape[0], x.shape[2], 5):
        raise ValueError("coordinates must be [batch, sequence, 5]")
    if len(spatial_scales) != 2:
        raise ValueError("entity_aware_girf_spatial_scales must contain Y/X scales")

    pair_count = x.shape[-1] // 2
    coords = coordinates.to(device=x.device, dtype=torch.float32)
    coords = coords.clone()
    coords[..., 3] *= float(spatial_scales[0])
    coords[..., 4] *= float(spatial_scales[1])
    pair_axes, _pair_band_ids, frequencies = entity_aware_pair_layout(
        pair_count,
        x.device,
        axis_weights,
        band_weights,
        long_range_bands,
        spatial_bands,
    )
    phase_values = coords.index_select(-1, pair_axes)
    angles = phase_values * frequencies.view(1, 1, -1)
    full_angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return x * full_angles.cos().to(x.dtype) + _rotate_half(x) * full_angles.sin().to(x.dtype)


def implicit_entity_graph(
    logits: torch.Tensor,
    text_segment_ids: torch.Tensor,
    key_entity_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate existing attention logits into a segment-to-entity graph.

    This diagnostic/training helper has no parameters and is not fed back into
    position encoding. ``logits`` may be ``[heads, query, key]`` or
    ``[query, key]``. Entity-size normalization prevents larger images from
    winning solely because they contain more visual tokens.
    """
    if logits.ndim == 3:
        logits = torch.logsumexp(logits.float(), dim=0) - math.log(logits.shape[0])
    elif logits.ndim != 2:
        raise ValueError("logits must be [query, key] or [heads, query, key]")
    if text_segment_ids.shape != (logits.shape[0],):
        raise ValueError("text_segment_ids must match the query dimension")
    if key_entity_ids.shape != (logits.shape[1],):
        raise ValueError("key_entity_ids must match the key dimension")

    segments = torch.unique(text_segment_ids[text_segment_ids >= 0], sorted=True)
    entities = torch.unique(key_entity_ids[key_entity_ids >= 0], sorted=True)
    graph_logits = torch.empty(
        segments.numel(), entities.numel(), device=logits.device, dtype=torch.float32
    )
    for row, segment in enumerate(segments):
        q_mask = text_segment_ids == segment
        for column, entity in enumerate(entities):
            k_mask = key_entity_ids == entity
            selected = logits[q_mask][:, k_mask]
            graph_logits[row, column] = torch.logsumexp(selected.flatten(), dim=0) - math.log(
                max(1, int(k_mask.sum()))
            )
    graph = torch.softmax(graph_logits, dim=-1) if entities.numel() else graph_logits
    return graph, segments, entities


def entity_graph_cross_entropy(
    graph: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Optional multi-target supervision for an implicit entity graph."""
    if graph.shape != target.shape:
        raise ValueError("graph and target must have the same [segments, entities] shape")
    target = target.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(target * graph.float().clamp_min(eps).log()).sum(dim=-1).mean()
