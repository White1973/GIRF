"""Paired NIAH data and loss for conservative GIRF recovery training.

The implementation deliberately leaves attention unchanged. A logical item
contains one or two order views. The model computes vocabulary logits only at
the answer prediction positions, keeping long multi-image training tractable
without changing the model or checkpoint structure.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer

from qwenvl.data.data_processor import (
    DataCollatorForSupervisedDataset,
)
from qwenvl.data.rope2d import get_rope_index_3


IGNORE_INDEX = -100


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _render(group: dict, variant: str) -> dict:
    """Render one stored layout using the established grouped-data schema."""
    order = group["layouts"][variant]
    images = [group["image"][index] for index in order]
    target_label = f"Image {order.index(group.get('target', 0)) + 1:02d}"
    header = "\n".join(f"Image {index + 1:02d}: <image>" for index in range(len(images)))
    task = group["task"]
    if task == "mcq_order_pair":
        options = group.get("options")
        answer_index = group.get("answer_index")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"{group['id']}: MCQ pair requires exactly four options")
        if not isinstance(answer_index, int) or not 0 <= answer_index < 4:
            raise ValueError(f"{group['id']}: invalid MCQ answer index")
        rendered_options = "\n".join(
            f"{letter}. {option}" for letter, option in zip("ABCD", options)
        )
        question = (
            "Use the image collection to answer the question. Exactly one image "
            "contains the evidence needed for the correct option.\n"
            + group["question"]
            + "\n\nChoices:\n"
            + rendered_options
            + "\nReturn only A, B, C, or D."
        )
        answer = "ABCD"[answer_index]
    elif task == "caption_match":
        question = (
            "Which image best matches this description?\n"
            + group["question"]
            + "\nReturn only the image label, such as Image 01."
        )
        answer = target_label
    elif task == "binding":
        question = f"Answer the following question about {target_label} only:\n{group['question']}"
        answer = group["answer"]
    elif task in {"retrieval_qa", "single"}:
        question, answer = group["question"], group["answer"]
    else:
        raise ValueError(f"Unsupported paired NIAH task: {task}")
    return {
        "id": f"{group['id']}/{variant}",
        "image": images,
        "conversations": [
            {"from": "human", "value": header + "\n" + question},
            {"from": "gpt", "value": answer},
        ],
    }


def make_pair_entries(groups: Sequence[dict]) -> list[dict]:
    """Turn each group into deterministic logical one/two-view entries."""
    entries = []
    for group in groups:
        variants = list(group["layouts"])
        if set(variants) == {"S"}:
            pairs = [("S",)]
        elif set(variants) == {"P0", "P1"}:
            pairs = [("P0", "P1")]
        elif set(variants) == {"A", "B", "D"}:
            # B moves the designated distractor and D moves the target. Both
            # comparisons are useful; retaining A in both matches the frozen
            # grouped-data protocol.
            pairs = [("A", "B"), ("A", "D")]
        elif set(variants) == {"target_before", "distractor_before"}:
            # Stage-1b uses the exact ordering contrast measured at evaluation.
            pairs = [("target_before", "distractor_before")]
        else:
            raise ValueError(f"Unexpected layouts for {group.get('id')}: {variants}")
        invariant = group["task"] in {"binding", "retrieval_qa", "mcq_order_pair"}
        for pair_index, pair in enumerate(pairs):
            entries.append({
                "id": f"{group['id']}/pair{pair_index}",
                "views": [_render(group, variant) for variant in pair],
                "invariant": invariant and len(pair) == 2,
                "task": group["task"],
            })
    return entries


def configure_fixed_grid(processor, grid_side: int) -> None:
    if grid_side < 1:
        raise ValueError("niah_grid_side must be positive")
    image_processor = processor.image_processor
    merge = int(getattr(image_processor, "merge_size", 2))
    patch = int(getattr(image_processor, "patch_size", 16))
    raw_grid_side = grid_side * merge
    pixel_side = raw_grid_side * patch
    pixels = pixel_side * pixel_side
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = pixels
    if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
        image_processor.size["shortest_edge"] = pixels
        image_processor.size["longest_edge"] = pixels


def encode_row(row: dict, processor, grid_side: int, max_length: int) -> dict:
    """Encode one view, require a fixed visual grid, and reject truncation."""
    image_paths = iter(row["image"])
    images = []
    messages = []
    for turn in row["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        content = []
        for segment in re.split(r"(<image>)", turn["value"]):
            if segment == "<image>":
                path = next(image_paths)
                pixel_side = grid_side * int(getattr(processor.image_processor, "merge_size", 2)) * int(
                    getattr(processor.image_processor, "patch_size", 16)
                )
                with Image.open(path) as source:
                    images.append(source.convert("RGB").resize(
                        (pixel_side, pixel_side), Image.Resampling.BICUBIC
                    ))
                content.append({"type": "image"})
            elif segment.strip():
                content.append({"type": "text", "text": segment.strip()})
        messages.append({"role": role, "content": content})
    try:
        next(image_paths)
    except StopIteration:
        pass
    else:
        raise ValueError(f"{row.get('id')}: unused images")

    prompt = processor.apply_chat_template(messages, tokenize=False)
    data = processor(text=[prompt], images=images, return_tensors="pt")
    labels = torch.full_like(data["input_ids"], IGNORE_INDEX)
    tokens = data["input_ids"][0].tolist()
    position = 0
    while position < len(tokens):
        if tokens[position] == 77091:  # assistant header in this checkpoint
            answer_start = position + 2
            answer_end = answer_start
            while answer_end < len(tokens) and tokens[answer_end] != 151645:
                answer_end += 1
            if answer_end < len(tokens):
                labels[0, answer_start : answer_end + 2] = data["input_ids"][
                    0, answer_start : answer_end + 2
                ]
                position = answer_end
        position += 1
    data["labels"] = labels
    length = int(data["input_ids"].shape[1])
    if length > max_length:
        raise ValueError(f"{row.get('id')}: {length} tokens exceed max_length={max_length}")

    merge = int(getattr(processor.image_processor, "merge_size", 2))
    expected_grid = [1, grid_side * merge, grid_side * merge]
    image_grid = data.get("image_grid_thw")
    if image_grid is None:
        raise ValueError(f"{row.get('id')}: no image grid")
    if image_grid.tolist() != [expected_grid] * len(row["image"]):
        raise ValueError(
            f"{row.get('id')}: image grid {image_grid.tolist()} != expected {expected_grid}"
        )
    position_ids, static_position_ids, _ = get_rope_index_3(
        merge,
        data["input_ids"],
        image_grid_thw=image_grid,
        position_embedding_type="mrope",
        position_embedding_options={},
    )
    data["position_ids"] = position_ids
    data["static_position_ids"] = static_position_ids
    return data


class PairedNIAHDataset(Dataset):
    def __init__(self, processor, data_args, model_max_length: int):
        groups = _read_jsonl(data_args.niah_manifest)
        if not groups:
            raise ValueError("Paired NIAH manifest is empty")
        if any(group.get("split") != "train" for group in groups):
            raise ValueError("Paired NIAH training manifest must contain train rows only")
        configure_fixed_grid(processor, data_args.niah_grid_side)
        self.processor = processor
        self.grid_side = data_args.niah_grid_side
        self.max_length = model_max_length
        self.entries = make_pair_entries(groups)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        return {
            "id": entry["id"],
            "task": entry["task"],
            "invariant": entry["invariant"],
            "views": [
                encode_row(view, self.processor, self.grid_side, self.max_length)
                for view in entry["views"]
            ],
        }


@dataclass
class PairedNIAHCollator:
    tokenizer: object

    def __post_init__(self) -> None:
        self.base = DataCollatorForSupervisedDataset(self.tokenizer)

    def __call__(self, instances: Sequence[dict]) -> dict:
        # One logical pair per device keeps answer positions shared across the
        # two views and bounds selected-logit memory. Effective batch size is
        # obtained through gradient accumulation/DDP.
        if len(instances) != 1:
            raise ValueError("Paired NIAH currently requires per_device_train_batch_size=1")
        item = instances[0]
        batch = self.base(item["views"])
        labels = batch.pop("labels")
        answer_masks = labels[:, 1:].ne(IGNORE_INDEX)
        reference = answer_masks[0]
        if any(not torch.equal(reference, mask) for mask in answer_masks[1:]):
            raise ValueError(f"{item['id']}: paired views have different answer geometry")
        prediction_positions = torch.nonzero(reference, as_tuple=False).flatten()
        if prediction_positions.numel() == 0:
            raise ValueError(f"{item['id']}: no supervised answer tokens")
        targets = torch.stack(
            [labels[view, prediction_positions + 1] for view in range(labels.shape[0])]
        )
        if bool((targets == IGNORE_INDEX).any()):
            raise ValueError(f"{item['id']}: invalid selected targets")
        if item["invariant"] and not torch.equal(targets[0], targets[1]):
            raise ValueError(f"{item['id']}: invariant pair has different answer tokens")
        batch["logits_to_keep"] = prediction_positions
        batch["niah_targets"] = targets
        batch["niah_invariant"] = torch.tensor(item["invariant"], dtype=torch.bool)
        if item["task"] == "mcq_order_pair":
            candidate_ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
            if any(len(ids) != 1 for ids in candidate_ids):
                raise ValueError("MCQ candidate letters must each be a single tokenizer token")
            batch["niah_candidate_token_ids"] = torch.tensor(
                [ids[0] for ids in candidate_ids], dtype=torch.long
            )
        return batch


def paired_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    invariant: bool,
    worst_weight: float,
    js_weight: float,
    temperature: float,
    candidate_token_ids: torch.Tensor | None = None,
    gap_weight: float = 0.0,
    gap_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute paired CE plus consistency of the correct-answer margin.

    For MCQ pairs, JS alone can be small even when both views are similarly
    uncertain or wrong.  The gap term compares correct-answer margins between
    the counterfactual order views and penalizes a difference above the small
    permitted tolerance.  Per-view CE and worst-view CE prevent the degenerate
    solution of making both margins equally poor.
    """
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("Expected logits [views,tokens,vocab] and matching targets")
    if temperature <= 0 or min(worst_weight, js_weight, gap_weight, gap_tolerance) < 0:
        raise ValueError("Loss weights/tolerance must be nonnegative and temperature positive")
    token_ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    view_ce = token_ce.mean(dim=1)
    mean_ce = view_ce.mean()
    if view_ce.numel() > 1:
        smooth_worst = temperature * (
            torch.logsumexp(view_ce / temperature, dim=0)
            - math.log(view_ce.numel())
        )
    else:
        smooth_worst = mean_ce.new_zeros(())

    js = mean_ce.new_zeros(())
    margin_gap = mean_ce.new_zeros(())
    correct_margin = mean_ce.new_zeros(())
    if invariant:
        if logits.shape[0] != 2:
            raise ValueError("Invariant consistency currently requires exactly two views")
        if candidate_token_ids is not None:
            if candidate_token_ids.ndim != 1 or candidate_token_ids.numel() < 2:
                raise ValueError("Candidate JS requires at least two candidate token IDs")
            # The first supervised position predicts the one-token A/B/C/D answer.
            log_a = F.log_softmax(logits[0, 0].float()[candidate_token_ids], dim=-1)
            log_b = F.log_softmax(logits[1, 0].float()[candidate_token_ids], dim=-1)
            candidate_logits = logits[:, 0].float()[:, candidate_token_ids]
            candidate_targets = targets[:, 0]
            candidate_positions = candidate_targets.unsqueeze(1).eq(candidate_token_ids.unsqueeze(0))
            if not bool(candidate_positions.any(dim=1).all()):
                raise ValueError("MCQ target is absent from candidate token IDs")
            answer_scores = candidate_logits.masked_select(candidate_positions).reshape(-1)
            other_scores = candidate_logits.masked_fill(candidate_positions, float("-inf")).max(dim=1).values
            margins = answer_scores - other_scores
            correct_margin = margins.mean()
            margin_gap = F.relu((margins[0] - margins[1]).abs() - gap_tolerance)
        else:
            log_a = F.log_softmax(logits[0].float(), dim=-1)
            log_b = F.log_softmax(logits[1].float(), dim=-1)
        log_m = torch.logaddexp(log_a, log_b) - math.log(2.0)
        js = 0.5 * (
            (log_a.exp() * (log_a - log_m)).sum(dim=-1).mean()
            + (log_b.exp() * (log_b - log_m)).sum(dim=-1).mean()
        )
    loss = mean_ce + worst_weight * smooth_worst + js_weight * js + gap_weight * margin_gap
    return loss, {
        "mean_ce": mean_ce.detach(),
        "worst": smooth_worst.detach(),
        "js": js.detach(),
        "margin_gap": margin_gap.detach(),
        "correct_margin": correct_margin.detach(),
        "view_gap": (view_ce.max() - view_ce.min()).detach(),
    }


class PairedNIAHTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs.pop("niah_targets")
        invariant = bool(inputs.pop("niah_invariant").item())
        candidate_token_ids = inputs.pop("niah_candidate_token_ids", None)
        outputs = model(**inputs, labels=None, return_dict=True)
        loss, metrics = paired_objective(
            outputs.logits,
            targets,
            invariant,
            self.args.niah_worst_weight,
            self.args.niah_js_weight,
            self.args.niah_worst_temperature,
            candidate_token_ids,
            self.args.niah_gap_weight,
            self.args.niah_gap_tolerance,
        )
        self.niah_last_microbatch = {key: float(value) for key, value in metrics.items()}
        return (loss, outputs) if return_outputs else loss


def make_paired_niah_module(processor, data_args, training_args) -> dict:
    if training_args.per_device_train_batch_size != 1:
        raise ValueError("Paired NIAH requires --per_device_train_batch_size 1")
    dataset = PairedNIAHDataset(processor, data_args, training_args.model_max_length)
    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": PairedNIAHCollator(processor.tokenizer),
    }
