#!/usr/bin/env python3
"""Collect per-dataset VLMEvalKit score artifacts into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_MARKERS = ("score", "rating", "acc", "result")
SKIP_MARKERS = ("prediction", "status", "tmp")


def normalize(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6)
    if pd.isna(value):
        return None
    return value


def load_artifact(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        return None
    return [
        {str(key): normalize(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def candidates(root: Path, dataset: str) -> list[Path]:
    files: list[Path] = []
    needle = dataset.lower()
    for path in root.rglob("*"):
        if not path.is_file() or needle not in path.name.lower():
            continue
        name = path.stem.lower()
        if any(marker in name for marker in SKIP_MARKERS):
            continue
        if any(marker in name for marker in RESULT_MARKERS):
            files.append(path)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def compact(value: Any, limit: int = 500) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    status = {}
    status_file = args.work_dir / "dataset_status.tsv"
    if status_file.exists():
        frame = pd.read_csv(status_file, sep="\t")
        status = {str(row.dataset): str(row.status) for row in frame.itertuples()}

    summary: dict[str, Any] = {}
    for dataset in args.datasets:
        artifacts = candidates(args.work_dir, dataset)
        selected = artifacts[0] if artifacts else None
        entry: dict[str, Any] = {
            "status": status.get(dataset, "UNKNOWN"),
            "artifact": str(selected) if selected else None,
            "metrics": None,
        }
        if selected:
            try:
                entry["metrics"] = load_artifact(selected)
            except Exception as exc:
                entry["parse_error"] = f"{type(exc).__name__}: {exc}"
        summary[dataset] = entry

    json_path = args.work_dir / "final_results.json"
    md_path = args.work_dir / "final_results.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Evaluation results", "", "| Dataset | Status | Metrics | Artifact |", "|---|---|---|---|"]
    for dataset, entry in summary.items():
        metrics = "not found" if entry["metrics"] is None else compact(entry["metrics"])
        metrics = metrics.replace("|", "\\|")
        artifact = entry["artifact"] or "-"
        lines.append(f"| {dataset} | {entry['status']} | `{metrics}` | `{artifact}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {md_path}")
    for dataset, entry in summary.items():
        print(f"{dataset}: {entry['status']} | {compact(entry['metrics']) if entry['metrics'] is not None else 'no score artifact'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
