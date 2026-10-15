#!/usr/bin/env python3
"""Compatibility entry point for the bundled VLMEvalKit checkout."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
VLMEVALKIT_ROOT = ROOT / "VLMEvalKit"
ENTRY_POINT = VLMEVALKIT_ROOT / "run.py"

if not ENTRY_POINT.is_file():
    raise SystemExit(f"Missing VLMEvalKit entry point: {ENTRY_POINT}")

sys.path.insert(0, str(VLMEVALKIT_ROOT))
try:
    runpy.run_path(str(ENTRY_POINT), run_name="__main__")
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing VLMEvalKit dependency '{exc.name}'. "
        "Install the evaluation environment with "
        "`bash scripts/install_eval_env.sh` or `pip install -e ./VLMEvalKit`."
    ) from None
