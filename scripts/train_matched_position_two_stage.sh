#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
case "${VARIANT}" in vanilla_rope|mrope|mrope_i|mmrope|circlerope|dipe|girf|girf_mrope_i|girf_mmrope|girf_circlerope) ;; *)
  echo "Usage: $0 {vanilla_rope|mrope|mrope_i|mmrope|circlerope|dipe|girf|girf_mrope_i|girf_mmrope|girf_circlerope}" >&2; exit 2;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}"
PRETRAIN_STEPS="${MATCHED_PRETRAIN_STEPS:-4362}"
SFT_STEPS="${MATCHED_SFT_STEPS:-5771}"
PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-matched_v1_${VARIANT}_pretrain_step${PRETRAIN_STEPS}_seed${SEED:-42}}"
SFT_RUN_NAME="${SFT_RUN_NAME:-matched_v1_${VARIANT}_sft_step${SFT_STEPS}_seed${SEED:-42}}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-${CHECKPOINT_ROOT}/${PRETRAIN_RUN_NAME}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${CHECKPOINT_ROOT}/${SFT_RUN_NAME}}"
RESUME="${RESUME:-0}"
TRAINING_PYTHON="${PYTHON_BIN:-/opt/conda/envs/dipe/bin/python}"

pretrain_is_complete() {
  "${TRAINING_PYTHON}" -c '
import json, sys
from pathlib import Path
state_path = Path(sys.argv[1]) / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(1)
state = json.loads(state_path.read_text())
raise SystemExit(0 if state.get("global_step") == int(sys.argv[2]) else 1)
' "${PRETRAIN_OUTPUT_DIR}" "${PRETRAIN_STEPS}"
}

if [[ "${RESUME}" == 1 ]] && pretrain_is_complete; then
  echo "[1/2] ${VARIANT} matched pretraining is complete; skipping -> ${PRETRAIN_OUTPUT_DIR}"
else
  echo "[1/2] ${VARIANT} matched pretraining -> ${PRETRAIN_OUTPUT_DIR}"
  RESUME="${RESUME}" RUN_NAME="${PRETRAIN_RUN_NAME}" OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}" \
    bash "${SCRIPT_DIR}/train_matched_position_4gpu.sh" "${VARIANT}" pretrain
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "[dry run] SFT will start only after pretraining completes successfully."
else
  echo "[2/2] ${VARIANT} matched SFT -> ${SFT_OUTPUT_DIR}"
fi
RESUME="${RESUME}" RUN_NAME="${SFT_RUN_NAME}" OUTPUT_DIR="${SFT_OUTPUT_DIR}" INIT_CKPT="${PRETRAIN_OUTPUT_DIR}" \
  bash "${SCRIPT_DIR}/train_matched_position_4gpu.sh" "${VARIANT}" sft
