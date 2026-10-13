#!/usr/bin/env bash
set -euo pipefail

# Shared Qwen3-1.7B launcher for matched position/cross-modal experiments.
# Usage: bash scripts/train_qwen3_17b_pretrain_then_sft.sh {mrope|mrope_i|girf}
VARIANT="${1:-}"
case "${VARIANT}" in mrope|mrope_i|girf) ;; *)
  echo "Usage: $0 {mrope|mrope_i|girf}" >&2
  exit 2
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODELS_ROOT="${BASE_MODELS_ROOT:-/yuyan/share/users/baisikai.bsk/models}"
PRETRAIN_STEPS="${MATCHED_PRETRAIN_STEPS:-4362}"
SFT_STEPS="${MATCHED_SFT_STEPS:-5771}"

export LLM_BACKBONE=qwen3
export LLM_PATH="${LLM_PATH:-${BASE_MODELS_ROOT}/Qwen3-1.7B}"
export VISION_PATH="${VISION_PATH:-${BASE_MODELS_ROOT}/siglip2-so400m-patch16-naflex}"
export PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODELS_ROOT}/Qwen3-VL-2B-Instruct}"
export DIPE_DATA_ROOT="${DIPE_DATA_ROOT:-/yuyan/share/users/baisikai.bsk/code/RoPE/DIPE/data}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/yuyan/share/users/baisikai.bsk/code/RoPE/DIPE0922/checkpoints}"
export PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-qwen3_17b_${VARIANT}_pretrain_step${PRETRAIN_STEPS}_seed${SEED:-42}}"
export SFT_RUN_NAME="${SFT_RUN_NAME:-qwen3_17b_${VARIANT}_sft_step${SFT_STEPS}_seed${SEED:-42}}"

for required in \
  "${LLM_PATH}/config.json" \
  "${VISION_PATH}/config.json" \
  "${PROCESSOR_PATH}/config.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required model config was not found: ${required}" >&2
    exit 2
  fi
done

exec bash "${SCRIPT_DIR}/train_matched_position_two_stage.sh" "${VARIANT}"
