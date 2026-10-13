#!/usr/bin/env bash
set -euo pipefail

# Evaluate the final checkpoint produced by:
#   bash scripts/sft_entity_aware_girf_v3_4gpu.sh
#
# GPU layout:
#   GPU 0,1: local Qwen2.5 judge replicas
#   GPU 2:   Entity-aware GIRF V3 model under evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ckpt/qwen25_3b_sft_mrope_entity_aware_girf_v3_pre2ep_sft1ep}"
export GIRF_SAVE_NAME="${GIRF_SAVE_NAME:-qwen25_3b_sft_mrope_entity_aware_girf_v3_20bench}"

# Physical GPU id:service port. Keep judge and evaluated model on separate GPUs.
export JUDGE_INSTANCES="${JUDGE_INSTANCES:-0:8200 1:8201}"
export BACKEND_INSTANCES="${BACKEND_INSTANCES:-2:8102}"
export JUDGE_BALANCER_PORT="${JUDGE_BALANCER_PORT:-7772}"
export BACKEND_BALANCER_PORT="${BACKEND_BALANCER_PORT:-7771}"
export API_NPROC="${API_NPROC:-2}"
export AUTO_START_JUDGE="${AUTO_START_JUDGE:-1}"
export REUSE_RESULTS="${REUSE_RESULTS:-1}"

export POSITION_EMBEDDING="mrope"
export CROSS_MODAL_POSITION="entity-aware-girf"

export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-HRBench4K HRBench8K POPE CountBenchQA ChartQA_TEST DocVQA_VAL RealWorldQA MMStar MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMVP MathVision_MINI MathVista_MINI BLINK OCRBench M4Bench MUIRBench VLM2Bench MM_NIAH_VAL MMDU}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "SFT checkpoint not found: ${MODEL_PATH}/config.json" >&2
    exit 2
fi

printf '%s\n' \
    "Checkpoint   : ${MODEL_PATH}" \
    "Judge GPUs  : ${JUDGE_INSTANCES}" \
    "Model GPU   : ${BACKEND_INSTANCES}" \
    "Datasets    : ${DATASETS_OVERRIDE}" \
    "Result name : ${GIRF_SAVE_NAME}"

cd "${PROJECT_ROOT}"
exec bash scripts/eval/run_entity_aware_girf_fresh2ep_sft1ep_full_eval.sh
