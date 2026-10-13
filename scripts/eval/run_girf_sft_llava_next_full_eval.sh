#!/bin/bash
# Evaluate girf_sft_llava_next_seed42 sequentially on image and video suites.
# Video benchmarks use 8 uniformly sampled frames to keep request size bounded.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIPE_ROOT}"

source scripts/eval/distribution_config.env
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/code/RoPE/DIPE/ckpt/girf_sft_llava_next_seed42}"
export VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-/personal/jiayu2026/code/RoPE/DIPE/VLMEvalKit}"
# Keep benchmark downloads and extracted media on the writable shared volume.
# VLMEvalKit otherwise falls back to ~/LMUData (that is /root/LMUData here).
export LMUData="${LMUData:-/personal/jiayu2026/data/LMUData}"
mkdir -p "${LMUData}"
SAVE_NAME="${GIRF_SAVE_NAME:-girf_sft_llava_next_seed42}"
POSITION_EMBEDDING="${POSITION_EMBEDDING:-mrope}"
CROSS_MODAL_POSITION="${CROSS_MODAL_POSITION:-girf}"
API_NPROC="${API_NPROC:-2}"
# This full evaluation uses three GPUs by default: two judge replicas on GPUs
# 0-1 and one model backend on GPU 2.
JUDGE_INSTANCES="${JUDGE_INSTANCES:-0:8200 1:8201}"
BACKEND_INSTANCES="${BACKEND_INSTANCES:-2:8102}"
export JUDGE_INSTANCES BACKEND_INSTANCES

# User-facing benchmark names map to labeled VLMEvalKit splits. DocVQA TEST and
# MMBench TEST do not expose labels, so VAL/DEV are used to produce local scores.
DATASETS=(
    LongVideoBench_8frame
    MLVU_8frame
    Video-MME_8frame
    HRBench4K
    HRBench8K
    POPE
    CountBenchQA
    ChartQA_TEST
    DocVQA_VAL
    RealWorldQA
    MMStar
    MMBench_DEV_EN_V11
    MMBench_DEV_CN_V11
    MMVP
    MathVision_MINI
    MathVista_MINI
)

# Skip only datasets for which Hugging Face explicitly denies gated access.
# Transient network/preflight failures remain in the list so they are visible
# during evaluation instead of being silently hidden.
if [[ "${SKIP_GATED_DATASET_CHECK:-0}" != "1" ]]; then
    mapfile -t DATASETS < <(
        /opt/conda/envs/eval/bin/python scripts/eval/filter_gated_datasets.py "${DATASETS[@]}"
    )
fi
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No accessible datasets remain after the gated-access preflight." >&2
    exit 4
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Checkpoint not found: ${MODEL_PATH}/config.json" >&2
    exit 2
fi

WORK_DIR="${DIPE_ROOT}/eval_output/${SAVE_NAME}_${POSITION_EMBEDDING//-/_}_${CROSS_MODAL_POSITION//-/_}"
mkdir -p "${WORK_DIR}"

printf "%s\n" \
    "Checkpoint : ${MODEL_PATH}" \
    "Method     : ${CROSS_MODAL_POSITION}" \
    "Datasets   : ${DATASETS[*]}" \
    "Work dir   : ${WORK_DIR}" \
    "API nproc  : ${API_NPROC}" \
    | tee "${WORK_DIR}/launch.log"

# The Qwen2.5 judge is needed by open-ended/math answer extraction. Start it
# automatically unless it is already reachable or AUTO_START_JUDGE=0 is set.
LOCAL_IP="${LOCAL_IP:-localhost}"
JUDGE_BALANCER_PORT="${JUDGE_BALANCER_PORT:-7772}"
API_KEY="${API_KEY:-sk-abc123}"
if ! curl -s -m 5 -H "Authorization: Bearer ${API_KEY}" \
    "http://${LOCAL_IP}:${JUDGE_BALANCER_PORT}/v1/models" | grep -q "qwen_25"; then
    if [[ "${AUTO_START_JUDGE:-1}" == "1" ]]; then
        echo "Judge is not running; starting it now." | tee -a "${WORK_DIR}/launch.log"
        bash scripts/eval/start_judge.sh --daemon 2>&1 | tee -a "${WORK_DIR}/judge_start.log"
        judge_rc=${PIPESTATUS[0]}
        if [[ ${judge_rc} -ne 0 ]]; then
            echo "Judge failed to start; see ${WORK_DIR}/judge_start.log" >&2
            exit ${judge_rc}
        fi
    else
        echo "Judge is unavailable and AUTO_START_JUDGE=0." >&2
        exit 3
    fi
fi

set +e
DATA_LIST="${DATASETS[*]}" \
SAVE_NAME="${SAVE_NAME}" \
API_NPROC="${API_NPROC}" \
bash scripts/eval/run_dipe_eval.sh \
    --model-path "${MODEL_PATH}" \
    --position-embedding "${POSITION_EMBEDDING}" \
    --cross-modal-position "${CROSS_MODAL_POSITION}" \
    2>&1 | tee -a "${WORK_DIR}/full_evaluation.log"
eval_rc=${PIPESTATUS[0]}
set -e

/opt/conda/envs/eval/bin/python scripts/eval/summarize_vlmeval_results.py \
    --work-dir "${WORK_DIR}" \
    --datasets "${DATASETS[@]}" \
    2>&1 | tee "${WORK_DIR}/final_summary.log"

echo "Final results: ${WORK_DIR}/final_results.md"
exit ${eval_rc}
