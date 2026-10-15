#!/usr/bin/env bash
# Evaluate the entity-aware GIRF checkpoint on the requested image, video,
# multi-image, and long-context benchmark suites.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIPE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIPE_ROOT}"

source scripts/eval/distribution_config.env

EVAL_PYTHON="${EVAL_PYTHON:-/opt/conda/envs/eval/bin/python}"
MODEL_PATH="${MODEL_PATH:-${DIPE_ROOT}/ckpt/qwen25_3b_sft_mrope_entity_aware_girf_fresh2ep_sft1ep}"
export VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-/personal/jiayu2026/code/RoPE/DIPE/VLMEvalKit}"
SAVE_NAME="${GIRF_SAVE_NAME:-qwen25_3b_sft_mrope_entity_aware_girf_fresh2ep_sft1ep_full_eval}"
POSITION_EMBEDDING="${POSITION_EMBEDDING:-mrope}"
CROSS_MODAL_POSITION="${CROSS_MODAL_POSITION:-entity-aware-girf}"
API_NPROC="${API_NPROC:-2}"

# Default GPU allocation: judge replicas on GPUs 0-1 and the evaluated model
# on GPU 2. Override these variables when those GPUs are occupied.
JUDGE_INSTANCES="${JUDGE_INSTANCES:-0:8200 1:8201}"
BACKEND_INSTANCES="${BACKEND_INSTANCES:-2:8102}"
export JUDGE_INSTANCES BACKEND_INSTANCES

DEFAULT_DATASETS=(
    LongVideoBench_8frame
    MLVU_8frame
    Video-MME_2fps_8K
    Video-MME_2fps_16K
    Video-MME_2fps_32K
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
    BLINK
    OCRBench
    M4Bench
    MUIRBench
    VLM2Bench
    MM_NIAH_VAL
    MMDU
    MemLens_32K
    MemLens_64K
    MemLens_128K
    MemLens_256K
)

# Example: DATASETS_OVERRIDE="POPE BLINK M4Bench" bash this_script.sh
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

if [[ ! -x "${EVAL_PYTHON}" ]]; then
    echo "Evaluation Python is unavailable: ${EVAL_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Checkpoint not found: ${MODEL_PATH}/config.json" >&2
    exit 2
fi
if [[ ! -f "${VLMEVALKIT_DIR}/run.py" ]]; then
    echo "VLMEvalKit entry point not found: ${VLMEVALKIT_DIR}/run.py" >&2
    exit 2
fi

# Catch misspelled or unavailable VLMEvalKit aliases before starting GPUs.
"${EVAL_PYTHON}" - "${DATASETS[@]}" <<'PY'
import sys
from vlmeval.dataset import supported_video_datasets
from vlmeval.dataset import SUPPORTED_DATASETS

requested = sys.argv[1:]
known = set(SUPPORTED_DATASETS) | set(supported_video_datasets)
missing = [name for name in requested if name not in known]
if missing:
    print("Unregistered VLMEvalKit datasets: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print(f"Dataset registration preflight passed ({len(requested)} datasets).")
PY
if [[ $? -ne 0 ]]; then
    exit 5
fi

# Skip only datasets for which Hugging Face explicitly denies gated access.
if [[ "${SKIP_GATED_DATASET_CHECK:-0}" != "1" ]]; then
    mapfile -t DATASETS < <(
        "${EVAL_PYTHON}" scripts/eval/filter_gated_datasets.py "${DATASETS[@]}"
    )
fi
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No accessible datasets remain after the gated-access preflight." >&2
    exit 4
fi

for dataset in "${DATASETS[@]}"; do
    case "${dataset}" in
        MemLens_64K|MemLens_128K|MemLens_256K)
            echo "WARNING: ${dataset} exceeds this checkpoint's configured 32768-token native context window." >&2
            ;;
    esac
done

WORK_DIR="${DIPE_ROOT}/eval_output/${SAVE_NAME}_${POSITION_EMBEDDING//-/_}_${CROSS_MODAL_POSITION//-/_}"
mkdir -p "${WORK_DIR}"

printf "%s\n" \
    "Checkpoint : ${MODEL_PATH}" \
    "Method     : ${CROSS_MODAL_POSITION}" \
    "Datasets   : ${DATASETS[*]}" \
    "Work dir   : ${WORK_DIR}" \
    "API nproc  : ${API_NPROC}" \
    | tee "${WORK_DIR}/launch.log"

# Start the local Qwen2.5 judge only when it is not already reachable.
LOCAL_IP="${LOCAL_IP:-127.0.0.1}"
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
            exit "${judge_rc}"
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
REUSE_RESULTS="${REUSE_RESULTS:-1}" \
bash scripts/eval/run_dipe_eval.sh \
    --model-path "${MODEL_PATH}" \
    --position-embedding "${POSITION_EMBEDDING}" \
    --cross-modal-position "${CROSS_MODAL_POSITION}" \
    2>&1 | tee -a "${WORK_DIR}/full_evaluation.log"
eval_rc=${PIPESTATUS[0]}
set -e

"${EVAL_PYTHON}" scripts/eval/summarize_vlmeval_results.py \
    --work-dir "${WORK_DIR}" \
    --datasets "${DATASETS[@]}" \
    2>&1 | tee "${WORK_DIR}/final_summary.log"

echo "Final results: ${WORK_DIR}/final_results.md"
exit "${eval_rc}"
