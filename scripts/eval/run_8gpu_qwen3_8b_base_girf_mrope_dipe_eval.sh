#!/usr/bin/env bash
# Concurrent 22-dataset evaluation of three Qwen3-8B-Base SFT checkpoints.
#   GPU 0-1: shared Qwen2.5-7B judge replicas
#   GPU 2-3: GIRF inference replicas
#   GPU 4-5: native MRoPE inference replicas
#   GPU 6-7: DIPE inference replicas

set -uo pipefail

CHECK_ONLY=0
case "${1:-}" in
  --check)
    CHECK_ONLY=1
    shift
    ;;
  --help|-h)
    cat <<'EOF'
Usage:
  bash scripts/eval/run_8gpu_qwen3_8b_base_girf_mrope_dipe_eval.sh --check
  bash scripts/eval/run_8gpu_qwen3_8b_base_girf_mrope_dipe_eval.sh

The check mode validates checkpoints, Qwen3 runtime structure, datasets, and
paths without touching GPUs or starting evaluation services.
EOF
    exit 0
    ;;
esac
if [[ $# -ne 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/cuda_env.sh"
prefer_kernel_nvidia_driver
cd "${PROJECT_ROOT}"

EVAL_PYTHON="${EVAL_PYTHON:-/opt/conda/envs/eval/bin/python}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${PROJECT_ROOT}/VLMEvalKit}"
LMUData="${LMUData:-${PROJECT_ROOT}/data/LMUData}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/h20}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen3_8b_base_girf_mrope_dipe_full_eval}"
REUSE_RESULTS="${REUSE_RESULTS:-0}"
APPEND_STATUS="${APPEND_STATUS:-${REUSE_RESULTS}}"
# Existing GPU processes are warnings by default. Set REQUIRE_IDLE_GPUS=1 to
# require exclusive ownership of all eight visible GPUs.
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-0}"

# The backend proxy must time out before its client so repeated slow requests
# become explicit HTTP 504s that run_dipe_eval.sh can count and recover from.
BACKEND_REQUEST_TIMEOUT="${BACKEND_REQUEST_TIMEOUT:-600}"
CLIENT_REQUEST_TIMEOUT="${CLIENT_REQUEST_TIMEOUT:-660}"
JUDGE_REQUEST_TIMEOUT="${JUDGE_REQUEST_TIMEOUT:-600}"
DATASET_TIMEOUT_SECONDS="${DATASET_TIMEOUT_SECONDS:-21600}"
DATASET_TIMEOUT_KILL_AFTER_SECONDS="${DATASET_TIMEOUT_KILL_AFTER_SECONDS:-60}"
BACKEND_RECOVERY_WAIT_SECONDS="${BACKEND_RECOVERY_WAIT_SECONDS:-60}"
MAX_NO_PROGRESS_SECONDS="${MAX_NO_PROGRESS_SECONDS:-1200}"
MAX_504_PER_DATASET="${MAX_504_PER_DATASET:-3}"
MAX_API_ERRORS_PER_DATASET="${MAX_API_ERRORS_PER_DATASET:-6}"
BACKEND_DEAD_GRACE_SECONDS="${BACKEND_DEAD_GRACE_SECONDS:-30}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
API_RETRY="${API_RETRY:-2}"
JUDGE_RETRY="${JUDGE_RETRY:-2}"
BACKEND_RESTART_ATTEMPTS="${BACKEND_RESTART_ATTEMPTS:-2}"
BACKEND_RESTART_READY_TIMEOUT="${BACKEND_RESTART_READY_TIMEOUT:-300}"
BACKEND_STOP_GRACE_SECONDS="${BACKEND_STOP_GRACE_SECONDS:-20}"
SHORT_ANSWER_MAX_TOKENS="${SHORT_ANSWER_MAX_TOKENS:-128}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-128}"
DIPE_MAX_TOKENS="${DIPE_MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
export DIPE_MAX_TOKENS

# Public image suite, including VStarBench, AI2D, TextVQA, InfoVQA, and
# MMSIBench_wo_circular. M4Bench expands to its five registered subsets.
DATASETS="${DATASETS:-HRBench4K HRBench8K VStarBench POPE CountBenchQA BLINK ChartQA_TEST DocVQA_VAL AI2D_TEST TextVQA_VAL InfoVQA_VAL OCRBench RealWorldQA MMStar MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMVP MathVision_MINI MathVista_MINI M4Bench MUIRBench MMSIBench_wo_circular}"

MODEL_PATHS=(
  "${CHECKPOINT_ROOT}/qwen3_8b_base_girf_sft_step5771_seed42"
  "${CHECKPOINT_ROOT}/qwen3_8b_base_mrope_sft_step5771_seed42"
  "${CHECKPOINT_ROOT}/qwen3_8b_base_dipe_sft_step5771_seed42"
)
MODEL_LABELS=(
  "qwen3_8b_base_girf_step5771"
  "qwen3_8b_base_mrope_step5771"
  "qwen3_8b_base_dipe_step5771"
)
PROCESSOR_PATHS=("${MODEL_PATHS[@]}")
POSITION_TYPES=(mrope mrope mrope)
CROSS_MODAL_TYPES=(girf native dipe)
GPU_LAYOUTS=("2:8252 3:8253" "4:8264 5:8265" "6:8276 7:8277")
BALANCER_PORTS=(7822 7824 7826)
JUDGE_INSTANCES="${JUDGE_INSTANCES:-0:8240 1:8241}"
JUDGE_BALANCER_PORT="${JUDGE_BALANCER_PORT:-7820}"
ALL_OWNED_PORTS=(8252 8253 8264 8265 8276 8277 7822 7824 7826 8240 8241 7820)

RUN_ROOT="${PROJECT_ROOT}/eval_output/${OUTPUT_PREFIX}_control"
PARALLEL_LOG_DIR="${RUN_ROOT}/parallel_logs"
LOCK_FILE="${LOCK_FILE:-${RUN_ROOT}/evaluation.lock}"

for required_path in "${EVAL_PYTHON}" "${VLMEVALKIT_DIR}/run.py" "${JUDGE_MODEL_PATH}/config.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required evaluation resource is unavailable: ${required_path}" >&2
    exit 2
  fi
done

for model_path in "${MODEL_PATHS[@]}"; do
  for required in config.json preprocessor_config.json tokenizer_config.json; do
    if [[ ! -f "${model_path}/${required}" ]]; then
      echo "Checkpoint file is unavailable: ${model_path}/${required}" >&2
      exit 2
    fi
  done
  if [[ ! -f "${model_path}/model.safetensors" && \
        ! -f "${model_path}/model.safetensors.index.json" ]]; then
    echo "Checkpoint weights are unavailable under: ${model_path}" >&2
    exit 2
  fi
done

for index in "${!MODEL_PATHS[@]}"; do
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dipe0920_eval_pycache}" \
    "${EVAL_PYTHON}" scripts/eval/validate_qwen3_checkpoint.py \
      --model-path "${MODEL_PATHS[$index]}" \
      --position-embedding "${POSITION_TYPES[$index]}" \
      --cross-modal-position "${CROSS_MODAL_TYPES[$index]}" \
      --require-exact-weight-set || exit 2
done

missing_data=0
for dataset in ${DATASETS}; do
  if [[ "${dataset}" == M4Bench ]]; then
    for subset in State_Invariance State_Comparison Spatial_Perception Instance_Comparison Detailed_Difference; do
      if [[ ! -s "${LMUData}/${subset}.tsv" ]]; then
        echo "Missing M4Bench subset: ${LMUData}/${subset}.tsv" >&2
        missing_data=1
      fi
    done
  elif [[ ! -s "${LMUData}/${dataset}.tsv" ]]; then
    echo "Missing evaluation data: ${LMUData}/${dataset}.tsv" >&2
    missing_data=1
  fi
done
if [[ "${missing_data}" -ne 0 ]]; then
  exit 2
fi

PYTHONPATH="${VLMEVALKIT_DIR}:${PYTHONPATH:-}" DATASETS="${DATASETS}" LMUData="${LMUData}" \
  LITELLM_LOCAL_MODEL_COST_MAP=True \
  "${EVAL_PYTHON}" - <<'PY' || exit 2
import os
from vlmeval.dataset import SUPPORTED_DATASETS

requested = os.environ["DATASETS"].split()
missing = sorted(set(requested) - set(SUPPORTED_DATASETS))
if missing:
    raise SystemExit(f"Unsupported VLMEvalKit datasets: {', '.join(missing)}")
print(f"VALID: VLMEvalKit registers all {len(requested)} requested dataset entries")
PY

if [[ "${CHECK_ONLY}" == 1 || "${DRY_RUN:-0}" == 1 ]]; then
  echo "CHECK PASSED: all three Qwen3-8B-Base checkpoints, runtime weights, datasets, and paths are valid."
  echo "models=${MODEL_LABELS[*]}"
  printf 'checkpoint=%s\n' "${MODEL_PATHS[@]}"
  echo "datasets=${DATASETS}"
  echo "layout: judge=[0,1], GIRF=[2,3], MRoPE=[4,5], DIPE=[6,7]"
  exit 0
fi

mkdir -p "${RUN_ROOT}" "${PARALLEL_LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Qwen3-8B-Base three-method evaluation owns: ${LOCK_FILE}" >&2
  exit 2
fi

detected_gpus="$(env -u CUDA_VISIBLE_DEVICES "${EVAL_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${detected_gpus}" -lt 8 ]]; then
  echo "Eight GPUs are required, but PyTorch detects ${detected_gpus}." >&2
  exit 2
fi

for port in "${ALL_OWNED_PORTS[@]}"; do
  if "${EVAL_PYTHON}" -c 'import socket,sys; s=socket.socket(); s.settimeout(.2); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "${port}"; then
    echo "Port ${port} is already in use. Stop the old evaluation before running this script." >&2
    exit 2
  fi
done

gpu_processes=""
if ! gpu_processes="$(env -u CUDA_VISIBLE_DEVICES nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)"; then
  if [[ "${REQUIRE_IDLE_GPUS}" == 1 ]]; then
    echo "Unable to query GPU process ownership with nvidia-smi; refusing strict-idle startup." >&2
    exit 2
  fi
  echo "WARNING: unable to query existing GPU processes; continuing because REQUIRE_IDLE_GPUS=0." >&2
elif [[ -n "${gpu_processes}" ]]; then
  echo "WARNING: existing GPU compute processes detected; continuing because REQUIRE_IDLE_GPUS=0:" >&2
  printf '%s\n' "${gpu_processes}" >&2
  if [[ "${REQUIRE_IDLE_GPUS}" == 1 ]]; then
    echo "REQUIRE_IDLE_GPUS=1, so startup is blocked." >&2
    exit 2
  fi
fi

CHILD_PIDS=()
JUDGE_STARTED=0
CLEANED_UP=0
JUDGE_PID_FILE="${RUN_ROOT}/judge_pids.txt"
JUDGE_BALANCER_PID_FILE="${RUN_ROOT}/judge_balancer_pid.txt"

model_work_dir() {
  local index="$1"
  local position_tag="${POSITION_TYPES[$index]//-/_}"
  printf '%s\n' \
    "${PROJECT_ROOT}/eval_output/${OUTPUT_PREFIX}_${MODEL_LABELS[$index]}_${position_tag}_${CROSS_MODAL_TYPES[$index]}"
}

terminate_owned_model_processes() {
  local signal="$1"
  local index work_dir model_path owned_pid saved_pid saved_cmd
  for index in "${!MODEL_LABELS[@]}"; do
    work_dir="$(model_work_dir "${index}")"
    model_path="${MODEL_PATHS[$index]}"
    while IFS= read -r owned_pid; do
      [[ "${owned_pid}" =~ ^[0-9]+$ ]] || continue
      [[ "${owned_pid}" -ne "$$" ]] || continue
      kill -"${signal}" "${owned_pid}" 2>/dev/null || true
    done < <(ps -eo pid=,comm=,args= | awk -v work="${work_dir}" -v model="${model_path}" '
      (index($0, work) && ($2 == "python" || $2 == "python3" || $2 == "timeout")) ||
      (index($0, model) && ($0 ~ /run_dipe_eval[.]sh/ || $0 ~ /transformers_backend[.]py/)) {print $1}')

    if [[ -f "${work_dir}/backend_pids.txt" ]]; then
      while IFS= read -r saved_pid; do
        [[ "${saved_pid}" =~ ^[0-9]+$ ]] || continue
        [[ -r "/proc/${saved_pid}/cmdline" ]] || continue
        saved_cmd="$(tr '\0' ' ' < "/proc/${saved_pid}/cmdline")"
        if [[ "${saved_cmd}" == *"transformers_backend.py"* && \
              "${saved_cmd}" == *"${model_path}"* ]]; then
          kill -"${signal}" "${saved_pid}" 2>/dev/null || true
        fi
      done < "${work_dir}/backend_pids.txt"
    fi
    if [[ -f "${work_dir}/balancer_pid.txt" ]]; then
      saved_pid="$(cat "${work_dir}/balancer_pid.txt")"
      if [[ "${saved_pid}" =~ ^[0-9]+$ && -r "/proc/${saved_pid}/cmdline" ]]; then
        saved_cmd="$(tr '\0' ' ' < "/proc/${saved_pid}/cmdline")"
        if [[ "${saved_cmd}" == *"least_con_balancer.py"* && \
              "${saved_cmd}" == *"--port ${BALANCER_PORTS[$index]}"* ]]; then
          kill -"${signal}" "${saved_pid}" 2>/dev/null || true
        fi
      fi
    fi
  done
}

cleanup() {
  if [[ "${CLEANED_UP}" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1
  local child_pid
  for child_pid in "${CHILD_PIDS[@]}"; do
    kill "${child_pid}" 2>/dev/null || true
  done
  terminate_owned_model_processes TERM
  sleep 1
  terminate_owned_model_processes KILL
  for child_pid in "${CHILD_PIDS[@]}"; do
    wait "${child_pid}" 2>/dev/null || true
  done
  if [[ "${JUDGE_STARTED}" -eq 1 ]]; then
    JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}" \
    JUDGE_BALANCER_PORT="${JUDGE_BALANCER_PORT}" \
    JUDGE_PID_FILE="${JUDGE_PID_FILE}" \
    JUDGE_BALANCER_PID_FILE="${JUDGE_BALANCER_PID_FILE}" \
      bash scripts/eval/start_judge.sh --stop >/dev/null 2>&1 || true
  fi
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

echo "Starting shared judge on GPUs 0-1..."
# Record ownership before startup so cleanup also covers a partially started
# judge stack.
JUDGE_STARTED=1
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}" \
JUDGE_INSTANCES="${JUDGE_INSTANCES}" \
JUDGE_BALANCER_PORT="${JUDGE_BALANCER_PORT}" \
JUDGE_LOG_DIR="${RUN_ROOT}/judge_logs" \
JUDGE_PID_FILE="${JUDGE_PID_FILE}" \
JUDGE_BALANCER_PID_FILE="${JUDGE_BALANCER_PID_FILE}" \
  bash scripts/eval/start_judge.sh --daemon
judge_rc=$?
if [[ "${judge_rc}" -ne 0 ]]; then
  echo "Judge startup failed with exit code ${judge_rc}." >&2
  exit "${judge_rc}"
fi

echo "models:   ${MODEL_LABELS[*]}"
echo "datasets: ${DATASETS}"
echo "layout:   judge=[0,1], GIRF=[2,3], MRoPE=[4,5], DIPE=[6,7]"
echo "guards: request=${CLIENT_REQUEST_TIMEOUT}s dataset=${DATASET_TIMEOUT_SECONDS}s no-progress=${MAX_NO_PROGRESS_SECONDS}s api-errors=${MAX_API_ERRORS_PER_DATASET} 504s=${MAX_504_PER_DATASET}"

run_one_model() {
  local index="$1"
  local label="${MODEL_LABELS[$index]}"
  local model_path="${MODEL_PATHS[$index]}"
  local processor_path="${PROCESSOR_PATHS[$index]}"
  local position_type="${POSITION_TYPES[$index]}"
  local cross_modal_type="${CROSS_MODAL_TYPES[$index]}"
  local save_name="${OUTPUT_PREFIX}_${label}"
  local work_dir
  work_dir="$(model_work_dir "${index}")"

  DATA_LIST="${DATASETS}" \
  SAVE_NAME="${save_name}" \
  EVAL_PYTHON="${EVAL_PYTHON}" \
  VLMEVALKIT_DIR="${VLMEVALKIT_DIR}" \
  LMUData="${LMUData}" \
  EVAL_MODEL_PATH_OVERRIDE="${model_path}" \
  EVAL_PROCESSOR_PATH_OVERRIDE="${processor_path}" \
  POSITION_EMBEDDING="${position_type}" \
  CROSS_MODAL_POSITION="${cross_modal_type}" \
  BACKEND_INSTANCES="${GPU_LAYOUTS[$index]}" \
  BACKEND_BALANCER_PORT="${BALANCER_PORTS[$index]}" \
  JUDGE_BALANCER_PORT="${JUDGE_BALANCER_PORT}" \
  JUDGE_NAME=qwen_25 \
  API_NPROC=2 \
  API_RETRY="${API_RETRY}" \
  JUDGE_RETRY="${JUDGE_RETRY}" \
  BACKEND_REQUEST_TIMEOUT="${BACKEND_REQUEST_TIMEOUT}" \
  CLIENT_REQUEST_TIMEOUT="${CLIENT_REQUEST_TIMEOUT}" \
  JUDGE_REQUEST_TIMEOUT="${JUDGE_REQUEST_TIMEOUT}" \
  DATASET_TIMEOUT_SECONDS="${DATASET_TIMEOUT_SECONDS}" \
  DATASET_TIMEOUT_KILL_AFTER_SECONDS="${DATASET_TIMEOUT_KILL_AFTER_SECONDS}" \
  BACKEND_RECOVERY_WAIT_SECONDS="${BACKEND_RECOVERY_WAIT_SECONDS}" \
  MAX_NO_PROGRESS_SECONDS="${MAX_NO_PROGRESS_SECONDS}" \
  MAX_504_PER_DATASET="${MAX_504_PER_DATASET}" \
  MAX_API_ERRORS_PER_DATASET="${MAX_API_ERRORS_PER_DATASET}" \
  BACKEND_DEAD_GRACE_SECONDS="${BACKEND_DEAD_GRACE_SECONDS}" \
  MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS}" \
  BACKEND_RESTART_ATTEMPTS="${BACKEND_RESTART_ATTEMPTS}" \
  BACKEND_RESTART_READY_TIMEOUT="${BACKEND_RESTART_READY_TIMEOUT}" \
  BACKEND_STOP_GRACE_SECONDS="${BACKEND_STOP_GRACE_SECONDS}" \
  DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS}" \
  SHORT_ANSWER_MAX_TOKENS="${SHORT_ANSWER_MAX_TOKENS}" \
  REUSE_RESULTS="${REUSE_RESULTS}" \
  APPEND_STATUS="${APPEND_STATUS}" \
    bash scripts/eval/run_dipe_eval.sh \
      --model-path "${model_path}" \
      --position-embedding "${position_type}" \
      --cross-modal-position "${cross_modal_type}"
  local run_rc=$?

  "${EVAL_PYTHON}" scripts/eval/summarize_vlmeval_results.py \
    --work-dir "${work_dir}" \
    --datasets ${DATASETS} || true
  return "${run_rc}"
}

for index in "${!MODEL_LABELS[@]}"; do
  label="${MODEL_LABELS[$index]}"
  echo "Launching ${label}; log=${PARALLEL_LOG_DIR}/${label}.log"
  run_one_model "${index}" >"${PARALLEL_LOG_DIR}/${label}.log" 2>&1 &
  CHILD_PIDS+=("$!")
done

failures=0
for index in "${!CHILD_PIDS[@]}"; do
  if wait "${CHILD_PIDS[$index]}"; then
    echo "Completed ${MODEL_LABELS[$index]}"
  else
    echo "Completed with failed/timed-out datasets: ${MODEL_LABELS[$index]}" >&2
    echo "Inspect: ${PARALLEL_LOG_DIR}/${MODEL_LABELS[$index]}.log" >&2
    failures=$((failures + 1))
  fi
done
CHILD_PIDS=()

echo "Evaluation finished. Per-model status tables:"
for index in "${!MODEL_LABELS[@]}"; do
  work_dir="$(model_work_dir "${index}")"
  echo "  ${MODEL_LABELS[$index]}: ${work_dir}/dataset_status.tsv"
done

if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} model queue(s) contained failures or timeouts; successful datasets were preserved." >&2
  exit 1
fi
