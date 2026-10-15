#!/bin/bash
# ============================================================================
# GIRF/position-method end-to-end evaluation orchestrator.
#
# Assumes the judge is already running via `start_judge.sh` (GPU 4-5, judge
# balancer on port 7772). This script then:
#   1) Launch one DIPE backend instance (GPU 6, port 8106, env=dipe).
#   2) Poll each backend's /health until ready.
#   3) Launch the backend load balancer (least_con_balancer.py) on port 7771,
#      forwarding to 8104-8107.
#   4) Run VLMEvalKit/run.py over the dataset list, pointing:
#        - the VLM under test (api_qwen2_5_siglip_3b_instruct) at 7771
#        - the judge (qwen_25, via LOCAL_LLM + --judge-base-url) at 7772
#   5) On exit, stop the backend instances + backend balancer.
#
# GPU layout (concurrent with the judge):
#   GPU 4-5 = judge (qwen_25, vLLM)       GPU 6 = DIPE backend
#
# Usage:
#   bash scripts/eval/run_dipe_eval.sh --position-embedding mrope-interleave
#   DATA_LIST="ChartQA_TEST HRBench4K POPE RealWorldQA MMStar HRBench8K" \
#       bash scripts/eval/run_dipe_eval.sh
#   SKIP_JUDGE_CHECK=1 bash ...                        # skip judge readiness probe
#   bash scripts/eval/run_dipe_eval.sh --stop          # kill backend + balancer only
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${SCRIPT_DIR}/distribution_config.env"
CALLER_SAVE_NAME="${SAVE_NAME:-}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Missing evaluation configuration: ${CONFIG_PATH}" >&2
    exit 2
fi
source "${CONFIG_PATH}"
if [[ -n "${CALLER_SAVE_NAME}" ]]; then
    SAVE_NAME="${CALLER_SAVE_NAME}"
fi

POSITION_EMBEDDING=${POSITION_EMBEDDING:-mrope}
CROSS_MODAL_POSITION=${CROSS_MODAL_POSITION:-dipe}
GIRF_RESIDUAL_ALPHA=${GIRF_RESIDUAL_ALPHA:-}
GIRF_SINGLE_IMAGE_NATIVE=${GIRF_SINGLE_IMAGE_NATIVE:-0}
EVAL_MODEL_PATH_OVERRIDE=${EVAL_MODEL_PATH_OVERRIDE:-${EVAL_MODEL_PATH}}
EVAL_PROCESSOR_PATH_OVERRIDE=${EVAL_PROCESSOR_PATH_OVERRIDE:-${EVAL_MODEL_PATH_OVERRIDE}}
STOP_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --position-embedding)
            POSITION_EMBEDDING="$2"
            shift 2
            ;;
        --model-path)
            EVAL_MODEL_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --cross-modal-position)
            CROSS_MODAL_POSITION="$2"
            shift 2
            ;;
        --stop)
            STOP_ONLY=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 2
            ;;
    esac
done
POSITION_TAG=${POSITION_EMBEDDING//-/_}
CROSS_MODAL_TAG=${CROSS_MODAL_POSITION//-/_}
if [[ -n "${GIRF_RESIDUAL_ALPHA}" ]]; then
    GIRF_ALPHA_TAG=${GIRF_RESIDUAL_ALPHA//./p}
    CROSS_MODAL_TAG="${CROSS_MODAL_TAG}_alpha${GIRF_ALPHA_TAG}"
fi
if [[ "${GIRF_SINGLE_IMAGE_NATIVE}" == 1 ]]; then
    CROSS_MODAL_TAG="${CROSS_MODAL_TAG}_single_native"
fi

LOCAL_IP=${LOCAL_IP:-localhost}
SERVICE_HOST=${SERVICE_HOST:-0.0.0.0}
SAVE_NAME=${SAVE_NAME:-qwen25_3b_sft_dipe}
MODEL_NAME=${MODEL_NAME:-api_qwen2_5_siglip_3b_instruct}
JUDGE_NAME=${JUDGE_NAME:-qwen_25}
# A single Transformers backend is synchronous.  Keep request fan-out modest
# unless this script is deliberately redesigned for multiple replicas.
API_NPROC=${API_NPROC:-8}
# Bound retries and optionally bound the wall time of one complete dataset.
# Failed/timed-out datasets are recorded and the loop continues.
API_RETRY=${API_RETRY:-6}
JUDGE_RETRY=${JUDGE_RETRY:-${API_RETRY}}
CLIENT_REQUEST_TIMEOUT=${CLIENT_REQUEST_TIMEOUT:-${BACKEND_REQUEST_TIMEOUT:-1800}}
JUDGE_REQUEST_TIMEOUT=${JUDGE_REQUEST_TIMEOUT:-600}
# Stop a single dataset/model run after repeated backend timeouts, then move
# on to the next dataset. Set to 0 to disable this guard.
MAX_504_PER_DATASET=${MAX_504_PER_DATASET:-3}
# Stop after repeated request/connect failures even when they are not HTTP 504.
MAX_API_ERRORS_PER_DATASET=${MAX_API_ERRORS_PER_DATASET:-8}
# Skip a dataset when its live log has not changed for this many seconds.
# Disabled by default; targeted launchers can opt in without changing older runs.
MAX_NO_PROGRESS_SECONDS=${MAX_NO_PROGRESS_SECONDS:-0}
# If every backend worker process exits during a dataset, allow a short grace
# period for cleanup/log flushing and then advance to the next dataset.
BACKEND_DEAD_GRACE_SECONDS=${BACKEND_DEAD_GRACE_SECONDS:-30}
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-5}
DATASET_TIMEOUT_SECONDS=${DATASET_TIMEOUT_SECONDS:-0}
DATASET_TIMEOUT_KILL_AFTER_SECONDS=${DATASET_TIMEOUT_KILL_AFTER_SECONDS:-60}
# Public image benchmarks expect short answers.  Keeping the old 2048-token
# ceiling lets a malformed repetitive answer occupy the synchronous backend for
# many minutes, even after the client has timed out.  The per-dataset value is
# passed explicitly to VLMEvalKit below.
DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-${DIPE_MAX_TOKENS:-2048}}
SHORT_ANSWER_MAX_TOKENS=${SHORT_ANSWER_MAX_TOKENS:-128}
SHORT_ANSWER_DATASETS=${SHORT_ANSWER_DATASETS:-"HRBench4K HRBench8K VStarBench POPE CountBenchQA BLINK ChartQA_TEST DocVQA_VAL AI2D_TEST TextVQA_VAL InfoVQA_VAL OCRBench RealWorldQA MMStar MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMVP MathVision_MINI MathVista_MINI M4Bench MUIRBench MMSIBench_wo_circular"}
# A request timeout does not cancel model.generate() in the synchronous
# Transformers server.  Restart the owned worker and balancer before advancing
# so one pathological sample cannot poison the rest of the dataset queue.
BACKEND_RESTART_ATTEMPTS=${BACKEND_RESTART_ATTEMPTS:-2}
BACKEND_RESTART_READY_TIMEOUT=${BACKEND_RESTART_READY_TIMEOUT:-300}
BACKEND_STOP_GRACE_SECONDS=${BACKEND_STOP_GRACE_SECONDS:-20}
# TSV is natively supported by VLMEvalKit and avoids an unnecessary Excel
# writer dependency for large prediction tables.
export PRED_FORMAT=${PRED_FORMAT:-tsv}

for numeric_setting in API_NPROC API_RETRY JUDGE_RETRY CLIENT_REQUEST_TIMEOUT \
    JUDGE_REQUEST_TIMEOUT MAX_504_PER_DATASET MAX_API_ERRORS_PER_DATASET \
    MAX_NO_PROGRESS_SECONDS BACKEND_DEAD_GRACE_SECONDS MONITOR_INTERVAL_SECONDS \
    DATASET_TIMEOUT_SECONDS DATASET_TIMEOUT_KILL_AFTER_SECONDS DEFAULT_MAX_TOKENS \
    SHORT_ANSWER_MAX_TOKENS BACKEND_RESTART_ATTEMPTS \
    BACKEND_RESTART_READY_TIMEOUT BACKEND_STOP_GRACE_SECONDS; do
    numeric_value="${!numeric_setting}"
    if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "${numeric_setting} must be a non-negative integer; got '${numeric_value}'." >&2
        exit 2
    fi
done
if [[ "${API_NPROC}" -eq 0 || "${CLIENT_REQUEST_TIMEOUT}" -eq 0 || \
      "${JUDGE_REQUEST_TIMEOUT}" -eq 0 || "${MONITOR_INTERVAL_SECONDS}" -eq 0 || \
      "${DATASET_TIMEOUT_KILL_AFTER_SECONDS}" -eq 0 || "${DEFAULT_MAX_TOKENS}" -eq 0 || \
      "${SHORT_ANSWER_MAX_TOKENS}" -eq 0 || "${BACKEND_RESTART_READY_TIMEOUT}" -eq 0 || \
      "${BACKEND_STOP_GRACE_SECONDS}" -eq 0 ]]; then
    echo "API_NPROC, monitor interval, and request/kill-after timeouts must be positive integers." >&2
    exit 2
fi
if [[ "${DATASET_TIMEOUT_SECONDS}" -gt 0 ]] && ! command -v timeout >/dev/null 2>&1; then
    echo "GNU timeout is required when DATASET_TIMEOUT_SECONDS is enabled." >&2
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required to isolate and terminate one dataset evaluation safely." >&2
    exit 2
fi
export PYTHONUNBUFFERED=1

# Ports (must be kept in sync with start_judge.sh).
BACKEND_BALANCER_PORT=${BACKEND_BALANCER_PORT:-7771}
JUDGE_BALANCER_PORT=${JUDGE_BALANCER_PORT:-7772}
API_KEY=${API_KEY:-sk-abc123}

# One backend replica. The full-evaluation entry point overrides this to GPU 2;
# standalone invocations retain the original GPU 6 default.
BACKEND_INSTANCES="${BACKEND_INSTANCES:-6:8106}"
# Comma-separated backend ports, derived from the layout, forwarded by the balancer.
BACKEND_PORTS=$(echo ${BACKEND_INSTANCES} | sed 's/\"//g' | tr ' ' '\n' | cut -d':' -f2 | paste -sd,)

# Datasets — the standard DIPE evaluation suite. Override via DATA_LIST when a
# smaller smoke test or an additional benchmark is desired.
if [[ -z "${DATA_LIST:-}" ]]; then
    DATA_LIST="ChartQA_TEST HRBench4K POPE RealWorldQA MMStar HRBench8K"
fi

OUTPUT_DIR="eval_output"
WORK_DIR="${OUTPUT_DIR}/${SAVE_NAME}_${POSITION_TAG}_${CROSS_MODAL_TAG}"
LOG_DIR="${OUTPUT_DIR}/logs/transformers_instances"
mkdir -p ${WORK_DIR} ${LOG_DIR}

# Keep worker logs and worker PIDs local to this evaluation. This is required
# when several checkpoints are evaluated concurrently on disjoint GPU sets.
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-${WORK_DIR}/backend_logs}"
BACKEND_PID_FILE="${BACKEND_PID_FILE:-${WORK_DIR}/backend_pids.txt}"
BALANCER_PID_FILE="${BALANCER_PID_FILE:-${WORK_DIR}/balancer_pid.txt}"
mkdir -p "${BACKEND_LOG_DIR}"
BALANCER_PID=""

launch_owned_backends() {
    # Clear inherited CUDA_VISIBLE_DEVICES so INSTANCE_CONFIG always names
    # physical GPUs. infer_backend.sh atomically replaces the owned PID list.
    env -u CUDA_VISIBLE_DEVICES \
        INSTANCE_CONFIG="${BACKEND_INSTANCES}" \
        BACKEND_LOG_DIR="${BACKEND_LOG_DIR}" \
        BACKEND_PID_FILE="${BACKEND_PID_FILE}" \
        POSITION_EMBEDDING="${POSITION_EMBEDDING}" \
        CROSS_MODAL_POSITION="${CROSS_MODAL_POSITION}" \
        GIRF_RESIDUAL_ALPHA="${GIRF_RESIDUAL_ALPHA}" \
        GIRF_SINGLE_IMAGE_NATIVE="${GIRF_SINGLE_IMAGE_NATIVE}" \
        EVAL_MODEL_PATH_OVERRIDE="${EVAL_MODEL_PATH_OVERRIDE}" \
        EVAL_PROCESSOR_PATH_OVERRIDE="${EVAL_PROCESSOR_PATH_OVERRIDE}" \
        bash scripts/eval/infer_backend.sh
}

stop_owned_backends() {
    if [[ ! -f "${BACKEND_PID_FILE}" ]]; then
        return
    fi
    while IFS= read -r worker_pid; do
        [[ "${worker_pid}" =~ ^[0-9]+$ ]] || continue
        if [[ -r "/proc/${worker_pid}/cmdline" ]]; then
            worker_cmd="$(tr '\0' ' ' < "/proc/${worker_pid}/cmdline")"
            if [[ "${worker_cmd}" == *"qwenvl/eval/transformers_backend.py"* ]]; then
                kill "${worker_pid}" 2>/dev/null || true
            fi
        fi
    done < "${BACKEND_PID_FILE}"
}

stop_owned_backends_and_wait() {
    local worker_pid worker_cmd elapsed
    local -a owned_pids=()
    if [[ -f "${BACKEND_PID_FILE}" ]]; then
        while IFS= read -r worker_pid; do
            [[ "${worker_pid}" =~ ^[0-9]+$ ]] || continue
            owned_pids+=("${worker_pid}")
        done < "${BACKEND_PID_FILE}"
    fi
    stop_owned_backends
    elapsed=0
    while [[ "${elapsed}" -lt "${BACKEND_STOP_GRACE_SECONDS}" ]]; do
        local still_alive=0
        for worker_pid in "${owned_pids[@]}"; do
            if kill -0 "${worker_pid}" 2>/dev/null; then
                still_alive=1
                break
            fi
        done
        [[ "${still_alive}" -eq 0 ]] && break
        sleep 1
        elapsed=$((elapsed + 1))
    done
    for worker_pid in "${owned_pids[@]}"; do
        if [[ -r "/proc/${worker_pid}/cmdline" ]]; then
            worker_cmd="$(tr '\0' ' ' < "/proc/${worker_pid}/cmdline")"
            if [[ "${worker_cmd}" == *"qwenvl/eval/transformers_backend.py"* ]]; then
                kill -KILL "${worker_pid}" 2>/dev/null || true
            fi
        fi
    done
    : > "${BACKEND_PID_FILE}"
}

stop_owned_balancer() {
    local balancer_cmd elapsed
    if [[ -n "${BALANCER_PID}" ]] && [[ -r "/proc/${BALANCER_PID}/cmdline" ]]; then
        balancer_cmd="$(tr '\0' ' ' < "/proc/${BALANCER_PID}/cmdline")"
        if [[ "${balancer_cmd}" == *"qwenvl/eval/least_con_balancer.py"* ]]; then
            kill "${BALANCER_PID}" 2>/dev/null || true
            elapsed=0
            while kill -0 "${BALANCER_PID}" 2>/dev/null && [[ "${elapsed}" -lt 10 ]]; do
                sleep 1
                elapsed=$((elapsed + 1))
            done
            kill -KILL "${BALANCER_PID}" 2>/dev/null || true
        fi
    fi
    BALANCER_PID=""
    : > "${BALANCER_PID_FILE}"
}

stop_owned_dataset_evaluators() {
    local signal="${1:-TERM}"
    local evaluator_pid
    while IFS= read -r evaluator_pid; do
        [[ "${evaluator_pid}" =~ ^[0-9]+$ ]] || continue
        [[ "${evaluator_pid}" -ne "$$" ]] || continue
        kill -"${signal}" "${evaluator_pid}" 2>/dev/null || true
    done < <(ps -eo pid=,comm=,args= | awk -v needle="${WORK_DIR}" \
        'index($0, needle) && ($2 == "python" || $2 == "python3" || $2 == "timeout") {print $1}')
}

cleanup() {
    echo ""
    echo "Cleaning up owned backend instances + balancer..."
    stop_owned_dataset_evaluators TERM
    sleep 1
    stop_owned_dataset_evaluators KILL
    stop_owned_balancer
    stop_owned_backends_and_wait
    echo "Done. Results under ${WORK_DIR}/"
}

# Keep evaluation dependencies isolated from the custom DIPE model runtime.
# The backend process is launched by infer_backend.sh with the `dipe` env;
# VLMEvalKit and the lightweight HTTP balancer run in the `eval` env.
EVAL_PYTHON="${EVAL_PYTHON:-$(command -v python3 || command -v python)}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${PROJECT_ROOT}/VLMEvalKit}"
if [[ ! -f "${VLMEVALKIT_DIR}/run.py" ]]; then
    echo "Missing VLMEvalKit entry point: ${VLMEVALKIT_DIR}/run.py" >&2
    exit 2
fi

# --- stop mode ----------------------------------------------------------------
if [[ ${STOP_ONLY} -eq 1 ]]; then
    echo "Stopping backends recorded in ${BACKEND_PID_FILE}..."
    stop_owned_backends
    if [[ -f "${BALANCER_PID_FILE}" ]]; then
        saved_balancer_pid="$(cat "${BALANCER_PID_FILE}")"
        if [[ "${saved_balancer_pid}" =~ ^[0-9]+$ ]] && \
           [[ -r "/proc/${saved_balancer_pid}/cmdline" ]]; then
            balancer_cmd="$(tr '\0' ' ' < "/proc/${saved_balancer_pid}/cmdline")"
            if [[ "${balancer_cmd}" == *"qwenvl/eval/least_con_balancer.py"* ]]; then
                kill "${saved_balancer_pid}" 2>/dev/null || true
            fi
        fi
    fi
    exit 0
fi

# Prevent two orchestrators from writing the same model/dataset work directory
# or overwriting its worker PID manifest at the same time.
RUN_LOCK_FILE="${WORK_DIR}/.evaluation.lock"
exec {RUN_LOCK_FD}>"${RUN_LOCK_FILE}"
if ! flock -n "${RUN_LOCK_FD}"; then
    echo "Another evaluation already owns this work directory: ${WORK_DIR}" >&2
    exit 2
fi

echo "###############################################"
echo "# GIRF/position evaluation orchestrator"
echo "###############################################"
echo "VLM under test : ${MODEL_NAME}  (DIPE_API_BASE -> :${BACKEND_BALANCER_PORT})"
echo "Judge          : ${JUDGE_NAME}  (judge base url -> :${JUDGE_BALANCER_PORT})"
echo "Backend GPUs   : ${BACKEND_INSTANCES}  (ports -> ${BACKEND_PORTS})"
echo "Datasets       : ${DATA_LIST}"
echo "API parallelism: ${API_NPROC}"
echo "API retries    : ${API_RETRY} (judge=${JUDGE_RETRY})"
echo "Request timeout: ${CLIENT_REQUEST_TIMEOUT}s (judge=${JUDGE_REQUEST_TIMEOUT}s)"
echo "Dataset timeout: ${DATASET_TIMEOUT_SECONDS}s (0 means disabled)"
echo "No-progress timeout: ${MAX_NO_PROGRESS_SECONDS}s (0 means disabled)"
echo "Generation caps: short=${SHORT_ANSWER_MAX_TOKENS}, default=${DEFAULT_MAX_TOKENS} tokens"
echo "Backend restart: ${BACKEND_RESTART_ATTEMPTS} attempt(s), ready timeout=${BACKEND_RESTART_READY_TIMEOUT}s"
echo "Prediction format: ${PRED_FORMAT}"
echo "Position PE  : ${POSITION_EMBEDDING}"
echo "Cross-modal : ${CROSS_MODAL_POSITION}"
echo "Frozen alpha: ${GIRF_RESIDUAL_ALPHA:-disabled}"
echo "Single-image native bypass: ${GIRF_SINGLE_IMAGE_NATIVE}"
echo "Model path   : ${EVAL_MODEL_PATH_OVERRIDE}"
echo "Processor    : ${EVAL_PROCESSOR_PATH_OVERRIDE}"
echo "Output         : ${WORK_DIR}"
echo "###############################################"

# --- 0) probe judge readiness ------------------------------------------------
if [[ -z "${SKIP_JUDGE_CHECK:-}" ]]; then
    echo "[0/4] Probing judge balancer on :${JUDGE_BALANCER_PORT} ..."
    if curl -s -m 10 -H "Authorization: Bearer ${API_KEY}" \
            "http://${LOCAL_IP}:${JUDGE_BALANCER_PORT}/v1/models" 2>/dev/null | grep -q "${JUDGE_NAME}"; then
        echo "      judge OK."
    else
        echo "      [WARN] judge not reachable on :${JUDGE_BALANCER_PORT}."
        echo "             Start it first:  bash scripts/eval/start_judge.sh --daemon"
        echo "             Continuing anyway (datasets falling back to exact_matching will still score; "
        echo "             others will record judge failures)."
    fi
fi

# --- 1) launch backend instances ---------------------------------------------
echo "[1/4] Launching DIPE backend instances..."
launch_owned_backends

# Install cleanup immediately after worker launch, so a readiness failure does
# not leave orphan workers behind.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- 2) wait for backend readiness -------------------------------------------
echo "[2/4] Waiting for backend instances to become ready (max ~5 min)..."
health_ok() {
    local port=$1
    curl -s -m 10 "http://${LOCAL_IP}:${port}/health" 2>/dev/null | grep -q "healthy"
}
backend_process_for_port_alive() {
    local target_port="$1"
    local worker_pid worker_cmd
    [[ -f "${BACKEND_PID_FILE}" ]] || return 1
    while IFS= read -r worker_pid; do
        [[ "${worker_pid}" =~ ^[0-9]+$ ]] || continue
        [[ -r "/proc/${worker_pid}/cmdline" ]] || continue
        worker_cmd="$(tr '\0' ' ' < "/proc/${worker_pid}/cmdline")"
        if [[ "${worker_cmd}" == *"qwenvl/eval/transformers_backend.py"* ]] && \
           [[ "${worker_cmd}" == *"--port ${target_port}"* ]] && \
           kill -0 "${worker_pid}" 2>/dev/null; then
            return 0
        fi
    done < "${BACKEND_PID_FILE}"
    return 1
}
READY_TIMEOUT=300
READY_INTERVAL=10
wait_for_all_backends() {
    local ready_timeout="${1:-${READY_TIMEOUT}}"
    local port elapsed all_ready=1
    for port in $(echo "${BACKEND_PORTS}" | tr ',' ' '); do
        elapsed=0
        until health_ok "${port}"; do
            if ! backend_process_for_port_alive "${port}"; then
                echo "      FAILED: backend process for port ${port} exited before becoming ready."
                echo "      Recent backend log output:"
                tail -80 "${BACKEND_LOG_DIR}"/*_port${port}.log 2>/dev/null || true
                all_ready=0
                break
            fi
            sleep "${READY_INTERVAL}"
            elapsed=$((elapsed + READY_INTERVAL))
            if [[ "${elapsed}" -ge "${ready_timeout}" ]]; then
                echo "      TIMEOUT: backend port ${port} not healthy after ${elapsed}s."
                all_ready=0
                break
            fi
            echo "      port ${port}: not ready (${elapsed}s)..."
        done
        health_ok "${port}" && echo "      port ${port}: READY."
    done
    [[ "${all_ready}" -eq 1 ]]
}
if ! wait_for_all_backends "${READY_TIMEOUT}"; then
    echo "Some backend instances failed. Aborting. Logs: ${BACKEND_LOG_DIR}/instance_*.log"
    exit 1
fi

# --- 3) launch backend balancer ----------------------------------------------
launch_owned_balancer() {
    echo "Starting backend balancer on :${BACKEND_BALANCER_PORT} (forwarding to ${BACKEND_PORTS})..."
    LD_LIBRARY_PATH="" BACKEND_PORTS="${BACKEND_PORTS}" \
        BACKEND_REQUEST_TIMEOUT="${BACKEND_REQUEST_TIMEOUT:-1800}" \
        BACKEND_RECOVERY_WAIT_SECONDS="${BACKEND_RECOVERY_WAIT_SECONDS:-30}" \
        "${EVAL_PYTHON}" qwenvl/eval/least_con_balancer.py \
        --port "${BACKEND_BALANCER_PORT}" --host "${SERVICE_HOST}" \
        >> "${BACKEND_LOG_DIR}/backend_balancer.log" 2>&1 &
    BALANCER_PID=$!
    printf '%s\n' "${BALANCER_PID}" > "${BALANCER_PID_FILE}"
    echo "      balancer pid ${BALANCER_PID}"
}

echo "[3/4] Launching backend balancer..."
launch_owned_balancer

# Give the balancer a moment to bind.
sleep 3

# Confirm the registered VLM can be reached through the balancer.
export DIPE_API_BASE="http://${LOCAL_IP}:${BACKEND_BALANCER_PORT}/v1/chat/completions"
if curl -s -m 10 -H "Authorization: Bearer ${API_KEY}" "${DIPE_API_BASE%/*}/models" 2>/dev/null | grep -q "."; then
    echo "      backend balancer OK at ${DIPE_API_BASE}"
else
    echo "      [WARN] backend balancer /v1/models not responding yet; proceeding."
fi

restart_owned_backend_stack() {
    local reason="${1:-backend recovery}"
    local attempt
    if [[ "${BACKEND_RESTART_ATTEMPTS}" -eq 0 ]]; then
        echo "Backend restart disabled; cannot recover after ${reason}." >&2
        return 1
    fi
    for ((attempt = 1; attempt <= BACKEND_RESTART_ATTEMPTS; attempt++)); do
        echo "Recovering backend after ${reason} (attempt ${attempt}/${BACKEND_RESTART_ATTEMPTS})..."
        stop_owned_balancer
        stop_owned_backends_and_wait
        if ! launch_owned_backends; then
            echo "Backend relaunch command failed on attempt ${attempt}." >&2
            continue
        fi
        if ! wait_for_all_backends "${BACKEND_RESTART_READY_TIMEOUT}"; then
            echo "Backend failed its readiness check on attempt ${attempt}." >&2
            continue
        fi
        launch_owned_balancer
        sleep 3
        if kill -0 "${BALANCER_PID}" 2>/dev/null && \
           curl -s -m 10 -H "Authorization: Bearer ${API_KEY}" \
               "${DIPE_API_BASE%/*}/models" 2>/dev/null | grep -q "."; then
            echo "Backend stack recovered; continuing with the next dataset."
            return 0
        fi
        echo "Backend balancer failed after restart attempt ${attempt}." >&2
    done
    echo "Backend recovery exhausted after ${BACKEND_RESTART_ATTEMPTS} attempt(s)." >&2
    return 1
}

dataset_max_tokens() {
    local dataset="$1"
    case " ${SHORT_ANSWER_DATASETS} " in
        *" ${dataset} "*) printf '%s\n' "${SHORT_ANSWER_MAX_TOKENS}" ;;
        *) printf '%s\n' "${DEFAULT_MAX_TOKENS}" ;;
    esac
}

# --- 4) run the evaluation ----------------------------------------------------
echo "[4/4] Running VLMEvalKit over: ${DATA_LIST}"
echo "============================================"

STATUS_FILE="${WORK_DIR}/dataset_status.tsv"
if [[ "${APPEND_STATUS:-0}" != 1 || ! -s "${STATUS_FILE}" ]]; then
    printf "dataset\tstatus\texit_code\tlog\n" > "${STATUS_FILE}"
fi
overall_rc=0
REUSE_ARGS=()
if [[ "${REUSE_RESULTS:-1}" == "1" ]]; then
    REUSE_ARGS+=(--reuse)
fi

any_backend_ready() {
    local candidate_port
    for candidate_port in $(echo "${BACKEND_PORTS}" | tr ',' ' '); do
        if curl -s -m 2 "http://${LOCAL_IP}:${candidate_port}/health" 2>/dev/null | grep -q "healthy"; then
            return 0
        fi
    done
    return 1
}

any_backend_process_alive() {
    local candidate_port
    for candidate_port in $(echo "${BACKEND_PORTS}" | tr ',' ' '); do
        if backend_process_for_port_alive "${candidate_port}"; then
            return 0
        fi
    done
    return 1
}

stop_dataset_eval() {
    local session_pid="$1"
    local session_id pid
    session_id="$(ps -o sid= -p "${session_pid}" 2>/dev/null | tr -d ' ' || true)"
    # Every dataset command is started with setsid. Kill that isolated session
    # first, then use the work-directory match as a fallback for any descendants.
    if [[ "${session_id}" =~ ^[0-9]+$ && "${session_id}" -eq "${session_pid}" ]]; then
        kill -TERM -- "-${session_id}" 2>/dev/null || true
    else
        kill -TERM "${session_pid}" 2>/dev/null || true
    fi
    while read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done < <(ps -eo pid=,comm=,args= | awk -v needle="$WORK_DIR" \
        'index($0, needle) && ($2 == "python" || $2 == "python3" || $2 == "timeout") {print $1}')
    sleep 2
    if [[ "${session_id}" =~ ^[0-9]+$ && "${session_id}" -eq "${session_pid}" ]]; then
        kill -KILL -- "-${session_id}" 2>/dev/null || true
    else
        kill -KILL "${session_pid}" 2>/dev/null || true
    fi
    while read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        kill -KILL "$pid" 2>/dev/null || true
    done < <(ps -eo pid=,comm=,args= | awk -v needle="$WORK_DIR" \
        'index($0, needle) && ($2 == "python" || $2 == "python3" || $2 == "timeout") {print $1}')
}

run_dataset_eval() {
    local dataset_log="$1"
    shift
    : > "${dataset_log}"
    setsid "$@" >>"${dataset_log}" 2>&1 &
    local session_pid=$!
    local current_count current_error_count
    local last_log_mtime now backend_dead_since
    local api_error_pattern
    api_error_pattern='HTTP (408|429|5[0-9][0-9])|Backend timeout|Read timed out|ReadTimeout|ConnectTimeout|ConnectionError|Connection refused|RemoteDisconnected|Failed to obtain answer via API'
    last_log_mtime=$(date +%s)
    backend_dead_since=0
    while kill -0 "$session_pid" 2>/dev/null; do
        if [[ "$MAX_504_PER_DATASET" -gt 0 && -f "$dataset_log" ]]; then
            current_count=$(tail -n 5000 "$dataset_log" | tr '\r' '\n' | grep -Ec 'HTTP 504|Backend timeout' || true)
            current_count=${current_count:-0}
            if [[ "$current_count" -ge "$MAX_504_PER_DATASET" ]]; then
                echo "504 threshold reached (${current_count}/${MAX_504_PER_DATASET}); skipping current dataset/model." \
                    | tee -a "$dataset_log"
                stop_dataset_eval "$session_pid"
                wait "$session_pid" 2>/dev/null || true
                return 75
            fi
        fi
        if [[ "$MAX_API_ERRORS_PER_DATASET" -gt 0 && -f "$dataset_log" ]]; then
            current_error_count=$(tail -n 5000 "$dataset_log" | tr '\r' '\n' | grep -Eic "${api_error_pattern}" || true)
            current_error_count=${current_error_count:-0}
            if [[ "$current_error_count" -ge "$MAX_API_ERRORS_PER_DATASET" ]]; then
                echo "API error threshold reached (${current_error_count}/${MAX_API_ERRORS_PER_DATASET}); skipping current dataset/model." \
                    | tee -a "$dataset_log"
                stop_dataset_eval "$session_pid"
                wait "$session_pid" 2>/dev/null || true
                return 77
            fi
        fi
        if [[ -f "$dataset_log" ]]; then
            current_log_mtime=$(stat -c %Y "$dataset_log" 2>/dev/null || true)
            if [[ "$current_log_mtime" =~ ^[0-9]+$ ]] && [[ "$current_log_mtime" -gt "$last_log_mtime" ]]; then
                last_log_mtime="$current_log_mtime"
            fi
        fi
        if [[ "$MAX_NO_PROGRESS_SECONDS" -gt 0 ]]; then
            now=$(date +%s)
            if [[ $((now - last_log_mtime)) -ge "$MAX_NO_PROGRESS_SECONDS" ]]; then
                echo "No log progress for ${MAX_NO_PROGRESS_SECONDS}s; skipping current dataset/model." \
                    | tee -a "$dataset_log"
                stop_dataset_eval "$session_pid"
                wait "$session_pid" 2>/dev/null || true
                return 76
            fi
        fi
        now=$(date +%s)
        if any_backend_process_alive; then
            backend_dead_since=0
        elif [[ "${backend_dead_since}" -eq 0 ]]; then
            backend_dead_since="${now}"
        elif [[ "${BACKEND_DEAD_GRACE_SECONDS}" -gt 0 && \
                $((now - backend_dead_since)) -ge "${BACKEND_DEAD_GRACE_SECONDS}" ]]; then
            echo "All backend worker processes have been dead for ${BACKEND_DEAD_GRACE_SECONDS}s; skipping current dataset/model." \
                | tee -a "$dataset_log"
            stop_dataset_eval "$session_pid"
            wait "$session_pid" 2>/dev/null || true
            return 78
        fi
        sleep "${MONITOR_INTERVAL_SECONDS}"
    done
    wait "$session_pid"
}

for DATA in ${DATA_LIST}; do
    echo "========== evaluate ${DATA} =========="
    dataset_log="${WORK_DIR}/${DATA}.log"
    if ! any_backend_ready; then
        if ! restart_owned_backend_stack "unhealthy backend before ${DATA}"; then
            dataset_status="SKIPPED_BACKEND_UNAVAILABLE"
            dataset_rc=69
            overall_rc=1
            printf "%s\t%s\t%s\t%s\n" \
                "${DATA}" "${dataset_status}" "${dataset_rc}" "${dataset_log}" \
                >> "${STATUS_FILE}"
            echo "Skipping ${DATA}: backend recovery failed." | tee "${dataset_log}"
            continue
        fi
    fi

    current_max_tokens="$(dataset_max_tokens "${DATA}")"
    echo "Generation limit for ${DATA}: ${current_max_tokens} tokens"

    eval_command=(
        env "DIPE_MAX_TOKENS=${current_max_tokens}"
        "PYTHONPATH=${VLMEVALKIT_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"
        "${EVAL_PYTHON}" "${VLMEVALKIT_DIR}/run.py"
        --data "${DATA}"
        --model "${MODEL_NAME}"
        --api-nproc "${API_NPROC}"
        --retry "${API_RETRY}"
        --timeout "${CLIENT_REQUEST_TIMEOUT}"
        --max-tokens "${current_max_tokens}"
        --judge "${JUDGE_NAME}"
        --judge-base-url "http://${LOCAL_IP}:${JUDGE_BALANCER_PORT}/v1"
        --judge-key "${API_KEY}"
        --judge-api-nproc "${API_NPROC}"
        --judge-retry "${JUDGE_RETRY}"
        --judge-timeout "${JUDGE_REQUEST_TIMEOUT}"
        --work-dir "${WORK_DIR}"
        "${REUSE_ARGS[@]}"
    )
    if [[ "${DATASET_TIMEOUT_SECONDS}" -gt 0 ]]; then
        run_dataset_eval "${dataset_log}" timeout --signal=TERM \
            --kill-after="${DATASET_TIMEOUT_KILL_AFTER_SECONDS}" \
            "${DATASET_TIMEOUT_SECONDS}" \
            "${eval_command[@]}"
    else
        run_dataset_eval "${dataset_log}" "${eval_command[@]}"
    fi
    dataset_rc=$?
    restart_reason=""
    if [[ ${dataset_rc} -eq 0 ]]; then
        dataset_status="SUCCESS"
    elif [[ ${dataset_rc} -eq 75 ]]; then
        dataset_status="SKIPPED_504_THRESHOLD"
        overall_rc=1
        restart_reason="repeated 504 responses in ${DATA}"
    elif [[ ${dataset_rc} -eq 76 ]]; then
        dataset_status="SKIPPED_NO_PROGRESS"
        overall_rc=1
        restart_reason="no progress in ${DATA}"
    elif [[ ${dataset_rc} -eq 77 ]]; then
        dataset_status="SKIPPED_API_ERROR_THRESHOLD"
        overall_rc=1
        restart_reason="repeated API errors in ${DATA}"
    elif [[ ${dataset_rc} -eq 78 ]]; then
        dataset_status="SKIPPED_BACKEND_DIED"
        overall_rc=1
        restart_reason="worker death in ${DATA}"
    elif [[ ${dataset_rc} -eq 124 || ${dataset_rc} -eq 137 ]]; then
        dataset_status="TIMEOUT"
        overall_rc=1
        restart_reason="dataset timeout in ${DATA}"
    else
        dataset_status="FAILED"
        overall_rc=1
        if ! any_backend_ready; then
            restart_reason="unhealthy backend after ${DATA} failed"
        fi
    fi
    printf "%s\t%s\t%s\t%s\n" \
        "${DATA}" "${dataset_status}" "${dataset_rc}" "${dataset_log}" \
        >> "${STATUS_FILE}"
    echo "Finished ${DATA}: ${dataset_status}; log: ${dataset_log}"
    if [[ -n "${restart_reason}" ]]; then
        restart_owned_backend_stack "${restart_reason}" || true
    fi
done

echo "All datasets processed. Results under ${WORK_DIR}/"
echo "Dataset status: ${STATUS_FILE}"
exit ${overall_rc}
