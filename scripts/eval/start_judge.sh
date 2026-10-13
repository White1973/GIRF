#!/bin/bash
# ============================================================================
# Start the LLM-as-judge service (one-shot): vLLM judge instances + balancer.
#
# Flow:
#   1) Launch 2 vLLM judge instances (Qwen2.5-7B-Instruct, TP=1) on ports 8200-8201.
#   2) Poll each instance's /v1/models until ready (max ~10 min).
#   3) Start the judge load balancer (least_con_balancer_vllm.py) on port 7772
#      (overridable via JUDGE_BALANCER_PORT). Kept distinct from the backend
#      balancer (7771) so judge + DIPE backend can run side by side.
#
# Env: `eval` conda env (PyTorch 2.8.0 + CUDA 12.6, vLLM 0.10.2).
# GPU: 2x1 = 2 GPUs (4-5). Leaves GPU 6 free for the DIPE backend.
#
# Usage:
#   bash scripts/eval/start_judge.sh              # foreground (blocks after balancer up)
#   bash scripts/eval/start_judge.sh --daemon     # background everything
#   bash scripts/eval/start_judge.sh --stop       # kill running judge + balancer
# ============================================================================

set -uo pipefail

CONFIG_PATH="scripts/eval/distribution_config.env"
source $CONFIG_PATH

LOCAL_IP=${LOCAL_IP:-localhost}
SERVICE_HOST=${SERVICE_HOST:-0.0.0.0}
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-7B-Instruct}
LOG_DIR="${JUDGE_LOG_DIR:-eval_output/logs/transformers_instances}"
mkdir -p ${LOG_DIR}

# `eval` env binaries.
EVAL_PYTHON="/opt/conda/envs/eval/bin/python"
EVAL_VLLM="/opt/conda/envs/eval/bin/vllm"
# Do not inherit the base image's Python-3.10/private runtime directories here.
# In particular, resolving libstdc++.so from aistudio_common instead of the
# system/PyTorch runtime can make vLLM's native extension exit with SIGSEGV.
TORCH_LIB="/opt/conda/envs/eval/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${TORCH_LIB}:/opt/conda/envs/eval/lib:/usr/local/cuda/lib64"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# The vLLM CLI imports packages that transitively import PyArrow.  On this
# image, Arrow's jemalloc background thread has repeatedly crashed before the
# API server can log anything.  The judge does not need Arrow allocations, so
# force the system pool and disable jemalloc background maintenance.
export ARROW_DEFAULT_MEMORY_POOL=system
export MALLOC_CONF="background_thread:false"

# Stop mode must not depend on a working CUDA runtime.  A caller that supplies
# PID files gets scoped cleanup; the legacy fallback remains for standalone use.
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping judge services..."
    if [[ -n "${JUDGE_PID_FILE:-}" || -n "${JUDGE_BALANCER_PID_FILE:-}" ]]; then
        if [[ -f "${JUDGE_PID_FILE:-}" ]]; then
            while IFS= read -r judge_pid; do
                [[ "${judge_pid}" =~ ^[0-9]+$ ]] || continue
                if [[ -r "/proc/${judge_pid}/cmdline" ]]; then
                    judge_cmd="$(tr '\0' ' ' < "/proc/${judge_pid}/cmdline")"
                    if [[ "${judge_cmd}" == *"vllm serve ${JUDGE_MODEL_PATH}"* ]]; then
                        kill "${judge_pid}" 2>/dev/null || true
                    fi
                fi
            done < "${JUDGE_PID_FILE}"
        fi
        if [[ -f "${JUDGE_BALANCER_PID_FILE:-}" ]]; then
            balancer_pid="$(cat "${JUDGE_BALANCER_PID_FILE}")"
            if [[ "${balancer_pid}" =~ ^[0-9]+$ ]] && [[ -r "/proc/${balancer_pid}/cmdline" ]]; then
                balancer_cmd="$(tr '\0' ' ' < "/proc/${balancer_pid}/cmdline")"
                if [[ "${balancer_cmd}" == *"least_con_balancer_vllm.py"* ]] && \
                   [[ "${balancer_cmd}" == *"--port ${JUDGE_BALANCER_PORT:-7772}"* ]]; then
                    kill "${balancer_pid}" 2>/dev/null || true
                fi
            fi
        fi
    else
        pkill -f "vllm serve ${JUDGE_MODEL_PATH}" 2>/dev/null && echo "Killed vLLM judge instances." || echo "No running vLLM judge instances."
        pkill -f "least_con_balancer_vllm.py" 2>/dev/null && echo "Killed judge balancer." || echo "No running judge balancer."
    fi
    exit 0
fi

# Fail before launching GPU workers when the numerical stack is incompatible.
# vLLM 0.10.2 imports numba in every EngineCore process; numba in this env
# requires NumPy <= 2.2.
if ! ${EVAL_PYTHON} -c \
    'import numpy, numba; from packaging.version import Version; assert Version(numpy.__version__) < Version("2.3"), f"NumPy {numpy.__version__} is incompatible with numba {numba.__version__}"' \
    2>/dev/null; then
    echo "ERROR: incompatible NumPy/Numba in the eval environment." >&2
    echo "Fix with: ${EVAL_PYTHON} -m pip install --force-reinstall numpy==2.2.6" >&2
    exit 4
fi

# Fail with an actionable message before spawning two silent vLLM processes.
if ! ${EVAL_PYTHON} -c \
    'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is False"; assert torch.cuda.device_count() > 0, "no CUDA devices visible"'; then
    echo "ERROR: PyTorch in the eval environment cannot access CUDA." >&2
    echo "Run: ${EVAL_PYTHON} -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'" >&2
    exit 5
fi

# Judge instances: whitespace-separated GPU:port pairs, TP=1 per instance.
# The full-evaluation entry point overrides this to GPUs 0-1; standalone
# invocations retain the original GPUs 4-5 default.
JUDGE_INSTANCES="${JUDGE_INSTANCES:-4:8200 5:8201}"
if [[ -n "${JUDGE_PID_FILE:-}" ]]; then
    mkdir -p "$(dirname "${JUDGE_PID_FILE}")"
    : > "${JUDGE_PID_FILE}"
fi
if [[ -n "${JUDGE_BALANCER_PID_FILE:-}" ]]; then
    mkdir -p "$(dirname "${JUDGE_BALANCER_PID_FILE}")"
    : > "${JUDGE_BALANCER_PID_FILE}"
fi
declare -A INSTANCE_CONFIG
declare -A INSTANCE_PIDS
INSTANCES=()
JUDGE_PORTS=""
instance_index=0
for instance in ${JUDGE_INSTANCES}; do
    INSTANCE_CONFIG[${instance_index}]="${instance}"
    INSTANCES+=("${instance_index}")
    port="${instance#*:}"
    JUDGE_PORTS="${JUDGE_PORTS:+${JUDGE_PORTS},}${port}"
    instance_index=$((instance_index + 1))
done

# cudaGetDeviceCount alone is too weak: it can succeed even when allocating a
# CUDA context or loading cuBLAS would crash.  Exercise each selected physical
# GPU before starting long-lived vLLM processes.
for i in "${INSTANCES[@]}"; do
    GPU_ID="${INSTANCE_CONFIG[$i]%%:*}"
    if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" "${EVAL_PYTHON}" -c \
        'import torch; x=torch.ones((64,64), device="cuda"); y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==64'; then
        echo "ERROR: CUDA allocation/cuBLAS preflight failed on physical GPU ${GPU_ID}." >&2
        echo "This is a driver/container/runtime problem; vLLM was not started." >&2
        exit 6
    fi
    echo "CUDA compute preflight passed on physical GPU ${GPU_ID}."
done

API_KEY="sk-abc123"
BALANCER_PORT=${JUDGE_BALANCER_PORT:-7772}

# --- pre-flight: free GPU check ----------------------------------------------
echo "============================================"
echo "Starting LLM judge (env=eval)"
echo "Judge model : ${JUDGE_MODEL_PATH}"
echo "served name : qwen_25"
echo "Local IP    : ${LOCAL_IP}"
echo "Bind host   : ${SERVICE_HOST}"
echo "============================================"

NCPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print (s+0)}')
if [[ "${NCPU:-0}" -gt 10000 ]]; then
    echo "WARNING: some GPUs appear busy (>10GB used). Judge layout is ${JUDGE_INSTANCES} (TP=1 each);"
    echo "         if those are held by another job, instances may fail to load. [5s]"
    echo "         Busy memory total (MB): ${NCPU}"
    sleep 5
fi

DAEMON=0
[[ "${1:-}" == "--daemon" ]] && DAEMON=1

# --- 1) launch vLLM instances -------------------------------------------------
for i in "${INSTANCES[@]}"; do
    CONFIG=${INSTANCE_CONFIG[$i]}
    GPU_ID=$(echo $CONFIG | cut -d':' -f1)
    PORT=$(echo $CONFIG | cut -d':' -f2)

    echo "[$(date +%H:%M:%S)] Starting judge instance $i on GPU $GPU_ID (TP=1), port $PORT..."

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        exec ${EVAL_VLLM} serve ${JUDGE_MODEL_PATH} \
            --served-model-name qwen_25 \
            --api-key ${API_KEY} \
            --tensor-parallel-size 1 \
            --pipeline-parallel-size 1 \
            --trust-remote-code \
            --dtype auto \
            --gpu_memory_utilization 0.70 \
            --enforce-eager \
            --port $PORT \
            --host ${SERVICE_HOST}
        rc=$?
        echo "vLLM instance ${i} exited with code ${rc}" >&2
        exit ${rc}
    ) > ${LOG_DIR}/judge_instance_${i}.log 2>&1 &

    INSTANCE_PIDS[$i]=$!
    if [[ -n "${JUDGE_PID_FILE:-}" ]]; then
        printf '%s\n' "${INSTANCE_PIDS[$i]}" >> "${JUDGE_PID_FILE}"
    fi
    echo "  -> pid ${INSTANCE_PIDS[$i]}"
done

# --- 2) wait for readiness ----------------------------------------------------
echo "============================================"
echo "Waiting for vLLM judge instances to become ready (max ~10 min)..."
echo "============================================"

HEALTH_TIMEOUT=600        # 10 min per instance
HEALTH_INTERVAL=15

is_instance_ready() {
    local port=$1
    # /v1/models returns served model list when the engine is up.
    curl -s -m 10 -H "Authorization: Bearer ${API_KEY}" \
        "http://${LOCAL_IP}:${port}/v1/models" 2>/dev/null \
        | grep -q "qwen_25"
}

all_ready=1
for i in "${INSTANCES[@]}"; do
    PORT=$(echo ${INSTANCE_CONFIG[$i]} | cut -d':' -f2)
    elapsed=0
    until is_instance_ready $PORT; do
        if ! kill -0 "${INSTANCE_PIDS[$i]}" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] FAILED: judge instance $i exited before becoming ready."
            echo "  Check log: ${LOG_DIR}/judge_instance_${i}.log"
            tail -40 "${LOG_DIR}/judge_instance_${i}.log" 2>/dev/null || true
            all_ready=0
            break
        fi
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
        if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
            echo "[$(date +%H:%M:%S)] TIMEOUT: judge instance $i (port $PORT) not ready after ${elapsed}s."
            echo "  Check log: ${LOG_DIR}/judge_instance_${i}.log"
            all_ready=0
            break
        fi
        echo "[$(date +%H:%M:%S)] instance $i (port $PORT): not ready yet (${elapsed}s)..."
    done
    if is_instance_ready $PORT; then
        echo "[$(date +%H:%M:%S)] instance $i (port $PORT): READY."
    fi
done

if [[ $all_ready -ne 1 ]]; then
    echo "Some judge instances failed to start. Aborting balancer launch."
    echo "Inspect logs under ${LOG_DIR}/judge_instance_*.log"
    exit 1
fi

# --- 3) launch the balancer ---------------------------------------------------
echo "============================================"
echo "All judge instances ready. Starting balancer on port ${BALANCER_PORT}..."
echo "============================================"

if [[ $DAEMON -eq 1 ]]; then
    # nohup + disown so the balancer survives this script's exit (true daemon).
    JUDGE_PORTS=${JUDGE_PORTS} nohup ${EVAL_PYTHON} qwenvl/eval/least_con_balancer_vllm.py \
        --port ${BALANCER_PORT} --host ${SERVICE_HOST} \
        > ${LOG_DIR}/judge_balancer.log 2>&1 &
    balancer_pid=$!
    if [[ -n "${JUDGE_BALANCER_PID_FILE:-}" ]]; then
        printf '%s\n' "${balancer_pid}" > "${JUDGE_BALANCER_PID_FILE}"
    fi
    disown 2>/dev/null
    echo "Judge balancer started in background (pid ${balancer_pid})."
    echo "  Endpoint: http://${LOCAL_IP}:${BALANCER_PORT}/v1/chat/completions"
    echo "  Log:      ${LOG_DIR}/judge_balancer.log"
    echo "  Stop with: bash $0 --stop"
else
    echo "Judge balancer running in foreground. Logs: ${LOG_DIR}/judge_balancer.log"
    echo "  Endpoint: http://${LOCAL_IP}:${BALANCER_PORT}/v1/chat/completions"
    JUDGE_PORTS=${JUDGE_PORTS} exec ${EVAL_PYTHON} qwenvl/eval/least_con_balancer_vllm.py --port ${BALANCER_PORT} --host ${SERVICE_HOST}
fi
