#!/bin/bash
# Launch the DIPE inference backends (transformers, OpenAI-compatible FastAPI).
# Default: one instance on GPU 6, port 8106. GPUs 4-5 are used by the judge.
# Uses the `eval` conda env shared by the evaluation stack.
#
# Override the GPU/port layout with INSTANCE_CONFIG (see load_balancer note):
#   INSTANCE_CONFIG='"6:8106"' bash scripts/eval/infer_backend.sh
#
# Pre-req: scripts/eval/distribution_config.env (LOCAL_IP, EVAL_MODEL_PATH, ...)

CONFIG_PATH="scripts/eval/distribution_config.env"
source $CONFIG_PATH
MODEL_PATH=${EVAL_MODEL_PATH_OVERRIDE:-$EVAL_MODEL_PATH}
POSITION_EMBEDDING=${POSITION_EMBEDDING:-mrope}
CROSS_MODAL_POSITION=${CROSS_MODAL_POSITION:-dipe}

LOCAL_IP=${LOCAL_IP:-localhost}
SERVICE_HOST=${SERVICE_HOST:-0.0.0.0}
LOG_DIR="${BACKEND_LOG_DIR:-eval_output/logs/transformers_instances}"
mkdir -p "${LOG_DIR}"

# A run-local PID manifest lets concurrent evaluation orchestrators clean up
# only the workers they own.
BACKEND_PID_FILE="${BACKEND_PID_FILE:-}"
if [[ -n "${BACKEND_PID_FILE}" ]]; then
    mkdir -p "$(dirname "${BACKEND_PID_FILE}")"
    : > "${BACKEND_PID_FILE}"
fi

# Explicit eval environment Python. The backend defaults to PyTorch SDPA because
# this environment intentionally does not require a separately compiled
# flash-attn wheel.
EVAL_PYTHON="${EVAL_PYTHON:-/opt/conda/envs/eval/bin/python}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
GIRF_RESOLUTION_GATE="${GIRF_RESOLUTION_GATE:-0}"
GIRF_ORDINAL_ENABLED="${GIRF_ORDINAL_ENABLED:-}"
GIRF_PI_MASK_ENABLED="${GIRF_PI_MASK_ENABLED:-}"
GIRF_PI_MASK_COMPONENTS="${GIRF_PI_MASK_COMPONENTS:-}"
GIRF_SOFT_VV_PENALTY="${GIRF_SOFT_VV_PENALTY:-}"
DISABLE_KV_CACHE="${DISABLE_KV_CACHE:-0}"

GIRF_ARGS=()
if [[ "${GIRF_RESOLUTION_GATE}" == "1" ]]; then
    GIRF_ARGS+=(--girf-resolution-gate)
fi
if [[ "${GIRF_ORDINAL_ENABLED}" == "1" ]]; then
    GIRF_ARGS+=(--girf-ordinal-enabled)
elif [[ "${GIRF_ORDINAL_ENABLED}" == "0" ]]; then
    GIRF_ARGS+=(--no-girf-ordinal-enabled)
fi
if [[ "${GIRF_PI_MASK_ENABLED}" == "1" ]]; then
    GIRF_ARGS+=(--girf-pi-mask-enabled)
elif [[ "${GIRF_PI_MASK_ENABLED}" == "0" ]]; then
    GIRF_ARGS+=(--no-girf-pi-mask-enabled)
fi
if [[ -n "${GIRF_PI_MASK_COMPONENTS}" ]]; then
    GIRF_ARGS+=(--girf-pi-mask-components "${GIRF_PI_MASK_COMPONENTS}")
fi
if [[ -n "${GIRF_SOFT_VV_PENALTY}" ]]; then
    GIRF_ARGS+=(--girf-soft-vv-penalty "${GIRF_SOFT_VV_PENALTY}")
fi
if [[ "${DISABLE_KV_CACHE}" == "1" ]]; then
    GIRF_ARGS+=(--disable-kv-cache)
fi

# Default: GPU 6, port 8106 (avoid GPUs 4-5 reserved for the judge).
if [[ -z "${INSTANCE_CONFIG:-}" ]]; then
    INSTANCE_CONFIG='"6:8106"'
fi

echo "============================================"
echo "Starting DIPE backend instances (env=eval)..."
echo "Model: ${MODEL_PATH}"
echo "Local IP: ${LOCAL_IP}"
echo "Bind host: ${SERVICE_HOST}"
echo "Instances: ${INSTANCE_CONFIG}"
echo "Position embedding: ${POSITION_EMBEDDING}"
echo "Cross-modal position: ${CROSS_MODAL_POSITION}"
echo "GIRF resolution gate: ${GIRF_RESOLUTION_GATE}"
echo "GIRF ordinal: ${GIRF_ORDINAL_ENABLED:-checkpoint-default}"
echo "GIRF PI mask: ${GIRF_PI_MASK_ENABLED:-checkpoint-default}"
echo "GIRF PI components: ${GIRF_PI_MASK_COMPONENTS:-checkpoint-default}"
echo "GIRF Soft-VV penalty: ${GIRF_SOFT_VV_PENALTY:-checkpoint-default}"
echo "Disable KV cache: ${DISABLE_KV_CACHE}"
echo "============================================"

i=0
for CONFIG in ${INSTANCE_CONFIG}; do
    CONFIG=${CONFIG//\"/}
    GPU_ID=$(echo $CONFIG | cut -d':' -f1)
    PORT=$(echo $CONFIG | cut -d':' -f2)

    echo "Starting instance $i on GPU $GPU_ID, port $PORT..."

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${EVAL_PYTHON}" qwenvl/eval/transformers_backend.py \
        --port "${PORT}" \
        --host "${SERVICE_HOST}" \
        --model-path "${MODEL_PATH}" \
        --position-embedding "${POSITION_EMBEDDING}" \
        --cross-modal-position "${CROSS_MODAL_POSITION}" \
        --attn-implementation "${ATTN_IMPLEMENTATION}" \
        "${GIRF_ARGS[@]}" \
        > "${LOG_DIR}/instance_${i}_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    worker_pid=$!
    if [[ -n "${BACKEND_PID_FILE}" ]]; then
        printf '%s\n' "${worker_pid}" >> "${BACKEND_PID_FILE}"
    fi
    echo "Instance $i started (pid ${worker_pid})."
    i=$((i + 1))
done

echo "All backend instances launched. Logs: ${LOG_DIR}/instance_*.log"
echo "Wait ~1-2 min for model load, then check the configured backend port(s)."
