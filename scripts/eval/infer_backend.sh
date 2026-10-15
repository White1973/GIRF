#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${SCRIPT_DIR}/distribution_config.env"
cd "${PROJECT_ROOT}"

INSTANCE_CONFIG="${INSTANCE_CONFIG:-${BACKEND_INSTANCES}}"
MODEL_PATH="${EVAL_MODEL_PATH_OVERRIDE:-${EVAL_MODEL_PATH}}"
POSITION_EMBEDDING="${POSITION_EMBEDDING:-mrope}"
CROSS_MODAL_POSITION="${CROSS_MODAL_POSITION:-girf}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-${PROJECT_ROOT}/eval_output/logs/backends}"
BACKEND_PID_FILE="${BACKEND_PID_FILE:-${BACKEND_LOG_DIR}/backend_pids.txt}"

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "A checkpoint directory containing config.json is required; got '${MODEL_PATH}'." >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/qwenvl/eval/transformers_backend.py" ]]; then
  echo "Missing GIRF backend: ${PROJECT_ROOT}/qwenvl/eval/transformers_backend.py" >&2
  exit 2
fi

mkdir -p "${BACKEND_LOG_DIR}" "$(dirname "${BACKEND_PID_FILE}")"
temporary_pid_file="${BACKEND_PID_FILE}.tmp.$$"
: > "${temporary_pid_file}"
trap 'rm -f "${temporary_pid_file}"' EXIT

for instance in ${INSTANCE_CONFIG}; do
  gpu="${instance%%:*}"
  port="${instance#*:}"
  if [[ "${gpu}" == "${port}" || ! "${gpu}" =~ ^[0-9]+$ || ! "${port}" =~ ^[0-9]+$ ]]; then
    echo "Invalid backend instance '${instance}'; expected GPU:PORT." >&2
    exit 2
  fi

  extra_args=()
  if [[ "${GIRF_RESOLUTION_GATE:-0}" == 1 ]]; then
    extra_args+=(--girf-resolution-gate --disable-kv-cache)
    ATTN_IMPLEMENTATION=eager
  fi
  case "${GIRF_ORDINAL_ENABLED:-}" in
    1|true|True) extra_args+=(--girf-ordinal-enabled) ;;
    0|false|False) extra_args+=(--no-girf-ordinal-enabled) ;;
    "") ;;
    *) echo "GIRF_ORDINAL_ENABLED must be true or false." >&2; exit 2 ;;
  esac
  case "${GIRF_PI_MASK_ENABLED:-}" in
    1|true|True) extra_args+=(--girf-pi-mask-enabled) ;;
    0|false|False) extra_args+=(--no-girf-pi-mask-enabled) ;;
    "") ;;
    *) echo "GIRF_PI_MASK_ENABLED must be true or false." >&2; exit 2 ;;
  esac
  [[ -n "${GIRF_PI_MASK_COMPONENTS:-}" ]] && \
    extra_args+=(--girf-pi-mask-components "${GIRF_PI_MASK_COMPONENTS}")
  [[ -n "${GIRF_SOFT_VV_PENALTY:-}" ]] && \
    extra_args+=(--girf-soft-vv-penalty "${GIRF_SOFT_VV_PENALTY}")

  CUDA_VISIBLE_DEVICES="${gpu}" "${MODEL_PYTHON}" \
    qwenvl/eval/transformers_backend.py \
    --model-path "${MODEL_PATH}" \
    --position-embedding "${POSITION_EMBEDDING}" \
    --cross-modal-position "${CROSS_MODAL_POSITION}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --host "${SERVICE_HOST}" --port "${port}" \
    "${extra_args[@]}" \
    > "${BACKEND_LOG_DIR}/backend_gpu${gpu}_port${port}.log" 2>&1 &
  printf '%s\n' "$!" >> "${temporary_pid_file}"
done

mv "${temporary_pid_file}" "${BACKEND_PID_FILE}"
trap - EXIT
echo "Started $(wc -l < "${BACKEND_PID_FILE}") backend worker(s); PID file: ${BACKEND_PID_FILE}"
