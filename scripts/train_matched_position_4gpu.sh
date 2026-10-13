#!/usr/bin/env bash
set -euo pipefail

# One controlled training entry for all position/cross-modal baselines.
# Usage: bash scripts/train_matched_position_4gpu.sh {vanilla_rope|mrope|mrope_i|mmrope|circlerope|dipe|girf|girf_mrope_i|girf_mmrope|girf_circlerope} {pretrain|sft}
VARIANT="${1:-}"
STAGE="${2:-}"
if [[ "${STAGE}" != pretrain && "${STAGE}" != sft ]]; then
  echo "Usage: $0 {vanilla_rope|mrope|mrope_i|mmrope|circlerope|dipe|girf|girf_mrope_i|girf_mmrope|girf_circlerope} {pretrain|sft}" >&2
  exit 2
fi

case "${VARIANT}" in
  vanilla_rope)
    # Original 1D RoPE over the flattened text/vision token sequence.
    # Native cross-modal attention keeps this arm as the paper's plain
    # Vanilla-RoPE baseline rather than Vanilla-RoPE + DIPE.
    POSITION_EMBEDDING=vanilla-rope
    POSITION_OPTIONS='{}'
    CROSS_MODAL=native
    ;;
  mrope)
    POSITION_EMBEDDING=mrope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":2.0}'
    CROSS_MODAL=native
    ;;
  mrope_i)
    POSITION_EMBEDDING=mrope-interleave
    POSITION_OPTIONS='{"mrope_section":[24,20,20],"temporal_stride":1.0,"spatial_reset":true}'
    CROSS_MODAL=native
    ;;
  mmrope)
    POSITION_EMBEDDING=mmrope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":1.0,"position_scale":[1.0,1.0,1.0]}'
    CROSS_MODAL=native
    ;;
  circlerope)
    POSITION_EMBEDDING=circlerope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":1.0,"move_to_origin":true,"move_to_positive":false,"dff_rate":0.0,"method":"circle","radius":10.0,"alpha":0.5}'
    CROSS_MODAL=native
    ;;
  dipe)
    POSITION_EMBEDDING=mrope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":2.0}'
    CROSS_MODAL=dipe
    ;;
  girf)
    POSITION_EMBEDDING=mrope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":2.0}'
    CROSS_MODAL=girf
    ;;
  girf_mrope_i)
    # MRoPE-I controls the TT/VV dynamic path. GIRF independently replaces
    # only the TV/VT relation path in the attention implementation.
    POSITION_EMBEDDING=mrope-interleave
    POSITION_OPTIONS='{"mrope_section":[24,20,20],"temporal_stride":1.0,"spatial_reset":true}'
    CROSS_MODAL=girf
    ;;
  girf_mmrope)
    # MMRoPE's frequency-axis allocation remains on the dynamic path; GIRF
    # supplies the cross-modal relation rotary coordinates.
    POSITION_EMBEDDING=mmrope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":1.0,"position_scale":[1.0,1.0,1.0]}'
    CROSS_MODAL=girf
    ;;
  girf_circlerope)
    # Circle-CIP transforms the dynamic visual grid. The same transformed
    # position ids are deliberately provided to GIRF's normalized local axes.
    POSITION_EMBEDDING=circlerope
    POSITION_OPTIONS='{"mrope_section":[16,24,24],"temporal_stride":1.0,"move_to_origin":true,"move_to_positive":false,"dff_rate":0.0,"method":"circle","radius":10.0,"alpha":0.5}'
    CROSS_MODAL=girf
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# New-machine layout. Keep every path overrideable for a different mount or
# environment, while making the persisted checkpoint configuration portable.
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${ROOT_DIR}/../../.." && pwd)}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/dipe/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/dipe/bin/torchrun}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/deepspeed/zero2.json}"
BASE_MODELS_ROOT="${BASE_MODELS_ROOT:-${WORKSPACE_ROOT}/models}"
LLM_PATH="${LLM_PATH:-${BASE_MODELS_ROOT}/Qwen2.5-3B}"
LLM_BACKBONE="${LLM_BACKBONE:-auto}"
VISION_PATH="${VISION_PATH:-${BASE_MODELS_ROOT}/siglip2-so400m-patch16-naflex}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODELS_ROOT}/Qwen3-VL-2B-Instruct}"
DIPE_DATA_ROOT="${DIPE_DATA_ROOT:-${ROOT_DIR}/../DIPE/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
ENFORCE_MODEL_MAX_LENGTH="${ENFORCE_MODEL_MAX_LENGTH:-False}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
ATTENTION_CHUNK_ELEMENTS="${ATTENTION_CHUNK_ELEMENTS:-4194304}"
REPORT_TO="${REPORT_TO:-none}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29581}"
RESUME="${RESUME:-0}"

if [[ "${RESUME}" != 0 && "${RESUME}" != 1 ]]; then
  echo "RESUME must be 0 (default) or 1; got '${RESUME}'." >&2
  exit 2
fi

latest_resumable_checkpoint() {
  local output_dir=$1 checkpoint
  checkpoint=$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' \
    -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)
  [[ -n "${checkpoint}" ]] || return 1
  [[ -f "${output_dir}/${checkpoint}/trainer_state.json" ]] || return 1
  printf '%s\n' "${output_dir}/${checkpoint}"
}

# These are optimizer-update budgets, not approximate directory labels.  With
# the frozen data and batch settings they correspond to the intended 2-epoch
# alignment stage and 1-epoch SFT stage.  All arms must use the same values.
PRETRAIN_STEPS="${MATCHED_PRETRAIN_STEPS:-4362}"
SFT_STEPS="${MATCHED_SFT_STEPS:-5771}"
if ! [[ "${PRETRAIN_STEPS}" =~ ^[1-9][0-9]*$ && "${SFT_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Matched step budgets must be positive integers." >&2
  exit 2
fi

if [[ "${STAGE}" == pretrain ]]; then
  DATASET=llava_pretrain
  DATA_ANNOTATION="${DIPE_DATA_ROOT}/llava_pretrain_data/blip_laion_cc_sbu_558k.json"
  EPOCHS=2.0
  MAX_STEPS="${PRETRAIN_STEPS}"
  TARGET_EFFECTIVE_BATCH="${PRETRAIN_EFFECTIVE_BATCH:-256}"
  LEARNING_RATE="${PRETRAIN_LEARNING_RATE:-1e-3}"
  SAVE_STEPS="${SAVE_STEPS:-1000}"
  INIT_CKPT=None
  TUNE_MM_LLM=False
  GRADIENT_CHECKPOINTING=False
  RUN_NAME="${RUN_NAME:-matched_v1_${VARIANT}_pretrain_step${PRETRAIN_STEPS}_seed${SEED}}"
else
  DATASET=llava_next_sft
  DATA_ANNOTATION="${DIPE_DATA_ROOT}/llava_next_data/llava_next_raw_format/llava_next_raw_format_processed.json"
  EPOCHS=1.0
  MAX_STEPS="${SFT_STEPS}"
  TARGET_EFFECTIVE_BATCH="${SFT_EFFECTIVE_BATCH:-128}"
  LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
  SAVE_STEPS="${SAVE_STEPS:-500}"
  PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-matched_v1_${VARIANT}_pretrain_step${PRETRAIN_STEPS}_seed${SEED}}"
  INIT_CKPT="${INIT_CKPT:-${CHECKPOINT_ROOT}/${PRETRAIN_RUN_NAME}}"
  TUNE_MM_LLM=True
  GRADIENT_CHECKPOINTING=True
  RUN_NAME="${RUN_NAME:-matched_v1_${VARIANT}_sft_step${SFT_STEPS}_seed${SEED}}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_NAME}}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_DEVICES}"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"
if [[ "${NPROC_PER_NODE}" -ne 4 && "${NPROC_PER_NODE}" -ne 8 ]]; then
  echo "Exactly four or eight CUDA devices are required; got '${CUDA_DEVICES}'." >&2
  exit 2
fi

# Match DIPE0920's effective-batch protocol for either supported GPU layout.
MICRO_BATCH_TOTAL=$((NPROC_PER_NODE * PER_DEVICE_BATCH))
if (( TARGET_EFFECTIVE_BATCH % MICRO_BATCH_TOTAL != 0 )); then
  echo "Target effective batch ${TARGET_EFFECTIVE_BATCH} is not divisible by ${NPROC_PER_NODE} GPUs x per-device batch ${PER_DEVICE_BATCH}." >&2
  exit 2
fi
GRAD_ACCUM=$((TARGET_EFFECTIVE_BATCH / MICRO_BATCH_TOTAL))

REQUIRED=("${PYTHON_BIN}" "${TORCHRUN_BIN}" "${DATA_ANNOTATION}"
  "${LLM_PATH}/config.json" "${VISION_PATH}/config.json"
  "${PROCESSOR_PATH}/config.json" "${ROOT_DIR}/${DEEPSPEED_CONFIG}")
if [[ "${STAGE}" == sft && "${DRY_RUN:-0}" != 1 ]]; then
  REQUIRED+=("${INIT_CKPT}/config.json" "${INIT_CKPT}/trainer_state.json")
fi
for path in "${REQUIRED[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

# SFT may only inherit the completed, matching pretraining arm.
if [[ "${STAGE}" == sft && "${DRY_RUN:-0}" != 1 ]]; then
  "${PYTHON_BIN}" -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
config = json.loads((p / "config.json").read_text())
state = json.loads((p / "trainer_state.json").read_text())
text = config.get("text_config", {})
value = lambda key: config.get(key, text.get(key))
expected_position, expected_cross = sys.argv[2], sys.argv[3]
expected_options, expected_steps = json.loads(sys.argv[4]), int(sys.argv[5])
if value("position_embedding_type") != expected_position or value("cross_modal_position_type") != expected_cross:
    raise SystemExit("SFT init checkpoint has the wrong position/cross-modal method")
options = value("position_embedding_options") or {}
if any(options.get(k) != v for k, v in expected_options.items()):
    raise SystemExit(f"SFT init checkpoint has the wrong position options: {options}")
actual_steps = state.get("global_step")
if actual_steps != expected_steps:
    raise SystemExit(f"Pretraining is incomplete: {actual_steps} != {expected_steps}")
' "${INIT_CKPT}" "${POSITION_EMBEDDING}" "${CROSS_MODAL}" "${POSITION_OPTIONS}" "${PRETRAIN_STEPS}"
fi

ARGS=(
  --deepspeed "${DEEPSPEED_CONFIG}"
  --model_name_or_path "${INIT_CKPT}"
  --llm_path "${LLM_PATH}"
  --llm_backbone "${LLM_BACKBONE}"
  --visual_encoder_path "${VISION_PATH}"
  --processor_path "${PROCESSOR_PATH}"
  --position_embedding_type "${POSITION_EMBEDDING}"
  --position_embedding_options "${POSITION_OPTIONS}"
  --cross_modal_position_type "${CROSS_MODAL}"
  --attention_chunk_elements "${ATTENTION_CHUNK_ELEMENTS}"
  --attn_implementation eager
  --dataset_use "${DATASET}"
  --data_flatten True
  --enforce_model_max_length "${ENFORCE_MODEL_MAX_LENGTH}"
  --tune_mm_vision False
  --tune_mm_mlp True
  --tune_mm_llm "${TUNE_MM_LLM}"
  --llm_trainable_last_n_layers "${LLM_TRAINABLE_LAST_N_LAYERS:--1}"
  --bf16 True
  --tf32 True
  --output_dir "${OUTPUT_DIR}"
  --overwrite_output_dir False
  --num_train_epochs "${EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH}"
  --per_device_eval_batch_size "${PER_DEVICE_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_pixels 1310720
  --min_pixels 4096
  --eval_strategy no
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_safetensors True
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0.0
  --warmup_ratio 0.03
  --max_grad_norm 1.0
  --lr_scheduler_type cosine
  --logging_steps 5
  --model_max_length "${MAX_LENGTH}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --dataloader_num_workers "${NUM_WORKERS}"
  --dataloader_persistent_workers True
  --seed "${SEED}"
  --data_seed "${SEED}"
  --run_name "${RUN_NAME}"
  --report_to "${REPORT_TO}"
)

echo "matched_v1 stage=${STAGE} variant=${VARIANT} position=${POSITION_EMBEDDING} cross_modal=${CROSS_MODAL}"
echo "init=${INIT_CKPT} output=${OUTPUT_DIR} gpus=${CUDA_DEVICES} effective_batch=$((NPROC_PER_NODE * PER_DEVICE_BATCH * GRAD_ACCUM))"
echo "dataset=${DATASET} max_steps=${MAX_STEPS} seed=${SEED}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'command:'
  printf ' %q' "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" qwenvl/train/train_qwen.py "${ARGS[@]}"
  printf '\n'
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ "${RESUME}" != 1 ]]; then
    echo "Output directory is not empty: ${OUTPUT_DIR}. Set RESUME=1 to continue from its latest checkpoint." >&2
    exit 2
  fi
  RESUME_CHECKPOINT=$(latest_resumable_checkpoint "${OUTPUT_DIR}") || {
    echo "RESUME=1 requires a complete checkpoint-<step>/trainer_state.json under: ${OUTPUT_DIR}" >&2
    exit 2
  }
  echo "RESUME=1: Trainer will continue from ${RESUME_CHECKPOINT}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export DIPE_DATA_ROOT
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-girf-matched-position-v1}"
DETECTED_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${DETECTED_GPUS}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "Requested ${NPROC_PER_NODE} GPUs, but PyTorch detects ${DETECTED_GPUS}." >&2
  exit 3
fi
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
exec "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" qwenvl/train/train_qwen.py "${ARGS[@]}"
