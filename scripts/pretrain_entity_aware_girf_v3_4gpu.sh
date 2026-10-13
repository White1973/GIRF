#!/usr/bin/env bash
set -euo pipefail

# Stage 1: train the multimodal projector with simplified Entity-aware GIRF V3.
# Defaults preserve the old 8-GPU pretrain global batch size:
#   4 GPUs * 8 samples/GPU * 8 accumulation steps = 256.

ROOT_DIR="${ROOT_DIR:-code/RoPE/DIPE_EntityAwareGIRF}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/dipe/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/dipe/bin/torchrun}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/deepspeed/zero2.json}"

LLM_PATH="${LLM_PATH:-models/Qwen2.5-3B}"
VISION_PATH="${VISION_PATH:-models/siglip2-so400m-patch16-naflex}"
PROCESSOR_PATH="${PROCESSOR_PATH:-models/Qwen3-VL-2B-Instruct}"
DATASET="${DATASET:-llava_pretrain}"
DATA_ANNOTATION="${DATA_ANNOTATION:-code/RoPE/DIPE/data/llava_pretrain_data/blip_laion_cc_sbu_558k.json}"

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-2.0}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SPATIAL_SCALES="${SPATIAL_SCALES:-[16, 16]}"
ATTENTION_CHUNK_ELEMENTS="${ATTENTION_CHUNK_ELEMENTS:-4194304}"
REPORT_TO="${REPORT_TO:-wandb}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"

RUN_NAME="${RUN_NAME:-qwen25_3b_pretrain_mrope_entity_aware_girf_v3_2ep}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ckpt/${RUN_NAME}}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_DEVICES}"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"
if [[ "${NPROC_PER_NODE}" -ne 4 ]]; then
  echo "This launcher requires exactly 4 CUDA devices; got '${CUDA_DEVICES}'." >&2
  exit 2
fi

for required in "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${DATA_ANNOTATION}" \
  "${LLM_PATH}/config.json" "${VISION_PATH}/config.json" \
  "${PROCESSOR_PATH}/config.json" "${ROOT_DIR}/${DEEPSPEED_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path does not exist: ${required}" >&2
    exit 2
  fi
done

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-entity-aware-girf-v3}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  DETECTED_GPUS=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
  if [[ "${DETECTED_GPUS}" -ne 4 ]]; then
    echo "Requested 4 GPUs (${CUDA_DEVICES}), but PyTorch detects ${DETECTED_GPUS}." >&2
    exit 3
  fi
fi

echo "stage=pretrain run=${RUN_NAME} method=entity-aware-girf version=v3"
echo "gpus=${CUDA_DEVICES} effective_batch=$((4 * PER_DEVICE_BATCH * GRAD_ACCUM)) output=${OUTPUT_DIR}"
echo "spatial_scales=${SPATIAL_SCALES} epochs=${EPOCHS} max_steps=${MAX_STEPS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: paths and configuration are valid; training was not launched."
  exit 0
fi

exec "${TORCHRUN_BIN}" \
  --nproc_per_node=4 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path None \
  --llm_path "${LLM_PATH}" \
  --visual_encoder_path "${VISION_PATH}" \
  --processor_path "${PROCESSOR_PATH}" \
  --position_embedding_type mrope \
  --cross_modal_position_type entity-aware-girf \
  --entity_aware_girf_spatial_scales "${SPATIAL_SCALES}" \
  --attention_chunk_elements "${ATTENTION_CHUNK_ELEMENTS}" \
  --attn_implementation eager \
  --dataset_use "${DATASET}" \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm False \
  --bf16 True \
  --tf32 True \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir False \
  --num_train_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
  --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_pixels $((32 * 32 * 1280)) \
  --min_pixels $((32 * 32 * 4)) \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_safetensors True \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --model_max_length "${MAX_LENGTH}" \
  --gradient_checkpointing False \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --dataloader_persistent_workers True \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  --report_to "${REPORT_TO}"
