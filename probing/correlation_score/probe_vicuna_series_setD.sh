#!/usr/bin/env bash
# probe_vicuna_series_setD.sh
### Vicuna / Mistral / LLaVA series probing — Set D (HaluEval context + RLHF-V image) ###
set -euo pipefail

# ===== 데이터 경로 =====
CTX_JSONL="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probing_data/setD_context_halueval.jsonl"
IMG_JSONL="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probing_data/setD_images_rlhfv.jsonl"

# ===== 스크립트 파일 =====
PY="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probe_vicuna_series.py"

# ===== 모델 =====
LLM_NAME="lmsys/vicuna-7b-v1.5"
LLM_NAME2="mistralai/Mistral-7B-v0.1"
VLM_NAME="llava-hf/llava-1.5-7b-hf"
VLM_NAME2="llava-hf/llava-v1.6-vicuna-7b-hf"

# ===== 공통 하이퍼파라미터 =====
EPOCHS=200
LR=1e-2
WD=1e-3
BS=512
TEST_SIZE=0.2
SEED=42
# 빠른 테스트는 100 등으로; 전체는 0
MAX_SAMPLES=0
OUT_ROOT="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/correlation_score/outputs/vicuna_cross_dataset_probing_set_d_cv5"
mkdir -p "${OUT_ROOT}"
echo "[INFO] Output root -> ${OUT_ROOT}"

# 1) LLM (pure text)
OUT_LLM="${OUT_ROOT}/mistral"
mkdir -p "${OUT_LLM}"
python3 "${PY}" \
  --mode llm \
  --llm_name "${LLM_NAME2}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_LLM}" \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} --strip_trailing_eos \
  2>&1 | tee "${OUT_LLM}/run.log"

# 1) LLM (pure text)
OUT_LLM="${OUT_ROOT}/vicuna"
mkdir -p "${OUT_LLM}"
python3 "${PY}" \
  --mode llm \
  --llm_name "${LLM_NAME}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_LLM}" \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} --strip_trailing_eos \
  2>&1 | tee "${OUT_LLM}/run.log"

# 2) VLM (text_only)
#OUT_VLM_TEXT="${OUT_ROOT}/llava1_5"
#mkdir -p "${OUT_VLM_TEXT}"
#python3 "${PY}" \
#  --mode vlm \
#  --vlm_name "${VLM_NAME}" \
#  --jsonl_path "${CTX_JSONL}" \
#  --out_dir "${OUT_VLM_TEXT}" \
#  --cond text_only \
#  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
#  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
#  --max_samples ${MAX_SAMPLES} \
#  2>&1 | tee "${OUT_VLM_TEXT}/run.log"

# 3) VLM (real images; PHD-format) -> --img 플래그 사용
OUT_VLM_IMG="${OUT_ROOT}/llava1.5"
mkdir -p "${OUT_VLM_IMG}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME}" \
  --jsonl_path "${IMG_JSONL}" \
  --out_dir "${OUT_VLM_IMG}" \
  --img \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} \
  2>&1 | tee "${OUT_VLM_IMG}/run.log"

OUT_VLM_IMG="${OUT_ROOT}/llavanxt"
mkdir -p "${OUT_VLM_IMG}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME2}" \
  --jsonl_path "${IMG_JSONL}" \
  --out_dir "${OUT_VLM_IMG}" \
  --img \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} \
  2>&1 | tee "${OUT_VLM_IMG}/run.log"

echo "🎉 All done! Results in: ${OUT_ROOT}"
