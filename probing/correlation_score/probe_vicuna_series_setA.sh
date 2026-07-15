#!/usr/bin/env bash
# probe_vicuna_series_setA.sh
### Vicuna / Mistral / LLaVA series probing — Set A (HaluEval context, VLM text_only) ###
set -euo pipefail

# ===== 데이터 경로 =====
# VLM도 실제 이미지 없이 동일한 HaluEval context jsonl을 텍스트로만 사용 (Set D와 같은 파일)
CTX_JSONL="/root/Desktop/workspace/miso/faithful-probing/probing/probing_data/setD_context_halueval.jsonl"

# ===== 스크립트 파일 =====
PY="/root/Desktop/workspace/miso/faithful-probing/probing/probe_vicuna_series.py"

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
OUT_ROOT="/root/Desktop/workspace/miso/faithful-probing/probing/correlation_score/outputs/vicuna_cross_dataset_probing_set_a_cv5"
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

# 2) VLM (text_only; 실제 이미지 없이 텍스트만 vision-language 모델에 입력)
OUT_VLM_TEXT="${OUT_ROOT}/llava1.5"
mkdir -p "${OUT_VLM_TEXT}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_VLM_TEXT}" \
  --cond text_only \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} \
  2>&1 | tee "${OUT_VLM_TEXT}/run.log"

OUT_VLM_TEXT="${OUT_ROOT}/llavanxt"
mkdir -p "${OUT_VLM_TEXT}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME2}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_VLM_TEXT}" \
  --cond text_only \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} \
  2>&1 | tee "${OUT_VLM_TEXT}/run.log"

echo "🎉 All done! Results in: ${OUT_ROOT}"
