#!/usr/bin/env bash
# probe_qwen_series_setB.sh
### Qwen2.5 / Mistral series probing — Set B (HaluEval context, VLM w/ black image) ###
set -euo pipefail

# ===== 데이터 경로 =====
# VLM에 실제 이미지 대신 검은 더미 이미지를 넣어 vision 경로만 활성화 (Set D와 같은 파일)
CTX_JSONL="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probing_data/setD_context_halueval.jsonl"

# ===== 스크립트 파일 =====
PY="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probe_qwen_series.py"

# ===== 모델 =====
LLM_NAME="Qwen/Qwen2.5-7B"
LLM_NAME2="mistralai/Mistral-7B-v0.1"
VLM_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
VLM_NAME2="Qwen/Qwen2.5-Omni-7B"

# ===== 공통 하이퍼파라미터 =====
EPOCHS=200
LR=1e-2
WD=1e-3
BS=512
TEST_SIZE=0.2
SEED=42
# 빠른 테스트는 100 등으로; 전체는 0
MAX_SAMPLES=0
OUT_ROOT="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/correlation_score/outputs/qwen_cross_dataset_probing_set_b_cv5"
mkdir -p "${OUT_ROOT}"
echo "[INFO] Output root -> ${OUT_ROOT}"

# 1) LLM (pure text)
OUT_LLM="${OUT_ROOT}/qwen2.5"
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

# 2) VLM (black; 검은 더미 이미지 336x336을 넣어 vision 경로 활성화)
OUT_VLM_BLACK="${OUT_ROOT}/qwen2.5_vl_instruct"
mkdir -p "${OUT_VLM_BLACK}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_VLM_BLACK}" \
  --cond black \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} \
  2>&1 | tee "${OUT_VLM_BLACK}/run.log"

OUT_VLM_BLACK="${OUT_ROOT}/qwen2.5_vl_omni"
mkdir -p "${OUT_VLM_BLACK}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME2}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_VLM_BLACK}" \
  --cond black \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES} \
  2>&1 | tee "${OUT_VLM_BLACK}/run.log"

echo "🎉 All done! Results in: ${OUT_ROOT}"
