#!/usr/bin/env bash
# probe_qwen_series_truthscore.sh
### Qwen2.5 series — TruthProbe용 Truth Score 산출 ###
### LLM Probing: HaluEval (292 samples) / MLLM Probing: RLHF-V (2,726 samples) ###
set -euo pipefail

# ===== 데이터 경로 =====
# HaluEval: setD와 동일한 context jsonl. 앞 292개만 사용 (MAX_SAMPLES_LLM)
CTX_JSONL="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probing_data/setD_context_halueval.jsonl"
# RLHF-V: setD와 동일한 image jsonl (2,726 rows 전체 사용)
IMG_JSONL="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probing_data/setD_images_rlhfv.jsonl"

# ===== 스크립트 파일 =====
PY="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/probe_qwen_series.py"

# ===== 모델 =====
LLM_NAME="Qwen/Qwen2.5-7B"
VLM_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
VLM_NAME2="Qwen/Qwen2.5-Omni-7B"

# ===== 공통 하이퍼파라미터 =====
EPOCHS=200
LR=1e-2
WD=1e-3
BS=512
TEST_SIZE=0.2
SEED=42
MAX_SAMPLES_LLM=292   # HaluEval: 292개만 사용
MAX_SAMPLES_VLM=0     # RLHF-V: 전체(2,726개) 사용
OUT_ROOT="/root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/outputs/qwen_truthprobe_score"
mkdir -p "${OUT_ROOT}"
echo "[INFO] Output root -> ${OUT_ROOT}"

# 1) LLM Probing (HaluEval, 292 samples)
OUT_LLM="${OUT_ROOT}/qwen2.5"
mkdir -p "${OUT_LLM}"
python3 "${PY}" \
  --mode llm \
  --llm_name "${LLM_NAME}" \
  --jsonl_path "${CTX_JSONL}" \
  --out_dir "${OUT_LLM}" \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES_LLM} --strip_trailing_eos \
  2>&1 | tee "${OUT_LLM}/run.log"

# 2) MLLM Probing (RLHF-V, 2,726 samples; 실제 이미지)
OUT_VLM="${OUT_ROOT}/qwen2.5_vl_instruct"
mkdir -p "${OUT_VLM}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME}" \
  --jsonl_path "${IMG_JSONL}" \
  --out_dir "${OUT_VLM}" \
  --img \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES_VLM} \
  2>&1 | tee "${OUT_VLM}/run.log"

OUT_VLM="${OUT_ROOT}/qwen2.5_vl_omni"
mkdir -p "${OUT_VLM}"
python3 "${PY}" \
  --mode vlm \
  --vlm_name "${VLM_NAME2}" \
  --jsonl_path "${IMG_JSONL}" \
  --out_dir "${OUT_VLM}" \
  --img \
  --probe_epochs ${EPOCHS} --probe_lr ${LR} --probe_weight_decay ${WD} \
  --probe_batch_size ${BS} --test_size ${TEST_SIZE} --seed ${SEED} \
  --max_samples ${MAX_SAMPLES_VLM} \
  2>&1 | tee "${OUT_VLM}/run.log"

echo "🎉 All done! Results in: ${OUT_ROOT}"
