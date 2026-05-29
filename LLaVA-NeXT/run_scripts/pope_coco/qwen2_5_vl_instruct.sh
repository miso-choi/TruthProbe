export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRUTH_SCORES_DIR="${REPO_ROOT}/truth_scores"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${REPO_ROOT}/data}"

# Base task yaml: lmms_eval/tasks/pope/pope.yaml
# Runtime task name/result key: ${TASK_ALIAS}

# vanilla
TASK_ALIAS="pope_vanilla"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager \
	--tasks pope \
	--task_alias "${TASK_ALIAS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_vl_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


# TruthProbe_LLM
TASK_ALIAS="pope_truthful_head_truth_llm_norm_0p3"
TRUTHFUL_HEAD_FILE="${TRUTH_SCORES_DIR}/converted_score_cv5_head_metrics_qwen2.5_halueval_292.npy"
AMPLIFY="0.3"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.normalize_amplifying_factor=${AMPLIFY}"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager \
	--tasks pope \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_vl_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


# TruthProbe_MLLM
TASK_ALIAS="pope_truthful_head_truth_mllm_norm_0p3"
TRUTHFUL_HEAD_FILE="${TRUTH_SCORES_DIR}/converted_score_cv5_head_metrics_qwen2.5_vl_instruct_rlhfv_2726.npy"
AMPLIFY="0.3"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.normalize_amplifying_factor=${AMPLIFY}"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager \
	--tasks pope \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_vl_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media
