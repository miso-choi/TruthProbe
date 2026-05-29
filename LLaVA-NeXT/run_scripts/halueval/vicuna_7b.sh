export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRUTH_SCORES_DIR="${REPO_ROOT}/truth_scores"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${REPO_ROOT}/data}"

# Base task yaml: lmms_eval/tasks/halueval_remain/halueval_remain.yaml
# Runtime task name/result key: ${TASK_ALIAS}

# Vicuna-7b --------------------------------------------------------------------
TASK_ALIAS="halueval_vanilla"

# vanilla
python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model vicuna \
	--model_args pretrained=lmsys/vicuna-7b-v1.5,attn_implementation=eager \
	--tasks halueval_remain \
	--task_alias "${TASK_ALIAS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix vicuna_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


TASK_ALIAS="halueval_truthful_head_truth_llm_cont_4p5"
TRUTHFUL_HEAD_FILE="${TRUTH_SCORES_DIR}/converted_score_cv5_head_metrics_vicuna_halueval_292.npy"
AMPLIFY="4.5"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"


# TruthProbe_LLM
python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model vicuna \
	--model_args pretrained=lmsys/vicuna-7b-v1.5,attn_implementation=eager \
	--tasks halueval_remain \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix vicuna_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media

# Qwen2.5 ---------------------------------------------------------------------
TASK_ALIAS="halueval_vanilla"

# vanilla
python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5 \
	--model_args pretrained=Qwen/Qwen2.5-7B,attn_implementation=eager \
	--tasks halueval_remain \
	--task_alias "${TASK_ALIAS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media



TASK_ALIAS="halueval_truthful_head_truth_llm_cont_6p0"
TRUTHFUL_HEAD_FILE="${TRUTH_SCORES_DIR}/converted_score_cv5_head_metrics_qwen2.5_halueval_292.npy"
AMPLIFY="6.0"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"

# TruthProbe_LLM
python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5 \
	--model_args pretrained=Qwen/Qwen2.5-7B,attn_implementation=eager \
	--tasks halueval_remain \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media
