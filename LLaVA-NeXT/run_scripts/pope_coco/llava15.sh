export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export LOCAL_DATA_ROOT=/root/Desktop/workspace/miso/hub/datasets/lavis

# Base task yaml: lmms_eval/tasks/pope/pope.yaml
# Runtime task name/result key: ${TASK_ALIAS}

# vanilla
TASK_ALIAS="pope_vanilla"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model llava \
	--model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager \
	--tasks pope \
	--task_alias "${TASK_ALIAS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix llava_v1.5_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


# TruthProbe_LLM
TASK_ALIAS="pope_truthful_head_truth_llm_norm_0p2"
TRUTHFUL_HEAD_FILE="/root/Desktop/workspace/miso/faithful-lmms-eval/notebooks/linear_probing_score/converted_score_cv5_head_metrics_vicuna_halueval_292.npy"
AMPLIFY="0.2"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.normalize_amplifying_factor=${AMPLIFY}"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model llava \
	--model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager \
	--tasks pope \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix llava_v1.5_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


# TruthProbe_MLLM
TASK_ALIAS="pope_truthful_head_truth_mllm_norm_0p1"
TRUTHFUL_HEAD_FILE="/root/Desktop/workspace/miso/faithful-lmms-eval/notebooks/linear_probing_score/converted_score_cv5_head_metrics_llava1.5_rlhfv_2726.npy"
AMPLIFY="0.1"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.normalize_amplifying_factor=${AMPLIFY}"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model llava \
	--model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager \
	--tasks pope \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix llava_v1.5_7b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media
