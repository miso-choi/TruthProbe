export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export LOCAL_DATA_ROOT=/root/Desktop/workspace/miso/hub/datasets/lavis

# Base task yaml: lmms_eval/tasks/halueval_remain/halueval_remain.yaml
# Runtime task name/result key: ${TASK_ALIAS}

TASK_ALIAS="halueval_vanilla"

# vanilla
python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5 \
	--model_args pretrained=Qwen/Qwen2.5-7B-Instruct,attn_implementation=eager \
	--tasks halueval_remain \
	--task_alias "${TASK_ALIAS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media



TASK_ALIAS="halueval_truthful_head_truth_basellm_baseLLM_cont_6p0"
TRUTHFUL_HEAD_FILE="/root/Desktop/workspace/miso/faithful-lmms-eval/notebooks/linear_probing_score/converted_score_cv5_head_metrics_qwen2.5_halueval_292.npy"
AMPLIFY="6.0"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"

# TruthProbe_BaseLLM
python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5 \
	--model_args pretrained=Qwen/Qwen2.5-7B-Instruct,attn_implementation=eager \
	--tasks halueval_remain \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media