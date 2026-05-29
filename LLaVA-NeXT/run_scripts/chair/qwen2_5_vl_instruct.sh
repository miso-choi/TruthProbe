export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export LOCAL_DATA_ROOT=/root/Desktop/workspace/miso/hub/datasets/lavis

# Base task yaml: lmms_eval/tasks/chair_max64/chair_max64.yaml
# Runtime task name/result key: ${TASK_ALIAS}

# vanilla
TASK_ALIAS="chair_max64_vanilla"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager \
	--tasks chair_max64 \
	--task_alias "${TASK_ALIAS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_vl_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


# TruthProbe_LLM
TASK_ALIAS="chair_max64_truthful_head_truth_llm_cont_4p5"
TRUTHFUL_HEAD_FILE="/root/Desktop/workspace/miso/faithful-lmms-eval/notebooks/linear_probing_score/converted_score_cv5_head_metrics_qwen2.5_halueval_292.npy"
AMPLIFY="4.5"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager \
	--tasks chair_max64 \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_vl_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media


# TruthProbe_MLLM
TASK_ALIAS="chair_max64_truthful_head_truth_mllm_cont_7p5"
TRUTHFUL_HEAD_FILE="/root/Desktop/workspace/miso/faithful-lmms-eval/notebooks/linear_probing_score/converted_score_cv5_head_metrics_qwen2.5_vl_instruct_rlhfv_2726.npy"
AMPLIFY="7.5"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"

python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager \
	--tasks chair_max64 \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix qwen2_5_vl_7b_instruct_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media
