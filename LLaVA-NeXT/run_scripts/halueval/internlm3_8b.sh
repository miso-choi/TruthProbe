export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INTERNLM3_MODEL_ARGS="pretrained=internlm/internlm3-8b-instruct,attn_implementation=eager,image_max_num=4,clear_cuda_cache_each_step=True"

# Base task yaml: lmms_eval/tasks/halueval_9708/halueval_answer1_w_k.yaml
# Runtime task name/result key: ${TASK_ALIAS}
TASK_ALIAS="halueval_answer1_w_k_truthful_head_ver16_cont_45_9708"
TRUTHFUL_HEAD_FILE="/root/Desktop/workspace/miso/faithful-lmms-eval/notebooks/linear_probing_score/converted_score_cv5_head_metrics_internlm3_8b_instruct_halueval_292.npy"
AMPLIFY="4.5"
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"

CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
	--num_processes=1 \
	-m lmms_eval \
	--model internlm3_8b \
	--model_args "${INTERNLM3_MODEL_ARGS}" \
	--tasks halueval_answer1_w_k_9708 \
	--task_alias "${TASK_ALIAS}" \
	--task_metadata_args "${TASK_METADATA_ARGS}" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix internlm3_8b_${TASK_ALIAS} \
	--output_path ./logs/ \
	--process_with_media
