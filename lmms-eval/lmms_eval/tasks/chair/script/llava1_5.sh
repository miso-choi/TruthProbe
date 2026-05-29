CUDA_VISIBLE_DEVICES=3 \
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b,attn_implementation=eager" \
    --tasks chair \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix chair \
    --output_path ./chiar_test_512_tokens/
