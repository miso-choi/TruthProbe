# all csv files in the folder

python /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/comparison_w_comparison/vicuna \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5

python /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/comparison_w_comparison/llava1.5 \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5


python /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/comparison_w_comparison/llavanxt \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5


python /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/comparison_w_comparison/qwen2.5 \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5


python /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/comparison_w_comparison/qwen2.5_vl_instruct \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5


python /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing_code/script_camera_ready/truthprobe_score/comparison_w_comparison/qwen2.5_vl_omni \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5

