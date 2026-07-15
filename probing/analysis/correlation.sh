# all csv files in the folder

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/correlation.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/correlation_score/outputs/qwen_cross_dataset_probing_set_a_cv5 \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref qwen2.5
