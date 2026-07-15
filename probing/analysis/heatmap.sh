# python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
#   --root ./csvs \
#   --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC" ##색상 코드& 끝나는 지점 지정(시작 끝만 0,1에 맞추면 됨)

# heatmap.py는 --root 아래 csv 파일명만으로 출력 폴더를 만들기 때문에, 여러 모델의 하위 폴더를
# 한 번에 넣으면 같은 이름(cv5_head_metrics)의 출력 폴더에 저장되며 서로 덮어씁니다.
# 그래서 correlation.sh와 달리 모델 하나당 한 번씩(--root를 그 모델 폴더로) 실행합니다.

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/truthprobe_score/comparison_w_comparison/vicuna \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/truthprobe_score/comparison_w_comparison/llava1.5 \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/truthprobe_score/comparison_w_comparison/llavanxt \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/truthprobe_score/comparison_w_comparison/qwen2.5 \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/truthprobe_score/comparison_w_comparison/qwen2.5_vl_instruct \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"

python /root/Desktop/workspace/miso/faithful-probing/probing/analysis/heatmap.py \
  --root /root/Desktop/workspace/miso/faithful-probing/probing/truthprobe_score/comparison_w_comparison/qwen2.5_vl_omni \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"
