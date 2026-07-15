# Probing 실행 가이드

`script_camera_ready/` 아래에서 head-level truth-probing을 돌리고, 그 결과(truth score)로
모델 간 correlation 및 heatmap을 뽑는 전체 절차를 정리합니다.

```
script_camera_ready/
├── probe_qwen_series.py              # Qwen2.5 / Mistral 계열 probing 스크립트
├── probe_vicuna_series.py            # Vicuna / Mistral / LLaVA 계열 probing 스크립트
├── probing_data/                     # probing에 쓰는 jsonl 데이터셋 (6개, A/B는 D의 파일을 공유)
└── correlation_score/
    ├── probe_qwen_series_set{A,B,C,D,E}.sh
    ├── probe_vicuna_series_set{A,B,C,D,E}.sh
    └── outputs/                      # 각 실행의 truth score(csv) 저장 위치
```

---

## 1. Probing dataset setting

세트는 A~E 다섯 가지입니다. C/D/E는 텍스트 context jsonl + 이미지 jsonl 한 쌍씩(총 6개 파일)을 쓰고,
A/B는 별도 파일 없이 **세트 D의 context jsonl을 그대로 재사용**합니다 (VLM 쪽 입력 방식만 다름).

| 세트 | LLM probing | MLLM(VLM) probing | 사용 파일 |
|---|---|---|---|
| **A** | HaluEval (텍스트 전용) | HaluEval, **text_only** (실제 이미지 없이 텍스트만) | `setD_context_halueval.jsonl` (세트 D와 파일 공유) |
| **B** | HaluEval (텍스트 전용) | HaluEval, **black image** (검은 더미 이미지로 vision 경로만 활성화) | `setD_context_halueval.jsonl` (세트 D와 파일 공유) |
| **C** | PHD-10k (텍스트 전용) | PHD-10k, 실제 이미지 | `setC_context_phd10k.jsonl`, `setC_images_phd10k.jsonl` |
| **D** | HaluEval (텍스트 전용) | RLHF-V, 실제 이미지 | `setD_context_halueval.jsonl`, `setD_images_rlhfv.jsonl` |
| **E** | HaluEval + PHD 혼합 (텍스트 전용) | RLHF-V + PHD 혼합, 실제 이미지 | `setE_context_halueval_phd.jsonl`, `setE_images_rlhfv_phd.jsonl` |

- A/B는 세트 D와 같은 HaluEval context 데이터를 쓰되, VLM 입력 방식만 `--cond text_only` /
  `--cond black`로 다릅니다 (세트 D는 `--img`로 RLHF-V 실제 이미지 사용). 그래서 A/B에는
  `IMG_JSONL` 자체가 필요 없습니다.
- **주의**: 실제 이미지를 쓰는 세트(C, D, E)의 이미지 jsonl(`setC_images_phd10k.jsonl`,
  `setD_images_rlhfv.jsonl`, `setE_images_rlhfv_phd.jsonl`)의 `image_path` 필드는 실제 이미지
  바이너리를 가리키는 절대경로이며, 그 이미지 파일 자체는 git에 올리지 않았습니다.
  (다운로드 방법은 추후 이 섹션에 추가 예정 — TODO)

새 데이터셋을 추가하려면 같은 명명 규칙(`set{X}_context_*.jsonl` / `set{X}_images_*.jsonl`)으로
`probing_data/`에 넣고, 아래 2번 절차에서 `CTX_JSONL`/`IMG_JSONL` 값을 그 경로로 바꾸면 됩니다.

---

## 2. Probing 실행 스크립트에서 실험별로 지정해야 하는 것

`correlation_score/probe_{qwen,vicuna}_series_set{A,B,C,D,E}.sh` 각 파일 상단에 아래 변수들이 있습니다.
(A/B는 VLM도 실제 이미지를 쓰지 않으므로 `IMG_JSONL` 변수 자체가 없습니다.)

```bash
# ===== 데이터 경로 =====
CTX_JSONL="..."   # LLM(텍스트 전용) probing에 쓸 jsonl. A/B/D는 동일한 HaluEval 파일을 공유
IMG_JSONL="..."   # VLM(실제 이미지) probing에 쓸 jsonl (C/D/E만 존재)

# ===== 모델 =====
LLM_NAME="..."    # 1번째 LLM (예: Qwen/Qwen2.5-7B, lmsys/vicuna-7b-v1.5)
LLM_NAME2="..."   # 2번째 LLM (보통 mistralai/Mistral-7B-v0.1)
VLM_NAME="..."    # 1번째 VLM
VLM_NAME2="..."   # 2번째 VLM

OUT_ROOT="..."    # 이 실행에서 나오는 모든 결과의 최상위 출력 경로
```

- **Probing 데이터**: `CTX_JSONL`(LLM), `IMG_JSONL`(VLM, C/D/E만)을 바꿔서 지정합니다. 두 값 모두
  `probing_data/`에 있는 jsonl 경로를 절대경로로 넣습니다.
- **Probing 대상 모델**: `LLM_NAME`/`LLM_NAME2`/`VLM_NAME`/`VLM_NAME2`를 HuggingFace 모델 ID로 지정합니다.
  스크립트 안에서 각 모델은 `python3 "${PY}" --mode {llm|vlm} --llm_name/--vlm_name ...` 형태로 호출됩니다.
  - `--mode llm` : 순수 텍스트 probing (`collect_llm`, `self_attn.o_proj` 입력을 head 단위로 수집)
  - `--mode vlm --cond text_only` : 실제 이미지 없이 텍스트만 VLM에 입력 (세트 A)
  - `--mode vlm --cond black` : 검은 더미 이미지(기본 336×336)를 넣어 vision 경로만 활성화 (세트 B)
  - `--mode vlm --img` : 실제 이미지가 있는 PHD 포맷으로 VLM probing (세트 C/D/E)
- **Output 경로**: `OUT_ROOT` 아래에 모델별 하위 폴더(`OUT_LLM="${OUT_ROOT}/mistral"` 등)가 자동으로 생기고,
  그 안에 `cv5_head_metrics.csv`(5-fold CV head-level metrics)와 `run.log`가 저장됩니다.
  **모든 모델의 결과 파일명이 동일하게 `cv5_head_metrics.csv`이므로, 절대 여러 모델 결과를 한 폴더에
  납작하게(flat) 복사하면 안 되고 반드시 모델별 하위 폴더 구조를 유지해야 합니다** (3번 참고).

**GPU 지정**: 스크립트 안에는 `CUDA_VISIBLE_DEVICES`가 없습니다. 실행할 때 밖에서 지정하세요.

```bash
CUDA_VISIBLE_DEVICES=0 bash script_camera_ready/correlation_score/probe_qwen_series_setA.sh
```

---

## 3. Truth score correlation / heatmap 시각화

Probing이 끝나면 각 모델의 truth score는 다음 위치에 생깁니다.

```
correlation_score/outputs/<set 이름>/<model>/cv5_head_metrics.csv
```

예: `correlation_score/outputs/qwen_cross_dataset_probing_set_c_cv5/mistral/cv5_head_metrics.csv`

### 3-1. Correlation 계산 + 시각화 — `script_rebuttal/correlation.sh`

두 개 이상 모델의 truth score를 비교하려면, **모델별 하위 폴더 구조를 유지한 채** 그 폴더들의
공통 상위 경로를 `--root`로 지정합니다. (`script_camera_ready/analysis/correlation.py`가 `--root` 아래를
재귀적으로 `*.csv` 탐색하고, 라벨은 `상위폴더명/파일명` 형태로 자동 구분하므로 파일명이 같아도 문제없습니다.)

```bash
python script_camera_ready/analysis/correlation.py \
  --root <모델별 하위폴더들을 담은 상위 경로> \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref <기준 모델 이름의 부분 문자열, 예: qwen2.5>
```

- `--bar_mode ref --bar_ref <substr>`: 지정한 문자열이 포함된 라벨을 기준으로 나머지 모델과의
  상관관계를 막대그래프로 표시
- `--bar_mode mean`: 각 파일이 자기 자신을 제외한 나머지와의 평균 상관관계를 표시
- `--save_csv`: 상관행렬 자체도 csv로 저장하고 싶으면 추가
- 결과: `GLOBAL_CORR_MATRIX_ACC.png`(상관행렬), `GLOBAL_CORR_BARS_*.png`(막대그래프, bar_mode 지정 시)가
  `--root` 아래에 저장됩니다.

`script_rebuttal/correlation.sh`를 열어 `--root`만 위 경로로 바꿔서 실행하면 됩니다.

### 3-2. Heatmap 시각화 — `script_rebuttal/heatmap.sh`

```bash
python script_camera_ready/analysis/heatmap.py \
  --root <csv가 있는 폴더> \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"
```

⚠️ **`heatmap.py`는 `correlation.py`와 달리 출력 폴더명을 "파일명만" 기준으로 만듭니다**
(`_plots/cv5_head_metrics/...`). 그래서 **`--root`에 여러 모델의 하위 폴더를 한꺼번에 넣으면
전부 같은 이름(`cv5_head_metrics`)의 출력 폴더에 저장되면서 서로 덮어씁니다.**
`heatmap.sh`는 **모델 하나당 한 번씩, 그 모델의 결과 폴더(csv 1개가 있는 폴더)를 `--root`로 지정해서
따로 실행**하세요.

```bash
# 예: mistral 모델 하나만 히트맵 뽑기
python script_camera_ready/analysis/heatmap.py \
  --root correlation_score/outputs/qwen_cross_dataset_probing_set_c_cv5/mistral \
  --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"
```

결과는 `<root>/_plots/cv5_head_metrics/cv5_head_metrics_HEAT_CONT.png`에 저장됩니다.

---

## TODO
- [ ] 이미지 데이터셋(PHD, COCO2017 subset) 다운로드/재구성 방법 문서화
