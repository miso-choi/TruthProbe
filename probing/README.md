# Faithful Probing: Head-Level Truthfulness Probing for LLMs & MLLMs

This repository probes the internal attention heads of language models (LLMs) and
vision-language models (VLMs/MLLMs) to test whether truthfulness is linearly decodable
from their hidden representations. For each (layer, head), we train a lightweight
probe on attention output activations and report per-head classification metrics
("truth scores"), which are then compared across models via correlation/heatmap analysis.

## Repository Structure

```
.
├── LLaVA-NeXT/                     # vendored LLaVA-NeXT (VLM inference dependency)
├── lmms-eval/                      # vendored lmms-eval (VLM evaluation dependency)
└── probing/                        # entry point for reproducing the paper's results
    ├── probe_vicuna_series.py  # Vicuna / Mistral / LLaVA-1.5 / LLaVA-NeXT probing
    ├── probe_qwen_series.py    # Qwen2.5 / Mistral / Qwen2.5-VL-Instruct / Qwen2.5-VL-Omni probing
    ├── probing_data/           # probing datasets (jsonl)
    ├── analysis/
    │   ├── correlation.py      # cross-model truth-score correlation + bar plots
    │   ├── correlation.sh
    │   ├── heatmap.py          # per-model (layer, head) truth-score heatmap
    │   └── heatmap.sh
    ├── correlation_score/      # Fig. 1: cross-model / cross-dataset truth-score correlation
    │   ├── probe_{vicuna,qwen}_series_set{A,B,C,D,E}.sh
    │   └── outputs/            # per-run truth scores (csv)
    └── truthprobe_score/       # truth-head scores consumed by the TruthProbe method
        ├── probe_{vicuna,qwen}_series_truthscore.sh
        └── outputs/
            ├── vicuna_truthprobe_score/{vicuna,llava1.5,llavanxt}/cv5_head_metrics.csv
            └── qwen_truthprobe_score/{qwen2.5,qwen2.5_vl_instruct,qwen2.5_vl_omni}/cv5_head_metrics.csv
```

## Setup

This repository is the probing component of [**TruthProbe**](https://github.com/miso-choi/TruthProbe),
an inference-time method for enhancing contextual truthfulness in LLMs/MLLMs. Set up
the same `truthprobe` conda environment described in the TruthProbe repo's
[installation instructions](https://github.com/miso-choi/TruthProbe#%EF%B8%8F-installation)
(conda env `truthprobe`, `lmms-eval` installed in editable mode, and the local
`transformers` build matching your model family), then reuse that environment here —
no separate setup is needed.

Models are pulled from the Hugging Face Hub on first run (e.g. `lmsys/vicuna-7b-v1.5`,
`mistralai/Mistral-7B-v0.1`, `llava-hf/llava-1.5-7b-hf`, `Qwen/Qwen2.5-7B`,
`Qwen/Qwen2.5-VL-7B-Instruct`), so make sure `huggingface-cli login` is configured if
any are gated.

## Probing Datasets

Probing uses five dataset configurations (A–E), built from HaluEval, PHD-10k, and
RLHF-V. These correspond to the probing setups behind **Fig. 1** of the paper (per-head
truthfulness probing accuracy across LLM/MLLM families and datasets). Sets A/B reuse
Set D's text context and only change how the VLM receives the image signal.

| Set | LLM probing | VLM probing | Files (in `probing_data/`) |
|---|---|---|---|
| **A** | HaluEval (text) | HaluEval, no image (`text_only`) | `setD_context_halueval.jsonl` |
| **B** | HaluEval (text) | HaluEval, dummy black image | `setD_context_halueval.jsonl` |
| **C** | PHD-10k (text) | PHD-10k, real image | `setC_context_phd10k.jsonl`, `setC_images_phd10k.jsonl` |
| **D** | HaluEval (text) | RLHF-V, real image | `setD_context_halueval.jsonl`, `setD_images_rlhfv.jsonl` |
| **E** | HaluEval + PHD (text) | RLHF-V + PHD, real image | `setE_context_halueval_phd.jsonl`, `setE_images_rlhfv_phd.jsonl` |

> **Note:** the `image_path` field in the image jsonls points to local image files
> (PHD-10k / RLHF-V), which are not distributed with this repository due to size and
> licensing. We plan to release the underlying probing data (HaluEval, ~10k rows; PHD
> context/images) via an external drive/hub link in this README. Until then, point
> `image_path` to your own copies of these datasets before running the VLM+real-image
> configurations (C, D, E).

## Running Probing

Each `correlation_score/probe_{vicuna,qwen}_series_set{A..E}.sh` script runs LLM and
VLM probing for one dataset configuration and one model family. Edit the paths/model
names at the top of the script to match your environment, then run with the GPU of
your choice:

```bash
cd probing
CUDA_VISIBLE_DEVICES=0 bash correlation_score/probe_vicuna_series_setD.sh
```

Each model's results are written to `${OUT_ROOT}/<model>/cv5_head_metrics.csv`
(5-fold CV per-head metrics) and `run.log`. Keep the per-model subfolder structure —
all models share the same output filename, so flattening the output directory will
overwrite results.

Underlying probing modes (see `--help` on either `probe_*_series.py`):
- `--mode llm`: pure-text probing
- `--mode vlm --cond text_only`: VLM with no image (Set A)
- `--mode vlm --cond black`: VLM with a dummy black image (Set B)
- `--mode vlm --img`: VLM with a real image, PHD-style jsonl (Sets C/D/E)

## Truth-Head Scores for TruthProbe

`truthprobe_score/` is separate from `correlation_score/`: instead of comparing
truth scores across datasets (Fig. 1), it produces the fixed per-head truth-score
files that the TruthProbe method itself gates/amplifies at inference time
(`--truthful_head_filepath` in the TruthProbe repo). It uses one fixed dataset per
modality — **HaluEval (292 samples)** for LLM probing and **RLHF-V (2,726 samples)**
for MLLM probing (Set D's jsonls, with `MAX_SAMPLES` set accordingly) — rather than
the A–E sweep.

```bash
cd probing
CUDA_VISIBLE_DEVICES=0 bash truthprobe_score/probe_vicuna_series_truthscore.sh   # Vicuna-7B (LLM) + LLaVA-1.5 / LLaVA-NeXT (MLLM)
CUDA_VISIBLE_DEVICES=0 bash truthprobe_score/probe_qwen_series_truthscore.sh     # Qwen2.5 (LLM) + Qwen2.5-VL-Instruct / Qwen2.5-VL-Omni (MLLM)
```

Results follow the same per-model-subfolder convention as `correlation_score/`:
`truthprobe_score/outputs/{vicuna,qwen}_truthprobe_score/<model>/cv5_head_metrics.csv`.
(Mistral is only used as a correlation baseline and is intentionally excluded here.)

## Analyzing Results

**Correlation across models** — point `--root` at a directory containing one
subfolder per model (each with its `cv5_head_metrics.csv`):

```bash
python probing/analysis/correlation.py \
  --root probing/correlation_score/outputs/<run_name> \
  --fname_regex ".*\.csv$" \
  --bar_mode ref --bar_ref qwen2.5
```

Outputs `GLOBAL_CORR_MATRIX_ACC.png` (correlation matrix) and, if `--bar_mode` is
set, `GLOBAL_CORR_BARS_*.png` under `--root`.

**Per-model heatmap** — run once per model, pointing `--root` at that model's own
output folder (a single csv), since `heatmap.py` names its output folder after the
csv filename and will overwrite across models if given a shared root:

```bash
python probing/analysis/heatmap.py \
  --root probing/correlation_score/outputs/<run_name>/<model>
```

Output: `<root>/_plots/cv5_head_metrics/cv5_head_metrics_HEAT_CONT.png`.
