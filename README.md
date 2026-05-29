# TruthProbe

Official implementation of **TruthProbe**, a plug-and-play inference-time method for enhancing contextual truthfulness in LLM and MLLM lineages.

TruthProbe supports inference with vanilla models as well as TruthProbe variants using truthful-head score files from either the base LLM or the target MLLM.

## ✨ Overview

This repository contains:

* Environment setup instructions for local `lmms-eval`, `LLaVA-NeXT`, and model-specific `transformers` versions.
* Task configurations for hallucination and truthfulness evaluation benchmarks.
* Inference scripts for running vanilla and TruthProbe-enhanced models.
* Support for multiple model families, including Vicuna, LLaVA, Qwen2.5-VL, Qwen3-VL, and InternVL3.

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/miso-choi/TruthProbe.git
cd TruthProbe
```

### 2. Create a conda environment

```bash
conda create -n truthprobe python=3.12 -y
conda activate truthprobe
```

### 3. Install `lmms-eval`

```bash
cd lmms-eval
pip install -e .
```

### 4. Install the required local `transformers`

TruthProbe uses local `transformers` source trees to support model-specific modifications.
Install the version required for the model family you want to run.

For example, for `transformers==4.53.2`:

```bash
cd ../LLaVA-NeXT/transformers-4.53.2
git checkout v4.53.2  # Optional if the folder is already on the correct version.
pip install -e .
```

| Model family                                      | Required local `transformers` |
| ------------------------------------------------- | ----------------------------- |
| Vicuna-family, Qwen2.5-family, and general models | `transformers-4.53.2`         |
| InternVL3-9B                                      | `transformers-4.37.2`         |
| Qwen3-VL family                                   | `transformers-4.57.0`         |

### 5. Install model-specific dependencies

For Qwen2.5-VL related models, install the following packages if needed:

```bash
pip install qwen_vl_utils audioread librosa moviepy qwen_omni_utils
```

> [!NOTE]
> If dependency conflicts occur, reinstall `torch` and `torchvision` with versions compatible with your CUDA, driver, and target model.

## 📁 Dataset Setup

TruthProbe currently uses the following tasks:

* `halueval` (samples used for probing are excluded)
* `pope`
* `pope_aokvqa`
* `chair` (max_new_tokens is set to 64)

Place the `data` directory in a location that can store the inference datasets
(e.g., COCO val2014 images are about 6 GB). The run scripts use
`<repo-root>/data` by default. If you place `data` somewhere else, set
`LOCAL_DATA_ROOT` to that location before running inference:

```bash
export LOCAL_DATA_ROOT=/path/to/data
```

Download the Val 2014 images from the official COCO website and place them under
`data/coco/images`.

## 🚀 Inference

```bash
conda activate truthprobe
cd TruthProbe/LLaVA-NeXT
```

Inference scripts are located under:

```bash
LLaVA-NeXT/run_scripts
```

### Run a script

For example, to run Qwen2.5-VL-Instruct on POPE:

```bash
bash run_scripts/pope/qwen2_5_vl_instruct.sh
```


## 🧪 TruthProbe Modes

Most scripts include three inference modes:

| Mode              | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `vanilla`         | Baseline inference without TruthProbe                      |
| `TruthProbe_LLM`  | TruthProbe using truthful-head scores from the base LLM    |
| `TruthProbe_MLLM` | TruthProbe using truthful-head scores from the target MLLM |

TruthProbe options are passed through `--task_metadata_args`.

Example with contrast amplification:

```bash
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.contrast_amplifying_factor=${AMPLIFY}"
```

Example with normalized amplification:

```bash
TASK_METADATA_ARGS="gate_truthful_head=true,truthful_head=true,truthful_head_filepath=${TRUTHFUL_HEAD_FILE},hyperparams.normalize_amplifying_factor=${AMPLIFY}"
```

## ✅ Before Running

Check the following values inside the target script:

* `HF_HOME`
* `HUGGINGFACE_HUB_CACHE`
* `HF_DATASETS_CACHE`
* `TRANSFORMERS_CACHE`
* `pretrained`
* `--tasks`
* `TASK_ALIAS`
* `TRUTHFUL_HEAD_FILE`
* `AMPLIFY`
* `--output_path`


## 📊 Outputs

By default, scripts write results under:

```bash
LLaVA-NeXT/logs
```

The results are saved as csv file for each model/task in `LLaVA-NeXT/logs/results_csv`.

## Acknowledgements

This work builds on the codebases of [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [Inference-Time Intervention / ITI](https://github.com/likenneth/honest_llama). We thank the authors and maintainers for making these resources publicly available.
