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

* `halueval_remain`
* `pope`
* `pope_aokvqa`
* `chair_max64`

Use `LOCAL_DATA_ROOT` as the root directory for local datasets:

```bash
LOCAL_DATA_ROOT=/path/to/local/datasets/lavis
```

Some task YAML files may contain machine-specific `dataset_path` values.
Before running evaluation on a new machine, update each YAML file to point to the correct local dataset path.

### Automatically downloaded dataset

The standard POPE task uses the Hugging Face dataset below:

```yaml
dataset_path: lmms-lab/POPE
```

This dataset is automatically downloaded by the Hugging Face `datasets` library when running `lmms-eval`, assuming network access and appropriate permissions.

Related task:

```text
pope
```

### Local datasets

The following datasets should be prepared locally.

#### HaluEval Remain

Related task:

```text
halueval_remain
```

Expected structure:

```text
${LOCAL_DATA_ROOT}/halueval_eval_9708/
└── annotations/
    ├── annotations_answer1.jsonl
    └── qa_evaluation_instruction.txt
```

YAML path:

```yaml
dataset_path: ${LOCAL_DATA_ROOT}/halueval_eval_9708
```

#### POPE-AOKVQA

Related task:

```text
pope_aokvqa
```

Expected structure:

```text
${LOCAL_DATA_ROOT}/coco/
├── images/
│   └── val2014/
│       └── COCO_val2014_*.jpg
└── pope_aokvqa_annotations/
    └── aokvqa_pope.jsonl
```

YAML path:

```yaml
dataset_path: ${LOCAL_DATA_ROOT}/coco
```

#### CHAIR Max64

Related task:

```text
chair_max64
```

Expected structure:

```text
${LOCAL_DATA_ROOT}/coco/
├── images/
│   └── val2014/
│       └── COCO_val2014_*.jpg
├── chair_annotations/
│   └── annotations_coco_500.jsonl
└── annotations/
    ├── captions_train2014.json
    ├── captions_val2014.json
    ├── instances_train2014.json
    └── instances_val2014.json
```

YAML path:

```yaml
dataset_path: ${LOCAL_DATA_ROOT}/coco
```

## 🚀 Inference

Inference scripts are located under:

```bash
LLaVA-NeXT/run_scripts
```

Activate the conda environment and move to the `LLaVA-NeXT` directory before running scripts:

```bash
conda activate truthprobe
cd TruthProbe/LLaVA-NeXT
```

### Script layout

| Benchmark   | Script directory          |
| ----------- | ------------------------- |
| CHAIR       | `run_scripts/chair`       |
| POPE-COCO   | `run_scripts/pope_coco`   |
| POPE-AOKVQA | `run_scripts/pope_aokvqa` |
| HaluEval    | `run_scripts/halueval`    |

Example scripts:

```bash
run_scripts/chair/qwen2_5_vl_instruct.sh
run_scripts/chair/qwen2_5_vl_omni.sh
run_scripts/chair/llava15.sh
run_scripts/pope_coco/qwen2_5_vl_instruct.sh
run_scripts/pope_aokvqa/llava_next.sh
run_scripts/halueval/vicuna_7b.sh
```

Some benchmark directories also include `all.sh`, which runs selected scripts within the directory.

### Run a script

For example, to run Qwen2.5-VL-Instruct on CHAIR:

```bash
bash run_scripts/chair/qwen2_5_vl_instruct.sh
```

Most scripts run `lmms_eval` through `accelerate`:

```bash
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model <model_name> \
    --model_args pretrained=<huggingface_model_id>,attn_implementation=eager \
    --tasks <task_name> \
    --task_alias "${TASK_ALIAS}" \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/ \
    --process_with_media
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

> [!IMPORTANT]
> Some scripts may still contain machine-specific absolute paths, especially for `TRUTHFUL_HEAD_FILE`.
> Update them to the corresponding paths in your local TruthProbe workspace before running inference.

## 📊 Outputs

By default, scripts write results under:

```bash
LLaVA-NeXT/logs
```

The output directory can be changed by modifying `--output_path` in each script.

## 📌 Notes

* Use the correct local `transformers` version for each model family.
* Ensure that all local dataset paths are updated before running tasks that require local files.
* For Hugging Face datasets and models, make sure your environment has network access and appropriate permissions.
