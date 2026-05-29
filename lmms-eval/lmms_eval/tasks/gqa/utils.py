import os

import datasets
from datasets import load_dataset


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process the dataset for hyperparameter optimization:
    - Shuffle the dataset with a fixed seed for reproducibility
    - Take only 10% of the data for faster hyperparameter search

    Environment variables:
    - LMMS_EVAL_RANDOM_SEED: Random seed for shuffling (default: 42)
    - LMMS_EVAL_SUBSET_RATIO: Fraction of data to use (default: 0.1 for 10%)
    - LMMS_EVAL_ENABLE_SUBSET: Enable subset selection (default: False)
    """
    random_seed = int(os.getenv("LMMS_EVAL_RANDOM_SEED", "42"))
    enable_subset = os.getenv("LMMS_EVAL_ENABLE_SUBSET", "False").lower() == "true"
    subset_ratio = float(os.getenv("LMMS_EVAL_SUBSET_RATIO", "0.1"))

    shuffled_dataset = dataset.shuffle(seed=random_seed)

    if enable_subset:
        subset_size = int(len(shuffled_dataset) * subset_ratio)
        result_dataset = shuffled_dataset.select(range(subset_size))

        print(f"Original dataset size: {len(dataset)}")
        print(f"Subset dataset size: {len(result_dataset)} ({subset_ratio*100}% for hyperparameter optimization)")
        print(f"Using random seed: {random_seed}")
    else:
        result_dataset = shuffled_dataset
        print(f"Dataset size: {len(dataset)} (full dataset, shuffled with seed {random_seed})")

    return result_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"
