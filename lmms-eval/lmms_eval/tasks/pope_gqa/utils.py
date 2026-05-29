import os
import datasets

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
    # Set random seed for reproducibility
    random_seed = int(os.getenv("LMMS_EVAL_RANDOM_SEED", "42"))
    
    # Check if subset is enabled
    enable_subset = os.getenv("LMMS_EVAL_ENABLE_SUBSET", "False").lower() == "true"
    subset_ratio = float(os.getenv("LMMS_EVAL_SUBSET_RATIO", "0.1"))
    
    # Shuffle the dataset with fixed seed
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    
    if enable_subset:
        # Take subset of the data for hyperparameter optimization
        subset_size = int(len(shuffled_dataset) * subset_ratio)
        result_dataset = shuffled_dataset.select(range(subset_size))
        
        print(f"Original dataset size: {len(dataset)}")
        print(f"Subset dataset size: {len(result_dataset)} ({subset_ratio*100}% for hyperparameter optimization)")
        print(f"Using random seed: {random_seed}")
    else:
        result_dataset = shuffled_dataset
        print(f"Dataset size: {len(dataset)} (full dataset, shuffled with seed {random_seed})")
    
    return result_dataset


import yaml
from pathlib import Path

# Read the YAML configuration to get dataset_path
with open(Path(__file__).parent / "pope.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    
    config = yaml.safe_load("".join(safe_data))
    dataset_path = config["dataset_path"]


def pope_doc_to_visual(doc):
    """
    Load image from path in the document.
    """
    import os
    from PIL import Image
    
    # Get image path from document
    image_path = doc.get("image_path", "")
    
    # Construct full path using dataset_path from YAML config
    full_image_path = os.path.join(dataset_path, 'testdev_balanced_images', image_path)
    
    # Load image
    if os.path.exists(full_image_path):
        return [Image.open(full_image_path).convert("RGB")]
    else:
        # Fallback or error handling
        raise FileNotFoundError(f"Image not found: {full_image_path}")


# def pope_doc_to_visual(doc):
#     # Assuming the 'doc' dictionary has a key 'image' with image data
#     return [doc["image"].convert("RGB")]


def pope_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def pope_process_results(doc, results):
    pred = results[0].lower().strip()

    # process pred to remove the punctuation or comma at the end
    import re
    m = re.match(r"^(yes|no)\b[.,]?", pred)
    if m:
        pred = m.group(1)   # yes 또는 no
    else:
        pred = "unknown"
        
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "pope_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def pope_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def pope_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "no" and pred == "yes":
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def pope_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "yes" and pred == "no":
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def pope_aggregate_f1_score(results):
    precision = pope_aggregate_precision(results)
    recall = pope_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def pope_aggregate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    for result in results:
        gt = result["ground_truth"]
        if gt == "yes":
            yes_count += 1
        elif gt == "no":
            no_count += 1
    yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    return yes_ratio
