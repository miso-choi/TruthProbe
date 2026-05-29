import yaml
import json
import random
from pathlib import Path
import os
from typing import Dict, List, Any

# Read the YAML configuration to get dataset_path
with open(Path(__file__).parent / "halueval_remain.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    
    config = yaml.safe_load("".join(safe_data))
    dataset_path = config["dataset_path"]
    use_instruction = config["metadata"]["use_instruction"]
    use_knowledge = config["metadata"]["use_knowledge"]

    instruction_path = os.path.join(dataset_path, "annotations", "qa_evaluation_instruction.txt")
    with open(instruction_path, "r") as f:
        instruction = f.read()

# Add the following functions to your existing utils.py file
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





def halueval_doc_to_text(doc, lmms_eval_specific_kwargs):
    """
    Convert HaluEval document to text prompt following the official evaluation format.
    
    Args:
        doc: Dictionary containing knowledge, question, right_answer, hallucinated_answer
        lmms_eval_specific_kwargs: Additional parameters (not used in this implementation)
    
    Returns:
        Formatted prompt string for the model
    """
    # Extract data from document
    knowledge = doc["knowledge"]
    question = doc["question"]
    # hallucinated_answer = doc["hallucinated_answer"]
    # right_answer = doc["right_answer"]
    selected_answer = doc["selected_answer"]
    # ground_truth = doc["ground_truth"]
    
    # Randomly select between hallucinated and right answer
    # if random.random() > 0.5:
    #     answer = hallucinated_answer
    #     ground_truth = "Yes"  # This indicates the answer contains hallucination
    # else:
    #     answer = right_answer
    #     ground_truth = "No"   # This indicates the answer is correct (no hallucination)
    
    # Store the ground truth in the document for later use in process_results
    # doc["answer"] = ground_truth
    
    # Debug: Print to verify the modification
    # print(f"DEBUG: Set doc['answer'] = {ground_truth} in doc_to_text")
    
    # Format the prompt according to HaluEval evaluation format
    if use_instruction:
        prompt = instruction + "\n\n#Knowledge:" + knowledge + "\n\n#Question#: " + question + "\n#Answer#: " + selected_answer + "\n#Your Judgement#:"
    else:
        if use_knowledge:
            prompt = "\n\n#Knowledge:" + knowledge + "\n\n#Question#: " + question + "\n#Answer#: " + selected_answer
        else:
            prompt = "\n\n#Question#: " + question + "\n#Answer#: " + selected_answer

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{prompt}"+f" {post_prompt}"
    



def halueval_process_results(doc, results):
    """
    Process results for HaluEval evaluation.
    This function integrates the evaluation logic from the official HaluEval code.
    
    Note: The ground truth ("answer") is set in halueval_doc_to_text:
    - "Yes" = the presented answer contains hallucination (model should detect it)
    - "No" = the presented answer is correct (model should not detect hallucination)
    """
    # Get the model's response and process it
    pred = results[0].lower().strip()

    # process pred to remove the punctuation or comma at the end
    import re
    m = re.match(r"^(yes|no)\b[.,]?", pred)
    if m:
        pred = m.group(1)   # yes 또는 no
    else:
        pred = "unknown"
        
    gt_ans = doc["ground_truth"].lower().strip()
    
    # Debug: Print to verify the modification persisted
    # print(f"DEBUG: Retrieved doc['answer'] = {gt_ans} in process_results")
    
    # Ensure ground truth is yes/no
    assert gt_ans in ["yes", "no"], f"Ground truth must be 'yes' or 'no', got: {gt_ans}"
    
    # Process prediction to ensure it's yes/no
    if 'yes' in pred:
        pred = 'yes'
    elif 'no' in pred:
        pred = 'no'
    else:
        # Default to 'no' if unclear (following HaluEval evaluation logic)
        pred = 'no'
    
    # Calculate score
    score = 1.0 if pred == gt_ans else 0.0
    
    # Return results for all metrics (they all use the same base data)
    return {
        "halueval_accuracy": {
            "score": score, 
            "prediction": pred, 
            "ground_truth": gt_ans,
            "full_response": results[0]  # Keep original response for debugging
        },
        "halueval_precision": {
            "score": score, 
            "prediction": pred, 
            "ground_truth": gt_ans,
            "full_response": results[0]
        },
        "halueval_recall": {
            "score": score, 
            "prediction": pred, 
            "ground_truth": gt_ans,
            "full_response": results[0]
        },
        "halueval_f1_score": {
            "score": score, 
            "prediction": pred, 
            "ground_truth": gt_ans,
            "full_response": results[0]
        },
        "halueval_yes_ratio": {
            "score": score, 
            "prediction": pred, 
            "ground_truth": gt_ans,
            "full_response": results[0]
        },
    }


def halueval_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def halueval_aggregate_precision(results):
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


def halueval_aggregate_recall(results):
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


def halueval_aggregate_f1_score(results):
    precision = halueval_aggregate_precision(results)
    recall = halueval_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def halueval_aggregate_yes_ratio(results):
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
