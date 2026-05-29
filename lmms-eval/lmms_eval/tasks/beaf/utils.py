import yaml
from pathlib import Path

# Read the YAML configuration to get dataset_path
with open(Path(__file__).parent / "beaf.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    
    config = yaml.safe_load("".join(safe_data))
    dataset_path = config["dataset_path"]



def beaf_doc_to_visual(doc):
    """
    Load image from path in the document.
    """
    import os
    from PIL import Image
    
    # Get image path from document
    image_path = doc.get("image_path", "")
    
    # Construct full path using dataset_path from YAML config
    full_image_path = os.path.join(dataset_path, 'images', image_path)
    
    # Load image
    if os.path.exists(full_image_path):
        return [Image.open(full_image_path).convert("RGB")]
    else:
        # Fallback or error handling
        raise FileNotFoundError(f"Image not found: {full_image_path}")



def beaf_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def beaf_process_results(doc, results): 
    """
    Process individual BEAF sample results.
    
    Args:
        doc: Document containing ground truth information
        results: Model prediction results
        
    Returns:
        dict: Dictionary with sample-level metrics and metadata for all BEAF metrics
    """
    pred = results[0].lower().strip().split(',')[0]
    gt_ans = doc["gt"].lower().strip()  # BEAF uses "gt" field
    
    # Parse prediction to yes/no
    if 'yes' in pred.split(',')[0]:
        pred_clean = 'yes'
    elif 'no' in pred.split(',')[0]:
        pred_clean = 'no'
    else:
        pred_clean = pred  # Keep original if unclear
    
    # Basic accuracy score
    score = 1.0 if pred_clean == gt_ans else 0.0

    # TP / FP / TN / FN
    answer = None
    if gt_ans == "yes" and pred_clean == "yes":
        answer = "TP"
    elif gt_ans == "yes" and pred_clean == "no":
        answer = "FN"
    elif gt_ans == "no" and pred_clean == "yes":
        answer = "FP"
    elif gt_ans == "no" and pred_clean == "no":
        answer = "TN"
    
    # Create the common result structure
    result_data = {
        "question_id": doc["idx"], 
        "score": score, 
        "answer": answer,
        "prediction": pred_clean, 
        "ground_truth": gt_ans,
        "image_path": doc["image_path"],
        "question": doc["question"],
        "orig_img": doc["orig_img"],
        "removed_q": doc["removed_q"]
    }
    
    return {
        "beaf_accuracy": result_data,
        "beaf_precision": result_data,
        "beaf_recall": result_data,
        "beaf_f1_score": result_data,
        "beaf_faithfulness_metrics": result_data
    }


def beaf_aggregate_accuracy(results):
    """Calculate overall accuracy from BEAF results."""
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def beaf_aggregate_precision(results):
    """Calculate precision from BEAF results."""
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


def beaf_aggregate_recall(results):
    """Calculate recall from BEAF results."""
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


def beaf_aggregate_f1_score(results):
    """Calculate F1 score from BEAF results."""
    precision = beaf_aggregate_precision(results)
    recall = beaf_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def beaf_aggregate_faithfulness_metrics(results):
    """
    Calculate BEAF faithfulness metrics (TU, IG, SB+, SB-, ID, F1_TUID).
    
    Args:
        results: List of processed BEAF results
        
    Returns:
        dict: Dictionary containing all faithfulness metrics
    """
    # Group results by image for faithfulness calculations
    # image_groups = {}
    # for result in results:
    #     image = result["image_path"]
    #     if image not in image_groups:
    #         image_groups[image] = []
    #     image_groups[image].append(result)
    orig_pairs = {}
    for result in results:
        if result["orig_img"]:
            if orig_pairs.get(result["image_path"], None) is None:
                orig_pairs[result["image_path"]] = {}
            orig_pairs[result["image_path"]][result["question"]] = result["answer"] # TP / FP / TN / FN
    
    # Calculate faithfulness metrics
    cnt = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
           'TU': 0, 'IG': 0, 'SBp': 0, 'SBn': 0, 'ID': 0}
    conv = {'TPTN': 'TU', 'FNFP': 'IG', 'TPFP': 'SBp', 'FNTN': 'SBn'}
    
    id_tot = 0
    
    for result in results:
        # Count basic metrics
        # pred = result["prediction"]
        # gt = result["ground_truth"]
        
        # if gt == "yes" and pred == "yes":
        #     cnt['TP'] += 1
        # elif gt == "no" and pred == "no":
        #     cnt['TN'] += 1
        # elif gt == "yes" and pred == "no":
        #     cnt['FN'] += 1
        # elif gt == "no" and pred == "yes":
        #     cnt['FP'] += 1
        cnt[result["answer"]] += 1
        
        if not result["orig_img"]:
            name = result["image_path"][:-7] + '.jpg'
            ori_ans = orig_pairs[name][result["question"]]

            # for TU, IG, SBp, SBn
            if result["removed_q"]:
                if conv.get(ori_ans + result["answer"], None) is not None:
                    key = conv[ori_ans + result["answer"]]
                    cnt[key] += 1
            # for ID
            else:
                id_tot += 1
                if ori_ans[0] != result["answer"][0]:
                    cnt['ID'] += 1
        
        
        
        # # Calculate faithfulness metrics for non-original images
        # if not result["orig_img"]:
        #     # Find corresponding original image result
        #     orig_image = result["image_path"][:-7] + '.jpg' 
        #     orig_results = [r for r in image_groups.get(orig_image, []) if r["orig_img"]]
            
        #     if orig_results:
        #         orig_result = orig_results[0]  # Should be only one original per image
        #         ori_ans = orig_result["prediction"]
                
        #         # Calculate TU, IG, SBp, SBn for removed questions
        #         if result["removed_q"]:
        #             combined_key = ori_ans + pred
        #             if combined_key in conv:
        #                 cnt[conv[combined_key]] += 1
        #         # Calculate ID for non-removed questions
        #         else:
        #             id_tot += 1
        #             if ori_ans[0] != pred[0]:  # First letter comparison
        #                 cnt['ID'] += 1
    
    # Calculate metrics
    total_samples = cnt['TP'] + cnt['FP'] + cnt['TN'] + cnt['FN']
    removed_samples = cnt['TU'] + cnt['IG'] + cnt['SBp'] + cnt['SBn']

    # Basic metrics
    acc = (cnt['TP'] + cnt['TN']) / total_samples * 100 if total_samples > 0 else 0
    precision = cnt['TP'] / (cnt['TP'] + cnt['FP']) * 100 if (cnt['TP'] + cnt['FP']) > 0 else 0
    recall = cnt['TP'] / (cnt['TP'] + cnt['FN']) * 100 if (cnt['TP'] + cnt['FN']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Faithfulness metrics
    tu = cnt['TU'] / removed_samples * 100 if removed_samples > 0 else 0
    ig = cnt['IG'] / removed_samples * 100 if removed_samples > 0 else 0
    sbp = cnt['SBp'] / removed_samples * 100 if removed_samples > 0 else 0
    sbn = cnt['SBn'] / removed_samples * 100 if removed_samples > 0 else 0
    id_ = cnt['ID'] / id_tot * 100 if id_tot > 0 else 0
    f1_tuid = 2 * tu * (100 - id_) / (tu + (100 - id_)) if (tu + (100 - id_)) > 0 else 0
    
    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1_PR": f1,
        "TU": tu,
        "IG": ig,
        "SBp": sbp,
        "SBn": sbn,
        "ID": id_,
        "F1_TUID": f1_tuid
    }


# Legacy evaluation function for backward compatibility
def beaf_eval(results):
    """
    Legacy BEAF evaluation function for backward compatibility.
    This function is deprecated - use the individual aggregate functions instead.
    
    Args:
        results: List of processed BEAF results
        
    Returns:
        dict: Dictionary containing all BEAF metrics
    """
    return beaf_aggregate_faithfulness_metrics(results)


