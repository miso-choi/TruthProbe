import yaml
from pathlib import Path

# Read the YAML configuration to get dataset_path
with open(Path(__file__).parent / "attn_diff.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    
    config = yaml.safe_load("".join(safe_data))
    dataset_path = config["dataset_path"]


def attn_diff_text_doc_to_visual(doc):
    """
    Load image from path in the document.
    """
    import os
    from PIL import Image
    
    # Get image path from document
    image_path = doc.get("image_path", "")
    
    # Construct full path using dataset_path from YAML config
    full_image_path = os.path.join(dataset_path, 'images',image_path)
    
    # Load image
    if os.path.exists(full_image_path):
        return [Image.open(full_image_path).convert("RGB")]
    else:
        # Fallback or error handling
        raise FileNotFoundError(f"Image not found: {full_image_path}")


def attn_diff_text_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    text_input = doc["text_input"].strip()
    return f"{pre_prompt}{text_input}{post_prompt}"


def attn_diff_text_process_results(doc, results):
    """
    Process attention difference results for saving.
    Handles the new incremental JSONL writing approach.
    """
    import json
    
    # results contains JSON strings from generate_until_with_head_masking
    # With the new approach, results are already saved to JSONL file
    # We just need to return a minimal acknowledgment
    
    all_sample_results = []
    for i, result in enumerate(results):
        try:
            # Parse JSON string back to dictionary
            sample_result = json.loads(result)
            # Check if this is the new completion acknowledgment format
            if "status" in sample_result and sample_result["status"] == "completed":
                # This is the new format where results are already saved to file
                sample_result["repeat_id"] = i
                all_sample_results.append(sample_result)
            else:
                # Legacy format - keep for backward compatibility
                sample_result["repeat_id"] = i
                all_sample_results.append(sample_result)
        except (json.JSONDecodeError, TypeError):
            # Fallback for non-JSON results
            sample_result = {"text_response": str(result), "repeat_id": i}
            all_sample_results.append(sample_result)
    
    return {
        "attn_diff_data": all_sample_results,
        "doc_id": doc.get("doc_id", "unknown"),
        "num_repeats": len(all_sample_results),
        "status": "completed"  # Indicate that processing is done
    }


def attn_diff_text_save_aggregation(results):
    """
    Aggregate function that handles the new incremental JSONL writing approach.
    Since results are already saved incrementally, this function just provides a summary.
    """
    import json
    import os
    from datetime import datetime
    
    # With the new approach, results are already saved to JSONL files during processing
    # This function just provides a summary and metadata
    
    # Count total samples processed
    total_samples = len(results)
    
    # Create a summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"attn_diff_summary_{timestamp}.json"
    
    # Extract output file paths from results (if available)
    output_files = []
    for result in results:
        if "attn_diff_data" in result:
            for sample_data in result["attn_diff_data"]:
                if "output_file" in sample_data:
                    output_files.append(sample_data["output_file"])
    
    # Remove duplicates
    output_files = list(set(output_files))
    
    summary_data = {
        "metadata": {
            "timestamp": timestamp,
            "total_samples_processed": total_samples,
            "output_files": output_files,
            "description": "Attention difference analysis results from head masking experiments (incremental JSONL format)"
        },
        "status": "completed"
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Attention difference analysis completed!")
    print(f"Total samples processed: {total_samples}")
    print(f"Results saved to JSONL files: {output_files}")
    print(f"Summary saved to: {summary_file}")
    
    # Return a simple metric (number of samples processed)
    return total_samples

