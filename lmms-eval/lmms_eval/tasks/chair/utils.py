"""
CHAIR (Caption Hallucination Assessment with Image Relevance) evaluation utilities.

This module provides functions for evaluating vision-language models using the CHAIR metric,
which measures object hallucination in generated captions by comparing them against
ground truth annotations from MSCOCO.
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import datasets
import yaml
from PIL import Image

from lmms_eval.tasks.chair.chair import CHAIR


# =============================================================================
# Configuration and Constants
# =============================================================================

# Environment variables for evaluation control
MAX_IMAGES = int(os.environ.get("CHAIR_MAX_IMAGES", "0")) or None
ANN_PATH = os.environ.get("CHAIR_ANN_PATH")
IMAGES_ROOT = os.environ.get("CHAIR_IMAGES_ROOT")

# Load configuration from YAML
def _load_config():
    """Load configuration from chair_max64.yaml, handling function definitions."""
    config_path = Path(__file__).parent / "chair_max64.yaml"
    with open(config_path, "r") as f:
        raw_data = f.readlines()
        # Remove function definitions since yaml load cannot handle them
        safe_data = [line for line in raw_data if "!function" not in line]
        config = yaml.safe_load("".join(safe_data))
        if isinstance(config.get("dataset_path"), str):
            config["dataset_path"] = os.path.expandvars(config["dataset_path"])
        return config

config = _load_config()
dataset_path = config["dataset_path"]
OUTPUT_DIR = Path(config.get("output_dir", "./chair_outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COCO_SPLIT = config.get("coco_split", "val2014")
TASK_NAME = config.get("task")

# Set annotation path if not provided via environment
if not ANN_PATH:
    ANN_PATH = os.path.join(dataset_path, "annotations")
ANN_PATH = os.path.abspath(ANN_PATH)

if not IMAGES_ROOT:
    IMAGES_ROOT = dataset_path


# =============================================================================
# Dataset Processing
# =============================================================================

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
    enable_subset = os.getenv("LMMS_EVAL_ENABLE_SUBSET", "False").lower() == "true"
    subset_ratio = float(os.getenv("LMMS_EVAL_SUBSET_RATIO", "0.1"))
    
    # Shuffle the dataset with fixed seed
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


# =============================================================================
# Document Processing Functions
# =============================================================================

def chair_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Load image from path in the document."""
    image_path = doc.get("image_path", "")
    full_image_path = os.path.join(dataset_path, 'images', image_path)
    
    if os.path.exists(full_image_path):
        return [Image.open(full_image_path).convert("RGB")]
    else:
        raise FileNotFoundError(f"Image not found: {full_image_path}")


def chair_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any]) -> str:
    """Generate caption prompt from document and kwargs."""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    text_input = (doc.get("text_input") or "").strip()
    return f"{pre_prompt}{text_input}{post_prompt}"


def chair_doc_to_target(doc: Dict[str, Any]) -> str:
    """Dummy target for generate_until."""
    return ""


def chair_process_results(doc: Dict[str, Any], results: Any) -> Dict[str, Dict[str, Any]]:
    """
    Convert model's caption generation results to CHAIR cap_file entry format.
    CHAIR expects flat entries with (image_id, caption) format.
    """
    if isinstance(results, (list, tuple)) and len(results) > 0:
        gen_caption = str(results[0])
    else:
        gen_caption = str(results)

    if "image_id" not in doc or doc["image_id"] is None:
        raise ValueError("COCO image_id not found. doc requires image_id field.")
    
    image_id = int(doc["image_id"])
    return {"CHAIR": {"image_id": image_id, "caption": gen_caption}}


# =============================================================================
# CHAIR Evaluation Core
# =============================================================================

# Global cache for CHAIR evaluations to avoid recomputation
_CHAIR_CACHE: Dict[Tuple[str, int, Tuple], Tuple[float, float, Dict[str, str]]] = {}


def _write_jsonl(entries: List[Dict[str, Any]], path: Path) -> None:
    """Write entries to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _make_cache_key(results_list: List[Dict[str, Any]], metric_name: str) -> Tuple[str, int, Tuple]:
    """Create cache key based on metric name and list signature to prevent unintended reuse."""
    n = len(results_list)
    sig = []
    for i in (0, n // 2, n - 1):
        if 0 <= i < n:
            s = results_list[i]
            sig.append((s.get("image_id"), len(str(s.get("caption", "")))))
    return (metric_name, n, tuple(sig))


def _compute_chair_once(results_list: List[Dict[str, Any]], metric_name: str = "CHAIR", task_name: Any = None) -> Tuple[float, float, Dict[str, str]]:
    """
    Compute CHAIR metrics once and cache the results.
    
    Args:
        results_list: List of sample entries from lmms-eval
        metric_name: Name of the metric for file naming and caching
        task_name: Current task name (can be from args parameter)
        
    Returns:
        Tuple of (CHAIRs, CHAIRi, file_paths_dict)
    """
    global _CHAIR_CACHE
    cache_key = _make_cache_key(results_list, metric_name)
    if cache_key in _CHAIR_CACHE:
        return _CHAIR_CACHE[cache_key]

    # Convert results to flat entries (no deduplication to preserve denominator)
    flat_entries = []
    for s in results_list:
        image_id = int(s["image_id"])
        caption = s["caption"]
        flat_entries.append({"image_id": image_id, "caption": caption})

    # Optional: Limit maximum images (affects denominator/numerator)
    if MAX_IMAGES is not None and len(flat_entries) > MAX_IMAGES:
        random.seed(42)
        flat_entries = random.sample(flat_entries, MAX_IMAGES)

    # Use the runtime task name when available so task_alias-based runs keep
    # their CHAIR artifact filenames aligned with the reported task name.
    current_task_name = task_name if task_name else TASK_NAME
    cap_file_path = OUTPUT_DIR / f"caps_for_chair_{COCO_SPLIT}_{metric_name}_{current_task_name}.jsonl"
    _write_jsonl(flat_entries, cap_file_path)

    # Run CHAIR evaluation (with automatic annotation caching)
    print(f"[CHAIR] Initializing evaluator with annotation path: {ANN_PATH}")
    evaluator = CHAIR(ANN_PATH)
    cap_dict = evaluator.compute_chair(str(cap_file_path), "image_id", "caption")

    # Save results
    summary = {
        "cap_file": str(cap_file_path),
        "overall_metrics": cap_dict.get("overall_metrics", {}),
        "total_images": len({e["image_id"] for e in flat_entries}),
        "split": COCO_SPLIT,
        "ann_path": ANN_PATH,
        "images_root": IMAGES_ROOT,
    }
    
    summary_path = OUTPUT_DIR / f"chair_summary_{COCO_SPLIT}_{metric_name}_{current_task_name}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    details_path = OUTPUT_DIR / f"chair_sentences_{COCO_SPLIT}_{metric_name}_{current_task_name}.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(cap_dict.get("sentences", []), f, ensure_ascii=False, indent=2)

    # Extract metrics
    overall = summary["overall_metrics"]
    chairs = float(overall.get("CHAIRs", 0.0))
    chairi = float(overall.get("CHAIRi", 0.0))

    # Print results
    print(f"[CHAIR] Completed ({metric_name})!")
    print(f"- cap_file:   {cap_file_path}")
    print(f"- summary:    {summary_path}")
    print(f"- sentences:  {details_path}")
    print(f"- CHAIRs: {chairs:.4f}, CHAIRi: {chairi:.4f}")
    
    if "Recall" in overall:
        print(f"- Recall: {float(overall['Recall']):.4f}")
    if "Len" in overall:
        print(f"- AvgLen(x0.01): {float(overall['Len']):.4f}")

    # Cache and return results
    result = (chairs, chairi, {
        "cap_file": str(cap_file_path),
        "summary_file": str(summary_path),
        "details_file": str(details_path),
    })
    _CHAIR_CACHE[cache_key] = result
    return result


# =============================================================================
# Aggregation Functions
# =============================================================================

def chair_agg_chair(results: List[Dict[str, Any]], args: Any = None) -> Dict[str, float]:
    """
    Combined aggregation function that computes both CHAIRs and CHAIRi metrics
    in a single call to avoid duplicate annotation processing.
    
    Returns:
        Dictionary with both CHAIRs and CHAIRi metrics
    """
    task_name = getattr(args, "tasks", None)
    chairs, chairi, _ = _compute_chair_once(results, metric_name="CHAIR", task_name=task_name)
    return {"CHAIRs": chairs, "CHAIRi": chairi}


# Legacy aggregation functions (kept for backward compatibility)
def chair_agg_chairs(results: List[Dict[str, Any]], args: Any = None) -> float:
    """Legacy function for CHAIRs metric only."""
    task_name = getattr(args, "tasks", None)
    chairs, _, _ = _compute_chair_once(results, metric_name="CHAIRs", task_name=task_name)
    return chairs


def chair_agg_chairi(results: List[Dict[str, Any]], args: Any = None) -> float:
    """Legacy function for CHAIRi metric only."""
    task_name = getattr(args, "tasks", None)
    _, chairi, _ = _compute_chair_once(results, metric_name="CHAIRi", task_name=task_name)
    return chairi