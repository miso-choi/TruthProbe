"""
OmniMedVQA (e.g. SARS-CoV-2 CT-scan): multi-choice VQA with 2 or 4 options (option_A..D).

Ground truth is text in ``gt_answer``; we normalize documents to expose ``answer`` for ``doc_to_target``.
Scoring follows MMMU-style parsing (letter or option text) and compares to ``gt_answer``.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
from loguru import logger as eval_logger
from PIL import Image

# Root folder that contains the relative paths in ``image_path`` (e.g. .../omnimedvqa/Images/...).
_DEFAULT_IMAGE_ROOT = os.environ.get(
    "OMNIMEDVQA_IMAGE_ROOT",
    "/root/Desktop/workspace/miso/hub/datasets/lavis/omnimedvqa",
)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    - Map ``gt_answer`` -> ``answer`` (for doc_to_target).
    - Build ``_omnimed_options``: non-empty options in order (2 or 4 choices).
    Optional env (same as pope): LMMS_EVAL_ENABLE_SUBSET, LMMS_EVAL_SUBSET_RATIO, LMMS_EVAL_RANDOM_SEED.
    """
    random_seed = int(os.getenv("LMMS_EVAL_RANDOM_SEED", "42"))
    enable_subset = os.getenv("LMMS_EVAL_ENABLE_SUBSET", "False").lower() == "true"
    subset_ratio = float(os.getenv("LMMS_EVAL_SUBSET_RATIO", "0.1"))

    def _prep(example: Dict[str, Any]) -> Dict[str, Any]:
        opts: List[str] = []
        for k in ("option_A", "option_B", "option_C", "option_D"):
            v = example.get(k)
            if v is not None and str(v).strip():
                opts.append(str(v).strip())
        example["answer"] = example.get("gt_answer", "")
        example["_omnimed_options"] = opts
        return example

    out = dataset.map(_prep)
    out = out.shuffle(seed=random_seed)
    if enable_subset:
        n = max(1, int(len(out) * subset_ratio))
        out = out.select(range(n))
        eval_logger.info(f"omnimedvqa: subset enabled, using {n} examples")
    return out


def _format_choices(options: List[str]) -> str:
    lines = []
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def omnimedvqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with only the letter of the correct option (e.g., A or B), or the exact option text.",
    )
    q = str(doc["question"]).strip()
    options = doc.get("_omnimed_options") or []
    if not options:
        eval_logger.warning(f"No options for question_id={doc.get('question_id')!r}")
        return f"{pre}{q}{post}"
    choices = _format_choices(options)
    return f"{pre}{q}\n{choices}{post}"


def omnimedvqa_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    root = os.environ.get("OMNIMEDVQA_IMAGE_ROOT", _DEFAULT_IMAGE_ROOT)
    rel = doc.get("image_path", "")
    path = os.path.join(os.path.join('/'.join(root.split('/')[:-1]), 'OmniMedVQA'), rel) if rel else ""
    if not path or not os.path.isfile(path):
        eval_logger.error(f"OmniMedVQA image not found: {path!r} (set OMNIMEDVQA_IMAGE_ROOT if needed)")
        raise FileNotFoundError(path)
    return [Image.open(path).convert("RGB")]


def _normalize(s: str) -> str:
    return s.strip().lower()


# --- MMMU-style multi-choice helpers (subset of lmms_eval.tasks.mmmu.utils; kept local to avoid heavy imports) ---


def get_multi_choice_info(options: List[str]):
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response: str, all_choices: List[str], index2ans: Dict[str, str]) -> str:
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]

    return pred_index


def _pred_matches_gt(pred_letter: str, index2ans: Dict[str, str], gt: str) -> bool:
    if not gt:
        return False
    gt_n = _normalize(str(gt))
    if pred_letter in index2ans and _normalize(index2ans[pred_letter]) == gt_n:
        return True
    if _normalize(pred_letter) == gt_n and len(gt_n) == 1:
        return True
    return False


def _strip_trailing_punct(s: str) -> str:
    return s.strip().rstrip(".,;:!?\"')]}")


def _score_from_direct_text(raw: str, options: List[str], gt: str) -> Optional[bool]:
    """
    True/False if we can decide from a short exact answer (Yes/No, CT, …) without letter parsing.
    None if inconclusive (delegate to parse_multi_choice_response).
    """
    if not raw or not gt:
        return None
    gt_n = _normalize(str(gt))
    line = _normalize(_strip_trailing_punct(raw.split("\n")[0].strip()))
    if not line:
        return None
    # Single-token or short exact match to ground truth (e.g. "Yes", "CT")
    if line == gt_n:
        return True
    # Exact match to one of the option strings
    matched_opts = [_normalize(o) for o in options if _normalize(o) == line]
    if len(matched_opts) == 1 and matched_opts[0] == gt_n:
        return True
    if len(matched_opts) == 1 and matched_opts[0] != gt_n:
        return False
    return None


def omnimedvqa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    raw = results[0] if results else ""
    options: List[str] = doc.get("_omnimed_options") or []
    gt = str(doc.get("answer", doc.get("gt_answer", "")))

    parsed_letter = ""
    pred_text = ""
    if len(options) < 2:
        score = 0.0
    else:
        index2ans, all_choices = get_multi_choice_info(options)
        direct = _score_from_direct_text(raw, options, gt)
        if direct is not None:
            score = 1.0 if direct else 0.0
            # Best-effort letter for logging: which option text was echoed
            line = _normalize(_strip_trailing_punct(raw.split("\n")[0].strip()))
            for letter, text in index2ans.items():
                if _normalize(text) == line:
                    parsed_letter = letter
                    pred_text = text
                    break
            if not parsed_letter and direct:
                for letter, text in index2ans.items():
                    if _normalize(text) == _normalize(gt):
                        parsed_letter = letter
                        pred_text = text
                        break
        else:
            parsed_letter = parse_multi_choice_response(raw, all_choices, index2ans)
            pred_text = index2ans.get(parsed_letter, "")
            score = 1.0 if _pred_matches_gt(parsed_letter, index2ans, gt) else 0.0

    qid = doc.get("question_id", "")
    payload = {
        "question_id": qid,
        "score": score,
        "ground_truth": gt,
        "prediction_raw": raw,
        "parsed_letter": parsed_letter if len(options) >= 2 else "",
        "parsed_text": pred_text if len(options) >= 2 else "",
    }
    return {"omnimedvqa_accuracy": payload}


def omnimedvqa_aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(float(r.get("score", 0.0)) for r in results) / len(results)
