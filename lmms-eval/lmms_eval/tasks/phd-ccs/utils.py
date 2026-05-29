# phd_ccs.py

import os
import re
import yaml
from pathlib import Path
from PIL import Image

# ------------------------
# YAML 로 dataset_path 읽기
# ------------------------
with open(Path(__file__).parent / "phd_ccs_yes.yaml", "r") as f:
    raw = f.readlines()
    safe = [ln for ln in raw if "!function" not in ln]
    config = yaml.safe_load("".join(safe))
    dataset_path = config["dataset_path"]  # 예: /root/.../Phd_ccs/data_ccs

# ------------------------
# 유틸: 이미지 찾기 (png/jpg 호환)
# ------------------------
_CAND_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

def _resolve_image(root_images_dir: str, name: str) -> str:
    p = os.path.join(root_images_dir, name)
    if os.path.exists(p):
        return p
    stem, _ = os.path.splitext(name)
    for e in _CAND_EXTS:
        q = os.path.join(root_images_dir, f"{stem}{e}")
        if os.path.exists(q):
            return q
    raise FileNotFoundError(f"Image not found: {name} (searched under {root_images_dir})")

# ------------------------
# doc → visual
# ------------------------
def phd_doc_to_visual(doc):
    """
    lmms-eval: return List[PIL.Image]
    doc 예시: {"image": "jpg_00035190.jpg", ...}
    """
    #기본: images, blank: veil_image, noise: noisy_image
    images_dir = os.path.join(dataset_path, "images")
    image_name = doc.get("image", "")
    full_path = _resolve_image(images_dir, image_name)
    return [Image.open(full_path).convert("RGB")]

# ------------------------
# doc → text
# ------------------------
def phd_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    # 필요시 yes/no만 답하라고 제한
    # post_prompt = post_prompt or "\nAnswer with a single word: yes or no."
    return f"{pre_prompt}{question}{post_prompt}"

# ------------------------
# 출력 정규화 (yes/no 판정)
# ------------------------
YES_TOKENS = {"yes", "yeah", "yep", "yup", "true", "correct"}
NO_TOKENS  = {"no", "nope", "nah", "false", "incorrect"}

def _norm_yesno(text: str) -> str:
    t = (text or "").strip().lower()

    # 1) 가장 앞 단어로 우선 판정
    first = re.split(r"\s|[.,!?;:()\"']", t.strip())[0] if t else ""
    if first in YES_TOKENS: return "yes"
    if first in NO_TOKENS:  return "no"

    # 2) 포함 판정 (동시 포함 방지)
    has_yes = any(tok in t for tok in YES_TOKENS)
    has_no  = any(tok in t for tok in NO_TOKENS)
    if has_yes and not has_no: return "yes"
    if has_no  and not has_yes: return "no"

    # 3) 마지막 fallback: 정확히 "yes"/"no" 토큰만 찾기
    if re.search(r"\byes\b", t): return "yes"
    if re.search(r"\bno\b",  t): return "no"

    # 4) 불명확 → 첫 토큰 반환(채점은 0 처리될 것)
    return first or ""

# ------------------------
# process_results: 항목별 결과
# ------------------------
def phd_process_results(doc, results):
    raw_pred = results[0] if results else ""
    pred = _norm_yesno(str(raw_pred))
    gt   = str(doc.get("answer", "")).strip().lower()
    assert gt in ("yes", "no"), f"Ground truth must be 'yes' or 'no', got: {gt}"

    score = 1.0 if pred == gt else 0.0
    # 메트릭 값은 숫자로!
    return {
        "phd_ccs_accuracy": score,

        # 상세 정보는 메트릭 키와 분리해서 보조 키로 남겨두세요(집계엔 안 쓰임)
        "phd_ccs_detail": {
            "question_id": doc.get("questionID") or doc.get("question_id") or doc.get("id"),
            "prediction": pred,
            "ground_truth": gt,
            "raw_text": str(raw_pred),
            "image": doc.get("image", "")
        }
    }


# ------------------------
# aggregation: 평균 정확도
# ------------------------
def phd_ccs_accuracy(results):
    """
    results: process_results에서 쌓인 dict들의 리스트
    평균 정확도(float) 반환
    """
    scores = []
    for r in results:
        item = r.get("phd_ccs_accuracy")
        if item is not None and "score" in item:
            scores.append(float(item["score"]))
    return sum(scores) / max(1, len(scores))

# ------------------------
# aggregation: F1 Score
# ------------------------
def phd_ccs_f1(results):
    """
    results: process_results에서 쌓인 dict들의 리스트
    F1 score 반환 (Yes-Recall, No-Recall 기반)
    """
    yes_total = no_total = 0
    yes_correct = no_correct = 0

    for r in results:
        detail = r.get("phd_ccs_detail", {})
        score = r.get("phd_ccs_accuracy", None)  # 1.0 or 0.0
        gt = detail.get("ground_truth")

        if gt == "yes":
            yes_total += 1
            if score == 1.0:
                yes_correct += 1
        elif gt == "no":
            no_total += 1
            if score == 1.0:
                no_correct += 1

    yes_recall = yes_correct / yes_total if yes_total > 0 else 0.0
    no_recall  = no_correct / no_total if no_total > 0 else 0.0

    if yes_recall + no_recall == 0:
        return 0.0

    f1 = 2 * (yes_recall * no_recall) / (yes_recall + no_recall)
    return f1
