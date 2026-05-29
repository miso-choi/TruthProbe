# utils.py — CHAIR (Maxlinn style) integration for lmms-eval
# - Uses flat jsonl ({"image_id","caption"}) caps
# - Per-metric cache keys (avoid mixing CHAIRs/CHAIRi)
# - No dedup of entries before evaluation (preserve denominator for CHAIRs)

import os
import json
import time
from pathlib import Path
import yaml

# lmms_eval/tasks/chair/chair.py 가 Maxlinn 스타일 구현이어야 합니다.
from lmms_eval.tasks.chair.chair import CHAIR

# =======================
# 설정 로딩 (YAML은 그대로)
# =======================
CFG_PATH = Path(__file__).parent / "chair.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    raw = f.readlines()
    # YAML에서 !function 같은 커스텀 태그가 있으면 깨지는 걸 방지
    safe = [ln for ln in raw if "!function" not in ln]
    config = yaml.safe_load("".join(safe)) or {}

def _err(msg: str):
    raise RuntimeError(f"[CHAIR Config Error] {msg}\n- YAML: {CFG_PATH}")

# 필수: dataset_path (chair 폴더 루트)
DATASET_PATH = config.get("dataset_path")
if not DATASET_PATH:
    _err("`dataset_path`가 없습니다. 예: /root/.../datasets/lavis/chair")
DATASET_PATH = os.path.abspath(DATASET_PATH)

# 선택: split 기본값 (COCO 규칙 이름, 예: val2014)
COCO_SPLIT = config.get("coco_split", "val2014")

# 출력 디렉토리 (없으면 생성)
OUTPUT_DIR = Path(config.get("output_dir", "./chair_outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 프롬프트 prefix/suffix (옵션)
DEFAULT_PRE  = config.get("pre_prompt", "")
DEFAULT_POST = config.get("post_prompt", "")

# =======================
# 이미지/어노테이션 루트 분리
# =======================
# - 의도: 주석/메타는 chair 폴더에 있고, 이미지는 coco 폴더에 있을 때를 지원
# - 환경변수로 오버라이드 가능 (없으면 dataset_path 사용)
#   예) export CHAIR_IMAGES_ROOT="/root/.../datasets/lavis/coco"
IMAGES_ROOT = os.environ.get("CHAIR_IMAGES_ROOT", DATASET_PATH)

# - 어노테이션 루트도 필요 시 분리
#   예) export CHAIR_ANN_PATH="/root/.../datasets/lavis/coco/annotations"
ANN_PATH = os.environ.get("CHAIR_ANN_PATH", os.path.join(DATASET_PATH, "annotations"))
ANN_PATH = os.path.abspath(ANN_PATH)

# (옵션) 평가 이미지 수 제한: export CHAIR_MAX_IMAGES=500
try:
    MAX_IMAGES = int(os.environ.get("CHAIR_MAX_IMAGES", "0")) or None
except ValueError:
    MAX_IMAGES = None

# =======================
# 경로/리소스 검증
# =======================
def _check_annotations_or_die(ann_dir: str, split: str):
    # Maxlinn CHAIR는 captions/instances train+val 모두 읽어 GT를 구성
    need = [
        f"instances_{split}.json",     # e.g., instances_val2014.json
        f"captions_{split}.json",      # e.g., captions_val2014.json
        "instances_train2014.json",
        "captions_train2014.json",
    ]
    missing = [fn for fn in need if not os.path.exists(os.path.join(ann_dir, fn))]
    if missing:
        _err(
            "주석(annotations) 파일이 없습니다.\n"
            f"- ann_dir: {ann_dir}\n"
            f"- 누락: {missing}\n"
            "  → CHAIR_ANN_PATH 환경변수로 어노테이션 폴더를 지정하거나,\n"
            "    dataset_path/annotations 에 COCO 공식 JSON을 배치하세요."
        )

# 어노테이션 검증 (여기서 막아야 이후 CHAIR가 정상 동작)
_check_annotations_or_die(ANN_PATH, COCO_SPLIT)

# =======================
# 내부 유틸: 이미지 경로 해석
# =======================
def _resolve_image_path(doc: dict) -> str:
    """
    이미지 경로를 COCO 규칙에 맞춰 해석한다.
    우선순위: doc['image_path'](절대/상대) > doc['file_name'] > doc['image_id']

    상대 경로일 때 시도 순서 (존재하는 첫 경로를 사용):
      A) {IMAGES_ROOT}/{COCO_SPLIT}/{rel}
      B) {IMAGES_ROOT}/images/{COCO_SPLIT}/{rel}
      C) {IMAGES_ROOT}/images/{rel}
      D) {DATASET_PATH}/{COCO_SPLIT}/{rel}
      E) {DATASET_PATH}/images/{COCO_SPLIT}/{rel}
      F) {DATASET_PATH}/images/{rel}
    """
    def _candidates(rel_path: str):
        return [
            os.path.join(IMAGES_ROOT, COCO_SPLIT, rel_path),
            os.path.join(IMAGES_ROOT, "images", COCO_SPLIT, rel_path),
            os.path.join(IMAGES_ROOT, "images", rel_path),
            os.path.join(DATASET_PATH, COCO_SPLIT, rel_path),
            os.path.join(DATASET_PATH, "images", COCO_SPLIT, rel_path),
            os.path.join(DATASET_PATH, "images", rel_path),
        ]

    # 1) image_path 직접 지정
    image_path = doc.get("image_path")
    if image_path:
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        tried = []
        for cand in _candidates(image_path):
            tried.append(cand)
            if os.path.exists(cand):
                return cand
        # image_id 규칙 파일명으로 재시도
        if "image_id" in doc and doc["image_id"] is not None:
            fn = f"COCO_{COCO_SPLIT}_{int(doc['image_id']):012d}.jpg"
            for cand in _candidates(fn):
                tried.append(cand)
                if os.path.exists(cand):
                    return cand
        raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다: {image_path}\nTried:\n" + "\n".join(tried))

    # 2) COCO file_name
    file_name = doc.get("file_name")
    if file_name:
        if os.path.isabs(file_name) and os.path.exists(file_name):
            return file_name
        tried = []
        for cand in _candidates(file_name):
            tried.append(cand)
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다: {file_name}\nTried:\n" + "\n".join(tried))

    # 3) image_id만 있을 때 규칙 파일명 생성
    image_id = doc.get("image_id")
    if image_id is not None:
        fn = f"COCO_{COCO_SPLIT}_{int(image_id):012d}.jpg"
        tried = []
        for cand in _candidates(fn):
            tried.append(cand)
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"이미지 경로를 찾을 수 없습니다(image_id 기반): {fn}\nTried:\n" + "\n".join(tried))

    raise ValueError("doc에는 최소 하나가 필요합니다: image_path / file_name / image_id")

# =======================
# 훅들 (YAML에서 !function utils.* 로 참조)
# =======================
def chair_doc_to_visual(doc: dict):
    """이미지 로드 훅"""
    from PIL import Image
    full_image_path = _resolve_image_path(doc)
    if not os.path.exists(full_image_path):
        raise FileNotFoundError(f"Image not found: {full_image_path}")
    return [Image.open(full_image_path).convert("RGB")]

def chair_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict):
    """캡션 프롬프트 생성 훅"""
    pre_prompt  = lmms_eval_specific_kwargs.get("pre_prompt", DEFAULT_PRE)
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", DEFAULT_POST)
    text_input  = (doc.get("text_input") or "").strip()
    return f"{pre_prompt}{text_input}{post_prompt}"

def chair_doc_to_target(doc: dict) -> str:
    """generate_until용 더미 타깃"""
    return ""

def chair_process_results(doc: dict, results):
    """
    모델의 캡션 생성 결과를 CHAIR cap_file의 한 엔트리로 변환.
    Maxlinn CHAIR는 평평한 엔트리( image_id, caption ) 형태를 기대.
    """
    if isinstance(results, (list, tuple)) and len(results) > 0:
        gen_caption = str(results[0])
    else:
        gen_caption = str(results)

    if "image_id" not in doc or doc["image_id"] is None:
        raise ValueError("COCO image_id를 찾을 수 없습니다. doc에 image_id가 필요합니다.")
    image_id = int(doc["image_id"])

    # metric별로 동일 엔트리를 돌려보내되, lmms-eval이 두 번 수집하므로
    # _compute_chair_once에서 metric_name으로 구분 처리함.
    return {
        "CHAIRs": {"image_id": image_id, "caption": gen_caption},
        "CHAIRi": {"image_id": image_id, "caption": gen_caption},
    }

# =======================
# CHAIR 계산(캐시) + Aggregation (Maxlinn 포맷)
# =======================
# 전역 캐시: metric/리스트 서명 기반으로 분리
_CHAIR_CACHE = {}  # key: (metric_name, n, signature) -> (chairs, chairi, paths)

def _write_jsonl(entries, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def _make_cache_key(results_list, metric_name: str):
    """metric 이름 + 리스트 서명으로 캐시 키 만들기 (의도치 않은 재사용 방지)"""
    n = len(results_list)
    sig = []
    for i in (0, n // 2, n - 1):
        if 0 <= i < n:
            s = results_list[i]
            sig.append((s.get("image_id"), len(str(s.get("caption", "")))))
    return (metric_name, n, tuple(sig))

def _compute_chair_once(results_list, metric_name="CHAIRs"):
    """
    results_list: sample_entry 리스트 (lmms-eval이 metric별로 전달)
    - 중복 제거 없이 그대로 jsonl로 덤프 (CHAIRs 분모 보존)
    - metric_name으로 캡파일/요약파일 분리 저장
    """
    global _CHAIR_CACHE
    cache_key = _make_cache_key(results_list, metric_name)
    if cache_key in _CHAIR_CACHE:
        return _CHAIR_CACHE[cache_key]

    # ⚠️ dedup 없이 그대로 사용 (분모 왜곡 방지)
    flat_entries = []
    for s in results_list:
        iid = int(s["image_id"])
        cap = s["caption"]
        flat_entries.append({"image_id": iid, "caption": cap})

    # (옵션) 최대 이미지 제한: 분모/분자에 영향 → 기본은 미사용 권장
    if MAX_IMAGES is not None and len(flat_entries) > MAX_IMAGES:
        import random
        random.seed(42)
        flat_entries = random.sample(flat_entries, MAX_IMAGES)

    ts = time.strftime("%Y%m%d_%H%M%S")
    cap_file_path = OUTPUT_DIR / f"caps_for_chair_{COCO_SPLIT}_{metric_name}_{ts}.jsonl"
    _write_jsonl(flat_entries, cap_file_path)

    # Maxlinn CHAIR 실행 (json/jsonl + key 지정)
    evaluator = CHAIR(ANN_PATH)  # __init__(coco_path)
    cap_dict = evaluator.compute_chair(str(cap_file_path), "image_id", "caption")

    # 결과 저장
    summary = {
        "cap_file": str(cap_file_path),
        "overall_metrics": cap_dict.get("overall_metrics", {}),
        "total_images": len({e["image_id"] for e in flat_entries}),
        "split": COCO_SPLIT,
        "ann_path": ANN_PATH,
        "images_root": IMAGES_ROOT,
    }
    summary_path = OUTPUT_DIR / f"chair_summary_{COCO_SPLIT}_{metric_name}_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    details_path = OUTPUT_DIR / f"chair_sentences_{COCO_SPLIT}_{metric_name}_{ts}.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(cap_dict.get("sentences", []), f, ensure_ascii=False, indent=2)

    overall = summary["overall_metrics"]
    chairs = float(overall.get("CHAIRs", 0.0))
    chairi = float(overall.get("CHAIRi", 0.0))

    print(f"[CHAIR] Completed ({metric_name})!")
    print(f"- cap_file:   {cap_file_path}")
    print(f"- summary:    {summary_path}")
    print(f"- sentences:  {details_path}")
    print(f"- CHAIRs: {chairs:.4f}, CHAIRi: {chairi:.4f}")
    if "Recall" in overall:
        print(f"- Recall: {float(overall['Recall']):.4f}")
    if "Len" in overall:
        print(f"- AvgLen(x0.01): {float(overall['Len']):.4f}")

    result = (chairs, chairi, {
        "cap_file": str(cap_file_path),
        "summary_file": str(summary_path),
        "details_file": str(details_path),
    })
    _CHAIR_CACHE[cache_key] = result
    return result

def chair_agg_chairs(results, args=None):
    chairs, _, _ = _compute_chair_once(results, metric_name="CHAIRs")
    return chairs

def chair_agg_chairi(results, args=None):
    _, chairi, _ = _compute_chair_once(results, metric_name="CHAIRi")
    return chairi
