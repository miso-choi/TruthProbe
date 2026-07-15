#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HaluEval probing CSV 시각화 — Correlation 전용 (Accuracy만)

- 입력: 여러 CSV에서 (layer, head, accuracy)만 사용
- 출력:
  1) GLOBAL_CORR_MATRIX_ACC.png  : 모든 파일 간 Accuracy 피어슨 상관 '행렬'
  2) GLOBAL_CORR_BARS_*.png      : (옵션) 막대그래프
     * --bar_mode mean : 각 파일의 (자기 자신 제외) 상관을 음수→0 클립 후 평균
     * --bar_mode ref  : 기준 파일(부분 문자열로 매칭)과의 상관을 음수→0 클립하여 표시
     * y축 하한은 항상 0
  3) (옵션) GLOBAL_CORR_MATRIX_ACC.csv : 상관행렬 수치 저장

예시:
python /root/Desktop/workspace/seonga/faithful-lmms-eval/probing_code/analysis/correlation.py \
  --root /root/Desktop/workspace/seonga/faithful-lmms-eval/lmms-eval/probe_merged/reproduce/csv \
  --fname_regex ".*\\.csv$" \
  --dpi 200 \
  --bar_mode ref \
  --bar_ref Vicuna
"""

import argparse
import os, re
from glob import glob
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 전역 폰트/스타일 업그레이드 (크게+굵게) ----
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})


# ---------------------------
# 유틸
# ---------------------------
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """대소문자 무시 완전일치 → 단어경계 부분정규식 순으로 탐색"""
    cols_lower = {c.lower(): c for c in df.columns}
    # 완전일치
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # 부분정규식(단어 경계)
    for c_lower, orig in cols_lower.items():
        for cand in candidates:
            if re.search(rf"\b{re.escape(cand)}\b", c_lower):
                return orig
    return None


def normalize_df_acc_only(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    """layer/head/accuracy만 강제 + 0/1-base 자동 보정"""
    dfc = df.copy()
    layer_col = find_col(dfc, ["layer", "layers", "l"])
    head_col  = find_col(dfc, ["head", "heads", "h", "head_idx"])
    acc_col   = find_col(dfc, ["accuracy", "acc", "accu", "acc_score"])

    missing = []
    if layer_col is None: missing.append("layer")
    if head_col  is None: missing.append("head")
    if acc_col   is None: missing.append("accuracy")
    if missing:
        raise ValueError(f"필수 컬럼을 찾지 못했습니다: {missing} / df.columns={list(dfc.columns)}")

    # 강제 숫자화
    for c in [layer_col, head_col]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    dfc[acc_col] = pd.to_numeric(dfc[acc_col], errors="coerce")
    dfc = dfc.dropna(subset=[layer_col, head_col, acc_col])

    # 0/1-base 자동 보정 (최소값이 1이고 0이 없으면 1 빼기)
    lvals = dfc[layer_col].astype(int)
    if (lvals.min() == 1) and (0 not in set(lvals)):
        dfc[layer_col] = (lvals - 1).astype(int)
    else:
        dfc[layer_col] = lvals.astype(int)

    hvals = dfc[head_col].astype(int)
    if (hvals.min() == 1) and (0 not in set(hvals)):
        dfc[head_col] = (hvals - 1).astype(int)
    else:
        dfc[head_col] = hvals.astype(int)

    return dfc, layer_col, head_col, acc_col


def label_from_path(fpath: str, root: Optional[str] = None, levels: int = 2) -> str:
    if root:
        try:
            rel = os.path.relpath(fpath, root)
        except ValueError:
            rel = fpath
    else:
        rel = fpath
    parts = rel.replace("\\", "/").split("/")
    base = os.path.splitext(parts[-1])[0]
    prefix = "/".join(parts[max(0, len(parts)-1-levels): -1])
    label = f"{prefix}/{base}" if prefix else base
    return label


def uniquify(names: List[str]) -> List[str]:
    seen = {}
    uniq = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            uniq.append(n)
        else:
            seen[n] += 1
            uniq.append(f"{n}#{seen[n]}")
    return uniq


# ---------------------------
# 전역 상관(Accuracy 전용)
# ---------------------------
def build_global_corr_matrix_accuracy_only(
    piv_by_file: Dict[str, pd.DataFrame],
    out_png: str,
    title: str,
    dpi: int = 200,
    label_root: Optional[str] = None,
    label_levels: int = 2
):
    files = list(piv_by_file.keys())
    if len(files) == 0:
        print("[WARN] 전역 상관(Accuracy): 파일이 없습니다.")
        return None

    # 1) 공통 (layer, head) 교집합
    common_layers = None
    common_heads  = None
    for f in files:
        L = set(piv_by_file[f].index.astype(int))
        H = set(piv_by_file[f].columns.astype(int))
        common_layers = L if common_layers is None else (common_layers & L)
        common_heads  = H if common_heads  is None else (common_heads  & H)

    if not common_layers or not common_heads:
        print("[WARN] 전역 상관(Accuracy): 공통 layer/head 교집합이 비었습니다.")
        return None

    common_layers = np.array(sorted(common_layers), dtype=int)
    common_heads  = np.array(sorted(common_heads), dtype=int)

    # 2) 좌표 순서 고정 (모든 파일 동일)
    #    meshgrid → ravel 순서가 모든 파일에서 동일하도록 보장
    #    (실제 값 추출은 reindex 후 values.ravel로 동일 순서 확보)
    #    coords 자체는 디버그용(필수는 아님)
    # L, H = np.meshgrid(common_layers, common_heads, indexing="ij")
    # coords = np.c_[L.ravel(), H.ravel()]

    # 3) 파일별 벡터 + 전역 유효 마스크
    matrices = []
    masks = []
    for f in files:
        piv = piv_by_file[f].copy()
        piv = piv.reindex(index=common_layers, columns=common_heads)
        A = piv.values.astype(float)   # (nL, nH)
        v = A.ravel()
        m = ~np.isnan(v)
        matrices.append(v)
        masks.append(m)

    # 4) 전역 유효 위치(모든 파일에서 동시에 유효)
    mask_all = np.logical_and.reduce(masks)
    if not np.any(mask_all):
        print("[WARN] 전역 상관(Accuracy): 공통 유효 위치가 없습니다 (전부 NaN).")
        return None

    X = np.stack([v[mask_all] for v in matrices], axis=1)  # (N_valid, K)
    if X.shape[0] < 2 or X.shape[1] < 2:
        print("[WARN] 전역 상관(Accuracy): 유효 표본/파일이 부족합니다.")
        return None

    corr = np.corrcoef(X, rowvar=False)  # (K, K)
    print("Correlation: ", corr)

    # 라벨
    col_names: List[str] = []
    for f in files:
        label = label_from_path(f, root=label_root, levels=label_levels)
        col_names.append(label)
    col_names = uniquify(col_names)

    # 5) 플롯
    k = corr.shape[0]
    cell = 0.72
    side = max(6.0, k * cell)
    fig_w = side * 1.6
    fig_h = side * 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, aspect="equal", cmap="coolwarm")

    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=13, fontweight="bold")
    ax.set_yticklabels(col_names, fontsize=13, fontweight="bold")

    for i in range(k):
        for j in range(k):
            if not np.isnan(corr[i, j]):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=20, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32, left=0.35, right=0.98, top=0.92)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 전역 상관(Accuracy) 행렬 저장: {out_png}")
    return corr, col_names


# ---------------------------
# 전역 상관 막대그래프 (음수→0 클립, y축 하한=0)
# ---------------------------
def plot_corr_bars(corr: np.ndarray, labels: List[str], out_png: str,
                   mode: str = "mean", ref_substr: str = "", dpi: int = 200):
    K = corr.shape[0]
    if K < 2:
        print("[WARN] 막대그래프: 유효 열/행이 2개 미만")
        return

    if mode == "mean":
        vals = []
        for i in range(K):
            mask = np.ones(K, dtype=bool); mask[i] = False
            vals.append(np.nanmean(np.clip(corr[i, mask], 0.0, None)))  # 음수 0 클립
        y = np.array(vals)
        x_labels = labels
        title = "Mean Pearson r per file (off-diagonal, negatives→0)"
    elif mode == "ref":
        if not ref_substr:
            print("[WARN] --bar_mode=ref 이지만 --bar_ref가 비어 있음 → 스킵")
            return
        ref_idx = None
        for i, lb in enumerate(labels):
            if ref_substr.lower() in lb.lower():
                ref_idx = i
                break
        if ref_idx is None:
            print(f"[WARN] 기준 라벨을 찾지 못했습니다: '{ref_substr}' → 스킵")
            return
        mask = np.ones(K, dtype=bool); mask[ref_idx] = False
        y = np.clip(corr[ref_idx, mask], 0.0, None)  # 음수 0 클립
        x_labels = [labels[i] for i in range(K) if i != ref_idx]
        title = f"Pearson r vs reference: {labels[ref_idx]} (negatives→0)"
    else:
        return

    n = len(x_labels)
    width = max(8.0, 0.65 * n)
    height = 6.2
    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(range(n), y)
    ax.set_ylim(0.0, 1.05)   # 바닥을 0으로
    ax.set_ylabel("Pearson r", fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30, left=0.10, right=0.98, top=0.92)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 전역 상관 막대그래프 저장: {out_png}")


# ---------------------------
# 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="CSV들이 들어있는 폴더 경로")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--fname_regex", type=str, default=r".*\.csv$")
    ap.add_argument("--label_levels", type=int, default=2, help="전역 행렬 라벨에 사용할 뒤쪽 폴더 단 수")
    ap.add_argument("--bar_mode", type=str, default="off", choices=["off", "mean", "ref"],
                    help="전역 상관 막대그래프: off(생략), mean(평균), ref(기준 파일 대비)")
    ap.add_argument("--bar_ref", type=str, default="",
                    help="--bar_mode=ref일 때 기준 파일을 찾기 위한 서브스트링(라벨에 부분일치)")
    ap.add_argument("--save_csv", action="store_true", help="상관행렬을 CSV로 저장")
    args = ap.parse_args()

    # 대상 CSV
    csv_files = [p for p in glob(os.path.join(args.root, "**", "*.csv"), recursive=True)
                 if re.search(args.fname_regex, os.path.basename(p))]
    csv_files = sorted(csv_files)
    if not csv_files:
        print(f"[ERR] CSV 없음: {args.root}")
        return

    print(f"[INFO] 발견된 CSV: {len(csv_files)}개")
    for p in csv_files:
        print(" -", p)

    out_root = os.path.join(args.root, "_plots")
    os.makedirs(out_root, exist_ok=True)

    piv_by_file: Dict[str, pd.DataFrame] = {}

    # 각 CSV에서 피벗(Accuracy만)
    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            df = pd.read_csv(csv_path)
            df, layer_col, head_col, acc_col = normalize_df_acc_only(df)
        except Exception as e:
            print(f"[WARN] 스킵({base}): {e}")
            continue

        df = df.sort_values([layer_col, head_col]).copy()
        piv_acc = df.pivot_table(index=layer_col, columns=head_col, values=acc_col, aggfunc="mean")
        piv_by_file[csv_path] = piv_acc

    # ===== 모든 파일 Accuracy 전역 상관 행렬 =====
    files = list(piv_by_file.keys())
    try:
        common_root = os.path.commonpath(files) if files else None
    except ValueError:
        common_root = None

    global_corr_png = os.path.join(out_root, "GLOBAL_CORR_MATRIX_ACC.png")
    res = build_global_corr_matrix_accuracy_only(
        piv_by_file,
        out_png=global_corr_png,
        title="HaluEval Correlation Across Files (Accuracy Only)",
        dpi=args.dpi,
        label_root=common_root,
        label_levels=args.label_levels
    )
    if res is not None:
        corr_mat, corr_labels = res
        # CSV 저장 옵션
        if args.save_csv:
            csv_out = os.path.join(out_root, "GLOBAL_CORR_MATRIX_ACC.csv")
            pd.DataFrame(corr_mat, index=corr_labels, columns=corr_labels).to_csv(csv_out, encoding="utf-8-sig")
            print(f"[OK] 전역 상관(Accuracy) 행렬 CSV 저장: {csv_out}")

        # ===== (옵션) 전역 상관 막대그래프 =====
        if args.bar_mode != "off":
            bar_png = os.path.join(out_root, f"GLOBAL_CORR_BARS_{args.bar_mode.upper()}.png")
            plot_corr_bars(
                corr=corr_mat,
                labels=corr_labels,
                out_png=bar_png,
                mode=args.bar_mode,
                ref_substr=args.bar_ref,
                dpi=args.dpi
            )

    print(f"\n[DONE] 결과는 {out_root} 밑에 저장됨 ✅")


if __name__ == "__main__":
    main()
