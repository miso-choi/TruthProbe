#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HaluEval probing CSV 시각화 — 연속 히트맵 (Gradient Stops 지원)

[핵심]
- 연속 히트맵 전용
- 색상 스톱을 직접 지정: --cont_stops "0:#FAFDD6,0.6:#91ADC8,1:#647FBC"
  * position: 0~1 또는 퍼센트(예: 60%)
  * HEX에 # 빠져도 자동 보정
- 기존 옵션도 유지:
  * --cont_colors "c1,c2,..." → 색만 주면 균등보간
  * --cont_cmap viridis       → 기본 colormap 사용

우선순위: cont_stops > cont_colors > cont_cmap
"""

import argparse
import os, re
from glob import glob
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# ---------- 스타일 ----------
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 50,
    "axes.labelsize": 34,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 30
})


# ---------- 유틸 ----------
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c_lower, orig in cols_lower.items():
        for cand in candidates:
            if re.search(rf"\b{re.escape(cand)}\b", c_lower):
                return orig
    return None


def normalize_df(df: pd.DataFrame, metric_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, str, str]:
    dfc = df.copy()
    layer_col = find_col(dfc, ["layer", "layers", "l"])
    head_col  = find_col(dfc, ["head", "heads", "h"])
    if metric_col is None or str(metric_col).lower() == "auto":
        candidates = ["accuracy", "acc", "f1", "score"]
        met_col = None
        for cand in candidates:
            mc = find_col(dfc, [cand])
            if mc is not None:
                met_col = mc
                break
        metric_col = met_col
    else:
        mc = find_col(dfc, [metric_col])
        metric_col = mc

    missing = []
    if layer_col is None: missing.append("layer")
    if head_col  is None: missing.append("head")
    if metric_col is None: missing.append("metric(accuracy/acc/f1/score 등)")
    if missing:
        raise ValueError(f"필수 컬럼을 찾지 못했습니다: {missing} / df.columns={list(dfc.columns)}")

    for c in [layer_col, head_col]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").astype("Int64")
    dfc[metric_col] = pd.to_numeric(dfc[metric_col], errors="coerce")
    dfc = dfc.dropna(subset=[layer_col, head_col, metric_col])
    return dfc, layer_col, head_col, metric_col


def parse_color_list(s: str) -> Optional[List[str]]:
    """'c1,c2,...' -> 색상 리스트 (#없는 HEX 자동 보정). 2개 미만이면 None."""
    if not s or not s.strip():
        return None
    raw = [c.strip() for c in s.split(",") if c.strip()]
    colors = []
    for c in raw:
        if re.fullmatch(r"[0-9A-Fa-f]{6}", c):
            colors.append("#" + c)
        else:
            colors.append(c)
    if len(colors) < 2:
        return None
    return colors


def parse_color_stops(s: str) -> Optional[List[Tuple[float, str]]]:
    """
    'p1:col1,p2:col2,...' -> [(pos, color), ...]
    - pos: 0~1 또는 '60%' 형태
    - color: 이름/HEX(자동 # 보정)
    - 잘못된 값은 무시, 결과가 2개 미만이면 None
    """
    if not s or not s.strip():
        return None
    out: List[Tuple[float, str]] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        p_str, col = token.split(":", 1)
        p_str, col = p_str.strip(), col.strip()
        if not p_str or not col:
            continue

        # pos 파싱
        if p_str.endswith("%"):
            try:
                val = float(p_str[:-1]) / 100.0
            except Exception:
                continue
        else:
            try:
                val = float(p_str)
            except Exception:
                continue

        # 색상 보정
        if re.fullmatch(r"[0-9A-Fa-f]{6}", col):
            col = "#" + col

        out.append((val, col))

    # 유효성 필터: 0~1 범위 내
    out = [(p, c) for (p, c) in out if np.isfinite(p) and 0.0 <= p <= 1.0]
    if not out:
        return None

    # 정렬 & 중복 pos 병합(마지막 색 우선)
    out.sort(key=lambda x: x[0])
    merged: List[Tuple[float, str]] = []
    for p, c in out:
        if not merged or abs(merged[-1][0] - p) > 1e-9:
            merged.append((p, c))
        else:
            merged[-1] = (p, c)

    if len(merged) < 2:
        return None
    return merged


# ---------- 연속 히트맵 ----------
def draw_continuous_heatmap(
    piv_values: pd.DataFrame,
    title: str,
    out_png: str,
    cmap: Union[str, ListedColormap, LinearSegmentedColormap] = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fix01: bool = False,
    dpi: int = 200,
):
    arr = piv_values.values.astype(float)
    if fix01:
        arr = np.clip(arr, 0.0, 1.0)
    vmin_eff = 0.0 if (fix01 and vmin is None) else (np.nanmin(arr) if vmin is None else vmin)
    vmax_eff = 1.0 if (fix01 and vmax is None) else (np.nanmax(arr) if vmax is None else vmax)

    nrows, ncols = piv_values.shape
    side_cells = max(nrows, ncols)
    cell_size = 0.55
    fig_w = max(5.0, side_cells * cell_size) + 12
    fig_h = max(5.0, side_cells * cell_size) + 10

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="equal", vmin=vmin_eff, vmax=vmax_eff, cmap=cmap)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Head", fontweight="bold")
    ax.set_ylabel("Layer", fontweight="bold")
    ax.set_xticks(range(ncols)); ax.set_yticks(range(nrows))
    ax.set_xticklabels(piv_values.columns, fontweight="bold")
    ax.set_yticklabels(piv_values.index, fontweight="bold")
    ax.set_box_aspect(1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=38) 
    #cbar.ax.set_ylabel("Value", rotation=270, labelpad=12, fontweight="bold")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18, left=0.14, right=0.90, top=0.90)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="CSV들이 들어있는 폴더 경로")
    ap.add_argument("--fname_regex", type=str, default=r".*\.csv$", help="파일명 정규식 필터")
    ap.add_argument("--dpi", type=int, default=200)

    # metric
    ap.add_argument("--metric_col", type=str, default="auto",
                    help="사용할 metric 컬럼명 (기본 auto: accuracy→acc→f1→score 탐색)")
    ap.add_argument("--fix01", action="store_true",
                    help="값을 [0,1]로 클리핑하고 vmin/vmax 기본을 0/1로 고정")

    # 연속 히트맵 옵션
    ap.add_argument("--cont_cmap", type=str, default="magma",
                    help="matplotlib colormap 이름 (예: magma, viridis, plasma, inferno, coolwarm 등)")
    ap.add_argument("--cont_colors", type=str, default="",
                    help="쉼표로 구분된 색 리스트 → 균등 보간 gradient. 예: '#FAFDD6,#91ADC8,#647FBC' 또는 'FAFDD6,91ADC8,647FBC'")
    ap.add_argument("--cont_stops", type=str, default="",
                    help="pos:color 쉼표 나열. pos는 0~1 또는 % 사용. 예: '0:#FAFDD6,0.6:#91ADC8,1:#647FBC' or '0%:FAFDD6,60%:91ADC8,100%:647FBC'")
    ap.add_argument("--cont_vmin", type=float, default=None)
    ap.add_argument("--cont_vmax", type=float, default=None)

    args = ap.parse_args()

    # CSV 수집
    csv_files = [p for p in glob(os.path.join(args.root, "**", "*.csv"), recursive=True)
                 if re.search(args.fname_regex, os.path.basename(p))]
    csv_files.sort()
    if not csv_files:
        print(f"[ERR] CSV 없음: {args.root}")
        return
    print(f"[INFO] 발견된 CSV: {len(csv_files)}개")
    for p in csv_files: print(" -", p)

    out_root = os.path.join(args.root, "_plots")
    os.makedirs(out_root, exist_ok=True)

    # 1) stops 우선
    cmap_obj: Optional[Union[ListedColormap, LinearSegmentedColormap, str]] = None
    stops = parse_color_stops(args.cont_stops)
    if stops is not None:
        positions, colors = zip(*stops)
        cmap_obj = LinearSegmentedColormap.from_list("custom_stops", list(zip(positions, colors)))
    else:
        # 2) colors (균등 보간)
        color_list = parse_color_list(args.cont_colors)
        if color_list is not None:
            cmap_obj = LinearSegmentedColormap.from_list("custom_colors", color_list)
        else:
            # 3) 기본 colormap 이름
            cmap_obj = args.cont_cmap

    # 파일별 처리
    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(out_root, base)
        os.makedirs(out_dir, exist_ok=True)

        try:
            df = pd.read_csv(csv_path)
            df, layer_col, head_col, met_col = normalize_df(df, metric_col=args.metric_col)
        except Exception as e:
            print(f"[WARN] 스킵({base}): {e}")
            continue

        df = df.sort_values([layer_col, head_col]).copy()
        piv = df.pivot_table(index=layer_col, columns=head_col, values=met_col, aggfunc="mean")

        out_png_cont = os.path.join(out_dir, f"{base}_HEAT_CONT.png")
        draw_continuous_heatmap(
            piv_values=piv,
            title=f"{base}",
            out_png=out_png_cont,
            cmap=cmap_obj,
            vmin=args.cont_vmin,
            vmax=args.cont_vmax,
            fix01=args.fix01,
            dpi=args.dpi
        )

        print(f"[OK] {base} 완료 -> {out_dir}")

    print(f"\n[DONE] 결과는 {out_root} 밑에 저장됨 ✅")


if __name__ == "__main__":
    main()
