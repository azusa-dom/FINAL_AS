#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/boot_auc_ap.py
──────────────────
1) 递归读取 PRED_DIR 下的 fold_*_predictions.csv
   · 自动识别真实标签列（label / true_label）
   · 若存在 “prob” 列直接使用
   · 否则从 logit_0 / logit_1 计算 P(class 1) = softmax(logits)

2) 合并 5-fold 验证集样本，计算
   · AUROC 与 AUPRC
   · 2 000 次 bootstrap → 95 % 置信区间

3) 结果写入 OUT_CSV，并在终端打印

依赖：numpy, pandas, scikit-learn, scipy
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ──────────── 用户可修改区 ────────────
from pathlib import Path

ROOT_DIR = Path("/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS")  # ← 手动写死
PRED_DIR = ROOT_DIR / "results/final_run/clinical_preds/clinical_data_fold"
OUT_CSV  = ROOT_DIR / "results/final_run/boot_metrics_ci.csv"

N_BOOTSTRAP = 2_000
ALPHA = 0.95          # 置信度
SEED = 42
# ─────────────────────────────────────


def find_label_col(columns: list[str]) -> str:
    """返回真实标签列名；若未找到则抛异常"""
    candidates = [c for c in columns if c.lower() in {"label", "true_label"}]
    if candidates:
        return candidates[0]

    fuzzy = [c for c in columns if "label" in c.lower()]
    if fuzzy:
        return fuzzy[0]

    raise KeyError("❌  未找到 label / true_label 列")


def logits_to_prob(df: pd.DataFrame) -> np.ndarray:
    """若没有 prob 列，则从 logit_0 / logit_1 计算 P(class 1)"""
    logit_cols = [c for c in df.columns if c.lower().startswith("logit")]
    if len(logit_cols) < 2:
        raise KeyError("❌  既无 prob 列，也不足两列 logit_* 用于 softmax")

    # 提取数字后缀，确定正负类列
    parsed = []
    for col in logit_cols:
        m = re.search(r"(\d+)", col)
        parsed.append((int(m.group(1)) if m else -1, col))
    parsed.sort(key=lambda x: x[0])  # 小编号当负类，大编号当正类

    neg_col, pos_col = parsed[0][1], parsed[-1][1]
    l0, l1 = df[neg_col].astype(float).values, df[pos_col].astype(float).values
    exp0, exp1 = np.exp(l0), np.exp(l1)
    return exp1 / (exp0 + exp1)


def collect_predictions(pred_dir: Path) -> pd.DataFrame:
    """读取所有 fold_*_predictions.csv → 返回合并后的 DataFrame(label, prob)"""
    pattern = pred_dir / "fold_*_predictions.csv"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"❌  未找到文件 {pattern}")

    dfs: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp)
        label_col = find_label_col(df.columns.tolist())

        # 取 prob 列或计算
        prob_series = None
        prob_cols = [c for c in df.columns if "prob" in c.lower() and "baseline" not in c.lower()]
        if prob_cols:
            # 优先精确匹配 "prob"
            prob_series = df[[c for c in prob_cols if c.lower() == "prob"][0]] if any(
                c.lower() == "prob" for c in prob_cols) else df[prob_cols[0]]
        else:
            prob_series = pd.Series(logits_to_prob(df), name="prob")

        dfs.append(pd.DataFrame({
            "label": df[label_col].astype(int),
            "prob": prob_series.astype(float)
        }))

    merged = pd.concat(dfs, ignore_index=True)
    print(f"✅  合并 {len(files)} 个文件，共 {len(merged)} 行")
    return merged


def bootstrap_ci(
        y: np.ndarray,
        p: np.ndarray,
        metric_fn,
        n_boot: int = 2_000,
        alpha: float = 0.95,
        seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """返回 (原始分数, (low, high))"""
    score_orig = metric_fn(y, p)

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:  # 全 1 或全 0 跳过
            continue
        scores.append(metric_fn(yb, pb))

    low_p = (1 - alpha) / 2 * 100
    high_p = (alpha + (1 - alpha) / 2) * 100
    ci_low, ci_high = np.percentile(scores, [low_p, high_p])
    return score_orig, (ci_low, ci_high)


def main() -> None:
    df = collect_predictions(PRED_DIR)
    y, p = df["label"].values, df["prob"].values

    auc, auc_ci = bootstrap_ci(y, p, roc_auc_score, N_BOOTSTRAP, ALPHA, SEED)
    ap, ap_ci = bootstrap_ci(y, p, average_precision_score, N_BOOTSTRAP, ALPHA, SEED)

    # 打印结果
    print(f"AUROC = {auc:.3f}  (95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f})")
    print(f"AUPRC = {ap:.3f}   (95% CI {ap_ci[0]:.3f}–{ap_ci[1]:.3f})")

    # 保存 CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "metric":   ["AUROC", "AUPRC"],
        "mean":     [auc, ap],
        "ci_lower": [auc_ci[0], ap_ci[0]],
        "ci_upper": [auc_ci[1], ap_ci[1]],
    }).to_csv(OUT_CSV, index=False)
    print(f"📄  Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
