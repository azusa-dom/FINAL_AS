#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/boot_auc_ap.py
──────────────────
1) Recursively reads fold_*_predictions.csv under PRED_DIR
   · Automatically identifies true label column (label / true_label)
   · Uses "prob" column directly if exists
   · Otherwise, calculates P(class 1) = softmax(logits) from logit_0 / logit_1

2) Merges 5-fold validation samples, calculates
   · AUROC and AUPRC
   · 2,000 bootstrap iterations → 95% confidence intervals

3) Writes results to OUT_CSV and prints to terminal

Dependencies: numpy, pandas, scikit-learn, scipy
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt # Needed for configure_cns_style

# --- Import Custom Theme (even if not plotting, for consistency) ---
import sys
PROJECT_ROOT_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, 'src'))

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into boot_auc_ap.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. boot_auc_ap.py will use default Matplotlib style.")


# ──────────── User Modifiable Area ────────────
# Updated paths based on your latest information
ROOT_DIR = Path("/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS")
PRED_DIR = ROOT_DIR / "results/clinical/clinical_model/clinical_preds" # <--- UPDATED THIS PATH
OUT_CSV  = ROOT_DIR / "results/final_run/boot_metrics_ci.csv"   # Output CSV path. Adjust if needed.

N_BOOTSTRAP = 2_000
ALPHA = 0.95          # Confidence level
SEED = 42
# ─────────────────────────────────────


def find_label_col(columns: list[str]) -> str:
    """Returns the true label column name; raises exception if not found"""
    candidates = [c for c in columns if c.lower() in {"label", "true_label"}]
    if candidates:
        return candidates[0]

    fuzzy = [c for c in columns if "label" in c.lower()]
    if fuzzy:
        return fuzzy[0]

    raise KeyError("❌ No 'label' or 'true_label' column found")


def logits_to_prob(df: pd.DataFrame) -> np.ndarray:
    """Calculates P(class 1) from logit_0 / logit_1 if 'prob' column is missing"""
    logit_cols = [c for c in df.columns if c.lower().startswith("logit")]
    if len(logit_cols) < 2:
        raise KeyError("❌ Neither 'prob' column nor two 'logit_*' columns available for softmax conversion")

    # Extract numeric suffix to determine negative/positive class columns
    parsed = []
    for col in logit_cols:
        m = re.search(r"(\d+)", col)
        parsed.append((int(m.group(1)) if m else -1, col))
    parsed.sort(key=lambda x: x[0])  # Smaller number for negative class, larger for positive

    neg_col, pos_col = parsed[0][1], parsed[-1][1]
    l0, l1 = df[neg_col].astype(float).values, df[pos_col].astype(float).values
    exp0, exp1 = np.exp(l0), np.exp(l1)
    return exp1 / (exp0 + exp1)


def collect_predictions(pred_dir: Path) -> pd.DataFrame:
    """Reads all fold_*_predictions.csv files → Returns merged DataFrame (label, prob)"""
    pattern = pred_dir / "fold_*_predictions.csv"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"❌ No files found matching {pattern}")

    dfs: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp)
        label_col = find_label_col(df.columns.tolist())

        prob_series = None
        prob_cols = [c for c in df.columns if "prob" in c.lower() and "baseline" not in c.lower()]
        if prob_cols:
            prob_series = df[[c for c in prob_cols if c.lower() == "prob"][0]] if any(
                c.lower() == "prob" for c in prob_cols) else df[prob_cols[0]]
        else:
            prob_series = pd.Series(logits_to_prob(df), name="prob")

        dfs.append(pd.DataFrame({
            "label": df[label_col].astype(int),
            "prob": prob_series.astype(float)
        }))

    merged = pd.concat(dfs, ignore_index=True)
    print(f"✅ Merged {len(files)} files, total {len(merged)} rows")
    return merged


def bootstrap_ci(
        y: np.ndarray,
        p: np.ndarray,
        metric_fn,
        n_boot: int = 2_000,
        alpha: float = 0.95,
        seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """Returns (original score, (low, high))"""
    score_orig = metric_fn(y, p)

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:  # Skip if bootstrap sample has only one class
            continue
        scores.append(metric_fn(yb, pb))

    low_p = (1 - alpha) / 2 * 100
    high_p = (alpha + (1 - alpha) / 2) * 100
    ci_low, ci_high = np.percentile(scores, [low_p, high_p])
    return score_orig, (ci_low, ci_high)


def main() -> None:
    if _theme_loaded: # Even if this script doesn't plot, apply theme for consistency
        configure_cns_style()

    df = collect_predictions(PRED_DIR)
    y, p = df["label"].values, df["prob"].values

    auc, auc_ci = bootstrap_ci(y, p, roc_auc_score, N_BOOTSTRAP, ALPHA, SEED)
    ap, ap_ci = bootstrap_ci(y, p, average_precision_score, N_BOOTSTRAP, ALPHA, SEED)

    # Print results
    print(f"AUROC = {auc:.3f}  (95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f})")
    print(f"AUPRC = {ap:.3f}   (95% CI {ap_ci[0]:.3f}–{ap_ci[1]:.3f})")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "metric":   ["AUROC", "AUPRC"],
        "mean":     [auc, ap],
        "ci_lower": [auc_ci[0], ap_ci[0]],
        "ci_upper": [auc_ci[1], ap_ci[1]],
    }).to_csv(OUT_CSV, index=False)
    print(f"📄 Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()