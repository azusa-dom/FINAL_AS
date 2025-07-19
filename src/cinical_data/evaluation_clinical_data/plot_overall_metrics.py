#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_overall_metrics.py
Generates overall ROC curve, PR curve, and predicted probability distribution
plots by aggregating predictions across all cross-validation folds.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import re

# ─────────── Import Custom CNSStyle Theme ───────────
PROJECT_ROOT_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, 'src'))

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into plot_overall_metrics.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. Will use default Matplotlib style.")
    plt.rcParams.update({
        "figure.dpi":      300, "savefig.dpi":     300, "figure.figsize":  (8, 6),
        "font.family":     "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":       12, "axes.titlesize":  16, "axes.labelsize":  14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

# ─────────── Helper Functions ───────────
def find_label_col(columns: list[str]) -> str:
    candidates = [c for c in columns if c.lower() in {"label", "true_label"}]
    if candidates:
        return candidates[0]
    raise KeyError("❌ No 'label' or 'true_label' column found")

def logits_to_prob(df: pd.DataFrame) -> np.ndarray:
    logit_cols = [c for c in df.columns if c.lower().startswith("logit")]
    if len(logit_cols) < 2:
        raise KeyError("❌ Neither 'prob' column nor two 'logit_*' columns available for softmax conversion")

    parsed = []
    for col in logit_cols:
        m = re.search(r"(\d+)", col)
        parsed.append((int(m.group(1)) if m else -1, col))
    parsed.sort(key=lambda x: x[0])

    neg_col, pos_col = parsed[0][1], parsed[-1][1]
    l0, l1 = df[neg_col].astype(float).values, df[pos_col].astype(float).values
    exp0, exp1 = np.exp(l0), np.exp(l1)
    return exp1 / (exp0 + exp1)

def collect_predictions_for_plots(pred_dir: str, n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    all_labels, all_probs = [], []
    found_folds = 0

    for i in range(n_folds):
        fold_path = os.path.join(pred_dir, f"fold_{i}_predictions.csv")
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            found_folds += 1

            label_col = find_label_col(df.columns.tolist())
            labels = df[label_col].values

            prob_series = df["prob"] if "prob" in df.columns else pd.Series(logits_to_prob(df), name="prob")
            all_labels.append(labels)
            all_probs.append(prob_series.values)

    if found_folds == 0:
        raise FileNotFoundError(f"❌ No fold_X_predictions.csv files found in {pred_dir}. Checked {n_folds} folds.")
        
    return np.hstack(all_labels), np.hstack(all_probs)

# ─────────── Plotting Functions ───────────
def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='tab:blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ ROC curve saved to → {save_path}")

def plot_pr_curve(y_true, y_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision, color='tab:green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision-Recall (PR) Curve')
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ PR curve saved to → {save_path}")

def plot_probability_distribution(y_score, y_true, save_path):
    plt.figure()
    sns.histplot(x=y_score, hue=y_true, stat="density", common_norm=False, kde=True,
                 palette="viridis", bins=20, line_kws={'lw': 2})
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Overall Predicted Probability Distribution')
    plt.xlim([-0.02, 1.02])
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Probability distribution saved to → {save_path}")

# ─────────── Main Execution ───────────
def main():
    if _theme_loaded:
        configure_cns_style()
        print("🎉 Custom CNSStyle theme applied for overall plots.")

    BASE_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
    PREDS_DIR = os.path.join(BASE_DIR, "results", "clinical", "clinical_model", "clinical_preds")
    SAVE_DIR = os.path.join(BASE_DIR, "results", "final_run", "figures_overall")

    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"DEBUG: Save directory created or exists: {SAVE_DIR}")
        if not os.path.isdir(SAVE_DIR):
            raise IOError(f"Directory not created/accessible: {SAVE_DIR}")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not create/access save directory '{SAVE_DIR}': {e}")
        print("Please check directory permissions and path components.")
        sys.exit(1)

    print(f"\n📦 Collecting predictions from: {PREDS_DIR}")
    try:
        true_labels, probabilities = collect_predictions_for_plots(PREDS_DIR)
        print(f"✅ Collected {len(true_labels)} predictions across all folds.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please check PREDS_DIR path. Aborting plot generation.")
        return
    except KeyError as e:
        print(f"❌ Error in prediction file format: {e}. Please check CSV column names. Aborting plot generation.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during prediction collection: {e}. Aborting plot generation.")
        return

    print("\n📊 Generating Overall ROC Curve...")
    plot_roc_curve(true_labels, probabilities, os.path.join(SAVE_DIR, "overall_roc_curve.png"))

    print("📊 Generating Overall PR Curve...")
    plot_pr_curve(true_labels, probabilities, os.path.join(SAVE_DIR, "overall_pr_curve.png"))

    print("📊 Generating Overall Probability Distribution...")
    plot_probability_distribution(probabilities, true_labels, os.path.join(SAVE_DIR, "overall_probability_distribution.png"))

    print("\n🎉 All overall plots generated!")

if __name__ == "__main__":
    main()
