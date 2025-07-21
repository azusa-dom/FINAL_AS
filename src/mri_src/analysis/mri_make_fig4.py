#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mri_make_fig4.py

生成 Figure4: ROC / PR / Calibration / Decision Curve (L2O-CV)
要求输入 l2o_predictions.csv （含列 subject_id, y_true, prob_raw, logit_raw）
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import torch # For sigmoid and optimization
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.optimize import minimize_scalar # For optimizing T

# --- Import Custom CNSStyle Theme ---
import sys
from pathlib import Path

# Assuming project root is 3 levels up from this script (src/mri_src/analysis/)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src')) # Add 'src' to path to find visualization.theme

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into mri_make_fig4.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. Plots will use default Matplotlib style.")
    # Fallback Matplotlib configuration if theme fails to load
    plt.rcParams.update({
        "figure.dpi":      300, "savefig.dpi":     300, "figure.figsize":  (8, 6),
        "font.family":     "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":       12, "axes.titlesize":  16, "axes.labelsize":  14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_ece(y_true, probs, n_bins=5): # Reduced bins for small N
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i+1])
        if idx.sum() == 0: 
            continue
        conf = probs[idx].mean()
        acc = y_true[idx].mean()
        ece += np.abs(acc - conf) * idx.mean()
    return ece

def decision_curve(y_true, probs, thresholds):
    # net benefit = TP/n - FP/n * (pt/(1-pt))
    y_true = np.array(y_true)
    probs = np.array(probs)
    n = len(y_true)
    
    # Calculate prevalence for treat-all line
    prevalence = y_true.mean()

    model_nb = []
    for pt in thresholds:
        # Avoid division by zero for (1-pt) at pt=1.0
        if pt == 1.0:
            nb = 0 # Net benefit at threshold 1.0 is 0
        else:
            pred_pos = probs >= pt
            tp = ((pred_pos) & (y_true==1)).sum()
            fp = ((pred_pos) & (y_true==0)).sum()
            nb = tp/n - fp/n * (pt/(1-pt))
        model_nb.append(nb)
        
    # Treat-all and Treat-none are fixed lines
    nb_treat_all = prevalence - (1-prevalence) * thresholds / (1-thresholds + 1e-8) # Add epsilon to prevent div by zero
    nb_treat_none = np.zeros_like(thresholds)
    
    return np.array(model_nb), np.array(nb_treat_all), np.array(nb_treat_none)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="Path to L2O predictions CSV (with y_true, prob_raw, logit_raw).")
    ap.add_argument("--out-dir", required=True, help="Output directory for Figure 4 plots.")
    args = ap.parse_args()
    
    # Apply custom theme if loaded
    if _theme_loaded:
        configure_cns_style()

    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pred_csv)
    if not {"subject_id", "y_true", "prob_raw", "logit_raw"}.issubset(df.columns):
        raise ValueError("CSV needs columns: subject_id, y_true, prob_raw, logit_raw")
    
    y = df.y_true.values.astype(int)
    p_raw = df.prob_raw.values.astype(float)
    logits_raw = df.logit_raw.values.astype(float)

    # --- Step 1: Direction Correction on Logits ---
    # First calculate raw_auc from p_raw to determine direction
    raw_auc_initial = roc_auc_score(y, p_raw)

    if raw_auc_initial < 0.5:
        logger.info(f"Initial raw AUROC ({raw_auc_initial:.3f}) < 0.5. Flipping logits and probabilities for direction correction.")
        logits_corrected = -logits_raw # Flip logits
        p_corrected = 1 - p_raw     # Flip probabilities as well
    else:
        logger.info(f"Initial raw AUROC ({raw_auc_initial:.3f}) >= 0.5. No direction correction needed.")
        logits_corrected = logits_raw
        p_corrected = p_raw
    
    # Store the final direction-corrected probability for general use in plots
    p_for_plots = p_corrected

    # --- Step 2: Temperature Scaling (on corrected logits) ---
    logger.info("Optimizing temperature for probability calibration...")
    
    # Convert to PyTorch tensors for optimization
    logits_tensor = torch.tensor(logits_corrected, dtype=torch.float32)
    labels_tensor = torch.tensor(y, dtype=torch.float32)

    # Define the NLL loss function for optimization
    def nll_loss_func(temp_val):
        if temp_val <= 0: return float('inf') # Temperature must be positive
        calibrated_logits = logits_tensor / temp_val
        loss = torch.nn.BCEWithLogitsLoss()(calibrated_logits, labels_tensor) # BCEWithLogitsLoss for binary
        return loss.item()

    # Optimize for temperature (using scalar minimization)
    # Bounds for temperature search: e.g., (0.1, 10.0)
    res_opt = minimize_scalar(nll_loss_func, bounds=(0.1, 10.0), method='bounded')
    optimal_T = res_opt.x
    logger.info(f"Optimal temperature (T) found: {optimal_T:.3f}")

    # Calculate calibrated probabilities using optimal_T
    logits_calibrated_tensor = logits_tensor / optimal_T
    p_calibrated = torch.sigmoid(logits_calibrated_tensor).numpy()

    # --- Step 3: Calculate Metrics for Plots ---
    # Metrics based on direction-corrected and calibrated probabilities
    # (or just direction-corrected if calibration is deemed separate)
    
    # For ROC/PR, it's generally best to use probabilities that have been direction-corrected but *before* calibration,
    # because calibration primarily affects absolute probabilities, not necessarily rank order (which AUC relies on).
    # However, if your final model *includes* calibration, then using p_calibrated is fine.
    # The prompt implies using the final calibrated version for these plots.
    
    # Let's use p_calibrated for all plots as it's the "best" representation of model output.
    
    # ---- ROC / PR ----
    fpr, tpr, _ = roc_curve(y, p_calibrated)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y, p_calibrated)
    pr_auc = auc(rec, prec)

    # ---- Calibration ----
    ece_raw = compute_ece(y, p_for_plots) # Use p_for_plots (direction-corrected but uncalibrated) for raw ECE
    ece_cal = compute_ece(y, p_calibrated)

    # ---- Decision curve ----
    # Restrict thresholds to 0.1-0.6 as requested
    dca_thresholds = np.linspace(0.1, 0.6, 60) # 60 points for smoother curve in restricted range
    nb_model, nb_all, nb_none = decision_curve(y, p_calibrated, dca_thresholds)

    # --- Final Plotting ---
    plt.figure(figsize=(18, 6)) # Wider figure to accommodate 3 plots side-by-side with better spacing
    plt.style.use('seaborn-v0_8-whitegrid') # Ensure seaborn style if CNS style isn't fully integrated here

    # Subplot 1: ROC Curve
    ax1 = plt.subplot(1,3,1) # Changed from 1,2,1
    ax1.plot(fpr,tpr,label=f"AUROC = {roc_auc:.3f}", color='blue', lw=2)
    ax1.plot([0,1],[0,1],'k--',linewidth=1)
    ax1.set_xlabel("False Positive Rate", fontsize=12); ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("A. Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold'); 
    ax1.legend(loc='lower right', frameon=False, fontsize=10)
    ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])
    ax1.tick_params(axis='both', labelsize=10)
    ax1.grid(alpha=0.4, linestyle=':')

    # Subplot 2: Precision-Recall Curve
    ax2 = plt.subplot(1,3,2) # Changed from 1,2,2
    ax2.plot(rec,prec,label=f"AUPRC = {pr_auc:.3f}", color='green', lw=2)
    ax2.set_xlabel("Recall", fontsize=12); ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("B. Precision-Recall (PR) Curve", fontsize=14, fontweight='bold'); 
    ax2.legend(loc='lower left', frameon=False, fontsize=10)
    ax2.set_xlim([-0.02, 1.02]); ax2.set_ylim([-0.02, 1.02])
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(alpha=0.4, linestyle=':')

    # Subplot 3: Calibration Curve
    ax3 = plt.subplot(1,3,3) # Added new subplot for Calibration
    # Use smaller markers and thinner lines for clarity
    ax3.plot([0,1], [0,1], 'k--', linewidth=1, label='Perfectly calibrated')
    
    # Ensure raw/calibrated dots are distinct and follow typical journal style
    # Use the CNSStyle palette if loaded, or default nice colors.
    ax3.plot(np.mean(y_true_for_calib_plot(y, p_for_plots), axis=1), # Mean predicted in bin
             np.mean(y_true_for_calib_plot(y, p_for_plots, mode='observed'), axis=1), # Observed fraction in bin
             'o-', color='orange', markersize=6, lw=1.5, label=f"Uncalibrated (ECE={ece_raw:.3f})")

    ax3.plot(np.mean(y_true_for_calib_plot(y, p_calibrated), axis=1), # Mean predicted in bin
             np.mean(y_true_for_calib_plot(y, p_calibrated, mode='observed'), axis=1), # Observed fraction in bin
             '^-', color='blue', markersize=6, lw=1.5, label=f"Calibrated (ECE={ece_cal:.3f})")
    
    ax3.set_xlabel("Mean predicted probability", fontsize=12); ax3.set_ylabel("Observed fraction of positives", fontsize=12)
    ax3.set_title(f"C. Reliability Diagram (Calibration)", fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', frameon=False, fontsize=10)
    ax3.set_xlim([-0.02, 1.02]); ax3.set_ylim([-0.02, 1.02])
    ax3.tick_params(axis='both', labelsize=10)
    ax3.grid(alpha=0.4, linestyle=':')

    plt.tight_layout(rect=[0,0,1,0.95]) # Adjust rect for overall suptitle
    plt.savefig(out_dir_path/"figure4_abc_roc_pr_cal.svg", dpi=300, format='svg') # Save as SVG
    plt.close()

    # Subplot 4: Decision Curve Analysis (Separate Figure)
    plt.figure(figsize=(6, 5)) # Adjusted size for DCA
    ax4 = plt.gca()
    ax4.plot(dca_thresholds, nb_model, label="Calibrated Model", lw=2, color='blue')
    ax4.plot(dca_thresholds, nb_all, 'r--', label="Treat-All", lw=1.5)
    ax4.plot(dca_thresholds, nb_none, 'k:', label="Treat-None", lw=1.5)
    
    ax4.set_xlabel("Risk Threshold", fontsize=12)
    ax4.set_ylabel("Net Benefit", fontsize=12)
    ax4.set_title("D. Decision Curve Analysis (DCA)", fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', frameon=False, fontsize=10)
    ax4.grid(alpha=0.4, linestyle=':')
    ax4.set_xlim([0.1, 0.6]) # Restrict threshold range as requested
    # Auto-adjust y-lim or set reasonable fixed range if known
    # ax4.set_ylim([-0.1, 0.2]) # Example fixed y-limit

    plt.tight_layout()
    plt.savefig(out_dir_path/"figure4d_dca.svg", dpi=300, format='svg') # Save as SVG
    plt.close()

    logger.info(f"Generated ROC/PR/Calibration plots to {out_dir_path/'figure4_abc_roc_pr_cal.svg'}")
    logger.info(f"Generated DCA plot to {out_dir_path/'figure4d_dca.svg'}")
    logger.info(f"Final AUROC={roc_auc:.3f}, AUPRC={pr_auc:.3f}")
    logger.info(f"ECE raw={ece_raw:.3f}, ECE cal={ece_cal:.3f}")


# Helper for calibration plot (replaces complex loop with pre-binned data)
def y_true_for_calib_plot(y_true, probs, n_bins=5, mode='predicted'):
    """
    Helper to bin probabilities and return mean predicted or observed fractions per bin.
    Used for calibration curve plotting (reliability diagrams).
    """
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    
    # Store tuples of (mean_predicted_in_bin, observed_fraction_in_bin)
    binned_data = []

    for i in range(n_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        
        # Select samples that fall into this bin
        in_bin = (probs >= lower_bound) & (probs < upper_bound)
        
        if np.sum(in_bin) == 0:
            # If bin is empty, matplotlib's calibration_curve skips it.
            # We can also skip it here, or plot empty points.
            continue
        
        mean_predicted = probs[in_bin].mean()
        observed_fraction = y_true[in_bin].mean()
        
        binned_data.append((mean_predicted, observed_fraction))
        
    if not binned_data:
        # Return empty arrays if no bins have data
        return np.array([]), np.array([])
        
    # Unzip the list of tuples
    mean_predicted_in_bins, observed_fractions_in_bins = zip(*binned_data)
    
    if mode == 'predicted':
        return np.array(mean_predicted_in_bins)
    elif mode == 'observed':
        return np.array(observed_fractions_in_bins)
    else:
        raise ValueError("Mode must be 'predicted' or 'observed'")


if __name__ == "__main__":
    main()