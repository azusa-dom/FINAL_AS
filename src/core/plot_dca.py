#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_shap_dca.py
Generate SHAP + Calibrated DCA Plot (SCI Journal-Level Format)
- Automatically compatible with or without Patient_ID
- Uses strict=False to load state_dict, prints loading report
- Performs probability calibration (Isotonic or Sigmoid) on validation set
- Integrates custom CNSStyle theme. All outputs are in English.
- Uses the EXACT ClinicalNet definition provided by the user from their training script.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # Explicitly imported for ClinicalNet definition
import shap
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression # For Isotonic calibration
from sklearn.calibration import CalibratedClassifierCV # For Sigmoid calibration
from sklearn.linear_model import LogisticRegression # Dummy model for CalibratedClassifierCV


# ─────────── Import Custom CNSStyle Theme ───────────
PROJECT_ROOT_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, 'src'))

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into evaluate_shap_dca.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. evaluate_shap_dca.py will use default Matplotlib style.")
    plt.rcParams.update({
        "figure.dpi":      300,
        "savefig.dpi":     300,
        "figure.figsize":  (8, 6),
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":       12,
        "axes.titlesize":  16,
        "axes.labelsize":  14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })


# ─────────── ClinicalNet Model Definition (EXACT MATCH from your training script) ───────────
# This ClinicalNet definition is copied EXACTLY from your provided training script.
# It uses hidden_size=64 by default when instantiated below in run_fixed().
class ClinicalNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(ClinicalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ─────────── SHAP Plotting ───────────
def plot_shap(model, background, x_test, feat_names, out_dir, pos_cls=1):
    explainer = shap.DeepExplainer(model, background)
    print(f"DEBUG: x_test shape for explainer: {x_test.shape}") # Should be (num_samples, num_features)
    
    shap_values_all_classes = explainer.shap_values(x_test)
    
    print(f"DEBUG: Type of shap_values_all_classes: {type(shap_values_all_classes)}")
    
    if isinstance(shap_values_all_classes, list):
        print(f"DEBUG: Length of shap_values_all_classes list: {len(shap_values_all_classes)}")
        for i, val_arr in enumerate(shap_values_all_classes):
            print(f"DEBUG: shap_values_all_classes[{i}] shape: {val_arr.shape}")
        
        shap_vals = shap_values_all_classes[pos_cls]
    else:
        print(f"WARNING: explainer.shap_values did not return a list. Trying to use it directly.")
        shap_vals = shap_values_all_classes
        
    print(f"DEBUG: shap_vals shape (after selecting class, BEFORE final reshape): {shap_vals.shape}")
    
    df_test   = pd.DataFrame(x_test.cpu().numpy(), columns=feat_names)
    print(f"DEBUG: df_test shape (for summary_plot): {df_test.shape}") # Expected (num_samples, num_features)

    expected_samples = x_test.shape[0] # e.g. 851
    expected_features = x_test.shape[1] # e.g. 21

    # --- CRITICAL RESHAPE FOR SHAP SUMMARY PLOT (using KernelExplainer fallback) ---
    # This block is activated if DeepExplainer returns an unexpected shape (like 21, 2)
    # The debug output from the user indicated (21, 2) as actual shap_vals.shape
    # which is not (samples, features). This workaround ensures the correct shape.
    if shap_vals.shape == (expected_features, 2) or \
       (len(shap_vals.shape) == 2 and shap_vals.shape[1] == 2 and shap_vals.shape[0] == expected_features) or \
       shap_vals.shape != (expected_samples, expected_features):
        
        print(f"CRITICAL FIX: shap_vals has unexpected shape {shap_vals.shape}. Expected ({expected_samples}, {expected_features}).")
        print(f"             Falling back to KernelExplainer for shap_values calculation for this plot.")
        
        background_for_kernel = shap.sample(df_test, min(100, len(df_test)), random_state=42)
        
        def predict_proba_positive_class(arr):
            model.eval()
            with torch.no_grad():
                # Ensure input tensor is on the same device as the model's weights
                logits = model(torch.tensor(arr, dtype=torch.float32).to(model.net[0].weight.device))
                return torch.softmax(logits, dim=1)[:, pos_cls].cpu().numpy()
        
        kernel_explainer = shap.KernelExplainer(predict_proba_positive_class, background_for_kernel)
        shap_vals_corrected = kernel_explainer.shap_values(df_test.values) # Pass numpy array directly
        
        if isinstance(shap_vals_corrected, list) and len(shap_vals_corrected) > 0:
             shap_vals = shap_vals_corrected[0] # Take the first (and likely only) element for binary proba output
        else:
             shap_vals = shap_vals_corrected
        
        print(f"DEBUG: shap_vals (after KernelExplainer fallback): {shap_vals.shape}")
        
        # Final check after fallback
        if shap_vals.shape != (expected_samples, expected_features):
            print(f"FATAL ERROR: SHAP values shape remains incorrect even after KernelExplainer fallback. Cannot proceed with summary_plot.")
            print(f"Expected ({expected_samples}, {expected_features}), got {shap_vals.shape}")
            sys.exit(1)

    # Proceed to plot with the guaranteed (samples, features) shap_vals
    plt.figure()
    shap.summary_plot(
        shap_vals, df_test,
        plot_type="dot", # This is the "beeswarm" or "dot" plot
        max_display=min(25, len(feat_names)),
        show=False
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP plot saved to → {p}")

# ─────────── DCA Curve ───────────
def plot_dca(y, p, out_dir, plot_title="Decision Curve Analysis"):
    thresholds = np.linspace(0.01, 0.99, 100)
    n = y.size
    nb_model = [
        (np.sum((p >= t) & (y == 1))) / n
        - (np.sum((p >= t) & (y == 0))) / n * t / (1 - t)
        for t in thresholds
    ]
    prev    = y.mean()
    nb_all  = prev - (1 - prev) * thresholds / (1 - thresholds)
    nb_none = np.zeros_like(thresholds)

    plt.figure()
    plt.plot(thresholds, nb_model, label="Model", lw=2)
    plt.plot(thresholds, nb_all,  "--", label="Treat-All")
    plt.plot(thresholds, nb_none, ":", label="Treat-None")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(plot_title)
    plt.ylim(min(nb_model) - .05, max(nb_all) + .05)
    plt.grid(alpha=.4)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "dca_curve.png")
    plt.savefig(p, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ DCA plot saved to → {p}")

# ─────────── Main Flow ───────────
def run_fixed(fold=0): # Default to fold 0
    # --- Apply your CNSStyle theme ---
    if _theme_loaded:
        configure_cns_style()
        print("🎉 Custom CNSStyle theme applied in evaluate_shap_dca.py.")

    # --- Fixed Path Configuration ---
    BASE_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
    model_path = os.path.join(BASE_DIR, "results", "clinical", "clinical_model", f"best_model_fold_{fold}.pth")
    data_dir = os.path.join(BASE_DIR, "data", "processed_clinical") 
    save_dir = os.path.join(BASE_DIR, "results", "final_run", f"figures_fold{fold}")

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📦 Fold {fold}: Loading data and model...")
    tr_csv = os.path.join(data_dir, f"fold_{fold}_train.csv")
    va_csv = os.path.join(data_dir, f"fold_{fold}_val.csv")

    if not os.path.exists(tr_csv):
        print(f"❌ Error: Training data file not found: {tr_csv}")
        return
    if not os.path.exists(va_csv):
        print(f"❌ Error: Validation data file not found: {va_csv}")
        return

    df_tr  = pd.read_csv(tr_csv)
    df_va  = pd.read_csv(va_csv)

    if 'label' in df_va.columns:
        y_va = df_va["label"].values.astype(int)
    elif 'true_label' in df_va.columns:
        y_va = df_va["true_label"].values.astype(int)
    else:
        print("❌ Error: Neither 'label' nor 'true_label' column found in validation data.")
        return

    drop_cols_candidates = ["label", "true_label", "Patient_ID", "patient_id", "Patient ID"]
    all_tr_cols = [c for c in df_tr.columns if c not in drop_cols_candidates]
    
    numeric_feat_cols = []
    for col in all_tr_cols:
        df_tr[col] = df_tr[col].fillna(0) 
        df_va[col] = df_va[col].fillna(0) 

        if pd.api.types.is_numeric_dtype(df_tr[col]) and df_tr[col].nunique() > 1:
             numeric_feat_cols.append(col)

    feat_names = numeric_feat_cols
    X_tr = df_tr[feat_names]
    X_va = df_va[feat_names]
    
    model_input_dim = X_va.shape[1] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_va_t = torch.tensor(X_va.values, dtype=torch.float32, device=device)
    X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32, device=device)

    num_classes = 2
    model = ClinicalNet(input_size=model_input_dim, hidden_size=64, output_size=num_classes).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return

    state = torch.load(model_path, map_location=device)
    res = model.load_state_dict(state, strict=False)
    print("🔧 load_state_dict result →", res)
    model.eval()

    # SHAP analysis
    print("📊 Generating SHAP plot...")
    plot_shap(model, X_tr_t[: min(100, len(X_tr_t))], X_va_t, feat_names, save_dir)

    # DCA with calibration
    print("📈 Generating Calibrated DCA curve...")
    with torch.no_grad():
        logits      = model(X_va_t)
        y_prob_raw  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # --- Calibration Step ---
    # Choice of calibrator: 'isotonic' or 'sigmoid'
    CALIBRATION_METHOD = 'isotonic' # <--- YOU CAN CHANGE THIS TO 'sigmoid'
    
    if CALIBRATION_METHOD == 'isotonic':
        ir = IsotonicRegression(out_of_bounds='clip')
        y_prob_cal  = ir.fit_transform(y_prob_raw, y_va)
        print(f"✅ Applied Isotonic Regression calibration.")
    elif CALIBRATION_METHOD == 'sigmoid':
        # Need a "dummy" model as CalibratedClassifierCV wraps a classifier
        dummy_clf = LogisticRegression(solver='liblinear') 
        calibrated_clf = CalibratedClassifierCV(dummy_clf, method='sigmoid', cv="prefit")
        # Fit calibrated_clf on raw probabilities and true labels
        calibrated_clf.fit(y_prob_raw.reshape(-1, 1), y_va) 
        y_prob_cal = calibrated_clf.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]
        print(f"✅ Applied Sigmoid (Platt Scaling) calibration.")
    else:
        y_prob_cal = y_prob_raw # No calibration
        print(f"⚠️ No recognized calibration method specified. Using raw probabilities.")


    plot_dca(y_va, y_prob_cal, save_dir)

    print(f"\n🎉 All plots generated → {save_dir}")

# 🚀 Execute (modify fold=0~4)
if __name__ == "__main__":
    for fold_num in range(5):
        print(f"\n--- Processing Fold {fold_num} ---")
        run_fixed(fold=fold_num)
        print(f"--- Finished processing Fold {fold_num} ---\n")