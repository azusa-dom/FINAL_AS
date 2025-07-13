#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_shap_dca.py
生成 SHAP + 校准后 DCA 图（SCI 期刊级格式）
- FIXED: Used the correct ClinicalNet architecture from the paper to resolve shape mismatch errors.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

# ─────────── Matplotlib SCI 期刊配置 ───────────
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

# ─────────── 正确的 ClinicalNet 模型定义 (根据论文) ───────────
class ClinicalNet(nn.Module):
    def __init__(self, input_dim=40, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, n_classes) # Outputting 2 logits for classification
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)

# ─────────── SHAP 绘图 ───────────
def plot_shap(model, background, x_test, feat_names, out_dir, pos_cls=1):
    explainer = shap.DeepExplainer(model, background)
    shap_vals = explainer.shap_values(x_test)[pos_cls]
    df_test   = pd.DataFrame(x_test.cpu().numpy(), columns=feat_names)

    plt.figure()
    shap.summary_plot(
        shap_vals, df_test,
        plot_type="dot",
        max_display=min(15, len(feat_names)), # Display up to 15 features
        show=False
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP plot saved to → {p}")

# ─────────── DCA 曲线 ───────────
def plot_dca(y, p, out_dir):
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
    plt.plot(thresholds, nb_model, label="Calibrated Model", lw=2)
    plt.plot(thresholds, nb_all,  "--", label="Treat-All")
    plt.plot(thresholds, nb_none, ":", label="Treat-None")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (Calibrated)")
    plt.ylim(np.min(nb_model) - .05, np.max(nb_all) + .05)
    plt.grid(alpha=.4)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "dca_curve_calibrated.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ Calibrated DCA plot saved to → {p}")

# ─────────── 主流程 ───────────
def run_analysis(model_path, data_dir, save_dir, fold=0):
    print(f"\n📦 Fold {fold}: Loading data and model...")
    tr_csv = os.path.join(data_dir, f"fold_{fold}_train.csv")
    va_csv = os.path.join(data_dir, f"fold_{fold}_val.csv")
    df_tr  = pd.read_csv(tr_csv)
    df_va  = pd.read_csv(va_csv)

    # Labels & Features
    y_va = df_va["label"].values.astype(int)
    drop_cols = ["label", "Patient_ID", "Patient ID"]
    X_va = df_va.drop(columns=[c for c in drop_cols if c in df_va.columns])
    X_tr = df_tr.drop(columns=[c for c in drop_cols if c in df_tr.columns])
    feat_names = X_va.columns.tolist()

    # To Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_va_t = torch.tensor(X_va.values, dtype=torch.float32, device=device)
    X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32, device=device)

    # Load Model (with the CORRECT architecture)
    model = ClinicalNet(input_dim=X_va_t.shape[1]).to(device)
    try:
        state = torch.load(model_path, map_location=device)
        # Using strict=True is better now that we have the correct model
        model.load_state_dict(state, strict=True)
        print("🔧 Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at {model_path}")
        return
    except RuntimeError as e:
        print(f"❌ ERROR: Mismatch while loading model weights. Ensure the architecture is correct. Details: {e}")
        return
        
    model.eval()

    # SHAP analysis
    print("📊 Generating SHAP plot...")
    background_data = X_tr_t[: min(100, len(X_tr_t))]
    plot_shap(model, background_data, X_va_t, feat_names, save_dir)

    # DCA with calibration
    print("📈 Generating Calibrated DCA curve...")
    with torch.no_grad():
        logits      = model(X_va_t)
        y_prob_raw  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    ir = IsotonicRegression(out_of_bounds='clip')
    y_prob_cal  = ir.fit_transform(y_prob_raw, y_va)

    plot_dca(y_va, y_prob_cal, save_dir)

    print(f"\n🎉 All plots for fold {fold} generated in → {save_dir}")

if __name__ == "__main__":
    FOLD_TO_ANALYZE = 0 
    
    # --- IMPORTANT: Please verify these paths are correct for your system ---
    BASE_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
    MODEL_PATH = os.path.join(BASE_DIR, "results", "final_run", f"best_model_fold_{FOLD_TO_ANALYZE}.pth")
    DATA_DIR = os.path.join(BASE_DIR, "results", "final_run", "processed_data")
    SAVE_DIR = os.path.join(BASE_DIR, "results", "final_run", f"figures_fold_{FOLD_TO_ANALYZE}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_DIR):
         print(f"⚠️ WARNING: Default paths not found. \nModel: {MODEL_PATH}\nData: {DATA_DIR}")
    else:
        run_analysis(
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR,
            fold=FOLD_TO_ANALYZE
        )