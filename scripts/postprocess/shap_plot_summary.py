#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_shap_dca.py
生成 SHAP + 校准后 DCA 图（SCI 期刊级格式）
- 自动兼容无 Patient_ID
- 使用 strict=False 加载 state_dict，打印加载报告
- 在验证集上以 Isotonic Regression 做概率校准
2025 · Git Expert
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

# ─────────── Matplotlib SCI 期刊配置（非 Times 字体） ───────────
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

# ─────────── 导入 ClinicalNet ───────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)
try:
    from src.models import ClinicalNet
except ImportError:
    # 兜底示例网络（仅供调试）
    class ClinicalNet(torch.nn.Module):
        def __init__(self, input_dim, hidden=64, n_classes=2):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, n_classes),
            )
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
        max_display=min(25, len(feat_names)),
        show=False
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP 图已保存 → {p}")

# ─────────── DCA 曲线 ───────────
def plot_dca(y, p, out_dir):
    thresholds = np.linspace(0.01, 0.99, 100)
    n = y.size
    nb_model = [
        ((p >= t) & (y == 1)).sum() / n
        - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
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
    plt.ylim(min(nb_model) - .05, max(nb_all) + .05)
    plt.grid(alpha=.4)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "dca_curve_calibrated.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ 校准后 DCA 图已保存 → {p}")

# ─────────── 主流程 ───────────
def run_fixed(fold=1):
    # —— 固定路径配置 —— 
    model_path = (
        f"/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/"
        f"results/final_run/best_model_fold_{fold}.pth"
    )
    data_dir = (
        "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/"
        "results/final_run/processed_data"
    )
    save_dir = (
        f"/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/"
        f"results/final_run/figures_fold{fold}"
    )

    print(f"\n📦 Fold {fold}: 载入数据和模型…")
    tr_csv = os.path.join(data_dir, f"fold_{fold}_train.csv")
    va_csv = os.path.join(data_dir, f"fold_{fold}_val.csv")
    df_tr  = pd.read_csv(tr_csv)
    df_va  = pd.read_csv(va_csv)

    # 标签 & 特征
    y_va = df_va["label"].values.astype(int)
    drop_va = [c for c in ["label", "Patient_ID"] if c in df_va.columns]
    drop_tr = [c for c in ["label", "Patient_ID"] if c in df_tr.columns]
    X_va = df_va.drop(columns=drop_va)
    X_tr = df_tr.drop(columns=drop_tr)
    feat_names = X_va.columns.tolist()

    # 张量化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_va_t = torch.tensor(X_va.values, dtype=torch.float32, device=device)
    X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32, device=device)

    # 加载模型（strict=False）
    model = ClinicalNet(input_dim=X_va_t.shape[1]).to(device)
    state = torch.load(model_path, map_location=device)
    res = model.load_state_dict(state, strict=False)
    print("🔧 load_state_dict 结果 →", res)
    model.eval()

    # SHAP
    print("📊 生成 SHAP 图…")
    plot_shap(model, X_tr_t[: min(100, len(X_tr_t))], X_va_t, feat_names, save_dir)

    # DCA with calibration
    print("📈 生成 校准后 DCA 曲线…")
    with torch.no_grad():
        logits      = model(X_va_t)
        y_prob_raw  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # 校准：Isotonic Regression
    ir = IsotonicRegression(out_of_bounds='clip')
    y_prob_cal  = ir.fit_transform(y_prob_raw, y_va)

    plot_dca(y_va, y_prob_cal, save_dir)

    print(f"\n🎉 所有图表已生成 → {save_dir}")

# 🚀 执行（修改 fold=0~4）
if __name__ == "__main__":
    run_fixed(fold=1)
