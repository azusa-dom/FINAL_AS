# ============================================
# plot_shap_interactions.py
# Generate SHAP dependence plot: ESR × HLA-B27
# ============================================

import os, shap, torch, pandas as pd, matplotlib.pyplot as plt
import torch.nn as nn

# ---------- 1. 路径 ----------
DATA_CSV  = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/results/final_run/processed_data/fold_0_val.csv"
MODEL_PTH = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/results/final_run/best_model_fold_0.pth"
OUT_DIR   = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/results/final_run/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 2. 读取特征 ----------
df = pd.read_csv(DATA_CSV)
X_full = df.drop(columns=[c for c in df.columns if c.lower() in {"label", "y", "disease"}])
print("Loaded validation matrix:", X_full.shape)     # e.g. (851, 23-1)=22

# ---------- 3. 读取 state_dict 并推断维度 ----------
state_dict = torch.load(MODEL_PTH, map_location="cpu")
in_dim  = state_dict['net.0.weight'].shape[1]    # e.g. 21
hid_dim = state_dict['net.0.weight'].shape[0]    # e.g. 64
out_dim = state_dict['net.6.weight'].shape[0]    # e.g. 2
print(f"state_dict dims → in={in_dim}, hidden={hid_dim}, out={out_dim}")

# 若 CSV 列数 > in_dim，自动裁剪（按列顺序）
X = X_full.iloc[:, :in_dim] if X_full.shape[1] > in_dim else X_full.copy()
print("Feature matrix used for SHAP:", X.shape)

# ---------- 4. 构建匹配网络 ----------
class ClinicalNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = ClinicalNet(in_dim, hid_dim, out_dim)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

# ---------- 5. 预测函数 ----------
def predict_fn(arr):
    with torch.no_grad():
        logits = model(torch.tensor(arr, dtype=torch.float32))
        if logits.shape[1] == 2:               # soft-max 二分类
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:                                  # 单输出 Sigmoid
            probs = torch.sigmoid(logits.squeeze(1))
        return probs.cpu().numpy()

# ---------- 6. SHAP 计算 ----------
background = shap.sample(X, min(100, len(X)), random_state=42)
explainer  = shap.KernelExplainer(predict_fn, background)
print("Computing SHAP values …")
shap_vals = explainer.shap_values(X, nsamples=1000)

# ---------- 7. 绘制依赖性交互图 ----------
feature_main     = "ESR"                  # 横轴
feature_interact = "HLA-B27_Positive"     # 点颜色

idx_main, idx_inter = map(X.columns.get_loc, (feature_main, feature_interact))

shap.dependence_plot(
    idx_main, shap_vals, X,
    interaction_index=idx_inter,
    show=False, alpha=0.6, dot_size=12
)

out_png = os.path.join(OUT_DIR, "shap_dependence_ESR_HLA.png")
plt.title(f"SHAP dependence: {feature_main} × {feature_interact}")
plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
print("Saved dependence plot to:", out_png)