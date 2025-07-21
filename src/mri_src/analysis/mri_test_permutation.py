#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mri_permutation_test_full.py

自包含 MRI 置换检验脚本：
- 提取每位受试者的聚合特征
- 计算 LOOCV AUC
- 做 n_perm 次标签置换检验，报告 p-value
- 绘制零假设分布直方图

用法示例：
  python src/mri_src/analysis/mri_test_permutation.py \
    --as-dir /Users/hydra/.../data/mri_AS \
    --healthy-dir /Users/hydra/.../data/mri_health/health1 \
    --healthy-dir /Users/hydra/.../data/mri_health/health2 \
    --n-perm 1000 \
    --batch-size 16 \
    --seed 42 \
    --device cpu
"""

import argparse
import random
import re
from pathlib import Path
import os # Import os for creating directories and checking path existence

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MRI subject-level permutation test")
    p.add_argument("--as-dir",      type=str, required=True,
                   help="AS SIJ 切片根目录")
    p.add_argument("--healthy-dir", action='append', required=True,
                   help="健康 SIJ 切片根目录，可多次指定")
    p.add_argument("--n-perm",     type=int, default=1000,
                   help="置换次数 (默认 1000)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="切片特征 batch size")
    p.add_argument("--seed",       type=int, default=42,
                   help="随机种子")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备")
    return p.parse_args()


class SliceDataset(Dataset):
    """递归读取所有图像，提取 subject_id 并打标签"""
    EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(self, as_root_dir, healthy_root_dirs, tf=None): # Changed signature
        self.samples = []
        self.as_root_dir = Path(as_root_dir)
        self.healthy_root_dirs = [Path(d) for d in healthy_root_dirs]
        self.tf = tf

        found_images = False

        # Process AS directory: Explicitly assign label 1
        if not self.as_root_dir.is_dir():
            print(f"⚠️ Warning: AS directory not found: {self.as_root_dir}", file=sys.stderr)
        else:
            print(f"[INFO] Collecting images from AS directory: {self.as_root_dir}")
            for img_path in self.as_root_dir.rglob("*"):
                if img_path.suffix.lower() in self.EXTS:
                    label = 1 # Always label 1 for AS images from this root
                    sid = img_path.parent.name # Subject ID is the immediate parent folder name
                    self.samples.append((str(img_path), label, sid))
                    found_images = True
        
        # Process Healthy directories: Explicitly assign label 0
        for healthy_dir in self.healthy_root_dirs:
            if not healthy_dir.is_dir():
                print(f"⚠️ Warning: Healthy directory not found: {healthy_dir}", file=sys.stderr)
                continue
            print(f"[INFO] Collecting images from Healthy directory: {healthy_dir}")
            for img_path in healthy_dir.rglob("*"):
                if img_path.suffix.lower() in self.EXTS:
                    label = 0 # Always label 0 for Healthy images from these roots
                    sid = img_path.parent.name # Subject ID is the immediate parent folder name
                    self.samples.append((str(img_path), label, sid))
                    found_images = True
        
        if not found_images:
            raise RuntimeError(f"No images found in the provided directories. Checked: {as_root_dir} and {healthy_root_dirs}. Please check paths and image extensions.")

        print(f"[INFO] Total slices collected: {len(self.samples)}")


    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tf: img = self.tf(img)
        return img, label, sid


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_loocv_auc(X, y):
    loo = LeaveOneOut()
    preds, trues = [], []
    
    if len(np.unique(y)) < 2: # Check if there's only one class overall
        return np.nan 

    for tr, va in loo.split(X):
        # Ensure that the training set has at least two classes to train LogisticRegression
        # And ensure at least two samples in training set for valid split
        if len(np.unique(y[tr])) < 2 or len(tr) < 2:
            continue # Skip this fold if it doesn't allow training
        
        clf = LogisticRegression(solver="liblinear", random_state=0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:,1][0]
        preds.append(p)
        trues.append(y[va][0])
    
    if len(trues) == 0:
        return np.nan # If no valid folds
    
    if len(np.unique(trues)) < 2:
        return np.nan # If only one class in combined true labels from all valid folds

    return roc_auc_score(trues, preds)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    # Pass individual AS and Healthy root directories to SliceDataset
    ds = SliceDataset(args.as_dir, args.healthy_dir, tf) # Changed how ds is initialized
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ext = FeatureExtractor().to(device).eval()
    feats, labels, sids = [], [], []
    with torch.no_grad():
        for imgs, labs, subjects in tqdm(loader, desc="FeatExt"):
            imgs = imgs.to(device)
            f = ext(imgs).cpu().numpy()
            feats.append(f)
            labels.extend(labs)
            sids.extend(subjects)
    X_slice = np.vstack(feats); y_slice = np.array(labels)

    # 聚合到受试者级
    subj_feats, subj_lab = {}, {}
    for feat, lab, sid in zip(X_slice, y_slice, sids):
        subj_feats.setdefault(sid, []).append(feat)
        subj_lab[sid] = lab # Assuming one label per subject
    
    # Get a consistent order of subject IDs and their corresponding features/labels
    # Sort by subject ID to ensure consistent permutation behavior across runs
    subj_ids_sorted = sorted(subj_feats.keys()) 
    X = np.vstack([np.mean(subj_feats[s], axis=0) for s in subj_ids_sorted])
    y = np.array([subj_lab[s] for s in subj_ids_sorted])
    
    # --- AGGREGATED SUBJECT DATA DEBUG ---
    print("\n--- Aggregated Subject Data Debug ---")
    print(f"Total subjects after aggregation: {len(y)}")
    print(f"Unique subject IDs found: {subj_ids_sorted}")
    print(f"Labels of aggregated subjects: {y.tolist()}")
    print(f"Unique classes in aggregated labels: {np.unique(y).tolist()}")
    print("------------------------------------\n")
    # --- END AGGREGATED SUBJECT DATA DEBUG ---

    # Check if there's sufficient data after aggregation
    if len(np.unique(y)) < 2: # Check unique classes *after* aggregation
        raise RuntimeError(f"Insufficient unique classes ({len(np.unique(y))}) among subjects after aggregation for permutation test. Need at least 2 classes.")
    if len(y) < 2: # Check unique subjects *after* aggregation
        raise RuntimeError(f"Insufficient unique subjects ({len(y)}) after aggregation for permutation test. Need at least 2 subjects.")

    print(f"[INFO] Subjects: {len(y)}, Pos: {y.sum()}, Neg: {len(y)-y.sum()}")

    obs_auc = compute_loocv_auc(X, y)
    
    if np.isnan(obs_auc):
        raise RuntimeError("Observed AUC could not be computed. This usually means LOOCV folds resulted in single-class training sets or insufficient validation data for AUC calculation. Check your data distribution and ensure sufficient samples per class for LOOCV.")

    # --- Determine which AUC to use for primary reporting and plot line ---
    direction_corrected_auc = obs_auc
    if obs_auc < 0.5:
        direction_corrected_auc = 1.0 - obs_auc
        print(f"\n[RESULT] Observed LOOCV AUC (raw) = {obs_auc:.3f}")
        print(f"[RESULT] Observed LOOCV AUC (Direction-Corrected) = {direction_corrected_auc:.3f} (model directionality likely reversed)")
    else:
        print(f"\n[RESULT] Observed LOOCV AUC = {obs_auc:.3f}") # This is already the direction-corrected if >= 0.5
    # --- End New ---

    # 置换检验
    perm_aucs = []
    rng = np.random.RandomState(args.seed)
    
    print(f"[INFO] Starting {args.n_perm} permutations...")
    for i in tqdm(range(args.n_perm), desc="Permutation Test"):
        y_perm = rng.permutation(y)
        perm_auc = compute_loocv_auc(X, y_perm)
        if not np.isnan(perm_auc): # Only add valid AUCs
            perm_aucs.append(perm_auc)
        # No need for intermediate prints in tqdm loop, tqdm handles progress

    perm_aucs = np.array(perm_aucs)

    if len(perm_aucs) == 0:
        raise RuntimeError("No valid permutation AUCs could be computed after shuffling labels. Permutation test aborted. This may indicate an issue with `compute_loocv_auc` or very small/imbalanced subject groups, making even shuffled folds non-computable.")

    # 1. 计算多少次置换检验的结果大于或等于观测值 (for the raw observed_auc)
    # This calculation should still use the raw obs_auc for statistical correctness
    # as the null distribution is symmetric around 0.5 for a well-behaved permutation test.
    # The p-value's interpretation of "extreme" (either very high or very low) should match.
    # For a one-sided test (e.g. is it *better* than chance), you'd use (perm_aucs >= obs_auc).sum()
    # For a two-sided test (is it *different* from chance), you'd need to consider deviations from 0.5.
    # Given your context, you care about "better than chance" or "significantly different from chance if performance is low"
    # The original p-value calculation (perm_aucs >= obs_auc) is typically for "greater than or equal to".
    # If the observed AUC is low (like 0.083), a two-sided p-value would check deviation from 0.5.
    
    count_extreme = (perm_aucs >= obs_auc).sum()
    
    # 2. 使用修正后的公式计算 p-value
    # The p-value for the raw AUC remains valid as the null distribution is symmetric.
    pval = (count_extreme + 1) / (len(perm_aucs) + 1) # Use len(perm_aucs) for N_perm

    print(f"\n[RESULT] Permutation p-value = {pval:.4f} (based on {len(perm_aucs)} valid permutations)")

    # 3. (可选) 绘制并保存零分布直方图
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 5), dpi=300)
        # 1. Histogram: discrete bars, light blue fill, gray edge
        sns.histplot(perm_aucs, kde=False, bins=12, # bins=12 for N=8 subjects (12 unique AUC values)
                     label='Permutation AUCs (Null Distribution)',
                     color='skyblue', edgecolor='gray')
        
        # 2. Plot both raw and direction-corrected AUCs
        plt.axvline(obs_auc, color='gray', linestyle='--', linewidth=2, label=f'Raw AUC = {obs_auc:.3f}')
        # Only plot the second line if it's different (i.e., if flipping happened)
        if obs_auc != direction_corrected_auc: 
            plt.axvline(direction_corrected_auc, color='red', linestyle='--', linewidth=2, label=f'Direction-Corrected AUC = {direction_corrected_auc:.3f}')
        
        # 3. Add p-value and N annotation
        # Adjust position if needed based on the actual plot
        plt.annotate(f'N = {len(y)}, {len(perm_aucs)} permutations, p = {pval:.2f}',
                     xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8)) # Add background box for clarity
        
        plt.title('Permutation Test Null Distribution of AUCs')
        plt.xlabel('AUC Score')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right') # Place legend in upper right
        plt.grid(alpha=0.4) # Keep grid for clarity
        
        # Ensure output directory exists
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        output_dir = os.path.join(project_root, "results", "mri_analysis")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "permutation_test_auc_distribution.png") # Keep PNG for now as it was requested, but SVG is better for papers
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Permutation distribution plot saved to: {save_path}")

    except ImportError:
        print("⚠️ Matplotlib/Seaborn not found, skipping plot generation.")

if __name__ == "__main__":
    main()