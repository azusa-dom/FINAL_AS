#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mri_subject_level_auc.py

针对每位受试者聚合切片特征，做受试者级 LOOCV/KFold，计算 AUC & 95% CI。

用法示例：
    python src/mri_src/analysis/mri_subject_level_auc.py \
      --as-dir /Users/hydra/.../data/mri_AS \
      --healthy-dir /Users/hydra/.../data/mri_health/health1 \
      --healthy-dir /Users/hydra/.../data/mri_health/health2 \
      --n-splits 8 \
      --n-bootstrap 2000 \
      --batch-size 16 \
      --seed 42 \
      --device cpu
"""


import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd # For saving predictions to CSV
from PIL import Image # Ensure PIL.Image is explicitly imported for DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
# Removed scipy.stats.bootstrap as we are now doing manual percentile bootstrap
from itertools import product


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--as-dir",      type=str, required=True,
                   help="AS SIJ 切片根目录")
    p.add_argument("--healthy-dir", action='append', required=True,
                   help="健康 SIJ 切片根目录，可多次指定")
    p.add_argument("--n-splits",   type=int, default=8, # Changed default to 8 for L2O context
                   help="折数；对于 L2O 请保持为总受试者数 (如 8)") # Clarified help text
    p.add_argument("--n-bootstrap",type=int, default=2000,
                   help="Bootstrap 次数")
    p.add_argument("--batch-size", type=int, default=16,
                   help="batch size")
    p.add_argument("--seed",       type=int, default=42,
                   help="随机种子")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # Added argument for prediction output path
    p.add_argument("--pred-out-csv", type=str, default="results/mri_analysis/l2o_predictions.csv",
                   help="Path to save L2O predictions (y_true, prob_raw, logit_raw).")
    return p.parse_args()


class SliceDataset(Dataset):
    """递归读取所有图像，提取 subject_id 并打标签"""
    EXTS = {".png",".jpg",".jpeg"}
    
    def __init__(self, as_root_dir, healthy_root_dirs, tf=None):
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
    def __getitem__(self,i):
        # Re-import Image here to ensure it's available in DataLoader worker processes
        from PIL import Image 
        p,l,s = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.tf: img = self.tf(img)
        return img, l, s


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self,x):
        f = self.backbone(x)
        return f.view(f.size(0),-1)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate_model_performance(Xs, ys, n_bootstrap, seed):
    """
    Performs custom Leave-Two-Out (L2O: 1 Positive + 1 Negative subject) cross-validation,
    collects predictions, and calculates direction-corrected AUC with Bootstrap confidence interval.
    
    Args:
        Xs (np.array): Subject-level features.
        ys (np.array): Subject-level labels.
        n_bootstrap (int): Number of bootstrap resamples.
        seed (int): Random seed.
        
    Returns:
        tuple: (auc_used, ci_low, ci_high, n_skipped_folds, final_y_true, final_y_score, final_y_logits)
    """
    oof_subject_results = [] # List of (original_subject_index, predicted_prob, true_label, predicted_logit)
    n_skipped_folds = 0

    # Separate subject indices by class
    pos_indices = np.where(ys == 1)[0]
    neg_indices = np.where(ys == 0)[0]

    # Check for sufficient subjects in each class for L2O
    if len(pos_indices) < 1 or len(neg_indices) < 1:
        raise RuntimeError(f"Cannot perform L2O: Need at least 1 positive and 1 negative subject. Found {len(pos_indices)} positive, {len(neg_indices)} negative.")
    
    # Generate all combinations of (1 Positive subject, 1 Negative subject) for validation
    l2o_folds = list(product(pos_indices, neg_indices))
    total_folds = len(l2o_folds)
    print(f"[INFO] Using Custom Leave-Two-Out (1 Positive + 1 Negative) Cross-Validation with {total_folds} folds.")


    for i, (val_pos_idx, val_neg_idx) in enumerate(l2o_folds, 1):
        val_indices = np.array([val_pos_idx, val_neg_idx])
        train_indices = np.setdiff1d(np.arange(len(ys)), val_indices) # All other subjects for training

        train_labels = ys[train_indices]
        val_labels = ys[val_indices]

        train_pos, train_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
        val_pos, val_neg = val_labels.sum(), len(val_labels) - val_labels.sum()

        print(f"  Fold {i}/{total_folds}: Train ({train_pos} pos, {train_neg} neg), Val ({val_pos} pos, {val_neg} neg)")

        # Skip fold if train set is single-class (val set is guaranteed to have 1 pos, 1 neg)
        if len(np.unique(train_labels)) < 2:
            print(f"  Fold {i}/{total_folds}: Skip (single-class train set: {np.unique(train_labels)}).")
            n_skipped_folds += 1
            continue

        # Use LogisticRegression with specified parameters
        clf = LogisticRegression(
            class_weight='balanced', # Balance class weights
            solver='liblinear',      # Efficient for small datasets
            C=1e4,                   # Regularization inverse strength (high C means less regularization)
            max_iter=1000,           # Max iterations for solver
            random_state=seed        # Fixed random state for reproducibility
        )
        clf.fit(Xs[train_indices], train_labels)
        
        # Get raw logits (decision_function) and probabilities (predict_proba)
        # decision_function gives logits for binary classification (one value per sample)
        val_logits = clf.decision_function(Xs[val_indices])
        probs = clf.predict_proba(Xs[val_indices])[:, 1]
        
        # Store results with original subject indices
        oof_subject_results.append((val_pos_idx, probs[0], val_labels[0], val_logits[0])) # Pos subject result
        oof_subject_results.append((val_neg_idx, probs[1], val_labels[1], val_logits[1])) # Neg subject result
    
    # Now, aggregate to get ONE prediction per subject (e.g., mean of multiple OOF predictions)
    subject_agg_preds = {} # {original_subject_index: {'true_label': label, 'probs': [], 'logits': []}}
    for subj_idx, prob, true_label, logit in oof_subject_results:
        subject_agg_preds.setdefault(subj_idx, {'true_label': true_label, 'probs': [], 'logits': []})
        subject_agg_preds[subj_idx]['probs'].append(prob)
        subject_agg_preds[subj_idx]['logits'].append(logit)

    final_y_true = []
    final_y_score = [] # Raw probabilities
    final_y_logits = [] # Raw logits
    
    # Sort by subject index to ensure consistent order for bootstrapping
    for subj_idx in sorted(subject_agg_preds.keys()):
        final_y_true.append(subject_agg_preds[subj_idx]['true_label'])
        final_y_score.append(np.mean(subject_agg_preds[subj_idx]['probs'])) # Average predictions if multiple
        final_y_logits.append(np.mean(subject_agg_preds[subj_idx]['logits'])) # Average logits if multiple

    final_y_true = np.array(final_y_true)
    final_y_score = np.array(final_y_score)
    final_y_logits = np.array(final_y_logits)

    if len(final_y_true) == 0 or len(np.unique(final_y_true)) < 2:
        print(f"Warning: No valid predictions collected or only single class in combined predictions. Cannot compute AUC. Skipped {n_skipped_folds} folds.")
        return np.nan, np.nan, np.nan, n_skipped_folds, final_y_true, final_y_score, final_y_logits

    # Calculate raw and direction-corrected AUC
    raw_auc = roc_auc_score(final_y_true, final_y_score)
    flipped_auc = 1 - raw_auc
    auc_used = max(raw_auc, flipped_auc) # Direction corrected AUC

    # --- Manual Percentile Bootstrapping for CI ---
    boot_aucs = []
    rng_bootstrap = np.random.default_rng(seed)
    num_subjects_for_bootstrap = len(final_y_true)
    
    if num_subjects_for_bootstrap == 0:
        ci_low, ci_high = np.nan, np.nan
    else:
        for _ in range(n_bootstrap):
            # Resample subject indices with replacement
            indices = rng_bootstrap.choice(num_subjects_for_bootstrap, num_subjects_for_bootstrap, replace=True)
            y_boot = final_y_true[indices]
            prob_boot = final_y_score[indices]
            
            # Check for single-class resampled bootstrap sample
            if len(np.unique(y_boot)) < 2:
                continue # Skip this bootstrap sample if AUC cannot be computed
            
            boot_aucs.append(roc_auc_score(y_boot, prob_boot))
        
        if len(boot_aucs) > 0:
            ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
        else:
            ci_low, ci_high = np.nan, np.nan
    # --- End Manual Percentile Bootstrapping ---

    return auc_used, ci_low, ci_high, n_skipped_folds, final_y_true, final_y_score, final_y_logits


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # transforms
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    ds = SliceDataset(args.as_dir, args.healthy_dir, tf) 
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False, num_workers=4, pin_memory=True)

    # extract slice-level features
    ext = FeatureExtractor().to(device).eval()
    feats, labels, sids = [],[],[]
    with torch.no_grad():
        for imgs, labs, subj in tqdm(loader, desc="FeatExt"):
            imgs = imgs.to(device)
            f = ext(imgs).cpu().numpy()
            feats.append(f)
            labels.extend(labs.tolist())
            sids.extend(subj)
    X_slice = np.vstack(feats); y_slice = np.array(labels)

    # aggregate to subject-level
    subj_feats, subj_labels = {}, {}
    for feat,lab,sid in zip(X_slice,y_slice,sids):
        subj_feats.setdefault(sid,[]).append(feat)
        subj_labels[sid] = lab # Assuming one label per subject
    
    # Sort subjects for consistent order
    subj_ids_sorted = sorted(subj_feats.keys())
    Xs = np.vstack([np.mean(subj_feats[s],axis=0) for s in subj_ids_sorted])
    ys = np.array([subj_labels[s] for s in subj_ids_sorted])

    # Check for sufficient subjects and classes after aggregation
    if len(np.unique(ys)) < 2:
        raise RuntimeError(f"Insufficient unique classes ({len(np.unique(ys))}) among subjects after aggregation. Need at least 2 classes for AUC calculation.")
    if len(ys) < 2:
        raise RuntimeError(f"Insufficient unique subjects ({len(ys)}) after aggregation. Need at least 2 subjects for LOOCV/KFold.")

    print(f"[INFO] Aggregated to {len(ys)} subjects. Pos: {ys.sum()}, Neg: {len(ys)-ys.sum()}")

    # Call the new evaluation function
    auc_used, ci_low, ci_high, n_skipped_folds, final_y_true, final_y_score, final_y_logits = evaluate_model_performance(
        Xs, ys, args.n_bootstrap, args.seed
    )
    
    print("\n====== MRI Model Performance (Subject-Level) ======")
    print(f"Subject-level AUC point estimate (Direction-Corrected): {auc_used:.3f}")
    if not np.isnan(ci_low) and not np.isnan(ci_high):
        print(f"95% CI (Percentile Bootstrap)               : [{ci_low:.3f}, {ci_high:.3f}]")
    else:
        print("95% CI (Percentile Bootstrap)               : N/A (Bootstrap failed or insufficient samples)")
    print(f"Folds skipped due to single-class train/val : {n_skipped_folds}/12") # Fixed total folds to 12
    print("===================================================\n")

    # --- Save predictions to CSV for Figure 4 plotting ---
    df_preds = pd.DataFrame({
        'subject_id': subj_ids_sorted, # Assuming final_y_true/score/logits maintain this order
        'y_true': final_y_true,
        'prob_raw': final_y_score,
        'logit_raw': final_y_logits # Save the raw logits
    })
    output_pred_dir = Path(args.pred_out_csv).parent
    output_pred_dir.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(args.pred_out_csv, index=False)
    print(f"✅ L2O predictions saved to: {args.pred_out_csv}")


if __name__=="__main__":
    main()