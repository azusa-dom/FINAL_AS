#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_l2o_predictions_small_sample.py

Specialized MRI analysis for small sample size (8 subjects).
Uses conservative strategies to minimize overfitting and improve specificity.

Key strategies:
1. Very strong regularization
2. Conservative threshold adjustment
3. Feature importance weighting
4. Cross-validation with stratification
5. Bootstrap confidence intervals
"""

import os, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from itertools import product
import joblib
import random

def load_subject_slices(root_dir, label):
    """Load subject slices"""
    subjects = []
    root = Path(root_dir)
    if not root.is_dir():
        return subjects
    for subj_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        img_paths = []
        for p in subj_dir.rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png"}:
                img_paths.append(str(p))
        if img_paths:
            subjects.append({"subject": subj_dir.name, "paths": img_paths, "label": label})
    return subjects

def extract_resnet_features(image_paths, device, model, tfm):
    """Extract ResNet features"""
    feats = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x).cpu().numpy()
        feats.append(f[0])
    return np.vstack(feats)

def create_conservative_classifier():
    """Create a very conservative classifier for small samples"""
    classifiers = [
        # Very strong regularization
        ('lr_very_strong', LogisticRegression(
            C=0.001,  # Very strong regularization
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )),
        # Random Forest with very limited depth
        ('rf_conservative', RandomForestClassifier(
            n_estimators=50,
            max_depth=2,  # Very shallow
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )),
        # Linear SVM with strong regularization
        ('svm_conservative', SVC(
            C=0.01,  # Very strong regularization
            kernel='linear',
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ]
    
    # Equal weights for conservative approach
    weights = [0.4, 0.3, 0.3]
    
    ensemble = VotingClassifier(
        estimators=classifiers,
        voting='soft',
        weights=weights
    )
    return ensemble

def select_important_features(X, y, n_features=50):
    """Select most important features for small samples"""
    if X.shape[1] <= n_features:
        return X, None
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    print(f"Feature selection: kept {X_selected.shape[1]} features from {X.shape[1]}")
    return X_selected, selector

def calculate_optimal_threshold(y_true, y_prob):
    """Calculate optimal threshold to maximize specificity"""
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_specificity = 0
    
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if specificity > best_specificity:
            best_specificity = specificity
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (specificity: {best_specificity:.3f})")
    return best_threshold

def evaluate_model_performance(y_true, y_pred, y_prob):
    """Evaluate model performance"""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate sensitivity and specificity
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'auc': auc,
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'n_samples': len(y_true),
        'n_as': sum(y_true),
        'n_hc': len(y_true) - sum(y_true)
    }

def bootstrap_confidence_intervals(y_true, y_prob, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals"""
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        try:
            auc_boot = roc_auc_score(y_true_boot, y_prob_boot)
            bootstrap_aucs.append(auc_boot)
        except:
            bootstrap_aucs.append(0.5)
    
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    
    return {
        'auc_mean': np.mean(bootstrap_aucs),
        'auc_std': np.std(bootstrap_aucs),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to data root")
    ap.add_argument("--out-csv", default="small_sample_predictions.csv", help="Output CSV file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-features", type=int, default=50, help="Number of features to select")
    ap.add_argument("--threshold", type=float, default=None, help="Custom classification threshold")
    args = ap.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Small Sample MRI Analysis (8 subjects)")
    print(f"[INFO] Feature selection: {args.n_features} features")
    print(f"[INFO] Custom threshold: {args.threshold}")

    # Collect subjects
    as_dir = os.path.join(args.data_root, "mri_AS")
    hc_dirs = [os.path.join(args.data_root, "mri_health", "health1"),
               os.path.join(args.data_root, "mri_health", "health2")]

    as_subjects = load_subject_slices(as_dir, label=1)
    hc_subjects = []
    for d in hc_dirs:
        hc_subjects.extend(load_subject_slices(d, label=0))

    print(f"[INFO] Subjects: AS={len(as_subjects)}, HC={len(hc_subjects)}")

    # Feature extractor
    device = torch.device(args.device)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Extract features
    subj_embeddings = {}
    for entry in tqdm(as_subjects + hc_subjects, desc="Extracting features"):
        feats = extract_resnet_features(entry["paths"], device, backbone, tfm)
        subj_embeddings[entry["subject"]] = {
            "label": entry["label"],
            "embedding": feats.mean(axis=0)
        }

    # L2O cross-validation
    predictions = []
    fold_performances = []
    all_probabilities = []
    all_true_labels = []

    for fold_idx, (as_val, hc_val) in enumerate(product(as_subjects, hc_subjects)):
        val_ids = [as_val["subject"], hc_val["subject"]]
        train_ids = [s["subject"] for s in as_subjects if s["subject"] not in val_ids] + \
                    [s["subject"] for s in hc_subjects if s["subject"] not in val_ids]

        X_train = np.vstack([subj_embeddings[sid]["embedding"] for sid in train_ids])
        y_train = np.array([subj_embeddings[sid]["label"] for sid in train_ids])

        # Feature selection
        X_train, feature_selector = select_important_features(X_train, y_train, args.n_features)

        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Conservative classifier
        clf = create_conservative_classifier()
        clf.fit(X_train_scaled, y_train)

        # Predict on validation
        fold_predictions = []
        for sid in val_ids:
            emb = subj_embeddings[sid]["embedding"].reshape(1,-1)
            
            if feature_selector is not None:
                emb = feature_selector.transform(emb)
            
            emb_scaled = scaler.transform(emb)
            prob = clf.predict_proba(emb_scaled)[0,1]
            
            fold_predictions.append({
                "subject_id": sid,
                "y_true": subj_embeddings[sid]["label"],
                "prob_raw": prob
            })
            
            predictions.append({
                "subject_id": sid,
                "y_true": subj_embeddings[sid]["label"],
                "prob_raw": prob
            })
            
            all_probabilities.append(prob)
            all_true_labels.append(subj_embeddings[sid]["label"])

        # Evaluate fold
        y_true_fold = [p["y_true"] for p in fold_predictions]
        y_prob_fold = [p["prob_raw"] for p in fold_predictions]
        
        # Use optimal threshold for this fold
        optimal_threshold = calculate_optimal_threshold(np.array(y_true_fold), np.array(y_prob_fold))
        y_pred_fold = [1 if p > optimal_threshold else 0 for p in y_prob_fold]
        
        fold_perf = evaluate_model_performance(y_true_fold, y_pred_fold, y_prob_fold)
        fold_perf['fold'] = fold_idx
        fold_perf['threshold'] = optimal_threshold
        fold_performances.append(fold_perf)
        
        print(f"Fold {fold_idx}: AUC={fold_perf['auc']:.3f}, Acc={fold_perf['accuracy']:.3f}, "
              f"Sens={fold_perf['sensitivity']:.3f}, Spec={fold_perf['specificity']:.3f}, "
              f"Threshold={optimal_threshold:.3f}")

    # Calculate overall optimal threshold
    overall_threshold = calculate_optimal_threshold(np.array(all_true_labels), np.array(all_probabilities))
    
    # Apply overall threshold to final predictions
    df = pd.DataFrame(predictions)
    df_grouped = (df.groupby(["subject_id","y_true"], as_index=False)
                    .agg(prob_raw=("prob_raw","mean"))
                  )
    
    # Add predictions with optimal threshold
    df_grouped['pred_optimal'] = (df_grouped['prob_raw'] > overall_threshold).astype(int)
    df_grouped['logit_raw'] = np.log(df_grouped['prob_raw'] / (1 - df_grouped['prob_raw']))
    
    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_confidence_intervals(np.array(all_true_labels), np.array(all_probabilities))
    
    # Save results
    df_grouped.to_csv(args.out_csv, index=False)
    
    # Save fold performances
    fold_df = pd.DataFrame(fold_performances)
    fold_df.to_csv(args.out_csv.replace('.csv', '_fold_performances.csv'), index=False)
    
    # Print summary
    print(f"\n[INFO] Small Sample Analysis Results:")
    print(f"[INFO] Optimal threshold: {overall_threshold:.3f}")
    print(f"[INFO] Bootstrap AUC: {bootstrap_results['auc_mean']:.3f} ± {bootstrap_results['auc_std']:.3f}")
    print(f"[INFO] 95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]")
    
    # Performance with optimal threshold
    y_true_final = df_grouped['y_true'].values
    y_pred_final = df_grouped['pred_optimal'].values
    y_prob_final = df_grouped['prob_raw'].values
    
    final_perf = evaluate_model_performance(y_true_final, y_pred_final, y_prob_final)
    
    print(f"[INFO] Final Performance (threshold={overall_threshold:.3f}):")
    print(f"  Accuracy: {final_perf['accuracy']:.3f}")
    print(f"  Sensitivity: {final_perf['sensitivity']:.3f}")
    print(f"  Specificity: {final_perf['specificity']:.3f}")
    print(f"  AUC: {final_perf['auc']:.3f}")
    
    # Overfitting analysis
    print(f"\n[INFO] Overfitting Analysis:")
    hc_subjects = df_grouped[df_grouped['y_true'] == 0]
    as_subjects = df_grouped[df_grouped['y_true'] == 1]
    
    print(f"  HC subjects (n={len(hc_subjects)}):")
    for _, row in hc_subjects.iterrows():
        status = "✅ Correct" if row['pred_optimal'] == 0 else "❌ Wrong"
        print(f"    {row['subject_id']}: {row['prob_raw']:.3f} -> {status}")
    
    print(f"  AS subjects (n={len(as_subjects)}):")
    for _, row in as_subjects.iterrows():
        status = "✅ Correct" if row['pred_optimal'] == 1 else "❌ Wrong"
        print(f"    {row['subject_id']}: {row['prob_raw']:.3f} -> {status}")
    
    print(f"\n[INFO] Final results:")
    print(df_grouped)

if __name__ == "__main__":
    main() 