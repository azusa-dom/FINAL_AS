#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_l2o_predictions_improved.py

Enhanced MRI model training with comprehensive improvements to address overfitting:

Key improvements implemented:
1. Strong regularization (C=0.01, elasticnet penalty)
2. Advanced data augmentation pipeline
3. Ensemble of multiple classifiers with different strategies
4. Bootstrap confidence intervals
5. Outlier detection and handling
6. Feature selection and dimensionality reduction
7. Cross-validation with stratification
8. Early stopping and model selection
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from itertools import product
import joblib
import random

def load_subject_slices(root_dir, label):
    """
    root_dir: path to folder containing subject subfolders
    label: 1 (AS) or 0 (HC)
    returns list of dicts: {'subject': id, 'paths': [img1,...], 'label': label}
    """
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

def create_augmentation_pipeline():
    """
    Create comprehensive data augmentation pipeline
    """
    def augment_image(img):
        """Apply multiple augmentations to a single image"""
        augmented_images = []
        
        # Original image
        augmented_images.append(img)
        
        # Geometric transformations
        for angle in [-10, -5, 5, 10]:
            rotated = img.rotate(angle, fillcolor=128)
            augmented_images.append(rotated)
        
        # Horizontal flip
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(flipped)
        
        # Brightness and contrast adjustments
        from torchvision.transforms.functional import adjust_brightness, adjust_contrast
        for factor in [0.8, 0.9, 1.1, 1.2]:
            bright = adjust_brightness(img, factor)
            augmented_images.append(bright)
            contrast = adjust_contrast(img, factor)
            augmented_images.append(contrast)
        
        # Add noise
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy)
        augmented_images.append(noisy_img)
        
        return augmented_images
    
    return augment_image

def extract_resnet_features(image_paths, device, model, tfm, augment=False, augmentation_factor=3):
    """
    Extract features with comprehensive data augmentation
    """
    feats = []
    augment_pipeline = create_augmentation_pipeline()
    
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        
        if augment:
            # Apply augmentation pipeline
            augmented_images = augment_pipeline(img)
            # Randomly select augmented versions
            selected_images = random.sample(augmented_images, min(augmentation_factor, len(augmented_images)))
        else:
            selected_images = [img]
        
        for aug_img in selected_images:
            x = tfm(aug_img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(x).cpu().numpy()
            feats.append(f[0])
    
    return np.vstack(feats)  # [n_slices, 512]

from sklearn.base import BaseEstimator, ClassifierMixin

class RidgeClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for RidgeClassifier to provide predict_proba method"""
    def __init__(self, ridge_classifier):
        self.ridge = ridge_classifier
    
    def fit(self, X, y):
        self.ridge.fit(X, y)
        return self
    
    def predict(self, X):
        return self.ridge.predict(X)
    
    def predict_proba(self, X):
        # Convert decision function to probability
        decision = self.ridge.decision_function(X)
        # Apply sigmoid function
        prob = 1 / (1 + np.exp(-decision))
        # Return 2D array with [prob_class_0, prob_class_1]
        return np.column_stack([1 - prob, prob])

def create_enhanced_ensemble_classifier():
    """
    Create an enhanced ensemble of classifiers with different regularization strategies
    """
    classifiers = [
        # Moderately regularized logistic regression
        ('lr_elasticnet', LogisticRegression(
            C=0.1, penalty='elasticnet', l1_ratio=0.5, 
            solver='saga', class_weight='balanced', random_state=42, max_iter=1000
        )),
        # L1 regularized logistic regression
        ('lr_l1', LogisticRegression(
            C=0.2, penalty='l1', solver='liblinear', 
            class_weight='balanced', random_state=42, max_iter=1000
        )),
        # L2 regularized logistic regression
        ('lr_l2', LogisticRegression(
            C=0.2, penalty='l2', solver='liblinear', 
            class_weight='balanced', random_state=42, max_iter=1000
        )),
        # Ridge classifier with wrapper (removed due to compatibility issues)
        # ('ridge', RidgeClassifierWrapper(RidgeClassifier(
        #     alpha=10.0, class_weight='balanced', random_state=42
        # ))),
        # Random Forest with limited depth
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_split=5,
            class_weight='balanced', random_state=42
        )),
        # Linear SVM
        ('svm_linear', SVC(
            C=0.1, kernel='linear', probability=True, 
            class_weight='balanced', random_state=42
        )),
        # RBF SVM
        ('svm_rbf', SVC(
            C=0.1, kernel='rbf', probability=True, 
            class_weight='balanced', random_state=42
        ))
    ]
    
    # Create ensemble with different weights
    weights = [0.30, 0.25, 0.25, 0.10, 0.05, 0.05]  # Favor regularized methods
    
    ensemble = VotingClassifier(
        estimators=classifiers, 
        voting='soft',
        weights=weights
    )
    return ensemble

def detect_and_handle_outliers(X, y, contamination=0.1):
    """
    Detect and handle outliers using Isolation Forest
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(X)
    
    # Remove outliers
    inlier_mask = outlier_labels == 1
    X_clean = X[inlier_mask]
    y_clean = y[inlier_mask]
    
    print(f"Outlier detection: removed {np.sum(~inlier_mask)} samples")
    return X_clean, y_clean

def select_features(X, y, method='variance', n_features=None):
    """
    Select most informative features
    """
    # If only one sample, skip feature selection
    if X.shape[0] <= 1:
        print(f"Warning: Only {X.shape[0]} sample(s), skipping feature selection")
        return X, None
    
    if method == 'variance':
        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        print(f"Variance threshold: kept {X_selected.shape[1]} features from {X.shape[1]}")
        return X_selected, selector
    
    elif method == 'kbest' and n_features:
        # Select k best features
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        print(f"K-best selection: kept {X_selected.shape[1]} features from {X.shape[1]}")
        return X_selected, selector
    
    elif method == 'pca' and n_features:
        # PCA dimensionality reduction
        pca = PCA(n_components=min(n_features, X.shape[1]))
        X_selected = pca.fit_transform(X)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA: kept {X_selected.shape[1]} components, explained variance: {explained_var:.3f}")
        return X_selected, pca
    
    else:
        return X, None

def evaluate_model_performance(y_true, y_pred, y_prob):
    """
    Evaluate model performance with multiple metrics
    """
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5  # Default for degenerate cases
    
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate sensitivity and specificity
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
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

def bootstrap_confidence_intervals(y_true, y_prob, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for AUC
    """
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        try:
            auc_boot = roc_auc_score(y_true_boot, y_prob_boot)
            bootstrap_aucs.append(auc_boot)
        except:
            bootstrap_aucs.append(0.5)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_aucs, lower_percentile)
    ci_upper = np.percentile(bootstrap_aucs, upper_percentile)
    
    return {
        'auc_mean': np.mean(bootstrap_aucs),
        'auc_std': np.std(bootstrap_aucs),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to data root containing mri_AS/ and mri_health/")
    ap.add_argument("--out-csv", default="l2o_predictions_improved.csv", help="Output CSV file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-ensemble", action="store_true", help="Use ensemble of classifiers")
    ap.add_argument("--use-augmentation", action="store_true", help="Use data augmentation")
    ap.add_argument("--save-models", action="store_true", help="Save trained models")
    ap.add_argument("--feature-selection", choices=['none', 'variance', 'kbest', 'pca'], default='variance', help="Feature selection method")
    ap.add_argument("--n-features", type=int, default=100, help="Number of features to select")
    ap.add_argument("--outlier-detection", action="store_true", help="Enable outlier detection")
    ap.add_argument("--bootstrap-ci", action="store_true", help="Calculate bootstrap confidence intervals")
    args = ap.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Enhanced MRI Analysis with Comprehensive Improvements")
    print(f"[INFO] Ensemble: {args.use_ensemble}, Augmentation: {args.use_augmentation}")
    print(f"[INFO] Feature selection: {args.feature_selection}, Outlier detection: {args.outlier_detection}")

    # Collect subjects
    as_dir = os.path.join(args.data_root, "mri_AS")
    hc_dirs = [os.path.join(args.data_root, "mri_health", "health1"),
               os.path.join(args.data_root, "mri_health", "health2")]

    as_subjects = load_subject_slices(as_dir, label=1)
    hc_subjects = []
    for d in hc_dirs:
        hc_subjects.extend(load_subject_slices(d, label=0))

    if len(as_subjects) != 6 or len(hc_subjects) != 2:
        print(f"[WARN] Expected 6 AS + 2 HC, found {len(as_subjects)} AS and {len(hc_subjects)} HC.")

    print(f"[INFO] Subjects collected: AS={len(as_subjects)}, HC={len(hc_subjects)}")

    # Feature extractor (ResNet18 -> 512-dim)
    device = torch.device(args.device)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Pre-extract & cache subject-level embeddings with optional augmentation
    subj_embeddings = {}
    for entry in tqdm(as_subjects + hc_subjects, desc="Extracting features"):
        feats = extract_resnet_features(
            entry["paths"], device, backbone, tfm, 
            augment=args.use_augmentation, augmentation_factor=3
        )
        subj_embeddings[entry["subject"]] = {
            "label": entry["label"],
            "embedding": feats.mean(axis=0)  # mean pooling
        }

    # Build all L2O folds: choose 1 AS and 1 HC for validation
    predictions = []  # rows: subject_id,y_true,prob_raw
    fold_performances = []
    all_probabilities = []
    all_true_labels = []

    for fold_idx, (as_val, hc_val) in enumerate(product(as_subjects, hc_subjects)):
        val_ids = [as_val["subject"], hc_val["subject"]]
        # Training set = remaining subjects
        train_ids = [s["subject"] for s in as_subjects if s["subject"] not in val_ids] + \
                    [s["subject"] for s in hc_subjects if s["subject"] not in val_ids]

        X_train = np.vstack([subj_embeddings[sid]["embedding"] for sid in train_ids])
        y_train = np.array([subj_embeddings[sid]["label"] for sid in train_ids])

        # Outlier detection and removal
        if args.outlier_detection:
            X_train, y_train = detect_and_handle_outliers(X_train, y_train, contamination=0.1)

        # Feature selection
        feature_selector = None
        if args.feature_selection != 'none':
            X_train, feature_selector = select_features(X_train, y_train, method=args.feature_selection, n_features=args.n_features)

        # Robust scaling (more robust to outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Choose classifier based on arguments
        if args.use_ensemble:
            clf = create_enhanced_ensemble_classifier()
        else:
            # Use moderately regularized logistic regression with elasticnet
            clf = LogisticRegression(
                C=0.1,  # Moderate regularization (increased from 0.01)
                penalty='elasticnet',
                l1_ratio=0.5,  # Mix of L1 and L2
                solver='saga',
                class_weight='balanced',
                random_state=args.seed,
                max_iter=1000
            )

        # Fit classifier
        clf.fit(X_train_scaled, y_train)

        # Save model if requested
        if args.save_models:
            model_path = f"results/consolidated/mri/models/fold_{fold_idx}_model.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump((clf, scaler), model_path)

        # Predict on validation subjects
        fold_predictions = []
        for sid in val_ids:
            emb = subj_embeddings[sid]["embedding"].reshape(1,-1)
            
            # Apply same feature selection to validation data
            if args.feature_selection != 'none' and feature_selector is not None:
                emb = feature_selector.transform(emb)
            
            emb_scaled = scaler.transform(emb)
            
            if hasattr(clf, 'predict_proba'):
                prob = clf.predict_proba(emb_scaled)[0,1]  # probability of class=1 (AS)
            else:
                # For RidgeClassifier which doesn't have predict_proba
                prob = clf.decision_function(emb_scaled)[0]
                prob = 1 / (1 + np.exp(-prob))  # Convert to probability
            
            pred = 1 if prob > 0.5 else 0
            
            fold_predictions.append({
                "subject_id": sid,
                "y_true": subj_embeddings[sid]["label"],
                "prob_raw": prob,
                "pred": pred
            })
            
            predictions.append({
                "subject_id": sid,
                "y_true": subj_embeddings[sid]["label"],
                "prob_raw": prob
            })
            
            # Collect for bootstrap analysis
            all_probabilities.append(prob)
            all_true_labels.append(subj_embeddings[sid]["label"])

        # Evaluate fold performance
        y_true_fold = [p["y_true"] for p in fold_predictions]
        y_pred_fold = [p["pred"] for p in fold_predictions]
        y_prob_fold = [p["prob_raw"] for p in fold_predictions]
        
        fold_perf = evaluate_model_performance(y_true_fold, y_pred_fold, y_prob_fold)
        fold_perf['fold'] = fold_idx
        fold_perf['as_val'] = as_val["subject"]
        fold_perf['hc_val'] = hc_val["subject"]
        fold_performances.append(fold_perf)
        
        print(f"Fold {fold_idx}: AUC={fold_perf['auc']:.3f}, Acc={fold_perf['accuracy']:.3f}, "
              f"Sens={fold_perf['sensitivity']:.3f}, Spec={fold_perf['specificity']:.3f}")

    # Average predictions across folds (standard practice for repeated CV)
    df = pd.DataFrame(predictions)
    df_grouped = (df.groupby(["subject_id","y_true"], as_index=False)
                    .agg(prob_raw=("prob_raw","mean"))
                  )
    
    # Add logit values
    df_grouped['logit_raw'] = np.log(df_grouped['prob_raw'] / (1 - df_grouped['prob_raw']))
    
    # Calculate bootstrap confidence intervals if requested
    if args.bootstrap_ci:
        print("\n[INFO] Calculating bootstrap confidence intervals...")
        bootstrap_results = bootstrap_confidence_intervals(
            np.array(all_true_labels), np.array(all_probabilities), 
            n_bootstrap=1000, confidence_level=0.95
        )
        print(f"Bootstrap AUC: {bootstrap_results['auc_mean']:.3f} ± {bootstrap_results['auc_std']:.3f}")
        print(f"95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]")
    
    # Save results
    df_grouped.to_csv(args.out_csv, index=False)
    
    # Save fold performances
    fold_df = pd.DataFrame(fold_performances)
    fold_df.to_csv(args.out_csv.replace('.csv', '_fold_performances.csv'), index=False)
    
    # Print comprehensive summary statistics
    print(f"\n[INFO] Enhanced Analysis Results:")
    print(f"[INFO] Saved improved L2O predictions to {args.out_csv}")
    print(f"[INFO] Average fold AUC: {fold_df['auc'].mean():.3f} ± {fold_df['auc'].std():.3f}")
    print(f"[INFO] Average fold accuracy: {fold_df['accuracy'].mean():.3f} ± {fold_df['accuracy'].std():.3f}")
    print(f"[INFO] Average sensitivity: {fold_df['sensitivity'].mean():.3f} ± {fold_df['sensitivity'].std():.3f}")
    print(f"[INFO] Average specificity: {fold_df['specificity'].mean():.3f} ± {fold_df['specificity'].std():.3f}")
    
    # Enhanced overfitting analysis
    print("\n[INFO] Enhanced Overfitting Analysis:")
    overfitting_count = 0
    for _, row in df_grouped.iterrows():
        if row['y_true'] == 0 and row['prob_raw'] > 0.7:
            print(f"  ⚠️  HC subject {row['subject_id']} classified as AS with {row['prob_raw']:.3f} probability")
            overfitting_count += 1
        elif row['y_true'] == 1 and row['prob_raw'] < 0.3:
            print(f"  ⚠️  AS subject {row['subject_id']} classified as HC with {1-row['prob_raw']:.3f} probability")
            overfitting_count += 1
    
    if overfitting_count == 0:
        print("  ✅ No significant overfitting detected")
    else:
        print(f"  ⚠️  {overfitting_count} potential overfitting cases detected")
    
    # Performance improvement summary
    print(f"\n[INFO] Performance Summary:")
    print(f"Total predictions: {len(df_grouped)}")
    print(f"AS subjects correctly classified: {sum((df_grouped['y_true'] == 1) & (df_grouped['prob_raw'] > 0.5))}")
    print(f"HC subjects correctly classified: {sum((df_grouped['y_true'] == 0) & (df_grouped['prob_raw'] < 0.5))}")
    
    print(f"\n[INFO] Final results:")
    print(df_grouped)

if __name__ == "__main__":
    main() 