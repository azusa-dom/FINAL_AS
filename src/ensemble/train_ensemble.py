#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ensemble.py

Training script for DDI-AS ensemble model
Demonstrates the integration of ClinicalNet and ImagingNet
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.clinical.training_clinical_data.train_clinical_ensemble import ClinicalNet
from src.mri.models.imaging_net import ImagingNet
from src.ensemble.ensemble_model import EnsembleModel, create_ensemble_model


def load_clinical_data(data_path: str):
    """Load clinical data"""
    # This is a placeholder - replace with actual data loading logic
    data = pd.read_csv(data_path)
    feature_cols = [col for col in data.columns if col not in ['label', 'patient_id']]
    X = data[feature_cols].values
    y = data['label'].values
    return X, y, feature_cols


def load_imaging_data(data_path: str):
    """Load imaging data"""
    # This is a placeholder - replace with actual data loading logic
    # In practice, this would load MRI images and convert to tensors
    pass


def train_clinical_net(X, y, n_folds=5):
    """Train ClinicalNet with cross-validation"""
    print("Training ClinicalNet...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    clinical_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train ClinicalNet
        clinical_net = ClinicalNet(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        clinical_net.fit(X_train, y_train)
        
        # Get predictions
        clinical_probs = clinical_net.predict_proba(X_val)
        
        # Store predictions
        for i, (idx, true_label) in enumerate(zip(val_idx, y_val)):
            clinical_predictions.append({
                'patient_id': idx,
                'true_label': true_label,
                'fold': fold_idx,
                'clinical_prob': clinical_probs[i, 1]
            })
    
    return pd.DataFrame(clinical_predictions)


def train_imaging_net(imaging_data, labels, n_folds=8):
    """Train ImagingNet with leave-two-out cross-validation"""
    print("Training ImagingNet...")
    
    # This is a placeholder - in practice, you would implement L2O-CV
    # For now, we'll create dummy predictions
    imaging_predictions = []
    
    for i, (data, label) in enumerate(zip(imaging_data, labels)):
        imaging_predictions.append({
            'patient_id': i,
            'true_label': label,
            'imaging_prob': np.random.random()  # Placeholder
        })
    
    return pd.DataFrame(imaging_predictions)


def evaluate_ensemble(clinical_df, imaging_df):
    """Evaluate ensemble model performance"""
    print("Evaluating ensemble model...")
    
    # Merge predictions (assuming same patient IDs)
    ensemble_df = clinical_df.merge(imaging_df, on='patient_id', suffixes=('_clinical', '_imaging'))
    
    # Calculate ensemble probabilities
    ensemble_df['ensemble_prob'] = 0.5 * ensemble_df['clinical_prob'] + 0.5 * ensemble_df['imaging_prob']
    
    # Calculate metrics
    y_true = ensemble_df['true_label_clinical'].values
    clinical_probs = ensemble_df['clinical_prob'].values
    imaging_probs = ensemble_df['imaging_prob'].values
    ensemble_probs = ensemble_df['ensemble_prob'].values
    
    # Individual model performance
    clinical_auc = roc_auc_score(y_true, clinical_probs)
    imaging_auc = roc_auc_score(y_true, imaging_probs)
    ensemble_auc = roc_auc_score(y_true, ensemble_probs)
    
    print(f"ClinicalNet AUROC: {clinical_auc:.3f}")
    print(f"ImagingNet AUROC: {imaging_auc:.3f}")
    print(f"Ensemble AUROC: {ensemble_auc:.3f}")
    print(f"Improvement: {ensemble_auc - clinical_auc:.3f}")
    
    return {
        'clinical_auc': clinical_auc,
        'imaging_auc': imaging_auc,
        'ensemble_auc': ensemble_auc,
        'improvement': ensemble_auc - clinical_auc
    }


def main():
    parser = argparse.ArgumentParser(description='Train DDI-AS ensemble model')
    parser.add_argument('--clinical_data', required=True, help='Path to clinical data')
    parser.add_argument('--imaging_data', required=True, help='Path to imaging data')
    parser.add_argument('--output_dir', default='results/ensemble', help='Output directory')
    parser.add_argument('--n_folds_clinical', type=int, default=5, help='Number of CV folds for clinical data')
    parser.add_argument('--n_folds_imaging', type=int, default=8, help='Number of L2O folds for imaging data')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    clinical_X, clinical_y, feature_cols = load_clinical_data(args.clinical_data)
    
    # Train ClinicalNet
    clinical_results = train_clinical_net(clinical_X, clinical_y, args.n_folds_clinical)
    clinical_results.to_csv(os.path.join(args.output_dir, 'clinical_predictions.csv'), index=False)
    
    # Train ImagingNet (placeholder)
    # imaging_results = train_imaging_net(imaging_data, imaging_labels, args.n_folds_imaging)
    # imaging_results.to_csv(os.path.join(args.output_dir, 'imaging_predictions.csv'), index=False)
    
    # For demonstration, create dummy imaging results
    imaging_results = pd.DataFrame({
        'patient_id': clinical_results['patient_id'],
        'true_label': clinical_results['true_label'],
        'imaging_prob': np.random.random(len(clinical_results))
    })
    imaging_results.to_csv(os.path.join(args.output_dir, 'imaging_predictions.csv'), index=False)
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_ensemble(clinical_results, imaging_results)
    
    # Save ensemble results
    ensemble_df = clinical_results.merge(imaging_results, on='patient_id', suffixes=('_clinical', '_imaging'))
    ensemble_df['ensemble_prob'] = 0.5 * ensemble_df['clinical_prob'] + 0.5 * ensemble_df['imaging_prob']
    ensemble_df.to_csv(os.path.join(args.output_dir, 'ensemble_predictions.csv'), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([ensemble_metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, 'ensemble_metrics.csv'), index=False)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Ensemble AUROC: {ensemble_metrics['ensemble_auc']:.3f}")
    print(f"Improvement over ClinicalNet: {ensemble_metrics['improvement']:.3f}")


if __name__ == "__main__":
    main() 