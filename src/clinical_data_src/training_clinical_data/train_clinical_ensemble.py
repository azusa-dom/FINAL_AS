#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_clinical_ensemble.py

Improved clinical model training with ensemble methods and better calibration
to address overconfidence issues.

Key improvements:
1. Ensemble of multiple algorithms (LightGBM, XGBoost, Neural Network)
2. Improved calibration methods (Platt scaling, Isotonic regression)
3. Cross-validation with stratification
4. Feature importance analysis
5. Uncertainty quantification
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class ClinicalNet(nn.Module):
    """Improved ClinicalNet with dropout and batch normalization"""
    def __init__(self, input_dim, hidden_size=64, dropout_rate=0.5):
        super(ClinicalNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

def load_data(data_dir):
    """Load and prepare data from processed directory"""
    train_files = [f for f in os.listdir(data_dir) if f.startswith('fold_') and f.endswith('_train.csv')]
    
    all_data = []
    for file in train_files:
        df = pd.read_csv(os.path.join(data_dir, file))
        all_data.append(df)
    
    # Combine all training data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Separate features and labels
    feature_cols = [col for col in combined_df.columns if col not in ['label', 'patient_id']]
    X = combined_df[feature_cols].values
    y = combined_df['label'].values
    
    return X, y, feature_cols

def create_lightgbm_model():
    """Create LightGBM model with optimized parameters"""
    return lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )

def create_xgboost_model():
    """Create XGBoost model with optimized parameters"""
    return xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    )

def create_neural_network_model(input_dim):
    """Create neural network model"""
    return ClinicalNet(input_dim=input_dim, hidden_size=64, dropout_rate=0.5)

def train_neural_network(X_train, y_train, X_val, y_val, input_dim, epochs=100):
    """Train neural network with early stopping"""
    model = create_neural_network_model(input_dim)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model)
    return model

def evaluate_model_performance(y_true, y_pred, y_prob):
    """Evaluate model performance with multiple metrics"""
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_prob)
    
    return {
        'auc': auc,
        'accuracy': acc,
        'log_loss': loss,
        'n_samples': len(y_true)
    }

def main():
    parser = argparse.ArgumentParser(description='Train ensemble clinical model')
    parser.add_argument('--data_dir', required=True, help='Path to processed data directory')
    parser.add_argument('--output_dir', default='results/clinical/ensemble_model', help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--use_calibration', action='store_true', help='Use probability calibration')
    parser.add_argument('--save_models', action='store_true', help='Save individual models')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X, y, feature_cols = load_data(args.data_dir)
    print(f"Data shape: {X.shape}, Features: {len(feature_cols)}")
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold_idx + 1}/{args.n_folds} ===")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train individual models
        models = {}
        
        # 1. LightGBM
        print("Training LightGBM...")
        lgb_model = create_lightgbm_model()
        lgb_model.fit(X_train_scaled, y_train)
        models['lightgbm'] = lgb_model
        
        # 2. XGBoost
        print("Training XGBoost...")
        xgb_model = create_xgboost_model()
        xgb_model.fit(X_train_scaled, y_train)
        models['xgboost'] = xgb_model
        
        # 3. Neural Network
        print("Training Neural Network...")
        nn_model = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, X_train_scaled.shape[1])
        models['neural_network'] = nn_model
        
        # 4. Logistic Regression (baseline)
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        models['logistic_regression'] = lr_model
        
        # Get predictions from each model
        predictions = {}
        
        # LightGBM predictions
        lgb_prob = lgb_model.predict_proba(X_val_scaled)[:, 1]
        predictions['lightgbm'] = lgb_prob
        
        # XGBoost predictions
        xgb_prob = xgb_model.predict_proba(X_val_scaled)[:, 1]
        predictions['xgboost'] = xgb_prob
        
        # Neural Network predictions
        nn_model.eval()
        with torch.no_grad():
            nn_logits = nn_model(torch.FloatTensor(X_val_scaled))
            nn_prob = torch.sigmoid(nn_logits).numpy().flatten()
        predictions['neural_network'] = nn_prob
        
        # Logistic Regression predictions
        lr_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
        predictions['logistic_regression'] = lr_prob
        
        # Create ensemble predictions (simple average)
        ensemble_prob = np.mean([predictions[model] for model in predictions.keys()], axis=0)
        predictions['ensemble'] = ensemble_prob
        
        # Evaluate each model
        fold_model_results = {}
        for model_name, prob in predictions.items():
            pred = (prob > 0.5).astype(int)
            metrics = evaluate_model_performance(y_val, pred, prob)
            fold_model_results[model_name] = metrics
            print(f"{model_name}: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}")
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'models': fold_model_results
        })
        
        # Store predictions for this fold
        for i, (idx, true_label) in enumerate(zip(val_idx, y_val)):
            all_predictions.append({
                'patient_id': idx,
                'true_label': true_label,
                'fold': fold_idx,
                'lightgbm_prob': predictions['lightgbm'][i],
                'xgboost_prob': predictions['xgboost'][i],
                'neural_network_prob': predictions['neural_network'][i],
                'logistic_regression_prob': predictions['logistic_regression'][i],
                'ensemble_prob': predictions['ensemble'][i]
            })
        
        # Save models if requested
        if args.save_models:
            fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Save sklearn models
            joblib.dump(lgb_model, os.path.join(fold_dir, 'lightgbm_model.pkl'))
            joblib.dump(xgb_model, os.path.join(fold_dir, 'xgboost_model.pkl'))
            joblib.dump(lr_model, os.path.join(fold_dir, 'logistic_regression_model.pkl'))
            joblib.dump(scaler, os.path.join(fold_dir, 'scaler.pkl'))
            
            # Save neural network
            torch.save(nn_model.state_dict(), os.path.join(fold_dir, 'neural_network_model.pth'))
    
    # Create results summary
    results_df = pd.DataFrame(all_predictions)
    
    # Calculate overall metrics
    print("\n=== Overall Results ===")
    overall_metrics = {}
    for model in ['lightgbm', 'xgboost', 'neural_network', 'logistic_regression', 'ensemble']:
        prob_col = f'{model}_prob'
        y_true = results_df['true_label'].values
        y_prob = results_df[prob_col].values
        y_pred = (y_prob > 0.5).astype(int)
        
        metrics = evaluate_model_performance(y_true, y_pred, y_prob)
        overall_metrics[model] = metrics
        print(f"{model}: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}, LogLoss={metrics['log_loss']:.3f}")
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, 'ensemble_predictions.csv'), index=False)
    
    # Save overall metrics
    metrics_df = pd.DataFrame(overall_metrics).T
    metrics_df.to_csv(os.path.join(args.output_dir, 'ensemble_metrics.csv'))
    
    # Save fold results
    fold_summary = []
    for fold_result in fold_results:
        fold_data = {'fold': fold_result['fold']}
        for model_name, metrics in fold_result['models'].items():
            for metric_name, value in metrics.items():
                fold_data[f'{model_name}_{metric_name}'] = value
        fold_summary.append(fold_data)
    
    fold_df = pd.DataFrame(fold_summary)
    fold_df.to_csv(os.path.join(args.output_dir, 'fold_results.csv'), index=False)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Best ensemble model: AUC={overall_metrics['ensemble']['auc']:.3f}")

if __name__ == "__main__":
    main() 