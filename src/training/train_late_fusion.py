# src/train_late_fusion.py (已修复)
# NOTE: Due to limited paired MRI and clinical data, this late-fusion module is
# provided for future experiments and is not used in the current workflow.

import pandas as pd
import numpy as np
import os
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
from pathlib import Path


def softmax(logits):
    """对logits执行softmax运算"""
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def load_and_merge_predictions(clinical_path, mri_path):
    """加载预测结果并合并为特征向量"""
    clinical_preds = pd.read_csv(clinical_path)
    mri_preds = pd.read_csv(mri_path)

    merged = pd.merge(clinical_preds, mri_preds, on="patient_id")
    features = merged.drop(columns=["patient_id", "label"])
    labels = merged["label"]

    return features.values, labels.values


def train_late_fusion(X, y, n_splits=5, random_state=42):
    """使用XGBoost进行晚期融合训练和交叉验证"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = {
        "roc_auc": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        metrics["roc_auc"].append(roc_auc_score(y_val, probs))
        metrics["accuracy"].append(accuracy_score(y_val, preds))
        metrics["f1"].append(f1_score(y_val, preds))
        metrics["precision"].append(precision_score(y_val, preds))
        metrics["recall"].append(recall_score(y_val, preds))

    for key, values in metrics.items():
        print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def main(args):
    X, y = load_and_merge_predictions(args.clinical_preds, args.mri_preds)
    train_late_fusion(X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late Fusion Model Trainer")
    parser.add_argument("--clinical_preds", type=str, required=True, help="Path to clinical prediction CSV")
    parser.add_argument("--mri_preds", type=str, required=True, help="Path to MRI prediction CSV")
    args = parser.parse_args()
    main(args)
