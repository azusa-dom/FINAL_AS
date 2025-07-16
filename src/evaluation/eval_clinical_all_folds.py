# src/evaluation/eval_clinical_all_folds.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

def load_predictions(pred_dir):
    all_logits, all_labels = [], []
    for i in range(5):
        df = pd.read_csv(os.path.join(pred_dir, f"fold_{i}_predictions.csv"))
        logits = df[[c for c in df.columns if c.startswith("logit_")]].values
        labels = df["true_label"].values
        all_logits.append(logits)
        all_labels.append(labels)
    return np.vstack(all_logits), np.hstack(all_labels)

def evaluate(pred_dir):
    logits, labels = load_predictions(pred_dir)
    probs = logits[:, 1]  # Positive class
    preds = (probs >= 0.5).astype(int)

    print("=== Evaluation ===")
    print("AUROC:", roc_auc_score(labels, probs))
    print("AUPRC:", average_precision_score(labels, probs))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    print("Classification Report:\n", classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Path to directory with fold_x_predictions.csv files")
    args = parser.parse_args()
    evaluate(args.pred_dir)
