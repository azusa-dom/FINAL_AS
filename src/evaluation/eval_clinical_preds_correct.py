# FINAL_AS/code/src/eval/eval_clinical_preds_correct.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

def load_all_predictions(preds_dir):
    df_all = []
    for i in range(5):
        path = os.path.join(preds_dir, f"fold_{i}_predictions.csv")
        df = pd.read_csv(path)
        df_all.append(df)
    return pd.concat(df_all, ignore_index=True)

if __name__ == "__main__":
    preds_dir = "FINAL_AS/output_clinical_model/clinical_preds"
    df_preds = load_all_predictions(preds_dir)
    print(f"✅ Total patients in evaluation set: {len(df_preds)}")

    y_true = df_preds["true_label"].values
    y_score = df_preds["logit_1"].values  # 用于二分类ROC
    y_pred = (y_score >= 0.5).astype(int)

    print("\n=== Evaluation ===")
    print(f"AUROC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"AUPRC: {average_precision_score(y_true, y_score):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
