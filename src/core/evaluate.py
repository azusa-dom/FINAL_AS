import torch
import pandas as pd
import numpy as np
import os
import argparse
from glob import glob
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(args):
    pred_files = glob(os.path.join(args.preds_dir, "fold_*_predictions.csv"))
    if not pred_files:
        print(f"❌ 在 {args.preds_dir} 中未找到预测文件")
        return

    all_dfs = [pd.read_csv(f) for f in pred_files]
    df_all = pd.concat(all_dfs, ignore_index=True)

    true_labels = df_all["true_label"]
    logit_cols = [c for c in df_all.columns if "logit_" in c]
    logits = df_all[logit_cols].values

    probs = torch.nn.Softmax(dim=1)(torch.tensor(logits)).numpy()
    pred_labels = np.argmax(probs, axis=1)

    # --- 总体指标 ---
    print("\n--- 交叉验证总体评估结果 ---")
    acc = accuracy_score(true_labels, pred_labels)
    print(f"🎯 总体准确率 (Accuracy): {acc:.4f}")

    try:
        if probs.shape[1] == 2:
            auc_score = roc_auc_score(true_labels, probs[:, 1])
        else:
            auc_score = roc_auc_score(true_labels, probs, multi_class="ovr", average="weighted")
        print(f"🎯 总体AUC分数: {auc_score:.4f}")
    except Exception as e:
        auc_score = -1
        print(f"⚠️ 无法计算AUC: {e}")

    print("\n📋 总体分类报告:")
    print(classification_report(true_labels, pred_labels))

    # --- 保存图表和指标 ---
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=np.unique(true_labels), 
                yticklabels=np.unique(true_labels))
    plt.title("总体混淆矩阵 (Overall Confusion Matrix)")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    cm_path = os.path.join(args.preds_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\n✅ 混淆矩阵已保存至: {cm_path}")

    metrics = {"accuracy": acc, "auc": auc_score}
    metrics_path = os.path.join(args.preds_dir, "metrics_summary.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"📊 总体指标已保存至: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估临床模型结果。")
    parser.add_argument("--preds_dir", required=True, help="预测结果文件所在目录。")
    args = parser.parse_args()
    evaluate(args)