import pandas as pd
import numpy as np
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import torch

# --- 设置出版级图表样式 (SCI Standard) ---
# 建议使用无衬线字体，如 Arial, Helvetica, 或 DejaVu Sans
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.dpi': 300,  # 提高图像分辨率
    'savefig.dpi': 300,
    'savefig.bbox': 'tight' # 保存时自动调整边界
})
# ---------------------------------------------

def plot_roc_curve(y_true, y_probs, output_path):
    """绘制并保存精美的ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='#FF6F61', lw=2.5, label=f'Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path)
    print(f"✅ ROC 曲线已保存至: {output_path}")
    plt.close()

def plot_pr_curve(y_true, y_probs, output_path):
    """绘制并保存精美的PR曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(7, 7))
    plt.step(recall, precision, where='post', color='#6B5B95', lw=2.5, label=f'Model (AP = {avg_precision:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.02])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path)
    print(f"✅ PR 曲线已保存至: {output_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path, class_names=['Negative', 'Positive']):
    """绘制并保存出版级的混淆矩阵热图"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = (np.asarray(["{0:d}\n({1:.1%})".format(value, percentage)
                         for value, percentage in zip(cm.flatten(), cm_percent.flatten())])
              ).reshape(cm.shape)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    print(f"✅ 混淆矩阵已保存至: {output_path}")
    plt.close()

def plot_calibration_curve(y_true, y_probs, output_path, n_bins=10):
    """【新增】绘制并保存校准曲线"""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    brier = brier_score_loss(y_true, y_probs)

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, "s-", color='#88B04B', label=f'Model (Brier Score = {brier:.3f})')
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.ylim([-0.05, 1.05])
    plt.title('Calibration Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path)
    print(f"✅ 校准曲线已保存至: {output_path}")
    plt.close()

def plot_probability_distribution(y_true, y_probs, output_path):
    """【新增】绘制并保存预测概率的分布图"""
    df = pd.DataFrame({'Probability': y_probs, 'True Label': y_true})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Probability', hue='True Label', bins=50, kde=True, palette=['#4A90E2', '#D0021B'])
    plt.xlabel("Predicted Probability of Positive Class")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Probabilities")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='True Label', labels=['Positive (1)', 'Negative (0)'])
    plt.savefig(output_path)
    print(f"✅ 概率分布图已保存至: {output_path}")
    plt.close()


def generate_visualizations(preds_dir):
    """主函数：加载所有预测结果并生成所有图表"""
    output_dir = os.path.join(preds_dir, 'publication_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    all_pred_files = glob(os.path.join(preds_dir, "fold_*_predictions.csv"))
    if not all_pred_files:
        print(f"❌ 错误: 在 {preds_dir} 中没有找到任何预测文件。")
        return

    df_all = pd.concat([pd.read_csv(f) for f in all_pred_files], ignore_index=True)
    
    true_labels = df_all["true_label"]
    logit_cols = [c for c in df_all.columns if "logit_" in c]
    logits = df_all[logit_cols].values

    probs_tensor = torch.nn.Softmax(dim=1)(torch.tensor(logits, dtype=torch.float32))
    positive_class_probs = probs_tensor[:, 1].numpy()
    pred_labels = np.argmax(logits, axis=1)

    print("\n--- 正在生成出版级可视化图表 ---")

    plot_roc_curve(true_labels, positive_class_probs, os.path.join(output_dir, 'sci_roc_curve.png'))
    plot_pr_curve(true_labels, positive_class_probs, os.path.join(output_dir, 'sci_pr_curve.png'))
    plot_confusion_matrix(true_labels, pred_labels, os.path.join(output_dir, 'sci_confusion_matrix.png'))
    plot_calibration_curve(true_labels, positive_class_probs, os.path.join(output_dir, 'sci_calibration_curve.png'))
    plot_probability_distribution(true_labels, positive_class_probs, os.path.join(output_dir, 'sci_probability_distribution.png'))

    print("\n🎉 所有图表已成功生成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为模型的交叉验证结果生成专业的、出版级的学术图表。")
    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="包含所有 'fold_k_predictions.csv' 文件的目录 (例如 'results/final_run/clinical_preds')"
    )
    args = parser.parse_args()
    generate_visualizations(args.preds_dir)