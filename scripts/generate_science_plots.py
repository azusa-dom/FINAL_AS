import pandas as pd
import numpy as np
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import torch
import shutil

# --- 严格遵循 Science 期刊标准进行配置 ---
SCIENCE_STYLE_CONFIG = {
    # 字体设置
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'], # 使用 Arial 字体
    'font.size': 10,
    
    # 图片设置
    'figure.dpi': 600, # 使用推荐的 600 DPI
    'savefig.dpi': 600,
    'savefig.format': 'tiff', # 保存为 TIFF 格式
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # 坐标轴设置
    'axes.linewidth': 1,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # 刻度设置
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    
    # 图例设置
    'legend.fontsize': 9,
    'legend.frameon': False,
    
    # 线条设置
    'lines.linewidth': 1.5, # 根据标准调整
    'lines.markersize': 5
}
plt.rcParams.update(SCIENCE_STYLE_CONFIG)

# Science 推荐的配色方案
COLORS = {
    'primary': '#1f77b4',     # 蓝色 - 主要数据
    'secondary': '#ff7f0e',   # 橙色 - 对比数据
    'neutral': '#7f7f7f',     # 灰色 - 中性/背景/Chance line
    'highlight': '#d62728'    # 红色 - 用于强调
}
# ---------------------------------------------

# 将英寸转换为 Matplotlib 使用的 figsize
def cm_to_inch(value):
    return value / 2.54

# 定义单栏图尺寸 (8.5 cm)
SINGLE_COLUMN_WIDTH = cm_to_inch(8.5)

def plot_roc_curve(y_true, y_probs, output_path):
    """绘制并保存 Science 标准的 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    plt.plot(fpr, tpr, color=COLORS['primary'], lw=plt.rcParams['lines.linewidth'], label=f'Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=COLORS['neutral'], lw=1, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path)
    print(f"✅ (Science) ROC 曲线已保存至: {output_path}")
    plt.close()

def plot_pr_curve(y_true, y_probs, output_path):
    """绘制并保存 Science 标准的 PR 曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    plt.step(recall, precision, where='post', color=COLORS['secondary'], lw=plt.rcParams['lines.linewidth'], label=f'Model (AP = {avg_precision:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.02])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path)
    print(f"✅ (Science) PR 曲线已保存至: {output_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path, class_names=['Negative', 'Positive']):
    """绘制并保存 Science 标准的混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names, 
                annot_kws={"size": 10})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    print(f"✅ (Science) 混淆矩阵已保存至: {output_path}")
    plt.close()

def plot_calibration_curve(y_true, y_probs, output_path, n_bins=10):
    """绘制并保存 Science 标准的校准曲线"""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    brier = brier_score_loss(y_true, y_probs)

    plt.figure(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    plt.plot(prob_pred, prob_true, "s-", color=COLORS['primary'], label=f'Model (Brier Score = {brier:.3f})')
    plt.plot([0, 1], [0, 1], linestyle=":", color=COLORS['neutral'], label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability (Fraction of Positives)")
    plt.ylabel("Observed Fraction of Positives")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title('Calibration Curve')
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path)
    print(f"✅ (Science) 校准曲线已保存至: {output_path}")
    plt.close()

def plot_probability_distribution(y_true, y_probs, output_path):
    """绘制并保存 Science 标准的预测概率分布图"""
    df = pd.DataFrame({'Probability': y_probs, 'True Label': y_true})
    
    plt.figure(figsize=(cm_to_inch(11.4), cm_to_inch(8.5))) # 1.5栏图
    sns.kdeplot(data=df, x='Probability', hue='True Label', fill=True, 
                palette=[COLORS['highlight'], COLORS['primary']], common_norm=False)
    plt.xlabel("Predicted Probability (Positive Class)")
    plt.ylabel("Density")
    plt.title("Distribution of Predicted Probabilities")
    plt.legend(title='True Label', labels=['Positive (1)', 'Negative (0)'])
    plt.grid(False)
    plt.savefig(output_path)
    print(f"✅ (Science) 概率分布图已保存至: {output_path}")
    plt.close()

def generate_visualizations(preds_dir):
    """主函数：加载所有预测结果并生成所有图表"""
    # --- 清理并创建新的输出目录 ---
    output_dir = os.path.join(preds_dir, 'science_plots')
    if os.path.exists(output_dir):
        print(f"🗑️ 正在删除旧的图表目录: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"✨ 已创建新的图表目录: {output_dir}")
    
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

    print("\n--- 正在按照 Science 标准生成可视化图表 ---")

    plot_roc_curve(true_labels, positive_class_probs, os.path.join(output_dir, 'roc_curve.tiff'))
    plot_pr_curve(true_labels, positive_class_probs, os.path.join(output_dir, 'pr_curve.tiff'))
    plot_confusion_matrix(true_labels, pred_labels, os.path.join(output_dir, 'confusion_matrix.tiff'))
    plot_calibration_curve(true_labels, positive_class_probs, os.path.join(output_dir, 'calibration_curve.tiff'))
    plot_probability_distribution(true_labels, positive_class_probs, os.path.join(output_dir, 'probability_distribution.tiff'))

    print("\n🎉 所有出版级图表已成功生成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为模型的交叉验证结果生成专业的、Science标准的学术图表。")
    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="包含所有 'fold_k_predictions.csv' 文件的目录 (例如 'results/final_run/clinical_preds')"
    )
    args = parser.parse_args()
    generate_visualizations(args.preds_dir)