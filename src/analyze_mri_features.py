#!/usr/bin/env python3
"""
analyze_mri_features.py

对 Engine 2 提取的 MRI 特征做定量分析并绘制黑/红风格柱状图：
- Silhouette Score (黑色)
- LogisticRegression 5-fold CV accuracy (红色)
- 保存结果到 JSON
- 绘制学术级黑白/红配色柱状图
"""

import os
import argparse
import json
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Quantitative analysis on extracted MRI features"
    )
    p.add_argument(
        "--feats-file", type=str,
        default="../outputs/mri_feats.npz",
        help="Path to .npz file with features, labels, paths"
    )
    p.add_argument(
        "--output-dir", type=str,
        default="../outputs",
        help="Directory to save analysis results"
    )
    p.add_argument(
        "--cv-folds", type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load features & labels
    data   = np.load(args.feats_file, allow_pickle=True)
    feats  = data["features"]
    labels = data["labels"]

    # 2) Binary labels: Healthy→0, AS→1
    y = np.array([1 if str(lab).startswith("1_") else 0 for lab in labels])

    # 3) Silhouette Score
    sil_score = silhouette_score(feats, y)
    print(f"Silhouette Score = {sil_score:.3f}")

    # 4) LogisticRegression CV
    clf    = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, feats, y,
                             cv=args.cv_folds,
                             scoring="accuracy")
    mean_acc = scores.mean()
    std_acc  = scores.std()
    print(f"{args.cv_folds}-fold CV accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

    # 5) Save JSON
    results = {
        "silhouette_score": float(sil_score),
        "cv_accuracy_mean": float(mean_acc),
        "cv_accuracy_std":  float(std_acc)
    }
    json_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to {json_path}")

    # 6) Academic-style black & red bar chart
    metrics = ["Silhouette", f"CV Acc ({args.cv_folds}-fold)"]
    values  = [sil_score, mean_acc]

    # 黑/红配色
    BAR_COLORS = ["black", "#D62728"]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(metrics, values,
                  color=BAR_COLORS,
                  edgecolor="black",
                  linewidth=0.8,
                  alpha=0.9)

    # 动态设置 y-limit，留出顶部空间防止数值重叠
    max_val = max(values)
    ax.set_ylim(0, max_val * 1.2)

    # 注释数值，黑色 bar 文字为白色，红色 bar 为黑色
    for bar, val, color in zip(bars, values, BAR_COLORS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max_val * 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="white" if color == "black" else "black"
        )

    # 样式化
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("MRI Feature Analysis Metrics", fontsize=16, pad=12)
    # 去除上/右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 加粗下/左边框
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    # y 方向网格线
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    # 刻度
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "analysis_results.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"✅ Plot saved to {fig_path}")

if __name__ == "__main__":
    main()
