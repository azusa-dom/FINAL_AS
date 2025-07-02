#!/usr/bin/env python3
"""
quantitative_mri_analysis.py

对预提取的 MRI 特征做定量分析：
1) Silhouette Score
2) Patient-group CV Accuracy (跳过只含单一类的折)
3) 保存 JSON 和绘制黑/红学术柱状图
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Quantitative MRI feature analysis (Silhouette & patient-group CV)"
    )
    p.add_argument(
        "--feats-file", type=str, default="../outputs/mri_feats.npz",
        help="Path to .npz with arrays 'features','labels','paths'"
    )
    p.add_argument(
        "--output-dir", type=str, default="../outputs",
        help="Directory to save JSON and figure"
    )
    p.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load data
    data   = np.load(args.feats_file, allow_pickle=True)
    feats  = data["features"]   # shape (N, D)
    labels = data["labels"]     # shape (N,), e.g. ['0_Healthy', '1_AS', ...]
    paths  = data["paths"]      # shape (N,), full paths to images

    # 2) Build binary target y and patient-group labels
    #    class folder at parent dirname: "0_Healthy" or "1_AS"
    class_folders = [os.path.basename(os.path.dirname(p)) for p in paths]
    y = np.array([int(cf.split("_")[0]) for cf in class_folders])  # 0 or 1

    #    patient ID from filename "KNEE_<pid>_...__augXX.png"
    basenames = [os.path.basename(p) for p in paths]
    pids = [bn.split("_")[1] for bn in basenames]
    #    group each sample by its unique patient (class + pid)
    groups = np.array([f"{cf.split('_')[0]}_{pid}"
                       for cf, pid in zip(class_folders, pids)])

    # 3) Silhouette Score (uses full dataset)
    sil_score = silhouette_score(feats, y)
    print(f"Silhouette Score = {sil_score:.3f}")

    # 4) Leave-One-Group-Out CV (skip folds with training set only one class)
    logo = LeaveOneGroupOut()
    accs = []
    skipped = 0
    for train_idx, test_idx in logo.split(feats, y, groups):
        y_tr = y[train_idx]
        # skip this fold if train has only one class
        if len(np.unique(y_tr)) < 2:
            skipped += 1
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(feats[train_idx], y_tr)
        preds = clf.predict(feats[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))

    if skipped:
        print(f"⚠️ Skipped {skipped} fold(s) with only one class in training set")

    mean_acc = float(np.mean(accs)) if accs else 0.0
    std_acc  = float(np.std(accs))  if accs else 0.0
    print(f"Patient-group CV accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

    # 5) Save JSON
    results = {
        "silhouette_score": sil_score,
        "patient_group_cv_mean_accuracy": mean_acc,
        "patient_group_cv_std_accuracy": std_acc,
        "n_valid_folds": len(accs),
        "n_skipped_folds": skipped
    }
    json_path = os.path.join(args.output_dir, "mri_quant_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Metrics saved to {json_path}")

    # 6) Plot academic-style black/red bar chart
    metrics = ["Silhouette", "Patient-CV Acc"]
    values  = [sil_score, mean_acc]
    colors  = ["black", "#D62728"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metrics, values,
                  color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)

    top = max(values) * 1.2
    ax.set_ylim(0, top)
    for bar, val, col in zip(bars, values, colors):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + top*0.03,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=12,
            color="white" if col == "black" else "black"
        )

    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("MRI Feature Analysis Metrics", fontsize=16, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "mri_quant_results.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"✅ Figure saved to {fig_path}")

if __name__ == "__main__":
    main()
