#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_tsne_as_vs_healthy.py
=================================
SCI-grade t-SNE visualisation of deep MRI features (AS vs Healthy).

• Inputs  : two root folders, one for AS cases, one for Healthy controls.
• Backend : ResNet-18 pretrained on ImageNet (fc removed → 512-d features).
• Outputs : tsne_as_vs_healthy.png  (slice-level)  ⭐主文推荐
            tsne_patient_mean.png    (subject-mean) ⭐可选补充

Author : <your name / institute>
Date   : 2025-07-05
"""

import os, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import models, transforms

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

IMG_EXTS = (".jpg", ".jpeg", ".png")   # 修改则同步改 collect() 判断


# ------------------------------------------------------------
# 0. CLI 解析
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="t-SNE visualisation of AS vs Healthy MRI slices")
    p.add_argument("--as-dir",      required=True,
                   help="Root dir of AS MRI images (n patients / sub-dirs)")
    p.add_argument("--healthy-dir", required=True,
                   help="Root dir of Healthy MRI images")
    p.add_argument("--outfile", default="tsne_as_vs_healthy.png",
                   help="Output PNG filename (slice-level)")
    p.add_argument("--agg", choices=["none", "mean"], default="none",
                   help="'mean' → aggregate per subject then t-SNE")
    p.add_argument("--device", default="cpu",
                   help="cuda or cpu (feature extraction only)")
    return p.parse_args()


# ------------------------------------------------------------
# 1. 收集文件 & 标签
# ------------------------------------------------------------
def collect(root_dir, label):
    """Return lists: paths, labels, patient_ids"""
    paths, labels, pids = [], [], []
    for root, dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, fn))
                labels.append(label)
                # patient id = immediate parent folder
                pids.append(os.path.basename(root))
    return paths, labels, pids


# ------------------------------------------------------------
# 2. 特征提取
# ------------------------------------------------------------
@torch.inference_mode()
def extract_feats(img_paths, device="cpu"):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()          # 512-d global features
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    for p in tqdm(img_paths, desc="Extracting feats"):
        img = Image.open(p).convert("RGB")
        x   = tf(img).unsqueeze(0).to(device)
        feat = model(x).cpu().numpy().ravel()
        feats.append(feat)
    return np.vstack(feats)   # [N, 512]


# ------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------
def main():
    args = parse_args()

    # 3.1 gather
    as_paths, as_labels, as_pids = collect(args.as_dir, "AS")
    h_paths,  h_labels,  h_pids  = collect(args.healthy_dir, "Healthy")

    paths   = as_paths + h_paths
    labels  = as_labels + h_labels
    pids    = as_pids  + h_pids

    print(f"[INFO] Found {len(paths)} images "
          f"({labels.count('AS')} AS, {labels.count('Healthy')} Healthy) "
          f"from {len(set(pids))} subjects")

    # 3.2 feats
    feats = extract_feats(paths, args.device)

    # 3.3 optional patient-mean aggregation
    if args.agg == "mean":
        df = pd.DataFrame({"pid": pids, "label": labels})
        feat_df = pd.DataFrame(feats)
        df_full = pd.concat([df, feat_df], axis=1)
        feats = (df_full.groupby("pid")
                         .mean()
                         .values)          # [n_subj, 512]
        labels = df_full.groupby("pid")["label"].first().tolist()
        pids   = df_full.groupby("pid").size().index.tolist()
        print(f"[INFO] Aggregated to {feats.shape[0]} subjects (mean vector)")

    # 3.4 t-SNE
    perp = max(5, min(30, feats.shape[0] // 2))
    emb  = TSNE(n_components=2, perplexity=perp,
                random_state=42, init="pca").fit_transform(feats)
    print(f"[INFO] t-SNE done, perplexity={perp}")

    # 3.5 plot
    df_plot = pd.DataFrame({
        "x": emb[:, 0], "y": emb[:, 1],
        "Label": labels, "Patient": pids
    })

    plt.figure(figsize=(6, 5), dpi=600)
    sns.scatterplot(data=df_plot, x="x", y="y",
                    hue="Label", style="Label",
                    palette={"AS": "tab:red", "Healthy": "tab:blue"},
                    s=60, edgecolor="k", linewidth=0.3)
    title = ("t-SNE of MRI Deep Features "
             f"({ 'patient-mean' if args.agg=='mean' else 'slice-level'})")
    plt.title(title, weight="bold", size=14)
    plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Class", frameon=False)
    plt.tight_layout()

    out_png = ("tsne_patient_mean.png" if args.agg == "mean" else args.outfile)
    plt.savefig(out_png)
    print(f"✔ Saved {out_png} (SCI-ready 600 dpi)")


if __name__ == "__main__":
    main()