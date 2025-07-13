#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_mri_slices_full.py

功能：
 1) ResNet-18 提取切片级深度特征
 2) t-SNE、UMAP（可选）与 PCA 三种二维可视化
 3) KDE：样本到各自类质心的欧氏距离分布

用法示例：
python viz_mri_slices_full.py \
  --as-dir      /Users/hydra/.../data/mri_AS \
  --healthy-dir /Users/hydra/.../data/mri_health/health1 \
  --healthy-dir /Users/hydra/.../data/mri_health/health2 \
  --save-dir    /Users/hydra/.../results/mri_viz \
  --device      cpu
"""
import os
import argparse
from glob import glob
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist

# 尝试导入 umap
try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--as-dir',      required=True, help='AS SIJ 切片根目录')
    p.add_argument('--healthy-dir', action='append', required=True,
                   help='健康 SIJ 切片根目录，可多次指定')
    p.add_argument('--save-dir',    required=True, help='结果保存目录')
    p.add_argument('--device',      default='cpu',   help='cpu 或 cuda')
    return p.parse_args()

def collect_paths(as_dir, healthy_dirs):
    exts = ('png','jpg','jpeg')
    paths, labels = [], []
    for ext in exts:
        found = glob(os.path.join(as_dir, '**', f'*.{ext}'), recursive=True)
        paths += found; labels += [1]*len(found)
    for hd in healthy_dirs:
        for ext in exts:
            found = glob(os.path.join(hd, '**', f'*.{ext}'), recursive=True)
            paths += found; labels += [0]*len(found)
    return paths, labels

def extract_feats(paths, device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    feats = []
    for p in tqdm(paths, desc='Extract feats'):
        img = Image.open(p).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x).cpu().numpy().reshape(-1)
        feats.append(f)
    return np.vstack(feats)

def plot_embedding(emb, labels, save_path, title):
    plt.figure(figsize=(6,6), dpi=300)
    sns.scatterplot(x=emb[:,0], y=emb[:,1],
                    hue=labels, palette=['#4C72B0','#DD8452'], s=25)
    plt.title(title, fontsize=14)
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.legend(title='Class', labels=['Healthy','AS'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(feats, labels, save_dir):
    n = len(feats)
    perp = min(30, max(5, (n - 1)//3))
    emb = TSNE(n_components=2, init='pca', random_state=42,
               perplexity=perp).fit_transform(feats)
    plot_embedding(emb, labels,
        os.path.join(save_dir,'tsne_slice_level.png'),
        f't-SNE (n={n}, perp={perp})')

def plot_umap(feats, labels, save_dir):
    if not _HAS_UMAP:
        print("⚠️  umap 未安装，跳过 UMAP 可视化")
        return
    emb = umap.UMAP(random_state=42).fit_transform(feats)
    plot_embedding(emb, labels,
        os.path.join(save_dir,'umap_slice_level.png'),
        'UMAP (random_state=42)')

def plot_pca(feats, labels, save_dir):
    emb = PCA(n_components=2, random_state=42).fit_transform(feats)
    plot_embedding(emb, labels,
        os.path.join(save_dir,'pca_slice_level.png'),
        'PCA (n_components=2)')

def plot_kde(feats, labels, save_dir):
    lab0 = feats[np.array(labels)==0]
    lab1 = feats[np.array(labels)==1]
    c0 = lab0.mean(axis=0)
    c1 = lab1.mean(axis=0)
    d0 = cdist(lab0, [c0], 'euclidean').flatten()
    d1 = cdist(lab1, [c1], 'euclidean').flatten()
    plt.figure(figsize=(6,4), dpi=300)
    sns.kdeplot(d0, fill=True, label='Healthy')
    sns.kdeplot(d1, fill=True, label='AS')
    plt.title('Distance to Class Centroid', fontsize=14)
    plt.xlabel('Euclidean Distance'); plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'kde_dist_centroid.png'))
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    paths, labels = collect_paths(args.as_dir, args.healthy_dir)
    if not paths:
        raise RuntimeError("❌ 未找到任何图片，请检查路径！")
    print(f"🖼️  Found {len(paths)} images ({sum(labels)} AS, {len(labels)-sum(labels)} Healthy)")

    feats = extract_feats(paths, args.device)

    print("🗺️  Generating t-SNE...")
    plot_tsne(feats, labels, args.save_dir)

    print("🌌 Generating UMAP / PCA...")
    plot_umap(feats, labels, args.save_dir)
    plot_pca(feats, labels, args.save_dir)

    print("📊 Generating KDE plot...")
    plot_kde(feats, labels, args.save_dir)

    print(f"✅ All plots saved under {args.save_dir}")

if __name__ == '__main__':
    main()