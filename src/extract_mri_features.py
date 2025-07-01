#!/usr/bin/env python3
"""
Engine 2: Exploratory MRI Feature Analysis
- Load pretrained CNN (ResNet) as feature extractor
- Traverse `--input-dir` 下每个子文件夹（patient ID），提取所有切片的高维特征
- 用 t-SNE 或 UMAP 将特征降到 2D，并按 patient 上色可视化
- 学术化排版并使用色盲安全配色
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 尝试导入 UMAP
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ImageNet 预处理（保持与训练时相同）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def get_feature_extractor(model_name: str, device: torch.device) -> nn.Module:
    """加载预训练模型并移除分类头，返回 feature extractor"""
    model = getattr(models, model_name)(pretrained=True)
    modules = list(model.children())[:-1]  # 去掉最后的 fc
    extractor = nn.Sequential(*modules).to(device)
    extractor.eval()
    return extractor

def extract_features(input_dir: str,
                     extractor: nn.Module,
                     device: torch.device):
    """
    遍历 input_dir 下的子文件夹（patient ID），提取每张图像的特征。
    Returns:
        feats: np.ndarray (N, D)
        labels: list of patient IDs
        paths: list of image paths
    """
    feats, labels, paths = [], [], []
    patients = sorted(d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d)))
    for pid in patients:
        pdir = os.path.join(input_dir, pid)
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                continue
            img = Image.open(os.path.join(pdir, fname)).convert('RGB')
            x = IMG_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = extractor(x)  # [1, C, 1, 1]
            vec = out.cpu().numpy().squeeze()  # [C]
            feats.append(vec)
            labels.append(pid)
            paths.append(os.path.join(pdir, fname))
    return np.vstack(feats), labels, paths

def visualize(feats: np.ndarray,
              labels: list,
              output_path: str,
              method: str = "tsne",
              perplexity: float = 5.0,
              n_neighbors: int = 15,
              min_dist: float = 0.1,
              random_state: int = 42):
    """用 t-SNE 或 UMAP 将 feats 降到 2D 并保存学术化彩色散点图"""
    # 降维
    if method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("请先安装 umap-learn: pip install umap-learn")
        reducer = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            random_state=random_state)
        emb = reducer.fit_transform(feats)
    else:
        reducer = TSNE(n_components=2,
                       perplexity=perplexity,
                       random_state=random_state)
        emb = reducer.fit_transform(feats)

    # 色盲安全调色板（Colorbrewer）
    COLORBLIND_PALETTE = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
    ]
    unique_ids = sorted(set(labels))
    colors = {uid: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
              for i, uid in enumerate(unique_ids)}

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    for pid in unique_ids:
        mask = [lab == pid for lab in labels]
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=pid,
                   color=colors[pid],
                   edgecolor="black",
                   lw=0.3,
                   alpha=0.8,
                   s=40)

    # 学术化排版
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=14)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=14)
    ax.set_title(f"Feature {method.upper()} Projection", fontsize=16)
    ax.legend(title="Patient ID",
              bbox_to_anchor=(1.02, 1),
              loc='upper left',
              frameon=False,
              fontsize=12,
              title_fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.grid(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=True, labelbottom=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Visualization saved to {output_path}")
    return emb

def save_features(output_dir: str,
                  feats: np.ndarray,
                  labels: list,
                  paths: list,
                  emb: np.ndarray):
    """保存特征、labels、paths 及降维结果到 .npz"""
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "mri_feats.npz"),
        features=feats,
        labels=np.array(labels, dtype='<U10'),
        paths=np.array(paths, dtype='<U128'),
        embedding=emb
    )
    print(f"✅ Features & embeddings saved to {output_dir}/mri_feats.npz")

def main():
    parser = argparse.ArgumentParser(
        description="Extract & visualize MRI features with pretrained CNN"
    )
    parser.add_argument("--input-dir",  required=True,
                        help="MRI 数据目录，每个子文件夹代表一名患者")
    parser.add_argument("--output-dir", default="outputs",
                        help="保存特征和可视化结果的目录")
    parser.add_argument("--model", choices=["resnet18","resnet34",
                                             "resnet50","resnet101"],
                        default="resnet50",
                        help="选择 CNN backbone")
    parser.add_argument("--method", choices=["tsne","umap"],
                        default="tsne",
                        help="降维方法")
    parser.add_argument("--perplexity", type=float, default=5.0,
                        help="t-SNE perplexity")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    extractor = get_feature_extractor(args.model, device)
    feats, labels, paths = extract_features(args.input_dir,
                                             extractor, device)

    emb = visualize(feats, labels,
                    os.path.join(args.output_dir,
                                 f"mri_{args.method}.png"),
                    method=args.method,
                    perplexity=args.perplexity,
                    n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist,
                    random_state=args.seed)

    save_features(args.output_dir, feats, labels, paths, emb)

if __name__ == "__main__":
    main()
