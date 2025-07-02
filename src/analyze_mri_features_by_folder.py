import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# 标签映射（可读形式）
LABEL_MAP = {
    "png_AS_knee_all": "AS Knee",
    "png_AS_sij": "AS SIJ",
    "png_healthy_knee": "Healthy Knee",
    "png_healthy_sij": "Healthy SIJ"
}

# 图像预处理（标准 ImageNet）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def get_feature_extractor(model_name, device):
    model = getattr(models, model_name)(weights="DEFAULT")
    modules = list(model.children())[:-1]
    extractor = nn.Sequential(*modules).to(device)
    extractor.eval()
    return extractor

def extract_features(input_dir, extractor, device):
    feats, readable_labels, paths = [], [], []
    for subfolder in sorted(os.listdir(input_dir)):
        subpath = os.path.join(input_dir, subfolder)
        if not os.path.isdir(subpath):
            continue
        readable_label = LABEL_MAP.get(subfolder, subfolder)
        for fname in sorted(os.listdir(subpath)):
            if "__aug" in fname:
                continue
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            fpath = os.path.join(subpath, fname)
            img = Image.open(fpath).convert('RGB')
            x = IMG_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = extractor(x)
            vec = out.cpu().numpy().squeeze()
            feats.append(vec)
            readable_labels.append(readable_label)
            paths.append(fpath)
    return np.vstack(feats), readable_labels, paths

def visualize(feats, labels, output_path, method="tsne", perplexity=5.0, n_neighbors=15, min_dist=0.1, random_state=42):
    if method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("请先安装 umap-learn：pip install umap-learn")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        emb = reducer.fit_transform(feats)
    else:
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        emb = reducer.fit_transform(feats)

    plt.rcParams.update({'font.family': 'Arial'})  # 或 'Times New Roman'

    COLORBLIND_PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999",
    ]
    unique_labels = sorted(set(labels))
    colors = {uid: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)] for i, uid in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for label in unique_labels:
        mask = [x == label for x in labels]
        ax.scatter(np.array(emb)[mask, 0], np.array(emb)[mask, 1],
                   label=label, color=colors[label], edgecolor="black", lw=0.3, alpha=0.8, s=40)

    ax.set_title("Figure 2. t-SNE Visualization of MRI Features by Diagnostic Group", fontsize=16, weight='bold', pad=12)
    ax.set_xlabel(f"{method.upper()} Component 1", fontsize=14, weight='bold')
    ax.set_ylabel(f"{method.upper()} Component 2", fontsize=14, weight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=11, title_fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ SCI-style visualization saved to {output_path}")
    return emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="包含多个子文件夹（每个子文件夹是一个分类标签）的图像路径")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=5.0)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    extractor = get_feature_extractor(args.model, device)
    feats, labels, paths = extract_features(args.input_dir, extractor, device)
    emb = visualize(feats, labels, os.path.join(args.output_dir, "mri_tsne_by_label.png"),
                    method=args.method, perplexity=args.perplexity, n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist, random_state=args.seed)

    np.savez_compressed(os.path.join(args.output_dir, "mri_features_by_label.npz"),
                        features=feats, labels=np.array(labels), paths=np.array(paths), embedding=emb)
    print(f"✅ Features & embeddings saved to {args.output_dir}/mri_features_by_label.npz")

if __name__ == "__main__":
    main()
