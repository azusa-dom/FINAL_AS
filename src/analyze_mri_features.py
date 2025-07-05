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
    feats, patient_ids, paths = [], [], []
    for fname in sorted(os.listdir(input_dir)):
        if "__aug" in fname:  # 跳过增强样本
            continue
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert('RGB')
        x = IMG_TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = extractor(x)
        vec = out.cpu().numpy().squeeze()
        feats.append(vec)
        # 从文件名提取患者ID (KNEE_1_... 或 SIJ_2_...)
        if fname.startswith(("KNEE", "SIJ")):
            pid = fname.split("_")[1]
        else:
            pid = "unknown"
        patient_ids.append(pid)
        paths.append(path)
    return np.vstack(feats), patient_ids, paths

def visualize(feats, patient_ids, output_path, method="tsne", perplexity=5.0, n_neighbors=15, min_dist=0.1, random_state=42):
    if method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("请先安装 umap-learn：pip install umap-learn")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        emb = reducer.fit_transform(feats)
    else:
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        emb = reducer.fit_transform(feats)
    # 配色
    COLORBLIND_PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999",
    ]
    unique_ids = sorted(set(patient_ids))
    colors = {uid: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)] for i, uid in enumerate(unique_ids)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for pid in unique_ids:
        mask = [x == pid for x in patient_ids]
        ax.scatter(np.array(emb)[mask, 0], np.array(emb)[mask, 1],
                   label=pid, color=colors[pid], edgecolor="black", lw=0.3, alpha=0.8, s=40)
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=14)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=14)
    ax.set_title("AS Patients MRI Feature Distribution", fontsize=16)
    ax.legend(title="Patient ID", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Visualization saved to {output_path}")
    return emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="1_AS 文件夹路径")
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
    feats, patient_ids, paths = extract_features(args.input_dir, extractor, device)
    emb = visualize(feats, patient_ids, os.path.join(args.output_dir, "as_mri_tsne.png"),
                    method=args.method, perplexity=args.perplexity, n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist, random_state=args.seed)
    # 可选：输出 npz 方便后续分析
    np.savez_compressed(os.path.join(args.output_dir, "as_mri_feats.npz"),
                        features=feats, patient_ids=np.array(patient_ids), paths=np.array(paths), embedding=emb)
    print(f"✅ Features & embeddings saved to {args.output_dir}/as_mri_feats.npz")

if __name__ == "__main__":
    main()
