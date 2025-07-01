import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

from src.models import get_feature_extractor


def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image)


def extract_features(df, image_root, feature_extractor, device):
    features = []
    labels = []
    ids = []

    for _, row in df.iterrows():
        patient_id = str(row["patient_id"]).zfill(3)  # 保证 '001' 这种格式
        label = row["label"]
        img_path = os.path.join(image_root, f"{patient_id}.png")

        if not os.path.exists(img_path):
            print(f"⚠️ 跳过缺失图像: {img_path}")
            continue

        img_tensor = load_image(img_path).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = feature_extractor(img_tensor)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy()[0])
            labels.append(label)
            ids.append(patient_id)

    return np.array(features), np.array(labels), ids


def visualize_tsne(features, labels, out_path="tsne.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idxs = labels == label
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=f"Label {label}", alpha=0.7)
    plt.legend()
    plt.title("t-SNE of MRI Features")
    plt.savefig(out_path)
    print(f"✅ t-SNE 图像已保存至: {out_path}")


def main(args):
    df = pd.read_csv(args.csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = get_feature_extractor(device=device)

    features, labels, ids = extract_features(df, args.image_root, extractor, device)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "features.npy"), features)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    np.save(os.path.join(args.output_dir, "patient_ids.npy"), np.array(ids))

    visualize_tsne(features, labels, os.path.join(args.output_dir, "tsne.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV with patient_id,label")
    parser.add_argument("--image_root", type=str, required=True, help="Root folder where patient_id.png images are stored")
    parser.add_argument("--output_dir", type=str, default="outputs/mri_features", help="Where to save features and t-SNE plot")
    args = parser.parse_args()
    main(args)

