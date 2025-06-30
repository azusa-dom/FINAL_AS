#!/usr/bin/env python3
"""
Extracts deep features from MRI slice images using a pre-trained ResNet50 model
and visualizes the feature distribution using t-SNE.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Local imports from your project structure
from src.models import get_feature_extractor


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MRI feature extraction and visualization")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="AS_Finetune_Data_balanced",
        help="Directory with preprocessed MRI slice images, organized in subfolders by class.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/mri_features",
        help="Directory to save extracted features and the t-SNE plot.",
    )
    parser.add_argument(
        "--plot_title",  ### NEW/MODIFIED ###
        type=str,
        default="t-SNE Visualization of MRI Deep Features",
        help="Title for the output t-SNE plot.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    """Main function to run the feature extraction and visualization pipeline."""
    args = parse_args()
    
    # --- Setup and Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  ### NEW/MODIFIED ###
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Initializing script with seed {args.seed} on device {device}.")
    print(f"[INFO] Output will be saved to: {out_dir}")

    # --- Model and Data Loading ---
    extractor = get_feature_extractor().to(device)

    # Define standard ImageNet transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load data using ImageFolder, which automatically finds classes from subdirectories
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ### NEW/MODIFIED ###
    print(f"[INFO] Found {len(dataset)} images in {len(dataset.classes)} classes: {dataset.class_to_idx}")
    if len(dataset) == 0:
        print(f"[ERROR] No images found in the specified directory: {args.data_dir}")
        return

    # --- Feature Extraction ---
    print("[INFO] Starting feature extraction...")
    all_feats, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feats = extractor(imgs).view(imgs.size(0), -1)  # Flatten the features
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            print(f"\r  -> Processing batch {i+1}/{len(loader)}", end="") ### NEW/MODIFIED ###
    
    print("\n[INFO] Feature extraction complete.")

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Save features and labels for potential future use
    np.save(out_dir / "mri_features.npy", feats)
    np.save(out_dir / "mri_labels.npy", labels)
    print(f"[INFO] Saved extracted features and labels to {out_dir}")

    # --- t-SNE Visualization ---
    ### NEW/MODIFIED ###
    # Dynamically set perplexity to be less than the number of samples
    n_samples = feats.shape[0]
    perplexity_value = min(30.0, float(n_samples - 1))
    if perplexity_value <= 0:
        print("[ERROR] Cannot run t-SNE with 1 or fewer samples.")
        return
        
    print(f"[INFO] Running t-SNE with perplexity = {perplexity_value:.1f}...")
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=perplexity_value, n_iter=1000)
    reduced = tsne.fit_transform(feats)

    # --- Plotting ---
    print("[INFO] Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid') ### NEW/MODIFIED ###
    fig, ax = plt.subplots(figsize=(10, 8)) ### NEW/MODIFIED ###
    
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('jet', len(unique_labels)) ### NEW/MODIFIED ###

    for i, lab_val in enumerate(unique_labels):
        idx = labels == lab_val
        # Use class names from the dataset for the legend
        class_name = dataset.classes[lab_val] ### NEW/MODIFIED ###
        ax.scatter(reduced[idx, 0], reduced[idx, 1], s=50, alpha=0.8, label=class_name, color=colors(i)) ### NEW/MODIFIED ###
        
    ax.set_title(args.plot_title, fontsize=16, weight='bold') ### NEW/MODIFIED ###
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12) ### NEW/MODIFIED ###
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12) ### NEW/MODIFIED ###
    ax.legend(title="Classes", fontsize=10) ### NEW/MODIFIED ###
    fig.tight_layout() ### NEW/MODIFIED ###
    
    plot_path = out_dir / "tsne_visualization.png"
    plt.savefig(plot_path, dpi=300) ### NEW/MOMDIFIED ###
    plt.close()
    
    print(f"[SUCCESS] t-SNE plot saved to {plot_path}")


if __name__ == "__main__":
    main()
