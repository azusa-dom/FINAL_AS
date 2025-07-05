import numpy as np
import nibabel as nib
import os
import re
import matplotlib.pyplot as plt

# --- 配置路径 ---
input_dir = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/mri_image_raw/sacroiliac_joint"
nii_output_dir = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/nii_sij"
png_output_dir = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/png_sij_clean"
os.makedirs(nii_output_dir, exist_ok=True)
os.makedirs(png_output_dir, exist_ok=True)


import os
import sys
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Image preprocessing ---
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class FeatureExtractor(nn.Module):
    """ResNet18-based feature extractor (removes final fc layer)."""
    def __init__(self, device):
        super().__init__()
        # Use updated weights parameter
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1]).to(device)
        self.encoder.eval() # Set to evaluation mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.encoder(x)       # [B, 512, 1, 1]
        return feat.view(feat.size(0), -1)  # [B, 512]

def extract_subject_features(dirs_labels, device, batch_size=8):
    """
    Scans all given directories recursively, extracts and aggregates features per subject.
    Args:
        dirs_labels (list): A list of (root_dir, label) tuples.
        device (torch.device): The device to run the model on.
        batch_size (int): Batch size for processing images.
    Returns:
        A tuple containing:
        - X (np.ndarray): Feature matrix of shape (n_subjects, 512).
        - y (np.ndarray): Label vector of shape (n_subjects,).
        - ids (list): A list of subject IDs.
    """
    extractor = FeatureExtractor(device)
    jobs = []  # list of (img_path, subject_id, label)
    exts = (".png", ".jpg", ".jpeg")

    print("📁 Scanning image directories...")
    for root_dir, label in dirs_labels:
        if not os.path.isdir(root_dir):
            print(f"⚠️ Warning: Directory not found, skipping: {root_dir}", file=sys.stderr)
            continue
        for root, _, files in os.walk(root_dir):
            # Assumes subject ID is the name of the directory containing the images
            sid = os.path.basename(root)
            # Ensure we are in a subject-specific subdirectory
            if sid == os.path.basename(root_dir):
                continue
            for fn in files:
                if fn.lower().endswith(exts):
                    jobs.append((os.path.join(root, fn), sid, label))

    if not jobs:
        print("❌ Error: No images found. Please check your --healthy-dir and --as-dir paths.", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Found {len(jobs)} images across {len(set(s for _, s, _ in jobs))} potential subjects.")
    
    # Group images by subject
    subject_images = {}
    for path, sid, label in jobs:
        subject_images.setdefault(sid, {'paths': [], 'label': label})['paths'].append(path)

    feats = {}
    labels = {}

    try:
        with torch.no_grad():
            for sid, data in tqdm(subject_images.items(), desc="🖼️  Extracting features per subject"):
                subject_feats = []
                img_paths = data['paths']
                label = data['label']
                
                for i in range(0, len(img_paths), batch_size):
                    batch_paths = img_paths[i:i+batch_size]
                    imgs = [IMG_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]
                    batch_t = torch.stack(imgs).to(device)
                    out = extractor(batch_t).cpu().numpy()
                    subject_feats.append(out)
                
                # Aggregate features for the subject (mean pooling)
                all_feats_np = np.vstack(subject_feats)
                feats[sid] = np.mean(all_feats_np, axis=0)
                labels[sid] = label

    except KeyboardInterrupt:
        print("\n⚠️ Extraction aborted by user.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ Error: Image file not found. {e}", file=sys.stderr)
        sys.exit(1)


    # Final assembly
    ids = list(feats.keys())
    X = np.array([feats[sid] for sid in ids])
    y = np.array([labels[sid] for sid in ids])
    
    return X, y, ids

def compute_distance_permutation_test(X: np.ndarray, y: np.ndarray, n_perm: int, seed: int):
    """
    Computes observed centroid distance and p-value from a permutation test.
    Returns:
        A tuple containing:
        - d_obs (float): The observed Euclidean distance between class centroids.
        - pval (float): The calculated p-value.
        - perm_distances (list): A list of distances from all permutations.
    """
    # Define labels for clarity
    HEALTHY_LABEL, AS_LABEL = 0, 1
    
    # Observed distance
    mu_healthy = X[y == HEALTHY_LABEL].mean(axis=0)
    mu_as = X[y == AS_LABEL].mean(axis=0)
    d_obs = np.linalg.norm(mu_healthy - mu_as)

    # Permutation test
    rng = np.random.RandomState(seed)
    perm_distances = []
    y_shuffled = np.copy(y)
    
    for _ in tqdm(range(n_perm), desc="🔀 Running permutations"):
        rng.shuffle(y_shuffled)
        mu0_p = X[y_shuffled == HEALTHY_LABEL].mean(axis=0)
        mu1_p = X[y_shuffled == AS_LABEL].mean(axis=0)
        perm_distances.append(np.linalg.norm(mu0_p - mu1_p))
    
    # Calculate p-value: (number of times permuted distance >= observed + 1) / (N + 1)
    count_extreme = np.sum(np.array(perm_distances) >= d_obs)
    pval = (count_extreme + 1) / (n_perm + 1)
    
    return d_obs, pval, perm_distances

def plot_permutation_results(d_obs, pval, perm_distances, output_file):
    """
    Generates and saves a publication-quality plot of the permutation test results.
    """
    print(f"🎨 Generating plot and saving to {output_file}...")
    
    # Set plot style for academic publications
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))

    # Plot the histogram of permuted distances
    ax = sns.histplot(x=perm_distances, kde=True, color="skyblue", stat="density")
    
    # Add a vertical line for the observed distance
    plt.axvline(d_obs, color='red', linestyle='--', linewidth=2, 
                label=f'Observed Distance\n($d_{{obs}} = {d_obs:.4f}$)')

    # Add text for the p-value
    # Place text based on data range to avoid overlap
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    text_x = d_obs + (xmax - xmin) * 0.05
    if text_x > xmax * 0.9: # Adjust if too far right
        text_x = xmin + (xmax-xmin) * 0.7
        
    plt.text(text_x, ymax * 0.8, f'$p$-value = {pval:.4f}', 
             fontdict={'size': 14, 'weight': 'bold', 'color': 'black'})

    # Add titles and labels
    plt.title('Permutation Test for Feature Space Separation', fontsize=16, fontweight='bold')
    plt.xlabel('Euclidean Distance Between Group Centroids', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    
    # Ensure layout is tight and save the figure with high resolution
    plt.tight_layout()
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot successfully saved.")
    except Exception as e:
        print(f"❌ Failed to save plot: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="SIJ Concept Validation via Distance-based Permutation Test.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--healthy-dir", action="append", required=True,
                        help="Root directory of healthy subjects (scanned recursively).\nEach subject's images should be in a separate subdirectory.")
    parser.add_argument("--as-dir", action="append", required=True,
                        help="Root directory of AS subjects (scanned recursively).\nEach subject's images should be in a separate subdirectory.")
    parser.add_argument("--output-file", type=str, default="permutation_test.png",
                        help="Path to save the output plot figure (e.g., 'results/test.png').")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for feature extraction.")
    parser.add_argument("--n-perm", type=int, default=5000,
                        help="Number of permutations for the test.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device ('cuda' or 'cpu').")
    args = parser.parse_args()

    # --- Setup ---
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"⚙️ Using device: {device}, Seed: {args.seed}")
    
    # Prepare directory list with labels (0 for healthy, 1 for AS)
    dirs_labels = [(d, 0) for d in args.healthy_dir] + [(d, 1) for d in args.as_dir]

    # --- Feature Extraction ---
    X, y, ids = extract_subject_features(dirs_labels, device, args.batch_size)
    n_healthy = np.sum(y == 0)
    n_as = np.sum(y == 1)
    
    if n_healthy == 0 or n_as == 0:
        print(f"❌ Error: Not enough subjects found. Healthy: {n_healthy}, AS: {n_as}. Need at least one of each.", file=sys.stderr)
        sys.exit(1)
        
    print(f"📊 Extracted features for {len(ids)} subjects (Healthy={n_healthy}, AS={n_as}).")

    # --- Permutation Test ---
    d_obs, pval, perm_distances = compute_distance_permutation_test(X, y, args.n_perm, args.seed)
    
    print("\n--- Results ---")
    print(f"▶ Observed Centroid Distance ($d_{{obs}}$): {d_obs:.4f}")
    print(f"▶ Permutation Test $p$-value: {pval:.4f} (based on {args.n_perm} permutations)")

    # --- Visualization ---
    if args.output_file:
        plot_permutation_results(d_obs, pval, perm_distances, args.output_file)

if __name__ == "__main__":
    main()