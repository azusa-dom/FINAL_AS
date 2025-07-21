#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_mri_slices_full.py

功能：
 1) ResNet-18 提取切片级深度特征
 2) t-SNE、UMAP（可选）与 PCA/KernelPCA 多种二维可视化
 3) KDE：样本到各自类质心的欧氏距离、余弦距离、马氏距离分布对比
 4) 量化评估 t-SNE/UMAP 投影后的簇可分性
 5) UMAP 参数敏感性扫描及最佳参数选择

用法示例：
# 运行所有可视化模式 (默认输出最优 UMAP 和 KDE)
python src/mri_src/visualization/viz_mri_slices_full.py \
  --as-dir data/mri_AS \
  --healthy-dir data/mri_health/health1 \
  --healthy-dir data/mri_health/health2 \
  --save-dir results/mri_visualization_full \
  --device cpu

# 仅运行 t-SNE 超参数循环
python src/mri_src/visualization/viz_mri_slices_full.py \
  --as-dir data/mri_AS \
  --healthy-dir data/mri_health/health1 \
  --healthy-dir data/mri_health/health2 \
  --save-dir results/tsne_tuning \
  --device cpu --mode tsne_tuning

# 仅运行 KDE 多距离对比 (包含正则化马氏距离)
python src/mri_src/visualization/viz_mri_slices_full.py \
  --as-dir data/mri_AS \
  --healthy-dir data/mri_health/health1 \
  --healthy-dir data/mri_health/health2 \
  --save-dir results/kde_distances \
  --device cpu --mode kde_distances

# 仅运行 UMAP 自动调优
python src/mri_src/visualization/viz_mri_slices_full.py \
  --as-dir data/mri_AS \
  --healthy-dir data/mri_health/health1 \
  --healthy-dir data/mri_health/health2 \
  --save-dir results/umap_tuning \
  --device cpu --mode umap_tuning

# 仅运行降维前置 PCA 和 Kernel PCA
python src/mri_src/visualization/viz_mri_slices_full.py \
  --as-dir data/mri_AS \
  --healthy-dir data/mri_health/health1 \
  --healthy-dir data/mri_health/health2 \
  --save-dir results/pca_kpca \
  --device cpu --mode pca_kpca
"""
import os
import argparse
from glob import glob
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans # For silhouette score
from sklearn.covariance import ShrunkCovariance # For regularized Mahalanobis distance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist # For Euclidean, Cosine
from scipy.stats import ks_2samp
from scipy.linalg import pinv # For Mahalanobis distance (if needed as fallback)

# Try importing umap
try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

# Global list to store quantitative results for the table
QUANT_RESULTS = []

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--as-dir',      required=True, help='AS SIJ 切片根目录')
    p.add_argument('--healthy-dir', action='append', required=True,
                   help='健康 SIJ 切片根目录，可多次指定')
    p.add_argument('--save-dir',    required=True, help='结果保存目录')
    p.add_argument('--device',      default='cpu',   help='cpu 或 cuda')
    p.add_argument('--mode',        type=str, default='all',
                   choices=['all', 'tsne_tuning', 'kde_distances', 'umap_tuning', 'pca_kpca'],
                   help='选择运行模式：all, tsne_tuning, kde_distances, umap_tuning, pca_kpca')
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

def get_silhouette_score(embedding, labels_true):
    """
    Computes Silhouette Score for 2-cluster KMeans on the embedding.
    Handles edge cases like single cluster or insufficient samples.
    """
    try:
        if len(np.unique(labels_true)) < 2:
            return np.nan # Cannot compute if only one true class
        
        # Silhouette score requires at least 2 clusters and > 1 sample per cluster
        # KMeans can sometimes produce less than k clusters if data points are identical
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels_pred = kmeans.fit_predict(embedding)

        if len(np.unique(cluster_labels_pred)) < 2: # KMeans did not find 2 distinct clusters
            return np.nan 
        
        # Check if any cluster is too small for silhouette calculation (must have >= 2 samples)
        if (np.bincount(cluster_labels_pred) < 2).any():
            return np.nan

        score = silhouette_score(embedding, cluster_labels_pred)
        return score
    except Exception as e:
        # print(f"⚠️  Could not compute Silhouette Score: {e}") # Suppress frequent warnings
        return np.nan


def plot_embedding(emb, labels, save_path, title, silhouette_val=None):
    plt.figure(figsize=(6,6), dpi=300)
    sns.scatterplot(x=emb[:,0], y=emb[:,1],
                    hue=labels, palette=['#4C72B0','#DD8452'], s=25)
    
    full_title = title
    if silhouette_val is not None and not np.isnan(silhouette_val):
        full_title += f'\nSilhouette: {silhouette_val:.3f}'
    
    plt.title(full_title, fontsize=14)
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.legend(title='Class', labels=['Healthy','AS'])
    plt.tight_layout()
    plt.savefig(save_path, format='svg') # Save as SVG
    plt.close()


def plot_tsne_tuning(feats, labels, save_dir, perplexity_values=[5, 10, 15]):
    fig, axes = plt.subplots(1, len(perplexity_values), figsize=(6 * len(perplexity_values), 6), dpi=300)
    if len(perplexity_values) == 1:
        axes = [axes]

    for i, perp in enumerate(perplexity_values):
        ax = axes[i]
        n_samples = len(feats)
        current_perp = min(perp, max(5, n_samples - 1))
        if current_perp != perp:
            print(f"Adjusting perplexity from {perp} to {current_perp} due to insufficient samples.")
        
        print(f"🗺️  Generating t-SNE with perplexity={current_perp}...")
        try:
            emb = TSNE(n_components=2, init='pca', random_state=42,
                       perplexity=current_perp, learning_rate=200, n_iter=1000,
                       n_jobs=-1).fit_transform(feats)
            
            silhouette_val = get_silhouette_score(emb, labels)
            
            QUANT_RESULTS.append({
                'Method': 't-SNE',
                'Parameters': f'perp={current_perp}',
                'Silhouette Score': f'{silhouette_val:.3f}' if not np.isnan(silhouette_val) else 'N/A',
                'KS P-value': 'N/A' # Add placeholder for consistent keys
            })

            sns.scatterplot(x=emb[:,0], y=emb[:,1],
                            hue=labels, palette=['#4C72B0','#DD8452'], s=25, ax=ax)
            
            full_title = f't-SNE (perplexity={current_perp})'
            if not np.isnan(silhouette_val):
                full_title += f'\nSilhouette: {silhouette_val:.3f}'
            
            ax.set_title(full_title, fontsize=12)
            ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
            ax.legend(title='Class', labels=['Healthy','AS'])
            ax.grid(alpha=0.3)
        except Exception as e:
            print(f"⚠️  t-SNE with perplexity={current_perp} failed: {e}")
            ax.set_title(f't-SNE (perp={current_perp}) - Error')
            ax.text(0.5, 0.5, 'Plotting Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            QUANT_RESULTS.append({ # Add result for failed plot as well
                'Method': 't-SNE',
                'Parameters': f'perp={current_perp}',
                'Silhouette Score': 'Error',
                'KS P-value': 'N/A'
            })


    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tsne_perplexity_tuning.svg')
    plt.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"✅ t-SNE tuning plots saved to: {save_path}")


def plot_umap_tuning(feats, labels, save_dir, n_neighbors_grid=[3,5,10], min_dist_grid=[0.01,0.1,0.5]):
    if not _HAS_UMAP:
        print("⚠️  umap 未安装，跳过 UMAP 可视化")
        return
    
    best_silhouette = -np.inf # Initialize with negative infinity
    best_params = {}
    best_emb = None

    fig_rows = len(n_neighbors_grid)
    fig_cols = len(min_dist_grid)
    fig, axes = plt.subplots(fig_rows, fig_cols, 
                             figsize=(6 * fig_cols, 6 * fig_rows), dpi=300)
    
    # Ensure axes is always 2D array
    if fig_rows == 1 and fig_cols == 1:
        axes = np.array([[axes]])
    elif fig_rows == 1:
        axes = axes[np.newaxis, :]
    elif fig_cols == 1:
        axes = axes[:, np.newaxis]

    for r, n_neighbors in enumerate(n_neighbors_grid):
        for c, min_dist in enumerate(min_dist_grid):
            ax = axes[r, c]
            print(f"🌌 Generating UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
            try:
                # Ensure n_neighbors does not exceed n_samples
                current_n_neighbors = min(n_neighbors, len(feats) - 1)
                if current_n_neighbors < 1: current_n_neighbors = 1 # Ensure n_neighbors >= 1 for UMAP
                if current_n_neighbors != n_neighbors:
                    print(f"Adjusting n_neighbors from {n_neighbors} to {current_n_neighbors} due to insufficient samples.")

                emb = umap.UMAP(n_neighbors=current_n_neighbors, min_dist=min_dist, 
                                random_state=42).fit_transform(feats)

                silhouette_val = get_silhouette_score(emb, labels)

                QUANT_RESULTS.append({
                    'Method': 'UMAP',
                    'Parameters': f'n_n={n_neighbors}, m_d={min_dist}',
                    'Silhouette Score': f'{silhouette_val:.3f}' if not np.isnan(silhouette_val) else 'N/A',
                    'KS P-value': 'N/A' # Add placeholder for consistent keys
                })

                if not np.isnan(silhouette_val) and silhouette_val > best_silhouette:
                    best_silhouette = silhouette_val
                    best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
                    best_emb = emb # Store the best embedding

                sns.scatterplot(x=emb[:,0], y=emb[:,1],
                                hue=labels, palette=['#4C72B0','#DD8452'], s=25, ax=ax)
                
                full_title = f'UMAP (n_n={n_neighbors}, m_d={min_dist})'
                if not np.isnan(silhouette_val):
                    full_title += f'\nSilhouette: {silhouette_val:.3f}'

                ax.set_title(full_title, fontsize=12)
                ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
                ax.legend(title='Class', labels=['Healthy','AS'])
                ax.grid(alpha=0.3)
            except Exception as e:
                print(f"⚠️  UMAP with n_n={n_neighbors}, m_d={min_dist} failed: {e}")
                ax.set_title(f'UMAP (n_n={n_neighbors}, m_d={min_dist}) - Error')
                ax.text(0.5, 0.5, 'Plotting Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                QUANT_RESULTS.append({ # Add result for failed plot as well
                    'Method': 'UMAP',
                    'Parameters': f'n_n={n_neighbors}, m_d={min_dist}',
                    'Silhouette Score': 'Error',
                    'KS P-value': 'N/A'
                })


    plt.tight_layout()
    save_path_grid = os.path.join(save_dir, 'umap_grid_tuning.svg')
    plt.savefig(save_path_grid, format='svg')
    plt.close(fig)
    print(f"✅ UMAP grid tuning plots saved to: {save_path_grid}")

    if best_params:
        print(f"\n✨ Best UMAP parameters found: {best_params} with Silhouette Score: {best_silhouette:.3f}")
        # Plotting the best UMAP separately for clear reporting
        plot_embedding(best_emb, labels, 
                       os.path.join(save_dir, f'umap_best_nn{best_params["n_neighbors"]}_md{best_params["min_dist"]}.svg'),
                       f'UMAP (Best Params: n_n={best_params["n_neighbors"]}, m_d={best_params["min_dist"]})',
                       best_silhouette)
        print(f"✅ Best UMAP plot saved to: {os.path.join(save_dir, f'umap_best_nn{best_params["n_neighbors"]}_md{best_params["min_dist"]}.svg')}")
    else:
        print("Could not find a best UMAP configuration with a valid Silhouette Score.")


def plot_pca_and_kernel_pca(feats_standardized, labels, save_dir):
    # Standard PCA
    print("🌌 Generating PCA (2 components) plot...")
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(feats_standardized)
    explained_variance_pca = pca.explained_variance_ratio_
    silhouette_pca = get_silhouette_score(emb_pca, labels)
    
    title_pca = (f'PCA (n_components=2)\n'
                 f'Explained Variance: PC1={explained_variance_pca[0]*100:.1f}%, '
                 f'PC2={explained_variance_pca[1]*100:.1f}%')
    
    QUANT_RESULTS.append({
        'Method': 'PCA',
        'Parameters': 'n_comp=2',
        'Silhouette Score': f'{silhouette_pca:.3f}' if not np.isnan(silhouette_pca) else 'N/A',
        'KS P-value': 'N/A' # Add placeholder for consistent keys
    })
    
    plot_embedding(emb_pca, labels, os.path.join(save_dir,'pca_slice_level.svg'), title_pca, silhouette_pca)

    # Kernel PCA
    print("🌌 Generating Kernel PCA (RBF) plot...")
    try:
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=None, random_state=42) 
        emb_kpca = kpca.fit_transform(feats_standardized)
        
        silhouette_kpca = get_silhouette_score(emb_kpca, labels)
        
        title_kpca = 'Kernel PCA (kernel=RBF, n_components=2)'
        QUANT_RESULTS.append({
            'Method': 'Kernel PCA',
            'Parameters': 'kernel=rbf, n_comp=2',
            'Silhouette Score': f'{silhouette_kpca:.3f}' if not np.isnan(silhouette_kpca) else 'N/A',
            'KS P-value': 'N/A' # Add placeholder for consistent keys
        })
        plot_embedding(emb_kpca, labels, os.path.join(save_dir,'kernel_pca_slice_level.svg'), 
                       title_kpca, silhouette_kpca)
    except Exception as e:
        print(f"⚠️  Kernel PCA failed: {e}")
        QUANT_RESULTS.append({ # Add result for failed plot as well
            'Method': 'Kernel PCA',
            'Parameters': 'kernel=rbf, n_comp=2',
            'Silhouette Score': 'Error',
            'KS P-value': 'N/A'
        })


def plot_kde_multi_distance(feats, labels, save_dir):
    lab0 = feats[np.array(labels)==0]
    lab1 = feats[np.array(labels)==1]
    
    # Calculate centroids
    c0 = lab0.mean(axis=0)
    c1 = lab1.mean(axis=0)

    plt.figure(figsize=(18, 5), dpi=300) # Wider figure for three subplots
    
    # --- Euclidean Distance ---
    ax1 = plt.subplot(1, 3, 1)
    d0_euclidean = cdist(lab0, [c0], 'euclidean').flatten()
    d1_euclidean = cdist(lab1, [c1], 'euclidean').flatten()
    ks_stat_euclidean, p_value_euclidean = ks_2samp(d0_euclidean, d1_euclidean)
    
    sns.kdeplot(d0_euclidean, fill=True, label='Healthy', ax=ax1)
    sns.kdeplot(d1_euclidean, fill=True, label='AS', ax=ax1)
    ax1.set_title('Euclidean Distance to Centroid', fontsize=12)
    ax1.set_xlabel('Distance'); ax1.set_ylabel('Density')
    ax1.legend()
    ax1.annotate(f'KS p = {p_value_euclidean:.2e}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
    QUANT_RESULTS.append({
        'Method': 'KDE (Euclidean)',
        'Parameters': 'to centroid',
        'Silhouette Score': 'N/A', # Add placeholder for consistent keys
        'KS P-value': f'{p_value_euclidean:.2e}'
    })


    # --- Cosine Distance (1 - cosine similarity) ---
    ax2 = plt.subplot(1, 3, 2)
    d0_cosine = cdist(lab0, [c0], 'cosine').flatten()
    d1_cosine = cdist(lab1, [c1], 'cosine').flatten()
    ks_stat_cosine, p_value_cosine = ks_2samp(d0_cosine, d1_cosine)
    
    sns.kdeplot(d0_cosine, fill=True, label='Healthy', ax=ax2)
    sns.kdeplot(d1_cosine, fill=True, label='AS', ax=ax2)
    ax2.set_title('Cosine Distance to Centroid', fontsize=12)
    ax2.set_xlabel('Distance'); ax2.set_ylabel('Density')
    ax2.legend()
    ax2.annotate(f'KS p = {p_value_cosine:.2e}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
    QUANT_RESULTS.append({
        'Method': 'KDE (Cosine)',
        'Parameters': 'to centroid',
        'Silhouette Score': 'N/A', # Add placeholder for consistent keys
        'KS P-value': f'{p_value_cosine:.2e}'
    })

    # --- Regularized Mahalanobis Distance ---
    ax3 = plt.subplot(1, 3, 3)
    try:
        n_samples_0 = len(lab0)
        n_samples_1 = len(lab1)
        n_features = feats.shape[1]

        if (n_samples_0 + n_samples_1) <= n_features: # Use total samples vs features for robust cov
            print(f"⚠️  Total samples ({n_samples_0 + n_samples_1}) less than or equal to features ({n_features}). Cannot reliably estimate covariance for Mahalanobis. Skipping.")
            ax3.set_title('Mahalanobis (Insufficient Samples)')
            QUANT_RESULTS.append({
                'Method': 'KDE (Mahalanobis)',
                'Parameters': 'to centroid (regularized)',
                'Silhouette Score': 'N/A', # Add placeholder for consistent keys
                'KS P-value': 'N/A (Insufficient Samples)'
            })
        else:
            # Pooled data for covariance estimation
            all_data_for_cov = np.vstack((lab0, lab1))
            
            # Use ShrunkCovariance for robust estimation, especially with small n or high dim
            shrinkage_estimator = ShrunkCovariance(assume_centered=False, random_state=42)
            shrinkage_estimator.fit(all_data_for_cov)
            
            # The precision matrix is the inverse of the covariance matrix
            VI = shrinkage_estimator.precision_
            
            d0_mahalanobis = cdist(lab0, [c0], 'mahalanobis', VI=VI).flatten()
            d1_mahalanobis = cdist(lab1, [c1], 'mahalanobis', VI=VI).flatten()
            
            ks_stat_mahalanobis, p_value_mahalanobis = ks_2samp(d0_mahalanobis, d1_mahalanobis)
            
            sns.kdeplot(d0_mahalanobis, fill=True, label='Healthy', ax=ax3)
            sns.kdeplot(d1_mahalanobis, fill=True, label='AS', ax=ax3)
            ax3.set_title('Regularized Mahalanobis Distance', fontsize=12)
            ax3.set_xlabel('Distance'); ax3.set_ylabel('Density')
            ax3.legend()
            ax3.annotate(f'KS p = {p_value_mahalanobis:.2e}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
            QUANT_RESULTS.append({
                'Method': 'KDE (Mahalanobis)',
                'Parameters': 'to centroid (regularized)',
                'Silhouette Score': 'N/A', # Add placeholder for consistent keys
                'KS P-value': f'{p_value_mahalanobis:.2e}'
            })

    except Exception as e:
        print(f"⚠️  Mahalanobis distance calculation failed: {e}")
        ax3.set_title('Mahalanobis (Error)')
        ax3.text(0.5, 0.5, 'Error calculating', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        QUANT_RESULTS.append({
            'Method': 'KDE (Mahalanobis)',
            'Parameters': 'to centroid (regularized)',
            'Silhouette Score': 'N/A', # Add placeholder for consistent keys
            'KS P-value': 'N/A (Error)'
        })

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'kde_multi_distance.svg')
    plt.savefig(save_path, format='svg')
    plt.close()
    print(f"✅ Multi-distance KDE plots saved to: {save_path}")


def main():
    args = parse_args()
    
    # Create mode-specific save directory
    mode_save_dir = os.path.join(args.save_dir, args.mode)
    os.makedirs(mode_save_dir, exist_ok=True)
    print(f"📊 Outputs will be saved to: {mode_save_dir}")

    paths, labels = collect_paths(args.as_dir, args.healthy_dir)
    if not paths:
        raise RuntimeError("❌ 未找到任何图片，请检查路径！")
    print(f"🖼️  Found {len(paths)} images ({sum(labels)} AS, {len(labels)-sum(labels)} Healthy)")

    feats = extract_feats(paths, args.device)

    print("📏 Standardizing features...")
    scaler = StandardScaler()
    feats_standardized = scaler.fit_transform(feats)
    
    # Check if number of samples is too low for some ops
    n_samples = feats_standardized.shape[0]
    n_features = feats_standardized.shape[1]

    # Decide which plotting function to call based on --mode
    if args.mode == 'tsne_tuning':
        plot_tsne_tuning(feats_standardized, labels, mode_save_dir, perplexity_values=[5, 10, 15])
    elif args.mode == 'kde_distances':
        plot_kde_multi_distance(feats_standardized, labels, mode_save_dir)
    elif args.mode == 'umap_tuning':
        plot_umap_tuning(feats_standardized, labels, mode_save_dir, 
                         n_neighbors_grid=[3,5,10], min_dist_grid=[0.01,0.1,0.5])
    elif args.mode == 'pca_kpca':
        # PCA 95% energy pre-reduction
        
        # Calculate max possible components for PCA (n_samples - 1)
        max_components_for_pca = n_samples - 1 if n_samples > 1 else 1

        # Fit PCA on all features to get explained variance ratios
        temp_pca_all_components = PCA(n_components=min(n_features, max_components_for_pca), random_state=42)
        temp_pca_all_components.fit(feats_standardized)
        
        cumulative_variance = np.cumsum(temp_pca_all_components.explained_variance_ratio_)
        
        # Find how many components explain >= 95% variance
        if np.any(cumulative_variance >= 0.95):
            n_components_95_explained = np.where(cumulative_variance >= 0.95)[0][0] + 1
        else:
            # If 95% isn't reached, take all possible components
            n_components_95_explained = max_components_for_pca 
            print(f"⚠️  95% explained variance not reached with {max_components_for_pca} components. Using all available.")

        # Final components for PCA95 is the minimum of explained and max allowed
        final_n_components_for_pca95 = min(n_components_95_explained, max_components_for_pca)

        pca95 = PCA(n_components=final_n_components_for_pca95, random_state=42)
        feats_pca95 = pca95.fit_transform(feats_standardized)
        print(f"📈 PCA for 95% energy reduced dimensions from {n_features} to {feats_pca95.shape[1]} "
              f"({pca95.explained_variance_ratio_.sum()*100:.1f}% explained variance).")
        
        # Default UMAP plot (n_neighbors=5, min_dist=0.1) after PCA 95%
        print("\n🌌 Generating UMAP (default params) after PCA 95% energy...")
        if _HAS_UMAP:
            n_neighbors_for_default_umap = min(5, len(feats_pca95) -1) if len(feats_pca95) > 1 else 1
            if n_neighbors_for_default_umap < 1: n_neighbors_for_default_umap = 1
            
            emb_umap_default = umap.UMAP(n_neighbors=n_neighbors_for_default_umap, min_dist=0.1, random_state=42).fit_transform(feats_pca95)
            silhouette_umap_default = get_silhouette_score(emb_umap_default, labels)
            plot_embedding(emb_umap_default, labels, 
                           os.path.join(mode_save_dir, 'umap_after_pca95_default_params.svg'),
                           f'UMAP (n_n={n_neighbors_for_default_umap}, m_d=0.1) after PCA (95% energy)', silhouette_umap_default)
            QUANT_RESULTS.append({
                'Method': 'UMAP (after PCA 95%)',
                'Parameters': f'n_n={n_neighbors_for_default_umap}, m_d=0.1',
                'Silhouette Score': f'{silhouette_umap_default:.3f}' if not np.isnan(silhouette_umap_default) else 'N/A',
                'KS P-value': 'N/A' # Add placeholder
            })
            print(f"✅ Default UMAP plot after PCA 95% saved to: {os.path.join(mode_save_dir, 'umap_after_pca95_default_params.svg')}")
        else:
            print("⚠️  UMAP not installed, skipping default UMAP after PCA.")

        # Default t-SNE plot (perplexity=15) after PCA 95%
        print("\n🗺️  Generating t-SNE (default perplexity) after PCA 95% energy...")
        n_samples_for_tsne_after_pca = len(feats_pca95)
        perplexity_for_default_tsne = min(15, max(5, n_samples_for_tsne_after_pca - 1))
        try:
            emb_tsne_default = TSNE(n_components=2, init='pca', random_state=42,
                                    perplexity=perplexity_for_default_tsne, learning_rate=200, n_iter=1000,
                                    n_jobs=-1).fit_transform(feats_pca95)
            silhouette_tsne_default = get_silhouette_score(emb_tsne_default, labels)
            plot_embedding(emb_tsne_default, labels, 
                           os.path.join(mode_save_dir, 'tsne_after_pca95_default_perp.svg'),
                           f't-SNE (perp={perplexity_for_default_tsne}) after PCA (95% energy)', silhouette_tsne_default)
            QUANT_RESULTS.append({
                'Method': 't-SNE (after PCA 95%)',
                'Parameters': f'perp={perplexity_for_default_tsne}',
                'Silhouette Score': f'{silhouette_tsne_default:.3f}' if not np.isnan(silhouette_tsne_default) else 'N/A',
                'KS P-value': 'N/A' # Add placeholder
            })
            print(f"✅ Default t-SNE plot after PCA 95% saved to: {os.path.join(mode_save_dir, 'tsne_after_pca95_default_perp.svg')}")
        except Exception as e:
            print(f"⚠️  Default t-SNE after PCA 95% failed: {e}")
            QUANT_RESULTS.append({
                'Method': 't-SNE (after PCA 95%)',
                'Parameters': f'perp={perplexity_for_default_tsne}',
                'Silhouette Score': 'Error',
                'KS P-value': 'N/A'
            })
        
        # Plot Kernel PCA on original standardized features
        plot_pca_and_kernel_pca(feats_standardized, labels, mode_save_dir)

    elif args.mode == 'all':
        print("Running all visualization modes...")

        # PCA and Kernel PCA
        plot_pca_and_kernel_pca(feats_standardized, labels, mode_save_dir)

        # Plot best UMAP (n=5, d=0.1) as a key visualization
        if _HAS_UMAP:
            n_neighbors_default = min(5, n_samples - 1) if n_samples > 1 else 1 # Ensure n_neighbors is valid
            if n_neighbors_default < 1: n_neighbors_default = 1 # UMAP requires n_neighbors >=1
            
            emb_umap_best = umap.UMAP(n_neighbors=n_neighbors_default, min_dist=0.1, random_state=42).fit_transform(feats_standardized)
            silhouette_umap_best = get_silhouette_score(emb_umap_best, labels)
            plot_embedding(emb_umap_best, labels, 
                           os.path.join(mode_save_dir, 'umap_best_default.svg'),
                           'UMAP (n_n=5, m_d=0.1) - Best Found', silhouette_umap_best)
            QUANT_RESULTS.append({
                'Method': 'UMAP (Default Best)',
                'Parameters': f'n_n={n_neighbors_default}, m_d=0.1',
                'Silhouette Score': f'{silhouette_umap_best:.3f}' if not np.isnan(silhouette_umap_best) else 'N/A',
                'KS P-value': 'N/A' # Add placeholder
            })
            print(f"✅ Default Best UMAP plot saved to: {os.path.join(mode_save_dir, 'umap_best_default.svg')}")
        else:
            print("⚠️  UMAP not installed, skipping default best UMAP plot.")


        # Multi-distance KDE (including regularized Mahalanobis)
        plot_kde_multi_distance(feats_standardized, labels, mode_save_dir)
    
    # Print quantitative results table at the end of the script execution
    print("\n" + "="*50)
    print("Quantitative Comparison Table")
    print("="*50)
    if QUANT_RESULTS:
        # Define all possible column headers and their default order
        all_headers = ['Method', 'Parameters', 'Silhouette Score', 'KS P-value']
        
        # Determine maximum width for each column based on all data
        col_widths = {header: len(header) for header in all_headers}
        for row in QUANT_RESULTS:
            for key in all_headers: # Iterate through all_headers to ensure all keys are considered
                val = row.get(key, 'N/A') # Use .get() with default 'N/A' if key is missing
                col_widths[key] = max(col_widths[key], len(str(val)))
        
        header_line = " | ".join(key.ljust(col_widths[key]) for key in all_headers)
        print(header_line)
        print("-+-".join("-" * col_widths[key] for key in all_headers))
        for row in QUANT_RESULTS:
            print(" | ".join(str(row.get(key, 'N/A')).ljust(col_widths[key]) for key in all_headers)) # Use .get() here too
    else:
        print("No quantitative results collected for this mode.")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()