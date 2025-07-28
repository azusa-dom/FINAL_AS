#!/usr/bin/env python3
"""
特征空间几何分析模块
实现论文2.4.3节描述的特征空间几何和距离统计
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def compute_distance_statistics(embeddings, labels, class_names=['HC', 'AS']):
    """
    计算特征空间中的距离统计
    
    Parameters:
    -----------
    embeddings : np.ndarray
        特征嵌入矩阵 (n_samples, n_features)
    labels : np.ndarray
        类别标签
    class_names : list
        类别名称
    
    Returns:
    --------
    dict
        距离统计结果
    """
    
    # 计算每个类别的中心
    unique_labels = np.unique(labels)
    class_centroids = {}
    
    for label in unique_labels:
        class_mask = labels == label
        class_centroids[label] = np.mean(embeddings[class_mask], axis=0)
    
    # 计算每个样本到其类中心的距离
    euclidean_distances = []
    cosine_distances = []
    sample_labels = []
    
    for i, (embedding, label) in enumerate(zip(embeddings, labels)):
        centroid = class_centroids[label]
        
        # 欧几里得距离
        euclidean_dist = np.linalg.norm(embedding - centroid)
        euclidean_distances.append(euclidean_dist)
        
        # 余弦距离
        cosine_sim = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
        cosine_dist = 1 - cosine_sim
        cosine_distances.append(cosine_dist)
        
        sample_labels.append(label)
    
    euclidean_distances = np.array(euclidean_distances)
    cosine_distances = np.array(cosine_distances)
    sample_labels = np.array(sample_labels)
    
    # 进行Kolmogorov-Smirnov检验
    ks_results = {}
    for label in unique_labels:
        mask = sample_labels == label
        other_mask = sample_labels != label
        
        # 欧几里得距离KS检验
        ks_euclidean = ks_2samp(
            euclidean_distances[mask], 
            euclidean_distances[other_mask]
        )
        
        # 余弦距离KS检验
        ks_cosine = ks_2samp(
            cosine_distances[mask], 
            cosine_distances[other_mask]
        )
        
        ks_results[label] = {
            'euclidean_ks': ks_euclidean,
            'cosine_ks': ks_cosine
        }
    
    return {
        'euclidean_distances': euclidean_distances,
        'cosine_distances': cosine_distances,
        'sample_labels': sample_labels,
        'class_centroids': class_centroids,
        'ks_results': ks_results
    }

def compute_embedding_projections(embeddings, labels, methods=['pca', 'kernel_pca', 'tsne', 'umap']):
    """
    计算多种降维投影方法
    
    Parameters:
    -----------
    embeddings : np.ndarray
        特征嵌入矩阵
    labels : np.ndarray
        类别标签
    methods : list
        要使用的降维方法
    
    Returns:
    --------
    dict
        各种投影结果和质量指标
    """
    
    results = {}
    
    # 标准化嵌入
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    for method in methods:
        print(f"计算 {method} 投影...")
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            projected = reducer.fit_transform(embeddings_scaled)
            
        elif method == 'kernel_pca':
            reducer = KernelPCA(n_components=2, kernel='rbf', gamma=1/embeddings.shape[1], random_state=42)
            projected = reducer.fit_transform(embeddings_scaled)
            
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=15, random_state=42)
            projected = reducer.fit_transform(embeddings_scaled)
            
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.1, random_state=42)
            projected = reducer.fit_transform(embeddings_scaled)
            
        else:
            print(f"跳过 {method} (不可用)")
            continue
        
        # 计算轮廓分数
        silhouette = silhouette_score(projected, labels)
        
        results[method] = {
            'projected': projected,
            'silhouette_score': silhouette
        }
        
        print(f"  {method} 轮廓分数: {silhouette:.3f}")
    
    return results

def plot_distance_distributions(distance_stats, output_path=None):
    """
    绘制距离分布图
    
    Parameters:
    -----------
    distance_stats : dict
        距离统计结果
    output_path : str, optional
        输出文件路径
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 欧几里得距离分布
    unique_labels = np.unique(distance_stats['sample_labels'])
    for label in unique_labels:
        mask = distance_stats['sample_labels'] == label
        axes[0].hist(distance_stats['euclidean_distances'][mask], 
                    alpha=0.7, label=f'Class {label}', bins=20)
    
    axes[0].set_xlabel('Euclidean Distance to Class Centroid')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Euclidean Distance Distribution')
    axes[0].legend()
    
    # 余弦距离分布
    for label in unique_labels:
        mask = distance_stats['sample_labels'] == label
        axes[1].hist(distance_stats['cosine_distances'][mask], 
                    alpha=0.7, label=f'Class {label}', bins=20)
    
    axes[1].set_xlabel('Cosine Distance to Class Centroid')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cosine Distance Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_embedding_projections(projection_results, labels, output_path=None):
    """
    绘制嵌入投影图
    
    Parameters:
    -----------
    projection_results : dict
        投影结果
    labels : np.ndarray
        类别标签
    output_path : str, optional
        输出文件路径
    """
    
    n_methods = len(projection_results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (method, result) in enumerate(projection_results.items()):
        if i >= 4:
            break
            
        projected = result['projected']
        silhouette = result['silhouette_score']
        
        scatter = axes[i].scatter(projected[:, 0], projected[:, 1], 
                                c=labels, cmap='viridis', alpha=0.7)
        axes[i].set_title(f'{method.upper()} (Silhouette: {silhouette:.3f})')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_geometry_report(distance_stats, projection_results, output_path=None):
    """
    生成几何分析报告
    
    Parameters:
    -----------
    distance_stats : dict
        距离统计结果
    projection_results : dict
        投影结果
    output_path : str, optional
        输出文件路径
    """
    
    report = []
    report.append("=== 特征空间几何分析报告 ===\n")
    
    # 距离统计
    report.append("1. 距离统计:")
    for label, ks_result in distance_stats['ks_results'].items():
        report.append(f"   类别 {label}:")
        report.append(f"     欧几里得距离 KS检验 p值: {ks_result['euclidean_ks'].pvalue:.6f}")
        report.append(f"     余弦距离 KS检验 p值: {ks_result['cosine_ks'].pvalue:.6f}")
    
    # 投影质量
    report.append("\n2. 投影质量 (轮廓分数):")
    for method, result in projection_results.items():
        report.append(f"   {method.upper()}: {result['silhouette_score']:.3f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text

if __name__ == "__main__":
    print("特征空间几何分析模块")
    print("请将此模块集成到现有的MRI分析流程中") 