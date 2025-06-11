import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- 关键改动：导入我们新建的 ClinicalDataset ---
from .dataset import ClinicalDataset 


def get_kfold_strafied_sampler(data_dir, n_splits=5, batch_size=32):
    """
    一个生成器，用于查找预先分割好的数据折并为每个折返回 DataLoader。
    """
    print(f"🔄 Loading {n_splits}-fold data from: {data_dir}")
    for i in range(n_splits):
        train_path = os.path.join(data_dir, f"fold_{i}_train.csv")
        val_path = os.path.join(data_dir, f"fold_{i}_val.csv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Data for fold {i} not found. Expected to find {train_path} and {val_path}"
            )

        # --- 关键改动：使用正确的Dataset类 ---
        train_dataset = ClinicalDataset(csv_path=train_path)
        val_dataset = ClinicalDataset(csv_path=val_path)
        
        # 创建 DataLoader 对象
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        yield train_loader, val_loader


def get_class_weights(dataset):
    """
    计算类别权重以处理不平衡的数据集。
    """
    # 现在它会正确地从 ClinicalDataset 中获取标签
    labels = dataset.labels
    unique_labels = np.unique(labels)
    
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    
    print(f"⚖️ Computed class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)
