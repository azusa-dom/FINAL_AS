import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 我们需要导入您的 Dataset class 来使用它
from .dataset import MRIDataset 


def get_kfold_strafied_sampler(data_dir, n_splits=5, batch_size=32):
    """
    一个生成器，用于查找预先分割好的数据折并为每个折返回 DataLoader。
    
    Args:
        data_dir (str): 'fold_X_train.csv' 和 'fold_X_val.csv' 所在的目录。
        n_splits (int): 要循环的折数。
        batch_size (int): DataLoader 的批量大小。
    
    Yields:
        (DataLoader, DataLoader): 一个包含训练集和验证集 DataLoader 的元组。
    """
    print(f"🔄 Loading {n_splits}-fold data from: {data_dir}")
    for i in range(n_splits):
        train_path = os.path.join(data_dir, f"fold_{i}_train.csv")
        val_path = os.path.join(data_dir, f"fold_{i}_val.csv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Data for fold {i} not found. Expected to find {train_path} and {val_path}"
            )

        # --- 这是需要修正的地方 ---
        # 使用正确的参数名 'csv_file' 来创建数据集对象
        train_dataset = MRIDataset(csv_file=train_path)
        val_dataset = MRIDataset(csv_file=val_path)
        # --- 修正结束 ---

        # 创建 DataLoader 对象
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        yield train_loader, val_loader


def get_class_weights(dataset):
    """
    计算类别权重以处理不平衡的数据集。

    Args:
        dataset (Dataset): 一个 PyTorch Dataset 对象，它有一个 'labels' 属性。
    
    Returns:
        torch.Tensor: 一个包含每个类别权重的张量。
    """
    # 从数据集的底层 dataframe 中访问标签
    labels = dataset.df[dataset.label_column].to_numpy()
    unique_labels = np.unique(labels)
    
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    
    print(f"⚖️ Computed class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)
