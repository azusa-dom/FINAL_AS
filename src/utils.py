import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from src.dataset import ClinicalDataset # 确保从我们修改过的dataset.py导入

def get_kfold_strafied_sampler(data_dir, n_splits=5, batch_size=32, id_column='patient_id', label_column='Disease'):
    """
    为K-折交叉验证创建数据加载器列表。
    这个版本现在可以将 id_column 和 label_column 参数传递给 ClinicalDataset。
    """
    kfold_loaders = []
    for i in range(n_splits):
        train_csv = os.path.join(data_dir, f"fold_{i}_train.csv")
        val_csv = os.path.join(data_dir, f"fold_{i}_val.csv")

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            # 这是一个预期的行为，如果文件不存在，说明需要先运行预处理脚本
            # 所以我们只打印信息而不是抛出错误，让主程序决定如何处理。
            print(f"提示: Fold {i} 的数据文件不存在, 需要先运行 `scripts/preprocess_clinical.py`。")
            return None # 返回None表示无法创建加载器

        # 【核心修改】在创建Dataset实例时，传入id_column和正确的label_column
        train_dataset = ClinicalDataset(csv_path=train_csv, label_column=label_column, id_column=id_column)
        val_dataset = ClinicalDataset(csv_path=val_csv, label_column=label_column, id_column=id_column)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        kfold_loaders.append((train_loader, val_loader))
        
    return kfold_loaders


def get_class_weights(dataset):
    """
    根据数据集中各类别的样本数量，计算类别权重。
    """
    if not hasattr(dataset, 'labels'):
        print("警告: 数据集没有 'labels' 属性, 无法计算类别权重。")
        return None

    labels = np.array(dataset.labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    if len(unique) < 2:
        print("警告: 数据集中只存在一个类别, 无法计算类别权重。")
        return None

    class_counts = dict(zip(unique, counts))
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = [total_samples / (num_classes * class_counts.get(i, 1)) for i in sorted(class_counts.keys())]
    
    print(f"INFO: Calculated class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)

