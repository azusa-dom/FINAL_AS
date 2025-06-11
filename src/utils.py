import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from .dataset import ClinicalDataset 

def get_kfold_strafied_sampler(data_dir, n_splits=5, batch_size=32):
    print(f"🔄 Loading {n_splits}-fold data from: {data_dir}")
    for i in range(n_splits):
        train_path = os.path.join(data_dir, f"fold_{i}_train.csv")
        val_path = os.path.join(data_dir, f"fold_{i}_val.csv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Data for fold {i} not found. Expected to find {train_path} and {val_path}"
            )

        train_dataset = ClinicalDataset(csv_path=train_path, label_column='pseudo_AS')
        val_dataset = ClinicalDataset(csv_path=val_path, label_column='pseudo_AS')  # ✅ 修复
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        yield train_loader, val_loader


def get_class_weights(dataset):
    labels = dataset.labels
    unique_labels = np.unique(labels)
    
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    
    print(f"⚖️ Computed class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)
