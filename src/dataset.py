import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os

class ClinicalDataset(Dataset):
    """专门用于加载和处理临床表格数据的Dataset类"""
    def __init__(self, csv_path, label_column='Disease'):
        """
        Args:
            csv_path (string): CSV文件的路径。
            label_column (string): 标签列的名称。
        """
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        
        # 标签编码逻辑
        self.unique_labels = self.df[self.label_column].astype('category').cat.categories
        self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
        print(f"INFO: Label mapping for {os.path.basename(csv_path)}: {self.label_to_int}")
        self.labels = self.df[self.label_column].map(self.label_to_int).values
        
        features_df = self.df.drop(columns=[self.label_column])
        self.features = features_df.select_dtypes(include=np.number).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor

class MRIDataset(Dataset):
    """
    这个类用于加载MRI影像数据，我们将在下一步中使用它。
    """
    def __init__(self, data_path, label_column='label', transform=None):
        self.df = pd.read_csv(data_path)
        self.label_column = label_column
        self.transform = transform
        # ... 您原有的其他代码

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pass # 暂时留空
