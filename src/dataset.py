import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

# ----------------- 新增这个类来处理临床表格数据 -----------------
class ClinicalDataset(Dataset):
    """专门用于加载和处理临床表格数据的Dataset类"""
    def __init__(self, csv_path, label_column='label'):
        """
        Args:
            csv_path (string): CSV文件的路径。
            label_column (string): 标签列的名称。
        """
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        
        # 将标签和其他特征分开
        self.labels = self.df[self.label_column].values
        self.features = self.df.drop(columns=[self.label_column]).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取特征和标签
        features = self.features[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long) # CrossEntropyLoss需要long类型的标签
        
        return features_tensor, label_tensor

# ----------------- 您原有的MRIDataset保持不变 -----------------
class MRIDataset(Dataset):
    """
    这个类用于加载MRI影像数据，我们将在下一步中使用它。
    （这里的代码保持您仓库中的原样）
    """
    def __init__(self, data_path, label_column='label', transform=None):
        # 这里的实现是您仓库中已有的，保持不变
        self.df = pd.read_csv(data_path) # 假设它也读取一个包含文件路径的 manifest 文件
        self.label_column = label_column
        self.transform = transform
        # ... 您原有的其他代码

    def __len__(self):
        # 您原有的代码
        return len(self.df)

    def __getitem__(self, idx):
        # 您原有的加载影像文件的代码
        # ...
        # return image, label
        pass # 暂时留空
