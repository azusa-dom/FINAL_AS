import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

# ----------------- 这是修正后的 ClinicalDataset 类 -----------------
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
        
        # --- 新增的标签编码逻辑 ---
        # 1. 获取所有唯一的字符串标签
        self.unique_labels = self.df[self.label_column].astype('category').cat.categories
        # 2. 创建一个从字符串标签到整数的映射
        self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
        print(f"INFO: Label mapping for {os.path.basename(csv_path)}: {self.label_to_int}")
        # 3. 使用这个映射将整个标签列转换为整数
        self.labels = self.df[self.label_column].map(self.label_to_int).values
        # --- 标签编码结束 ---
        
        # 将原始的标签列和其他非数值特征（如果有的话）从特征中移除
        features_df = self.df.drop(columns=[self.label_column])
        # 选择所有数值类型的列作为最终特征
        self.features = features_df.select_dtypes(include=np.number).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取特征和已经编码为整数的标签
        features = self.features[idx]
        label = self.labels[idx] # <-- 现在这是一个整数了
        
        # 转换为PyTorch张量
        features_tensor = torch.tensor(features, dtype=torch.float32)
        # 这行代码现在可以正常工作了
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor

# ----------------- 您原有的MRIDataset保持不变 -----------------
class MRIDataset(Dataset):
    """
    这个类用于加载MRI影像数据，我们将在下一步中使用它。
    （这里的代码保持您仓库中的原样）
    """
    def __init__(self, data_path, label_column='label', transform=None):
        self.df = pd.read_csv(data_path)
        self.label_column = label_column
        self.transform = transform
        # ... 您原有的其他代码

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 您原有的加载影像文件的代码
        pass # 暂时留空
