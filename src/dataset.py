#!/usr/bin/env python3
"""
数据集模块
包含临床数据和MRI数据的Dataset类
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
import os
from typing import Optional, Tuple, List

class ClinicalDataset(TensorDataset):
    """
    临床数据集类
    用于加载和处理临床表格数据
    """
    
    def __init__(self, csv_path: str, label_column: str = "label", id_column: str = "Patient_ID"):
        """
        初始化临床数据集
        
        Parameters:
        -----------
        csv_path : str
            CSV文件路径
        label_column : str
            标签列名
        id_column : str
            ID列名
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 提取特征和标签
        self.feature_columns = [col for col in df.columns if col not in [label_column, id_column]]
        self.features = torch.tensor(df[self.feature_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_column].values, dtype=torch.long)
        self.ids = df[id_column].values if id_column in df.columns else None
        
        # 获取唯一标签
        self.unique_labels = sorted(df[label_column].unique())
        
        # 调用父类构造函数
        super().__init__(self.features, self.labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.ids is not None:
            return self.features[idx], self.labels[idx], self.ids[idx]
        else:
            return self.features[idx], self.labels[idx]

class MRIDataset(Dataset):
    """
    MRI数据集类
    用于加载和处理MRI图像数据
    """
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 label_map: Optional[dict] = None):
        """
        初始化MRI数据集
        
        Parameters:
        -----------
        data_dir : str
            数据目录路径
        transform : callable, optional
            图像变换函数
        label_map : dict, optional
            标签映射字典
        """
        self.data_dir = data_dir
        self.transform = transform
        self.label_map = label_map or {}
        
        # 收集所有图像文件
        self.image_paths = []
        self.labels = []
        self.ids = []
        
        self._collect_data()
    
    def _collect_data(self):
        """收集数据文件"""
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    self.image_paths.append(file_path)
                    
                    # 从路径推断标签
                    label = self._infer_label_from_path(file_path)
                    self.labels.append(label)
                    
                    # 从路径推断ID
                    patient_id = self._extract_patient_id(file_path)
                    self.ids.append(patient_id)
    
    def _infer_label_from_path(self, file_path: str) -> int:
        """从文件路径推断标签"""
        path_lower = file_path.lower()
        
        # 根据路径中的关键词推断标签
        if 'as' in path_lower or 'ankylosing' in path_lower:
            return 1  # AS
        elif 'health' in path_lower or 'control' in path_lower:
            return 0  # 健康对照
        else:
            # 默认标签
            return 0
    
    def _extract_patient_id(self, file_path: str) -> str:
        """从文件路径提取患者ID"""
        # 从路径中提取患者ID
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if part.startswith('patient') or part.startswith('subj'):
                return part
        return os.path.basename(file_path).split('.')[0]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签和ID
        label = self.labels[idx]
        patient_id = self.ids[idx]
        
        return image, label, patient_id

class SliceDataset(Dataset):
    """
    切片数据集类
    用于处理MRI切片数据
    """
    
    def __init__(self, 
                 slice_paths: List[str],
                 labels: List[int],
                 transform=None):
        """
        初始化切片数据集
        
        Parameters:
        -----------
        slice_paths : List[str]
            切片文件路径列表
        labels : List[int]
            标签列表
        transform : callable, optional
            图像变换函数
        """
        self.slice_paths = slice_paths
        self.labels = labels
        self.transform = transform
        
        assert len(slice_paths) == len(labels), "切片路径和标签数量不匹配"
    
    def __len__(self):
        return len(self.slice_paths)
    
    def __getitem__(self, idx):
        # 加载切片图像
        image_path = self.slice_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label

def create_clinical_dataloader(csv_path: str, 
                              batch_size: int = 32,
                              shuffle: bool = True,
                              label_column: str = "label",
                              id_column: str = "Patient_ID") -> torch.utils.data.DataLoader:
    """
    创建临床数据加载器
    
    Parameters:
    -----------
    csv_path : str
        CSV文件路径
    batch_size : int
        批次大小
    shuffle : bool
        是否打乱数据
    label_column : str
        标签列名
    id_column : str
        ID列名
    
    Returns:
    --------
    torch.utils.data.DataLoader
        数据加载器
    """
    dataset = ClinicalDataset(csv_path, label_column, id_column)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # 避免多进程问题
    )

def create_mri_dataloader(data_dir: str,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         transform=None) -> torch.utils.data.DataLoader:
    """
    创建MRI数据加载器
    
    Parameters:
    -----------
    data_dir : str
        数据目录路径
    batch_size : int
        批次大小
    shuffle : bool
        是否打乱数据
    transform : callable, optional
        图像变换函数
    
    Returns:
    --------
    torch.utils.data.DataLoader
        数据加载器
    """
    dataset = MRIDataset(data_dir, transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # 避免多进程问题
    ) 