# 文件: src/dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image # Pillow库，用于读取图片。如果未安装，请运行: pip install Pillow

# ==============================================================================
# --- 您原有的代码 (完全保留，未做任何改动) ---
# ==============================================================================

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

# ==============================================================================
# --- 新增的代码 (用于处理MRI影像，不影响上面的代码) ---
# ==============================================================================

class ASFineTuneDataset(Dataset):
    """
    专门用于混合强直性脊柱炎(AS)和健康影像进行微调的数据集类。
    它会读取一个包含 '0_Healthy' 和 '1_AS' 子文件夹的根目录。
    """
    def __init__(self, root_dir, transform=None):
        """
        初始化函数，用于扫描文件目录并创建样本列表。

        Args:
            root_dir (string): 数据集的主目录路径 (例如: 'path/to/AS_Finetune_Data/')。
            transform (callable, optional): 应用于每个样本的可选变换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # 初始化一个空列表，用来存放 (图片路径, 标签) 的元组

        # 定义类别名称和它们对应的整数标签
        class_map = {"0_Healthy": 0, "1_AS": 1}

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"指定的根目录不存在: {self.root_dir}")

        # 遍历根目录下的每个类别文件夹
        for class_name, label in class_map.items():
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"警告: 找不到类别文件夹 {class_path}，将跳过。")
                continue

            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image_path = os.path.join(class_path, file_name)
                    self.samples.append((image_path, label))

        if not self.samples:
            print(f"警告: 在目录 {self.root_dir} 中没有找到任何图片文件。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"错误：无法读取图片 {image_path}。错误信息: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)

        return image, label
