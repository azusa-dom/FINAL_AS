import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image # Pillow库，用于读取图片。如果未安装，请运行: pip install Pillow

# ==============================================================================
# --- ClinicalDataset (已根据修改意见更新) ---
# ==============================================================================

class ClinicalDataset(Dataset):
    """专门用于加载和处理临床表格数据的Dataset类"""
    def __init__(self, csv_path, label_column='Disease', id_column='patient_id'):
        """
        Args:
            csv_path (string): CSV文件的路径。
            label_column (string): 标签列的名称。
            id_column (string): 患者ID列的名称。
        """
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        # 检查ID列是否存在
        self.id_column = id_column if id_column in self.df.columns else None
        
        # 标签编码逻辑
        # 检查标签列是否存在，如果不存在则可能是一个没有标签的测试集
        if self.label_column in self.df.columns:
            self.unique_labels = self.df[self.label_column].astype('category').cat.categories
            self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
            print(f"INFO: Label mapping for {os.path.basename(csv_path)}: {self.label_to_int}")
            self.labels = self.df[self.label_column].map(self.label_to_int).values
        else:
            self.labels = [0] * len(self.df) # 如果没有标签列，用0作为占位符
            print(f"警告: 在文件 {os.path.basename(csv_path)} 中未找到标签列 '{self.label_column}'。")


        # 根据是否存在ID列来处理特征和ID
        if self.id_column and self.id_column in self.df.columns:
            self.patient_ids = self.df[self.id_column].tolist()
            # 从特征中移除标签列和ID列
            features_df = self.df.drop(columns=[col for col in [self.label_column, self.id_column] if col in self.df.columns])
        else:
            # 如果没有ID列，则用索引作为占位符
            self.patient_ids = list(range(len(self.df)))
            features_df = self.df.drop(columns=[self.label_column], errors='ignore')
            
        self.features = features_df.select_dtypes(include=np.number).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        # 获取当前样本的patient_id
        pid = self.patient_ids[idx]

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 返回ID作为额外的数据
        return features_tensor, label_tensor, pid

# ==============================================================================
# --- ASFineTuneDataset (您原有的代码，保持不变) ---
# ==============================================================================

class ASFineTuneDataset(Dataset):
    """
    专门用于混合强直性脊柱炎(AS)和健康影像进行微调的数据集类。
    它会读取一个包含 '0_Healthy' 和 '1_AS' 子文件夹的根目录。
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        class_map = {"0_Healthy": 0, "1_AS": 1}

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"指定的根目录不存在: {self.root_dir}")

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
