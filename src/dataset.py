import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class ClinicalDataset(Dataset):
    """专门用于加载和处理临床表格数据的Dataset类"""

    def __init__(self, csv_path, label_column="Disease", id_column="patient_id"):
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        self.id_column = id_column if id_column in self.df.columns else None

        if self.label_column in self.df.columns:
            self.unique_labels = (
                self.df[self.label_column].astype("category").cat.categories
            )
            self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
            print(
                f"INFO: Label mapping for {os.path.basename(csv_path)}: {self.label_to_int}"
            )
            self.labels = self.df[self.label_column].map(self.label_to_int).values
        else:
            self.labels = np.zeros(len(self.df), dtype=int)
            print(
                f"警告: 在文件 {os.path.basename(csv_path)} 中未找到标签列 '{self.label_column}'。"
            )

        if self.id_column and self.id_column in self.df.columns:
            self.patient_ids = self.df[self.id_column].values
            features_df = self.df.drop(
                columns=[
                    col
                    for col in [self.label_column, self.id_column]
                    if col in self.df.columns
                ]
            )
        else:
            self.patient_ids = np.arange(len(self.df))
            features_df = self.df.drop(columns=[self.label_column], errors="ignore")

        self.features = features_df.select_dtypes(include=np.number).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 这一部分和之前一样
        features = self.features[idx]
        label = self.labels[idx]
        pid = self.patient_ids[idx]

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 【终极调试代码】在返回前，强制检查所有元素的有效性
        if torch.isnan(features_tensor).any():
            raise ValueError(
                f"错误! 在索引 {idx} (patient_id: {pid}) 处，特征数据(features)中包含NaN!"
            )

        if pid is None:
            raise ValueError(f"错误! 在索引 {idx} 处，patient_id 为 None!")

        return features_tensor, label_tensor, pid


# --- ASFineTuneDataset (保持不变) ---
class ASFineTuneDataset(Dataset):
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
                if file_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
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
