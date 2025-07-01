# src/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet 的标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 训练集 transform（module-level，所以可序列化）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 验证/测试集 transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class MRIImageFolderDataset(Dataset):
    """
    MRI Folder 数据集：自动把灰度图做成 3 通道 RGB，使用 ImageNet 预处理。
    root_dir 目录下应包含多个子文件夹，每个子文件夹名为类别：
        e.g. 0_Healthy/, 1_AS/
    """

    def __init__(self, root_dir, train: bool = True):
        self.root_dir = root_dir
        self.train = train
        # 选择对应的 transform
        self.transform = train_transform if train else val_transform

        # 类别与索引
        classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # 收集所有 (图像路径, label)
        self.samples = []
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 读为灰度后 convert('RGB') 会自动复制成 3 通道
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label
