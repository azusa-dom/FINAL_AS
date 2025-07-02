import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet 的归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 顶层定义 transform 保证多进程可序列化
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class MRIImageFolderDataset(Dataset):
    """
    按文件夹组织的 MRI 数据集：
      root_dir/
        0_Healthy/
        1_AS/
    自动将灰度图 convert('RGB') → 3 通道，再做 ImageNet 预处理。
    """
    def __init__(self, root_dir, train: bool = True):
        self.root_dir = root_dir
        self.transform = train_transform if train else val_transform

        # 子文件夹即类别
        classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # 收集 (path,label) 列表
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                    self.samples.append((os.path.join(cls_dir, fname),
                                         self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')  # 灰度→RGB
        img = self.transform(img)
        return img, label
