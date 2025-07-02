#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mri_bootstrap_auc_nogroup.py

自动扫描根目录下所有以 png_ 开头的子文件夹，
将其中含 'AS' 的视为阳性 (label=1)，其余为阴性 (label=0)；
提取 ResNet18 特征，使用 StratifiedKFold (支持 LOOCV) 训练
LogisticRegression 并计算 AUC 及 95% Bootstrap CI。

用法示例：
    python scripts/mri_bootstrap_auc_nogroup.py \
        --data-dir /Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/mri_image_modified \
        --n-splits 5 \
        --n-bootstrap 2000 \
        --batch-size 16 \
        --seed 42
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MRI AUC + 95% CI via bootstrap (no grouping)")
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="MRI 图像根目录，内部应包含若干以 png_ 开头的子文件夹"
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="交叉验证折数（默认 5）。如想 LOOCV，请设置为样本总数。"
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap 重采样次数（默认 2000）"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="特征提取时的 batch size"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备"
    )
    return p.parse_args()


class MRIDataset(Dataset):
    """自动扫描 data_dir 下所有以 png_ 开头的文件夹并打标签"""
    def __init__(self, root_dir, transform=None):
        self.samples = []
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"未找到根目录: {root_dir}")
        # 扫描所有以 png_ 开头的子目录
        subdirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("png_")]
        if not subdirs:
            raise FileNotFoundError(f"{root_dir} 下未找到任何以 'png_' 开头的文件夹。")
        for sub in subdirs:
            label = 1 if "AS" in sub.name else 0
            for img_path in sub.rglob("*.png"):
                self.samples.append((str(img_path), label))
        if not self.samples:
            raise ValueError(f"{root_dir} 下各子目录内未找到任何 PNG 文件。")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class FeatureExtractor(nn.Module):
    """ResNet18 去掉最后全连接层，仅作特征提取"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        f = self.backbone(x)            # [B, 512, 1, 1]
        return f.view(f.size(0), -1)    # [B, 512]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_ci(y_true, y_score, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    auc_ = roc_auc_score(y_true, y_score)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(boot) < n_bootstrap * 0.8:
        raise RuntimeError(f"有效的 bootstrap 次数不足：{len(boot)}/{n_bootstrap}")
    low, high = np.percentile(boot, [2.5, 97.5])
    return auc_, low, high


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"\n[INFO] Device: {device}")
    print("[INFO] 加载数据集并提取特征...")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    ds = MRIDataset(args.data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 特征提取器
    extractor = FeatureExtractor().to(device).eval()

    feats, labels = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Feature Extract"):
            imgs = imgs.to(device)
            f = extractor(imgs)
            feats.append(f.cpu().numpy())
            labels.extend(labs.numpy().tolist())

    X = np.vstack(feats)
    y = np.array(labels)
    n_samples = len(y)
    print(f"[INFO] 样本总数: {n_samples}，正例: {y.sum()}，负例: {n_samples - y.sum()}")

    # 分层交叉验证
    n_splits = min(args.n_splits, n_samples)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    print(f"[INFO] 使用 StratifiedKFold({n_splits} 折)")

    y_true_all, y_score_all = [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        clf = LogisticRegression(solver="liblinear", random_state=args.seed)
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        y_true_all.extend(y[val_idx].tolist())
        y_score_all.extend(probs.tolist())
        print(f"  Fold {fold}: 验证样本 = {len(val_idx)}")

    y_true_all = np.array(y_true_all)
    y_score_all = np.array(y_score_all)

    print("\n[INFO] 计算 AUC 及 95% Bootstrap 置信区间...")
    auc_pt, ci_l, ci_u = bootstrap_ci(
        y_true_all, y_score_all,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed
    )

    print("\n====== MRI 模型性能 ======")
    print(f"样本总数      : {n_samples}")
    print(f"AUC 点估计    : {auc_pt:.3f}")
    print(f"95% 置信区间  : [{ci_l:.3f}, {ci_u:.3f}]")
    print("============================\n")


if __name__ == "__main__":
    main()
