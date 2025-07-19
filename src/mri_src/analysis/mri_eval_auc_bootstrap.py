#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mri_bootstrap_auc_group.py

递归扫描 data-dir 下所有图像文件（.png/.jpg/.jpeg），
按受试者 ID 分组交叉验证 (GroupKFold)，避免同一受试者切片泄漏，
提取 ResNet18 特征，用 LogisticRegression 训练，
并报告 AUC 点估计及其 95% Bootstrap 置信区间。

用法示例：
    python scripts/mri_bootstrap_auc_group.py \
      --data-dir /Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/mri_image_modified \
      --n-splits 5 \
      --n-bootstrap 2000 \
      --batch-size 16 \
      --seed 42
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(
        description="MRI AUC + 95% CI via bootstrap with GroupKFold"
    )
    p.add_argument("--data-dir",   type=str, required=True,
                   help="根目录，递归扫描所有子文件夹下的 .png/.jpg/.jpeg")
    p.add_argument("--n-splits",   type=int, default=5,
                   help="GroupKFold 的折数（默认 5），LOOCV 请设为受试者数")
    p.add_argument("--n-bootstrap",type=int, default=2000,
                   help="Bootstrap 重采样次数（默认 2000）")
    p.add_argument("--batch-size", type=int, default=16,
                   help="特征提取 batch size")
    p.add_argument("--seed",       type=int, default=42,
                   help="随机种子，保证可复现")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备：cuda 或 cpu")
    return p.parse_args()


class MRIDataset(Dataset):
    """递归读取所有图像，提取 subject_id 并打标签"""
    IMG_EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root_dir, transform=None):
        self.samples = []
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"根目录不存在: {root_dir}")

        for img_path in root.rglob("*"):
            if img_path.suffix.lower() in self.IMG_EXTS:
                # 提取受试者 ID
                stem = img_path.stem
                m = re.match(r"^(KNEE_\d+|SIJ_\d+)", stem)
                if m:
                    subject_id = m.group(1)
                else:
                    # 去掉尾部括号
                    subject_id = re.sub(r"\s*\(.*\)$", "", stem)
                # 打标签：父文件夹名含 'AS' 即为阳性
                label = 1 if "AS" in img_path.parent.name.upper() else 0
                self.samples.append((str(img_path), label, subject_id))

        if not self.samples:
            raise ValueError(f"{root_dir} 下未找到任何图像文件。")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, sid


class FeatureExtractor(nn.Module):
    """ResNet18 backbone，无最终 fc，用于提取512维特征"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        f = self.backbone(x)       # [B,512,1,1]
        return f.view(f.size(0), -1)  # [B,512]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_ci(y_true, y_score, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    auc0 = roc_auc_score(y_true, y_score)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(boots) < n_bootstrap * 0.8:
        raise RuntimeError(
            f"有效 bootstrap 次数太少: {len(boots)}/{n_bootstrap}"
        )
    low, high = np.percentile(boots, [2.5, 97.5])
    return auc0, low, high


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

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
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # 提取所有切片的特征、标签、subject_id
    extractor = FeatureExtractor().to(device).eval()
    feats, labels, groups = [], [], []
    with torch.no_grad():
        for imgs, labs, sids in tqdm(loader, desc="Feature Extract"):
            imgs = imgs.to(device)
            f = extractor(imgs).cpu().numpy()  # [B,512]
            feats.append(f)
            labels.extend(labs.tolist())
            groups.extend(sids)

    X = np.vstack(feats)
    y = np.array(labels)
    print(f"[INFO] Total samples: {len(y)}, Pos: {y.sum()}, Neg: {len(y)-y.sum()}")

    # 分组交叉验证
    unique_sids = list(set(groups))
    n_splits = min(args.n_splits, len(unique_sids))
    gkf = GroupKFold(n_splits=n_splits)
    print(f"[INFO] Using GroupKFold with {n_splits} splits on {len(unique_sids)} subjects")

    y_true, y_score = [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        clf = LogisticRegression(solver="liblinear", random_state=args.seed)
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[va_idx])[:, 1]
        y_true.extend(y[va_idx].tolist())
        y_score.extend(probs.tolist())
        print(f"  Fold {fold}: val samples = {len(va_idx)}")

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    print("[INFO] Computing AUC & Bootstrap 95% CI...")
    auc_pt, ci_l, ci_u = bootstrap_ci(
        y_true, y_score,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed
    )

    print("\n====== MRI Model Performance ======")
    print(f"Subject-level AUC point estimate : {auc_pt:.3f}")
    print(f"95% CI                          : [{ci_l:.3f}, {ci_u:.3f}]")
    print("====================================\n")


if __name__ == "__main__":
    main()
