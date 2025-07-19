#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mri_permutation_test_full.py

自包含 MRI 置换检验脚本：
- 提取每位受试者的聚合特征
- 计算 LOOCV AUC
- 做 n_perm 次标签置换检验，报告 p-value

用法示例：
  chmod +x scripts/mri_permutation_test_full.py
  python scripts/mri_permutation_test_full.py \
    --data-dir /Users/hydra/.../data/mri_image_modified \
    --n-perm 1000 \
    --batch-size 16 \
    --seed 42 \
    --device cpu
"""

import argparse, random, re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MRI subject-level permutation test")
    p.add_argument("--data-dir",   type=str, required=True,
                   help="根目录，递归扫描所有子文件夹下的图像")
    p.add_argument("--n-perm",     type=int, default=1000,
                   help="置换次数 (默认 1000)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="切片特征 batch size")
    p.add_argument("--seed",       type=int, default=42,
                   help="随机种子")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备")
    return p.parse_args()


class SliceDataset(Dataset):
    EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root_dir, tf=None):
        self.samples = []
        for f in Path(root_dir).rglob("*"):
            if f.suffix.lower() in self.EXTS:
                label = 1 if "AS" in f.parent.name.upper() else 0
                m = re.match(r"^(KNEE_\d+|SIJ_\d+)", f.stem)
                sid = m.group(1) if m else re.sub(r"\s*\(.*\)$", "", f.stem)
                self.samples.append((str(f), label, sid))
        if not self.samples:
            raise RuntimeError(f"No images in {root_dir}")
        self.tf = tf

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tf: img = self.tf(img)
        return img, label, sid


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_loocv_auc(X, y):
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, va in loo.split(X):
        clf = LogisticRegression(solver="liblinear", random_state=0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:,1][0]
        preds.append(p)
        trues.append(y[va][0])
    return roc_auc_score(trues, preds)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = SliceDataset(args.data_dir, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ext = FeatureExtractor().to(device).eval()
    feats, labels, sids = [], [], []
    with torch.no_grad():
        for imgs, labs, subjects in tqdm(loader, desc="FeatExt"):
            imgs = imgs.to(device)
            f = ext(imgs).cpu().numpy()
            feats.append(f)
            labels.extend(labs)
            sids.extend(subjects)
    X_slice = np.vstack(feats); y_slice = np.array(labels)

    # 聚合到受试者级
    subj_feats, subj_lab = {}, {}
    for feat, lab, sid in zip(X_slice, y_slice, sids):
        subj_feats.setdefault(sid, []).append(feat)
        subj_lab[sid] = lab
    subj_ids = list(subj_feats)
    X = np.vstack([np.mean(subj_feats[s], axis=0) for s in subj_ids])
    y = np.array([subj_lab[s] for s in subj_ids])
    print(f"[INFO] Subjects: {len(y)}, Pos: {y.sum()}, Neg: {len(y)-y.sum()}")

    obs_auc = compute_loocv_auc(X, y)
    print(f"\n[RESULT] Observed LOOCV AUC = {obs_auc:.3f}")

    # 置换检验
    perm_aucs = []
    rng = np.random.RandomState(args.seed)
    for i in range(args.n_perm):
        y_perm = rng.permutation(y)
        perm_aucs.append(compute_loocv_auc(X, y_perm))
        if (i+1) % 100 == 0:
            print(f"[INFO] Permutations {i+1}/{args.n_perm}")
    perm_aucs = np.array(perm_aucs)
    pval = (perm_aucs >= obs_auc).sum() / args.n_perm
    print(f"\n[RESULT] Permutation p-value = {pval:.4f}")

if __name__ == "__main__":
    main()
