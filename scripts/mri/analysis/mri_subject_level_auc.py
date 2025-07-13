#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mri_subject_level_auc.py

针对每位受试者聚合切片特征，做受试者级 LOOCV/KFold，计算 AUC & 95% CI。

用法示例：
    python scripts/mri_subject_level_auc.py \
      --data-dir /Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/mri_image_modified \
      --n-splits 19 \
      --n-bootstrap 2000 \
      --batch-size 16 \
      --seed 42
"""

import argparse, random, re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   type=str, required=True,
                   help="MRI 图像根目录")
    p.add_argument("--n-splits",   type=int, default=19,
                   help="折数；留一法请设为受试者数")
    p.add_argument("--n-bootstrap",type=int, default=2000,
                   help="Bootstrap 次数")
    p.add_argument("--batch-size", type=int, default=16,
                   help="batch size")
    p.add_argument("--seed",       type=int, default=42,
                   help="随机种子")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class SliceDataset(Dataset):
    """按切片加载并打标签，返回 (img, label, subject_id)"""
    EXTS = {".png",".jpg",".jpeg"}
    def __init__(self, root_dir, tf=None):
        self.samples = []
        for f in Path(root_dir).rglob("*"):
            if f.suffix.lower() in self.EXTS:
                # label by parent folder name
                label = 1 if "AS" in f.parent.name.upper() else 0
                # subject_id from filename
                m = re.match(r"^(KNEE_\d+|SIJ_\d+)", f.stem)
                sid = m.group(1) if m else re.sub(r"\s*\(.*\)$","",f.stem)
                self.samples.append((str(f),label,sid))
        assert self.samples, "没找到切片"
        self.tf = tf

    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,l,s = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.tf: img = self.tf(img)
        return img, l, s


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self,x):
        f = self.backbone(x)
        return f.view(f.size(0),-1)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def bootstrap_ci(y,probs,n=2000,seed=42):
    rng = np.random.RandomState(seed)
    auc0 = roc_auc_score(y,probs)
    boots=[]
    for _ in range(n):
        idx = rng.randint(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        boots.append(roc_auc_score(y[idx],probs[idx]))
    low,high = np.percentile(boots,[2.5,97.5])
    return auc0,low,high


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # transforms
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = SliceDataset(args.data_dir, tf)
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False, num_workers=4)

    # extract slice-level features
    ext = FeatureExtractor().to(device).eval()
    feats, labels, sids = [],[],[]
    with torch.no_grad():
        for imgs, labs, subj in tqdm(loader, desc="FeatExt"):
            imgs = imgs.to(device)
            f = ext(imgs).cpu().numpy()
            feats.append(f)
            labels.extend(labs.tolist())
            sids.extend(subj)
    X = np.vstack(feats); y = np.array(labels)

    # aggregate to subject-level
    subj_feats, subj_labels = {}, {}
    for feat,lab,sid in zip(X,y,sids):
        subj_feats.setdefault(sid,[]).append(feat)
        subj_labels[sid] = lab
    subj_ids = list(subj_feats)
    Xs = np.vstack([np.mean(subj_feats[s],axis=0) for s in subj_ids])
    ys = np.array([subj_labels[s] for s in subj_ids])

    # choose CV
    if args.n_splits==len(subj_ids):
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_true,y_score = [],[]
    for tr,va in cv.split(Xs, ys):
        clf = LogisticRegression(solver="liblinear", random_state=args.seed)
        clf.fit(Xs[tr], ys[tr])
        probs = clf.predict_proba(Xs[va])[:,1]
        y_true.extend(ys[va].tolist())
        y_score.extend(probs.tolist())

    y_true = np.array(y_true); y_score = np.array(y_score)
    auc0,low,high = bootstrap_ci(y_true,y_score,n=args.n_bootstrap,seed=args.seed)
    print(f"\nSubject-level AUC: {auc0:.3f} 95% CI: [{low:.3f},{high:.3f}]\n")


if __name__=="__main__":
    main()
