#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear_probe_sij.py

Use ImageNet-pretrained ResNet-18 features + Linear Probe on SIJ MRI (6 AS vs 2 Healthy)
with LOOCV AUC and permutation test. No extra pretraining needed.
"""

import os, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sij-as-dir',      required=True, help='AS SIJ images (jpg/png)')
    p.add_argument('--sij-healthy-dir', action='append', required=True,
                   help='Healthy SIJ images (jpg/png), can repeat')
    p.add_argument('--n-perm', type=int, default=5000, help='Permutations for p-value')
    p.add_argument('--device', default='cpu', help='cpu or cuda')
    return p.parse_args()

def extract_resnet_features(paths, model, transform, device):
    feats = []
    model.eval()
    with torch.no_grad():
        for p in tqdm(paths, desc='Extracting features'):
            img = Image.open(p).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            f = model(x).view(-1).cpu().numpy()
            feats.append(f)
    return np.stack(feats, axis=0)

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) 构建 ResNet-18 特征提取器
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.to(device)

    # 2) 收集 SIJ 图像路径和标签
    paths, labels = [], []
    # AS -> 1
    for root,_,fs in os.walk(args.sij_as_dir):
        for fn in fs:
            if fn.lower().endswith(('.jpg','.png')):
                paths.append(os.path.join(root, fn)); labels.append(1)
    # Healthy -> 0
    for hd in args.sij_healthy_dir:
        for root,_,fs in os.walk(hd):
            for fn in fs:
                if fn.lower().endswith(('.jpg','.png')):
                    paths.append(os.path.join(root, fn)); labels.append(0)
    y = np.array(labels)

    # 3) 特征提取
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    X = extract_resnet_features(paths, resnet, transform, device)

    # 4) LOOCV + AUC
    loo = LeaveOneOut()
    preds, trues = [], []
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:,1][0]
        preds.append(p); trues.append(y[test_idx][0])
    auc = roc_auc_score(trues, preds)
    print(f'LOOCV AUC = {auc:.3f}')

    # 5) 置换检验
    obs_delta = np.mean([p for p,l in zip(preds,y) if l==1]) \
              - np.mean([p for p,l in zip(preds,y) if l==0])
    count = 0
    for _ in range(args.n_perm):
        yp = np.random.permutation(y)
        delta = np.mean([p for p,l in zip(preds,yp) if l==1]) \
              - np.mean([p for p,l in zip(preds,yp) if l==0])
        if delta >= obs_delta: 
            count += 1
    pval = (count+1)/(args.n_perm+1)
    print(f'Permutation p-value = {pval:.4f}')

if __name__ == '__main__':
    main()