#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser("5-fold CV + EarlyStop + WD 微调 ResNet50")
    p.add_argument("--data_dir",     type=str, default="AS_Finetune_Data_balanced",
                   help="平衡后数据集根目录，包含 0_Healthy/ 和 1_AS/")
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--max_epochs",   type=int,   default=10)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers",  type=int,   default=0,
                   help="DataLoader 的 num_workers，设为0可避免多进程问题")
    p.add_argument("--n_splits",     type=int,   default=5,
                   help="GroupKFold 折数")
    p.add_argument("--patience",     type=int,   default=3,
                   help="EarlyStopping 的容忍轮数")
    return p.parse_args()

# —— 一次性运行，生成 prefix → placeholder ID 映射模板 —— 
def extract_unique_prefixes(data_dir):
    files = list(Path(data_dir).rglob("*.png"))
    prefixes = sorted(set("_".join(p.stem.split("_")[:2]) for p in files))
    print("\n🧩 Detected prefixes:\n")
    for i, pref in enumerate(prefixes, 1):
        print(f'    "{pref}": "P{str(i).zfill(3)}",')
    print("\n✅ 请复制上面内容到 custom_id_map，然后注释掉此行调用。")
    sys.exit(0)

# —— 在此处粘贴一次性生成的映射 —— 
custom_id_map = {
    # "KNEE_1":  "P001",
    # "SPINE_1": "P001",  # 合并同一病人
    # "SIJ_1":   "P002",
    # "SIJ_2":   "P003",
    # "sub-01":  "P004",
    # "sub-02":  "P005",
    # ... 继续粘贴并手动合并
}

def get_subject_id(filepath):
    prefix = "_".join(Path(filepath).stem.split("_")[:2])
    return custom_id_map.get(prefix, prefix)

def build_model(num_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feats, num_classes)
    )
    return model.to(device)

def main():
    args = parse_args()
    # extract_unique_prefixes(args.data_dir)  # ← 第一次生成映射时取消注释
    device = torch.device(args.device)
    print(f"\nUsing device: {device}\n")

    # 1) 数据增强
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.2,0.2), scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    # 2) 全量数据加载（不指定 transform）
    full_ds = datasets.ImageFolder(args.data_dir, transform=None)
    samples = full_ds.samples
    paths  = [p for p,_ in samples]
    labels = [l for _,l in samples]
    groups = [get_subject_id(p) for p in paths]

    print(f"Total images: {len(paths)}")
    print(f"Unique subjects: {len(set(groups))}")
    print(f"Class mapping: {full_ds.class_to_idx}\n")

    # 3) GroupKFold 交叉验证
    gkf = GroupKFold(n_splits=args.n_splits)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(paths, labels, groups), 1):
        print(f"=== Fold {fold}/{args.n_splits} ===")
        cnt = Counter(labels[i] for i in val_idx)
        print(f"Val distribution: {cnt}")

        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        train_ds.dataset.transform = train_tf
        val_ds.dataset.transform   = val_tf

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

        model = build_model(len(full_ds.classes), device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, args.max_epochs+1):
            # 训练
            model.train()
            total_loss = 0.0
            for imgs, labs in train_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * imgs.size(0)
            train_loss = total_loss / len(train_loader.dataset)

            # 验证
            model.eval()
            correct = 0
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labs).sum().item()
            val_acc = correct / len(val_loader.dataset)

            print(f"Epoch {epoch}/{args.max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

            # EarlyStopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"best_fold{fold}.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"🛑 Early stopping after {args.patience} no-improve epochs")
                    break

        print(f"✅ Fold {fold} best Val Acc: {best_val_acc:.4f}\n")
        fold_accuracies.append(best_val_acc)

    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print("=== CV Summary ===")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f" Fold {i}: {acc:.4f}")
    print(f" Average Val Acc: {avg_acc:.4f}\n")

if __name__ == "__main__":
    from collections import Counter
    main()
