import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

def parse_args():
    p = argparse.ArgumentParser("GroupKFold 微调 ResNet")
    p.add_argument("--data_dir",    type=str, default="AS_Finetune_Data_balanced",
                   help="数据集根目录，包含 0_Healthy/ 和 1_AS/")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_splits",    type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1. 数据增强 & 预处理
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.2,0.2), scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 2. 载入全量 Dataset（不设置 transform）
    full_ds = datasets.ImageFolder(args.data_dir, transform=None)
    print("Classes:", full_ds.class_to_idx)

    # 3. 准备分组信息：按文件名前缀 sub-XX 或 SIJ_XX
    paths  = [p for p,_ in full_ds.samples]
    labels = [l for _,l in full_ds.samples]
    groups = [Path(p).stem.split("_")[0] for p in paths]
    print(f"Total images: {len(paths)}, Subjects: {len(set(groups))}")

    # 4. GroupKFold 拆分（这里只用第 1 折）
    gkf = GroupKFold(n_splits=args.n_splits)
    train_idx, val_idx = next(gkf.split(paths, labels, groups))
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    # 分别赋予不同的 transform
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform   = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    print(f"Fold 1/{args.n_splits} — Train images: {len(train_ds)}, Val images: {len(val_ds)}")

    # 5. 构建模型：ResNet50，只解冻 layer4 + fc
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for name,param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(full_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr)

    # 6. 训练 + 验证循环
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        # 训练
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs} "
              f"Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_resnet50_groupk.pth")

    print("Finished. Best Val Acc:", best_acc)

if __name__ == "__main__":
    main()
