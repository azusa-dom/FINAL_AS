import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser("5-fold CV + EarlyStop + WD 微调 ResNet")
    p.add_argument("--data_dir",    type=str, default="AS_Finetune_Data_balanced")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--max_epochs",  type=int, default=10)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_splits",    type=int, default=5)
    p.add_argument("--patience",    type=int, default=3,
                   help="EarlyStopping 的容忍轮次")
    return p.parse_args()

# ✅ 你可以在这里自定义受试者 ID 映射
custom_id_map = {
    "KNEE_1": "P001",
    "SPINE_1": "P001",
    "KNEE_2": "P002",
    "SPINE_2": "P002",
    "SJI_1": "P003",
    "SJI_2": "P004",
    "sub-01": "P005",
    "sub-02": "P006",
    "sub-03": "P007",
    "sub-04": "P008",
    "sub-05": "P009",
    "sub-06": "P010",
    "sub-07": "P011",
    "sub-08": "P012",
    "sub-09": "P013",
    "sub-10": "P014",
    # ... 补充完整你数据中的所有前缀
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
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 数据增强
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.2,0.2), scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # 加载数据
    full_ds = datasets.ImageFolder(args.data_dir, transform=None)
    samples = full_ds.samples
    paths  = [p for p, _ in samples]
    labels = [l for _, l in samples]
    groups = [get_subject_id(p) for p in paths]

    print(f"Total images: {len(paths)}")
    print(f"Unique subjects: {len(set(groups))}")
    print(f"Class mapping: {full_ds.class_to_idx}")

    gkf = GroupKFold(n_splits=args.n_splits)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(paths, labels, groups), 1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")

        # 统计每类验证样本数
        val_label_count = Counter([labels[i] for i in val_idx])
        print(f"Val class distribution: {val_label_count}")

        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        train_ds.dataset.transform = train_tf
        val_ds.dataset.transform   = val_tf

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

        model = build_model(num_classes=len(full_ds.classes), device=device)
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
            total_loss = 0
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

            print(f"Epoch {epoch}/{args.max_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

            # EarlyStopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"best_fold{fold}.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"🛑 Early stopped after {args.patience} epochs with no improvement.")
                    break

        print(f"✅ Fold {fold} best ValAcc: {best_val_acc:.4f}")
        fold_accuracies.append(best_val_acc)

    # 交叉验证结果汇总
    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\n=== Cross-Validation Summary ===")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {i}: {acc:.4f}")
    print(f"  Average: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
