import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

def parse_args():
    p = argparse.ArgumentParser(description="微调 ResNet50 on balanced AS vs Healthy")
    p.add_argument("--data_dir",    type=str, default="AS_Finetune_Data_balanced",
                   help="平衡后数据集根目录，含 0_Healthy 和 1_AS 两个子文件夹")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 的 num_workers，设为0可避免多进程错误")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------------
    # 1. Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(10),
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

    # -----------------------------------------------------------------------------
    # 2. Dataset 和 80/20 划分
    full_ds = datasets.ImageFolder(args.data_dir)
    print("Classes:", full_ds.class_to_idx)
    n = len(full_ds)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split].tolist(), indices[split:].tolist()

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform   = val_tf

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)

    print(f"Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")

    # -----------------------------------------------------------------------------
    # 3. Model: ResNet50, 冻结除 layer4 + fc 外的所有层
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(full_ds.classes))
    model = model.to(device)

    # -----------------------------------------------------------------------------
    # 4. 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=args.lr)

    # -----------------------------------------------------------------------------
    # 5. 训练 & 验证循环
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        # ——— 训练 ———
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ——— 验证 ———
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       "best_resnet50_balanced.pth")

    print("训练结束，最佳验证准确率：", best_acc)

if __name__ == "__main__":
    main()
