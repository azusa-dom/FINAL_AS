# src/train_mri.py

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_mri import get_mri_model
from dataset import MRIImageFolderDataset

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage fine-tuning for MRI classification"
    )
    p.add_argument("--train-dir",    type=str, required=True,
                   help="训练集根目录 (含 0_Healthy/ 1_AS/)")
    p.add_argument("--val-dir",      type=str, required=True,
                   help="验证集根目录")
    p.add_argument("--model-dir",    type=str, default="../checkpoints",
                   help="保存最佳模型的目录")
    p.add_argument("--epochs",       type=int,   default=30,
                   help="总训练轮数")
    p.add_argument("--batch-size",   type=int,   default=8,
                   help="Batch size")
    p.add_argument("--freeze-epochs",type=int,   default=3,
                   help="只训练 head 的轮数，之后自动解冻 backbone")
    p.add_argument("--lr-head",      type=float, default=1e-4,
                   help="head 的学习率")
    p.add_argument("--lr-backbone",  type=float, default=1e-5,
                   help="backbone 的学习率 (微调阶段)")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="Adam 权重衰减")
    p.add_argument("--patience",     type=int,   default=5,
                   help="早停耐心轮数")
    p.add_argument("--pretrained",   action="store_true",
                   help="加载 ImageNet 预训练权重")
    return p.parse_args()

def build_optimizer(model, lr_head, lr_backbone, weight_decay, freeze_backbone):
    """
    根据是否冻结 backbone 来构建 optimizer。
    freeze_backbone=True 只训练 head，使用 lr_head；
    否则为 head/backbone 分配不同的 lr。
    """
    if freeze_backbone:
        # 只优化最后的 fc 层
        params = model.fc.parameters()
        return optim.Adam(params, lr=lr_head, weight_decay=weight_decay)
    else:
        # head 和 backbone 分组优化
        head_params = list(model.fc.parameters())
        backbone_params = [
            p for n, p in model.named_parameters()
            if "fc" not in n and p.requires_grad
        ]
        return optim.Adam([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,      "lr": lr_head}
        ], weight_decay=weight_decay)

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Dataset & DataLoader
    train_ds = MRIImageFolderDataset(args.train_dir, train=True)
    val_ds   = MRIImageFolderDataset(args.val_dir,   train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Model instantiation (保留 3 通道 conv1)
    model = get_mri_model(
        num_classes=2,
        in_channels=3,
        pretrained=args.pretrained,
        freeze_backbone=(args.freeze_epochs > 0),
        dropout_p=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # 初始阶段：只训练 head
    optimizer = build_optimizer(
        model,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        freeze_backbone=(args.freeze_epochs > 0)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2
    )

    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # 解冻阶段：在第 freeze_epochs+1 轮自动解冻
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone for full fine-tuning")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = build_optimizer(
                model,
                lr_head=args.lr_head,
                lr_backbone=args.lr_backbone,
                weight_decay=args.weight_decay,
                freeze_backbone=False
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=2
            )

        # 训练
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_loss /= len(val_ds)
        val_acc  = correct / len(val_ds)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # LR 调度 & 早停
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            save_path = os.path.join(
                args.model_dir, f"best_model_epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"⛔ Early stopping at epoch {epoch}")
                break

    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
