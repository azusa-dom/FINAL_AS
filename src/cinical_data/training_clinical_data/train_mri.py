import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.models_mri import get_mri_model
from core.dataset import MRIImageFolderDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train MRI classification model")
    parser.add_argument("--train-dir", type=str, required=True, help="Directory with training images (mri_AS)")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory with validation images (mri_health)")
    parser.add_argument("--model-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    return parser.parse_args()

def build_optimizer(model, lr_head, lr_backbone, weight_decay, freeze_backbone):
    if freeze_backbone:
        return optim.Adam(model.fc.parameters(), lr=lr_head, weight_decay=weight_decay)
    else:
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
        return optim.Adam([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head}
        ], weight_decay=weight_decay)

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Using device: {device}\n")

    train_ds = MRIImageFolderDataset(args.train_dir, train=True)
    val_ds   = MRIImageFolderDataset(args.val_dir, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_mri_model(
        num_classes=2,
        in_channels=3,
        pretrained=args.pretrained,
        freeze_backbone=(args.freeze_epochs > 0),
        dropout_p=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay, freeze_backbone=(args.freeze_epochs > 0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n🔁 Epoch {epoch}/{args.epochs}")

        # 解冻阶段
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("🔓 Unfreezing backbone for fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = build_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay, freeze_backbone=False)

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

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.model_dir, f"best_model_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model to: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print("\n🏁 Training finished.")

if __name__ == "__main__":
    main()
