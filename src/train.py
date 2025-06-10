import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import MRIDataset, split_patient_kfold, create_dataloader
from .models import get_model, ClinicalMLP, ResNet18Encoder
from .utils import seed_everything, save_json


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for img, clin, label in loader:
        img, clin, label = img.to(device), clin.to(device), label.to(device)
        optimizer.zero_grad()
        if isinstance(model, ClinicalMLP):
            output = model(clin)
        elif isinstance(model, ResNet18Encoder):
            output = model(img)
        else:
            output = model(img, clin)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for img, clin, label in loader:
            img, clin, label = img.to(device), clin.to(device), label.to(device)
            if isinstance(model, ClinicalMLP):
                output = model(clin)
            elif isinstance(model, ResNet18Encoder):
                output = model(img)
            else:
                output = model(img, clin)
            loss = criterion(output, label)
            total_loss += loss.item() * label.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return total_loss / total, correct / total


def run_fold(fold, train_idx, val_idx, df, args, device):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_csv = Path(f"train_{fold}.csv")
    val_csv = Path(f"val_{fold}.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_loader = create_dataloader(train_csv, args.img_dir, args.batch_size, True, True, args.mode)
    val_loader = create_dataloader(val_csv, args.img_dir, args.batch_size, False, False, args.mode)

    clin_feat_dim = train_df.shape[1] - 2
    kwargs = {}
    if args.model == "clin_only":
        kwargs["in_features"] = clin_feat_dim
    elif args.model == "early_fusion":
        kwargs["clin_feat_dim"] = clin_feat_dim
    model = get_model(args.model, **kwargs)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"[Fold {fold}] Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = Path(args.save_dir) / f"fold{fold}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return best_acc


def main(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MRIDataset(args.csv, args.img_dir, mode=args.mode, train=True, augment=True)
    df = dataset.df
    folds = split_patient_kfold(df, 5)
    accs = []
    for i, (train_idx, val_idx) in enumerate(folds):
        acc = run_fold(i, train_idx, val_idx, df, args, device)
        accs.append(acc)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_json({"accuracy": float(sum(accs) / len(accs))}, Path(args.save_dir) / "cv_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--csv", type=Path, default=Path("data/clinical.csv"))
    parser.add_argument("--img_dir", type=Path, default=Path("data/mri"))
    parser.add_argument("--mode", type=str, default="png")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    main(args)
