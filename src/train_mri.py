#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


def parse_args():
    parser = argparse.ArgumentParser(description="Train MRI branch with 5-fold CV")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="AS_Finetune_Data_balanced",
        help="Root directory of preprocessed MRI images",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/mri_model",
        help="Directory to save trained MRI models",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_subject_id(path):
    return Path(path).stem.split("_")[0]


def build_model(num_classes, device):
    model = models.resnet50(pretrained=False)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_feats, num_classes))
    return model.to(device)


def train_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(all_labels, all_probs)
        print(f"  Epoch {epoch+1}/{epochs} Validation AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            yield model.state_dict(), best_auc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    paths = [p for p, _ in dataset.samples]
    labels = [l for _, l in dataset.samples]
    groups = [get_subject_id(p) for p in paths]

    gkf = GroupKFold(n_splits=args.n_splits)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(paths, labels, groups), 1):
        print(f"\n--- Fold {fold}/{args.n_splits} ---")
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = build_model(len(dataset.classes), device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_weights = None
        best_auc = 0.0
        for weights, auc in train_fold(
            model, train_loader, val_loader, criterion, optimizer, device, args.epochs
        ):
            best_weights = weights
            best_auc = auc

        model_path = Path(args.model_dir) / f"best_model_fold_{fold-1}.pth"
        torch.save(best_weights, model_path)
        print(
            f"Saved best model of fold {fold} with AUC {best_auc:.4f} to {model_path}"
        )


if __name__ == "__main__":
    main()
