import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

# ✅ 使用正确的类名 ClinicalDataset（你在 dataset.py 里定义的）
from .dataset import ClinicalDataset
from .models import SimpleResNet, SimpleCNN, SimpleMLP
from .utils import get_class_weights

# ✅ 替代原先的 get_kfold_strafied_sampler
def get_kfold_stratified_sampler(data_dir, batch_size=32, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    folds = []

    for fold in range(n_splits):
        train_csv = os.path.join(data_dir, f'fold_{fold}_train.csv')
        val_csv = os.path.join(data_dir, f'fold_{fold}_val.csv')

        train_dataset = ClinicalDataset(csv_path=train_csv, label_column="Disease")
        val_dataset = ClinicalDataset(csv_path=val_csv, label_column="Disease")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        folds.append((train_loader, val_loader))

    return folds

# ===========================
# 主训练函数
# ===========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kfold_loader = get_kfold_stratified_sampler(args.data_dir)

    preds_output_dir = os.path.join(args.model_dir, 'clinical_preds')
    os.makedirs(preds_output_dir, exist_ok=True)

    for fold, (train_loader, val_loader) in enumerate(kfold_loader):
        print(f"\n===== Fold {fold} =====")

        # 模型初始化
        if args.model_name == 'resnet':
            model = SimpleResNet(num_classes=2).to(device)
        elif args.model_name == 'cnn':
            model = SimpleCNN(num_classes=2).to(device)
        else:
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch[0].shape[1]
            model = SimpleMLP(input_dim=input_dim, num_classes=2).to(device)

        # 损失函数和优化器
        class_weights = get_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [T]"):
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            fold_true_labels = []
            fold_pred_logits = []

            with torch.no_grad():
                for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [V]"):
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    fold_true_labels.append(labels.cpu().numpy())
                    fold_pred_logits.append(outputs.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(args.model_dir, f"best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Best model for fold {fold} saved to {model_save_path}")

        print(f"💾 Saving predictions for Fold {fold}...")
        fold_true_labels = np.concatenate(fold_true_labels, axis=0)
        fold_pred_logits = np.concatenate(fold_pred_logits, axis=0)

        num_classes = fold_pred_logits.shape[1]
        logit_columns = [f'logit_{i}' for i in range(num_classes)]

        df_preds = pd.DataFrame(fold_pred_logits, columns=logit_columns)
        df_preds['true_label'] = fold_true_labels

        preds_save_path = os.path.join(preds_output_dir, f'fold_{fold}_predictions.csv')
        df_preds.to_csv(preds_save_path, index=False)
        print(f"✅ Predictions saved to {preds_save_path}")


# ===========================
# 主程序入口
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on clinical tabular data.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, choices=["mlp", "cnn", "resnet"], default="resnet")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    train(args)
