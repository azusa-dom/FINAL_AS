#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

print("✅✅✅ Running training script with Temperature Scaling & Mondrian-style plots ✅✅✅")

class ClinicalNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(ClinicalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class _ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(_ECELoss, self).__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confs, preds = torch.max(softmaxes, 1)
        accs = preds.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for i in range(self.n_bins):
            lo, hi = i / self.n_bins, (i + 1) / self.n_bins
            in_bin = confs.gt(lo) & confs.le(hi)
            prop = in_bin.float().mean()
            if prop.item() > 0:
                ece += torch.abs(confs[in_bin].mean() - accs[in_bin].float().mean()) * prop
        return ece

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, loader, device):
        self.to(device)
        nll = nn.CrossEntropyLoss().to(device)
        ece_crit = _ECELoss(n_bins=10).to(device)
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                logits_list.append(self.model(x))
                labels_list.append(y)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        print("Before ECE:", float(ece_crit(logits, labels)))
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def _eval():
            optimizer.zero_grad()
            loss = nll(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        print("After ECE:", float(ece_crit(logits / self.temperature, labels)))
        return self

class ClinicalDataset(TensorDataset):
    def __init__(self, df, label_column, id_column):
        self.labels = torch.tensor(df[label_column].values, dtype=torch.long)
        self.patient_ids = df[id_column].values if id_column in df.columns else np.full(len(df), 'N/A', dtype=object)
        feat_cols = [c for c in df.columns if c not in [label_column, id_column]]
        self.features = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        self.unique_labels = df[label_column].unique()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.patient_ids[idx]

def get_kfold_loaders(data_dir, id_column, label_column, n_splits=5, batch_size=32):
    loaders = []
    first_df = pd.read_csv(os.path.join(data_dir, 'fold_0_train.csv'))
    feat_cols = [c for c in first_df.columns if c != label_column and c != id_column]

    for i in range(n_splits):
        train_df = pd.read_csv(os.path.join(data_dir, f'fold_{i}_train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, f'fold_{i}_val.csv'))

        for df in (train_df, val_df):
            for col in feat_cols:
                if col not in df.columns:
                    df[col] = 0.0

        train_cols = feat_cols + [label_column]
        val_cols = feat_cols + [label_column]

        if id_column in train_df.columns:
            train_cols.append(id_column)
        if id_column in val_df.columns:
            val_cols.append(id_column)

        train_df = train_df[train_cols]
        val_df = val_df[val_cols]

        train_ds = ClinicalDataset(train_df, label_column=label_column, id_column=id_column)
        val_ds = ClinicalDataset(val_df, label_column=label_column, id_column=id_column)

        loaders.append((
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        ))

    return loaders

def get_class_weights(dataset):
    counts = torch.bincount(dataset.labels)
    return 1.0 / (counts.float() + 1e-6)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    kfold_loaders = get_kfold_loaders(args.data_dir, args.id_column, args.label_column)

    os.makedirs(args.model_dir, exist_ok=True)
    plots_dir = os.path.join(args.model_dir, 'calibration_plots')
    os.makedirs(plots_dir, exist_ok=True)

    mondrian = {'perfect': '#000000', 'uncal': '#0033A0', 'cal': '#D7141A'}

    for fold, (train_loader, val_loader) in enumerate(kfold_loaders):
        print(f"\n=== Fold {fold} ===")
        num_classes = len(train_loader.dataset.unique_labels)
        input_dim = train_loader.dataset.features.shape[1]
        model = ClinicalNet(input_dim, hidden_size=64, output_size=num_classes).to(device)

        weights = get_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_loss, no_improve = float('inf'), 0
        for epoch in range(args.epochs):
            model.train()
            for feats, labels, _ in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(feats), labels)
                loss.backward(); optimizer.step()

            model.eval()
            val_loss = sum(criterion(model(feats.to(device)), labels.to(device)).item()
                           for feats, labels, _ in val_loader) / len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
                best_state = model.state_dict()
            else:
                no_improve += 1
            if no_improve >= 3:
                print("⚠️  Early stopping")
                break

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_fold_{fold}.pth'))

        print("-- Calibrating temperature --")
        calibrated_model = ModelWithTemperature(model).set_temperature(val_loader, device)

        model.eval(); calibrated_model.eval()
        all_uncal, all_cal, all_labels = [], [], []
        with torch.no_grad():
            for feats, labels, _ in val_loader:
                feats = feats.to(device)
                up = torch.softmax(model(feats), dim=1)[:,1].cpu().numpy()
                cp = torch.softmax(calibrated_model(feats), dim=1)[:,1].cpu().numpy()
                all_uncal.append(up); all_cal.append(cp); all_labels.append(labels.numpy())
        uncal_probs = np.concatenate(all_uncal)
        cal_probs   = np.concatenate(all_cal)
        labels_arr  = np.concatenate(all_labels)

        plt.figure(figsize=(8, 6))
        ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
        ax2 = plt.subplot2grid((3,1), (2,0))

        ax1.plot([0,1], [0,1], '-', color=mondrian['perfect'], linewidth=2)
        un_frac, un_mean = calibration_curve(labels_arr, uncal_probs, n_bins=5)
        ca_frac, ca_mean = calibration_curve(labels_arr, cal_probs,   n_bins=5)
        ax1.plot(un_mean, un_frac, 'o-', color=mondrian['uncal'], linewidth=2, markersize=8, label='Uncalibrated')
        ax1.plot(ca_mean, ca_frac, '^-', color=mondrian['cal'], linewidth=2, markersize=8, label='Calibrated')

        ax2.hist(uncal_probs, bins=5, range=(0,1), histtype='step', linewidth=2, linestyle='-', color=mondrian['uncal'], label='Uncalibrated')
        ax2.hist(cal_probs, bins=5, range=(0,1), histtype='step', linewidth=2, linestyle='--', color=mondrian['cal'], label='Calibrated')

        for ax in (ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.tick_params(axis='both', which='major', labelsize=12)

        ax1.set_ylabel('Fraction of positives', fontsize=14, fontweight='bold')
        ax1.set_ylim(-0.02, 1.02)
        ax1.legend(loc='lower right', fontsize=12, frameon=False)
        ax1.set_title(f'Calibration Plot (Fold {fold})', fontsize=16, fontweight='bold')

        ax2.set_xlabel('Mean predicted value', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper center', ncol=2, fontsize=12, frameon=False)

        plt.tight_layout()
        save_path = os.path.join(plots_dir, f'fold_{fold}_mondrian_calibration.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved calibration plot: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ClinicalNet with Temperature Scaling & Mondrian-style calibration plots")
    parser.add_argument("--data_dir", required=True, help="Path to directory with fold_*_train.csv and fold_*_val.csv files.")
    parser.add_argument("--model_dir", required=True, help="Directory to save models and plots.")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column.")
    parser.add_argument("--id_column", type=str, default="patient_id", help="Name of the ID column.")
    args = parser.parse_args()
    train(args)
