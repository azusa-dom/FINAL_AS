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

print("✅✅✅ Running training script with Temperature Scaling & Mondrian-style plots (v2 - Corrected) ✅✅✅")

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
    def __init__(self, n_bins=15): # Using 15 bins to match manuscript
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
                acc_in_bin = accs[in_bin].float().mean()
                avg_conf_in_bin = confs[in_bin].mean()
                ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop
        return ece

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

    def set_temperature(self, loader, device):
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)
        
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                logits_list.append(self.model(x))
                labels_list.append(y)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        ece_before = ece_criterion(logits, labels).item()
        print(f"Before temperature scaling ECE: {ece_before:.4f}")

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        
        ece_after = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        print(f"After temperature scaling ECE: {ece_after:.4f}")
        
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
    # This logic assumes all fold files exist and have consistent columns.
    # A robust implementation might check each file.
    first_df = pd.read_csv(os.path.join(data_dir, 'fold_0_train.csv'))
    feat_cols = [c for c in first_df.columns if c != label_column and c != id_column]

    for i in range(n_splits):
        train_df = pd.read_csv(os.path.join(data_dir, f'fold_{i}_train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, f'fold_{i}_val.csv'))

        train_ds = ClinicalDataset(train_df, label_column=label_column, id_column=id_column)
        val_ds = ClinicalDataset(val_df, label_column=label_column, id_column=id_column)

        loaders.append((
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        ))
    return loaders, feat_cols

def get_class_weights(dataset):
    counts = torch.bincount(dataset.labels)
    weights = 1.0 / (counts.float() + 1e-6)
    return weights / weights.sum() # Normalize weights

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    kfold_loaders, feature_names = get_kfold_loaders(args.data_dir, args.id_column, args.label_column, n_splits=args.n_splits)

    os.makedirs(args.model_dir, exist_ok=True)
    plots_dir = os.path.join(args.model_dir, 'calibration_plots')
    os.makedirs(plots_dir, exist_ok=True)
    # **NEW**: Create a directory for calibrated predictions
    preds_dir = os.path.join(args.model_dir, 'clinical_preds')
    os.makedirs(preds_dir, exist_ok=True)

    for fold, (train_loader, val_loader) in enumerate(kfold_loaders):
        print(f"\n=== Fold {fold} ===")
        input_dim = train_loader.dataset.features.shape[1]
        num_classes = len(train_loader.dataset.unique_labels)
        
        model = ClinicalNet(input_dim, hidden_size=64, output_size=num_classes).to(device)
        
        weights = get_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_loss = float('inf')
        no_improve_epochs = 0
        
        for epoch in range(args.epochs):
            model.train()
            for feats, labels, _ in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(feats), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            current_val_loss = 0
            with torch.no_grad():
                for feats, labels, _ in val_loader:
                    feats, labels = feats.to(device), labels.to(device)
                    current_val_loss += criterion(model(feats), labels).item()
            
            val_loss = current_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_model_state = model.state_dict()
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= args.patience:
                print(f"⚠️  Early stopping after {args.patience} epochs with no improvement.")
                break

        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'best_model_fold_{fold}.pth'))

        print("-- Calibrating temperature on validation set --")
        calibrated_model = ModelWithTemperature(model).set_temperature(val_loader, device)

        # **NEW**: Save calibrated predictions for the validation set of this fold
        calibrated_model.eval()
        fold_preds = []
        with torch.no_grad():
            for feats, labels, patient_ids in val_loader:
                feats = feats.to(device)
                calibrated_logits = calibrated_model(feats)
                calibrated_probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()
                
                for i in range(len(patient_ids)):
                    fold_preds.append({
                        'patient_id': patient_ids[i],
                        'true_label': labels[i].item(),
                        'prob': calibrated_probs[i, 1], # Probability of positive class (AS)
                        'logit_0': calibrated_logits[i, 0].item(),
                        'logit_1': calibrated_logits[i, 1].item(),
                    })
        
        preds_df = pd.DataFrame(fold_preds)
        preds_df.to_csv(os.path.join(preds_dir, f'fold_{fold}_predictions.csv'), index=False)
        print(f"✅ Saved calibrated predictions for fold {fold} to {preds_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ClinicalNet with Temperature Scaling & Mondrian-style calibration plots")
    parser.add_argument("--data_dir", required=True, help="Path to directory with fold_*_train.csv and fold_*_val.csv files.")
    parser.add_argument("--model_dir", required=True, help="Directory to save models and plots.")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Epochs to wait for improvement before early stopping.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column.")
    parser.add_argument("--id_column", type=str, default="Patient_ID", help="Name of the ID column (case-sensitive).")
    args = parser.parse_args()
    train(args)