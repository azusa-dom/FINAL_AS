# Final_AS/code/src/train.py
# FINAL, SELF-CONTAINED TRAINING SCRIPT - v3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

# --- VERIFICATION ---
print("✅✅✅ Successfully running the FINAL, ALL-IN-ONE training script! ✅✅✅")

# --- Model Definition (Included directly in this file) ---
class ClinicalNet(nn.Module):
    """A simple MLP with Dropout to prevent overfitting."""
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

# --- Data Loading (Final Robust Version) ---
class ClinicalDataset(TensorDataset):
    """A Dataset class that is robust to a missing ID column."""
    def __init__(self, df, label_column, id_column):
        self.labels = torch.tensor(df[label_column].values, dtype=torch.long)
        
        # --- CORE FIX: Handle missing patient_id column ---
        if id_column in df.columns:
            self.patient_ids = df[id_column].values
        else:
            # Create a placeholder if the ID column is not found
            self.patient_ids = np.full(len(df), fill_value="N/A", dtype=object)
        
        feature_cols = [c for c in df.columns if c not in [label_column, id_column]]
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.unique_labels = df[label_column].unique()
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.patient_ids[idx]

def get_kfold_loaders(data_dir, id_column, label_column, n_splits=5, batch_size=32):
    loaders = []
    all_feature_columns = None  # 用于统一列顺序和补缺

    # 先读取一个 fold，记录标准列名（不含 ID 和 label）
    first_df = pd.read_csv(os.path.join(data_dir, "fold_0_train.csv"))
    all_feature_columns = [c for c in first_df.columns if c not in [label_column, id_column]]

    for i in range(n_splits):
        train_df = pd.read_csv(os.path.join(data_dir, f"fold_{i}_train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, f"fold_{i}_val.csv"))

        for df in [train_df, val_df]:
            # 补全缺失列（全填0）
            for col in all_feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            # 移除额外列，重新排序
            kept_cols = all_feature_columns + [label_column, id_column]
            df.drop(columns=[c for c in df.columns if c not in kept_cols], inplace=True)
            existing_cols = [c for c in kept_cols if c in df.columns]
            df = df[existing_cols]


        # 创建 Dataset 和 DataLoader
        train_ds = ClinicalDataset(train_df, label_column=label_column, id_column=id_column)
        val_ds = ClinicalDataset(val_df, label_column=label_column, id_column=id_column)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        loaders.append((train_loader, val_loader))

    return loaders


def get_class_weights(dataset):
    class_counts = torch.bincount(dataset.labels)
    class_weights = 1. / (class_counts.float() + 1e-6)
    return class_weights

# --- Main Training Logic ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    kfold_loaders = get_kfold_loaders(
        args.data_dir, id_column=args.id_column, label_column=args.label_column
    )
    
    preds_out_dir = os.path.join(args.model_dir, "clinical_preds")
    os.makedirs(preds_out_dir, exist_ok=True)

    for fold, (train_loader, val_loader) in enumerate(kfold_loaders):
        print(f"\n===== Fold {fold} =====")
        num_classes = len(train_loader.dataset.unique_labels)
        sample_feats, _, _ = next(iter(train_loader))
        input_dim = sample_feats.shape[1]

        model = ClinicalNet(input_size=input_dim, hidden_size=64, output_size=num_classes).to(device)

        weights = get_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_val_loss, best_model_state = float("inf"), None
        patience, no_improve_epochs = 3, 0

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for feats, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [T]"):
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(feats)
                loss = criterion(outputs, labels)
                loss.backward(); optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feats, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [V]"):
                    feats, labels = feats.to(device), labels.to(device)
                    outputs = model(feats)
                    val_loss += criterion(outputs, labels).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= patience:
                print(f"⚠️  Early stopping triggered at Epoch {epoch+1}")
                break
        
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"best_model_fold_{fold}.pth"))
        
        model.eval()
        all_logits, all_labels, all_ids = [], [], []
        with torch.no_grad():
            for feats, labels, ids in val_loader:
                outputs = model(feats.to(device))
                all_logits.append(outputs.cpu().numpy())
                all_labels.append(labels.numpy())
                all_ids.append(np.array(ids))
        
        final_logits = np.concatenate(all_logits)
        final_labels = np.concatenate(all_labels)
        final_ids = np.concatenate(all_ids)
        
        logit_cols = [f"logit_{i}" for i in range(num_classes)]
        df_preds = pd.DataFrame(final_logits, columns=logit_cols)
        df_preds["true_label"] = final_labels
        df_preds["patient_id"] = final_ids
        
        preds_save_path = os.path.join(preds_out_dir, f"fold_{fold}_predictions.csv")
        df_preds.to_csv(preds_save_path, index=False)
        print(f"📦 Fold {fold} predictions saved to: {preds_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-in-one script to train a model on clinical data.")
    parser.add_argument("--data_dir", required=True, help="Directory with cross-validation fold data.")
    parser.add_argument("--model_dir", required=True, help="Directory to save models and predictions.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column.")
    parser.add_argument("--id_column", type=str, default="patient_id", help="Name of the patient ID column.")
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    train(args)