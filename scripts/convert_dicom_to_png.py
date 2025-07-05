#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simclr_linear_probe_sij.py

Self-Supervised Pretraining (SimCLR) on unlabeled pelvis MRI + Linear Probe on SIJ MRI
with subject-level Leave-One-Out Cross-Validation (LOOCV) AUC and permutation test.

Workflow:
1. Pre-train a ResNet18 encoder using SimCLR on a large, unlabeled dataset of pelvis MRIs.
   (This step is skipped if the output encoder file already exists).
2. Load the pre-trained encoder.
3. For the downstream task, load SIJ images from AS and Healthy cohorts.
   - The subject ID is inferred from the parent directory name of the image files.
4. Extract features for every SIJ image and aggregate them (mean pooling) to get a
   single feature vector per subject.
5. Perform a linear probe using Logistic Regression with LOOCV on the subject-level features.
6. Calculate the LOOCV AUC score.
7. Run a permutation test on the LOOCV predictions to assess statistical significance.

Dependencies:
  pip install torch torchvision lightly scikit-learn tqdm pillow

Usage:
  python simclr_linear_probe_sij.py \
    --pelvis-dir /path/to/pelvis_png_images \
    --sij-as-dir /path/to/mri_AS \
    --sij-healthy-dir /path/to/health1 \
    --sij-healthy-dir /path/to/health2 \
    --encoder-out encoder_simclr.pth \
    --batch-size-pre 64 --epochs-pre 50 \
    --n-perm 5000 \
    --device cpu
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import lightly
from lightly.data import LightlyDataset, ImageCollateFunction
from lightly.models.simclr import SimCLR

def parse_args():
    p = argparse.ArgumentParser(description="SimCLR Pretrain + Linear Probe on SIJ MRI")
    p.add_argument('--pelvis-dir', required=True, help='Directory with pelvis images for self-supervised pretraining')
    p.add_argument('--sij-as-dir', required=True, help='Root directory with AS SIJ images (subject folders inside)')
    p.add_argument('--sij-healthy-dir', action='append', required=True, help='Root directory with Healthy SIJ images (subject folders inside)')
    p.add_argument('--encoder-out', default='encoder_simclr.pth', help='Path to save/load SimCLR encoder weights')
    p.add_argument('--batch-size-pre', type=int, default=64, help='Batch size for SimCLR pretraining')
    p.add_argument('--epochs-pre', type=int, default=50, help='Epochs for SimCLR pretraining')
    p.add_argument('--n-perm', type=int, default=5000, help='Number of permutations for p-value test')
    p.add_argument('--force-pretrain', action='store_true', help='Force pretraining even if encoder file exists')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device: cpu or cuda')
    return p.parse_args()

# -------- 1. Pretrain SimCLR -------- #
def pretrain_simclr(pelvis_dir, encoder_out, batch_size, epochs, device):
    print("--- Starting SimCLR Pre-training ---")
    # Dataset & DataLoader
    dataset = LightlyDataset(input_dir=pelvis_dir)
    collate_fn = ImageCollateFunction(input_size=224, vf_prob=0.5, hf_prob=0.5, rr_prob=0.5)
    # Set num_workers=0 for better cross-platform compatibility, especially on CPU
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=True)

    # Model initialization with backbone
    backbone = models.resnet18()
    backbone.fc = nn.Identity() # Remove final classification layer
    
    # === THIS IS THE FIX ===
    # Pass the backbone to the SimCLR model constructor
    model = SimCLR(backbone=backbone)
    # =======================
    model.to(device)

    criterion = lightly.loss.NTXentLoss() # SimCLRLoss is an alias
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        # The collate function returns a tuple of ((x0, x1), labels, fnames)
        for (x0, x1), _, _ in tqdm(loader, desc=f"SimCLR Epoch {ep}/{epochs}"):
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = model(x0, x1)
            loss = criterion(z0, z1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep}: Average Loss = {avg_loss:.4f}")
        
    # Save encoder backbone
    torch.save(model.backbone.state_dict(), encoder_out)
    print(f"✅ Saved pre-trained encoder weights to {encoder_out}")

# -------- 2. Linear Probe -------- #
def extract_features_by_subject(root_dirs_labels, encoder, transform, device):
    """Extracts features and aggregates them by subject ID."""
    encoder.eval()
    
    subject_data = {} # {sid: {'paths': [...], 'label': 0/1}}
    print("\n🔎 Scanning SIJ directories to identify subjects and images...")
    for root_dir, label in root_dirs_labels:
        # Handle case where directory might not exist
        if not os.path.isdir(root_dir):
            print(f"⚠️ Warning: Directory not found, skipping: {root_dir}")
            continue
        for subject_id in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject_id)
            if os.path.isdir(subject_path):
                # Use subject path as a more unique key if subject_ids can overlap (e.g., 'health1/S01', 'health2/S01')
                unique_sid = f"{os.path.basename(root_dir)}_{subject_id}"
                subject_data[unique_sid] = {'paths': [], 'label': label}
                for fn in os.listdir(subject_path):
                    if fn.lower().endswith(('.jpg', '.png', '.jpeg')):
                        subject_data[unique_sid]['paths'].append(os.path.join(subject_path, fn))

    print(f"Found {len(subject_data)} subjects.")
    
    # Extract features for all images
    subject_features = {sid: [] for sid in subject_data}
    with torch.no_grad():
        for sid, data in tqdm(subject_data.items(), desc="🖼️  Extracting image features"):
            if not data['paths']: continue
            for path in data['paths']:
                img = Image.open(path).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                feature_vec = encoder(x).flatten().cpu().numpy()
                subject_features[sid].append(feature_vec)

    # Aggregate features by subject (mean pooling)
    X_subject, y_subject, subject_ids = [], [], []
    print("🧠 Aggregating features to subject-level...")
    for sid, features_list in subject_features.items():
        if features_list:
            X_subject.append(np.mean(np.array(features_list), axis=0))
            y_subject.append(subject_data[sid]['label'])
            subject_ids.append(sid)
            
    return np.array(X_subject), np.array(y_subject), subject_ids

def run_loocv_probe(X, y, n_perm):
    """Performs LOOCV linear probe and permutation test."""
    print("\n--- Starting Linear Probe with LOOCV ---")
    loo = LeaveOneOut()
    preds_proba, true_labels = [], []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Predict probability of the positive class (AS=1)
        prob = clf.predict_proba(X_test)[0, 1]
        preds_proba.append(prob)
        true_labels.append(y_test[0])
        
    auc = roc_auc_score(true_labels, preds_proba)
    print(f"📈 LOOCV AUC = {auc:.4f}")

    # --- Permutation Test ---
    print(f"🔀 Running permutation test ({n_perm} iterations)...")
    
    # Observed difference in mean predicted probability between classes
    preds_proba = np.array(preds_proba)
    true_labels = np.array(true_labels)
    obs_stat = (np.mean(preds_proba[true_labels == 1]) - np.mean(preds_proba[true_labels == 0]))
    
    count_extreme = 0
    # Use tqdm for the permutation loop as it can be slow
    for _ in tqdm(range(n_perm), desc="Permutations"):
        shuffled_labels = np.random.permutation(true_labels)
        perm_stat = (np.mean(preds_proba[shuffled_labels == 1]) - np.mean(preds_proba[shuffled_labels == 0]))
        if abs(perm_stat) >= abs(obs_stat):
            count_extreme += 1
            
    pval = (count_extreme + 1) / (n_perm + 1)
    print(f"📊 Permutation Test p-value = {pval:.4f}")
    return auc, pval

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    # --- 1. PRE-TRAINING ---
    if args.force_pretrain or not os.path.exists(args.encoder_out):
        pretrain_simclr(
            pelvis_dir=args.pelvis_dir,
            encoder_out=args.encoder_out,
            batch_size=args.batch_size_pre,
            epochs=args.epochs_pre,
            device=device
        )
    else:
        print(f"☑️ Found existing encoder '{args.encoder_out}'. Skipping pre-training.")

    # --- 2. FEATURE EXTRACTION & PROBING ---
    # Load the backbone of the pre-trained model
    encoder = models.resnet18()
    encoder.fc = nn.Identity() # The encoder is just the backbone
    encoder.load_state_dict(torch.load(args.encoder_out, map_location=device))
    encoder.to(device)
    encoder.eval()

    # Define the same transformation used for probing
    tf_probe = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare SIJ data paths with labels
    sij_dirs_labels = [(args.sij_as_dir, 1)] + [(hd, 0) for hd in args.sij_healthy_dir]

    # Extract features at the subject level
    X_subjects, y_subjects, sids = extract_features_by_subject(sij_dirs_labels, encoder, tf_probe, device)
    
    n_healthy = np.sum(y_subjects == 0)
    n_as = np.sum(y_subjects == 1)
    print(f"Loaded {len(sids)} subjects for probing (Healthy={n_healthy}, AS={n_as}).")

    if n_healthy < 1 or n_as < 1:
        print("❌ Error: Need at least one subject from each class to run the probe.")
    else:
        run_loocv_probe(X_subjects, y_subjects, args.n_perm)