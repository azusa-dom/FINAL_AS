#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_l2o_predictions.py

Generate subject-level Leave-Two-Out (1 AS + 1 HC) predictions and save
`l2o_predictions.csv` for downstream Figure 4 plotting.

Directory layout assumed:
  data_root/
    mri_AS/patient1/...jpg
    mri_AS/patient2/...jpg
    ...
    mri_health/health1/subjA/...jpg
    mri_health/health2/subjB/...jpg

Output CSV columns:
  subject_id,y_true,prob_raw
"""

import os, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from itertools import product

def load_subject_slices(root_dir, label):
    """
    root_dir: path to folder containing subject subfolders
    label: 1 (AS) or 0 (HC)
    returns list of dicts: {'subject': id, 'paths': [img1,...], 'label': label}
    """
    subjects = []
    root = Path(root_dir)
    if not root.is_dir():
        return subjects
    for subj_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        img_paths = []
        for p in subj_dir.rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png"}:
                img_paths.append(str(p))
        if img_paths:
            subjects.append({"subject": subj_dir.name, "paths": img_paths, "label": label})
    return subjects

def extract_resnet_features(image_paths, device, model, tfm):
    feats = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x).cpu().numpy()
        feats.append(f[0])
    return np.vstack(feats)  # [n_slices, 512]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to data root containing mri_AS/ and mri_health/")
    ap.add_argument("--out-csv", default="l2o_predictions.csv", help="Output CSV file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    # Collect subjects
    as_dir = os.path.join(args.data_root, "mri_AS")
    hc_dirs = [os.path.join(args.data_root, "mri_health", "health1"),
               os.path.join(args.data_root, "mri_health", "health2")]

    as_subjects = load_subject_slices(as_dir, label=1)
    hc_subjects = []
    for d in hc_dirs:
        hc_subjects.extend(load_subject_slices(d, label=0))

    if len(as_subjects) != 6 or len(hc_subjects) != 2:
        print(f"[WARN] Expected 6 AS + 2 HC, found {len(as_subjects)} AS and {len(hc_subjects)} HC.")

    print(f"[INFO] Subjects collected: AS={len(as_subjects)}, HC={len(hc_subjects)}")

    # Feature extractor (ResNet18 -> 512-dim)
    device = torch.device(args.device)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Pre-extract & cache subject-level embeddings (mean pooled)
    subj_embeddings = {}
    for entry in tqdm(as_subjects + hc_subjects, desc="Extracting features"):
        feats = extract_resnet_features(entry["paths"], device, backbone, tfm)  # [n_slices,512]
        subj_embeddings[entry["subject"]] = {
            "label": entry["label"],
            "embedding": feats.mean(axis=0)  # mean pooling
        }

    # Build all L2O folds: choose 1 AS and 1 HC for validation
    predictions = []  # rows: subject_id,y_true,prob_raw

    for as_val in as_subjects:
        for hc_val in hc_subjects:
            val_ids = [as_val["subject"], hc_val["subject"]]
            # Training set = remaining subjects
            train_ids = [s["subject"] for s in as_subjects if s["subject"] not in val_ids] + \
                        [s["subject"] for s in hc_subjects if s["subject"] not in val_ids]

            X_train = np.vstack([subj_embeddings[sid]["embedding"] for sid in train_ids])
            y_train = np.array([subj_embeddings[sid]["label"] for sid in train_ids])

            # Logistic regression with balanced class weights
            clf = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', random_state=args.seed)
            clf.fit(X_train, y_train)

            # Predict on validation subjects
            for sid in val_ids:
                emb = subj_embeddings[sid]["embedding"].reshape(1,-1)
                prob = clf.predict_proba(emb)[0,1]  # probability of class=1 (AS)
                predictions.append({
                    "subject_id": sid,
                    "y_true": subj_embeddings[sid]["label"],
                    "prob_raw": prob
                })

    # Because each subject appears in multiple validation folds (each pairing),
    # we average its probabilities across folds (standard practice for repeated CV).
    df = pd.DataFrame(predictions)
    df_grouped = (df.groupby(["subject_id","y_true"], as_index=False)
                    .agg(prob_raw=("prob_raw","mean"))
                  )
    df_grouped.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved L2O predictions to {args.out_csv}")
    print(df_grouped)

if __name__ == "__main__":
    main()
