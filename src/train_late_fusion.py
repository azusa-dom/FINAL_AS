import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from .dataset import MRIDataset, split_patient_kfold, create_dataloader
from .models import FeatureExtractorLateFusion
from .utils import seed_everything, save_json


def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for img, clin, label in loader:
            img, clin = img.to(device), clin.to(device)
            img_feat, clin_feat = model(img, clin)
            feat = torch.cat([img_feat, clin_feat], dim=1)
            feats.append(feat.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def run_fold(fold, train_idx, test_idx, df, args, device):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    train_path = Path(f"lf_train_{fold}.csv")
    test_path = Path(f"lf_test_{fold}.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_loader = create_dataloader(train_path, args.img_dir, args.batch_size, False, False, args.mode)
    test_loader = create_dataloader(test_path, args.img_dir, args.batch_size, False, False, args.mode)

    model = FeatureExtractorLateFusion()
    model.to(device)
    train_feat, train_lbl = extract_features(model, train_loader, device)
    test_feat, test_lbl = extract_features(model, test_loader, device)

    clf = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr)
    clf.fit(train_feat, train_lbl)
    preds = clf.predict_proba(test_feat)[:, 1]
    acc = accuracy_score(test_lbl, preds > 0.5)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, Path(args.save_dir) / f"fold{fold}_late_fusion.xgb")
    return acc


def main(args):
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = MRIDataset(args.csv, args.img_dir, mode=args.mode, train=True, augment=True).df
    folds = split_patient_kfold(df, 5)
    accs = []
    for i, (train_idx, test_idx) in enumerate(folds):
        acc = run_fold(i, train_idx, test_idx, df, args, device)
        accs.append(acc)
    Path('results').mkdir(exist_ok=True)
    save_json({'accuracy': float(np.mean(accs))}, Path('results') / 'late_fusion_cv.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('data/clinical.csv'))
    parser.add_argument('--img_dir', type=Path, default=Path('data/mri'))
    parser.add_argument('--mode', type=str, default='png')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_dir', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    main(args)

