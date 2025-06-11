import pandas as pd
import numpy as np
import os
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
from pathlib import Path

def softmax(logits):
    """对logits执行softmax运算"""
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def load_and_merge_predictions(clinical_preds_dir, mri_preds_dir, n_folds=5, clinical_target_index=1):
    """
    加载并合并所有折叠的临床和MRI预测结果。

    返回:
        一个包含 'true_label', 'clinical_score', 'mri_score' 的完整DataFrame。
    """
    all_folds_data = []
    print("--- 开始加载和合并预测文件 ---")
    for i in range(n_folds):
        try:
            # 1. 加载临床预测
            clin_path = os.path.join(clinical_preds_dir, f"fold_{i}_predictions.csv")
            df_clin = pd.read_csv(clin_path)
            logit_cols = [col for col in df_clin.columns if col.startswith("logit_")]
            if not logit_cols:
                raise ValueError(f"Fold {i} 的临床预测文件中未找到 logit 列。")

            logits = df_clin[logit_cols].values
            probs = softmax(logits)
            df_clin["clinical_score"] = probs[:, clinical_target_index]
            df_clin = df_clin.rename(columns={"true_label": "true_label"})

            # 2. 加载MRI预测
            mri_path = os.path.join(mri_preds_dir, f"fold_{i}_predictions.csv")
            df_mri = pd.read_csv(mri_path)
            df_mri = df_mri.rename(columns={"preds": "mri_score", "labels": "true_label"})

            # 3. 合并
            df_merged = pd.concat([df_clin[['true_label', 'clinical_score']], df_mri['mri_score']], axis=1)
            all_folds_data.append(df_merged)
            print(f"  ✅ 成功加载 Fold {i} 的数据。")

        except FileNotFoundError as e:
            print(f"❌ 错误: 找不到文件 {e.filename}")
            return None
        except Exception as e:
            print(f"❌ 错误: Fold {i} 数据处理失败 - {e}")
            return None

    if not all_folds_data:
        print("❌ 错误: 未能加载任何预测数据。")
        return None

    full_df = pd.concat(all_folds_data, ignore_index=True)
    print(f"--- ✅ 数据合并完成，总样本数: {len(full_df)} ---")
    return full_df

def main(args):
    fusion_df = load_and_merge_predictions(args.clinical_preds_dir, args.mri_preds_dir,
                                           args.n_splits, clinical_target_index=args.clinical_target_index)
    if fusion_df is None:
        return

    X = fusion_df[['clinical_score', 'mri_score']].values
    y = fusion_df['true_label'].values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []}

    print("\n--- 开始训练和评估融合模型 ---")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        clf.fit(X_train, y_train)

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba > 0.5).astype(int)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred_class))
        metrics['auc'].append(roc_auc_score(y_test, y_pred_proba))
        metrics['f1'].append(f1_score(y_test, y_pred_class))
        metrics['precision'].append(precision_score(y_test, y_pred_class))
        metrics['recall'].append(recall_score(y_test, y_pred_class))

        print(f"  Fold {fold+1} | AUC: {metrics['auc'][-1]:.4f}, Acc: {metrics['accuracy'][-1]:.4f}")

    print("\n========== Fusion CV Summary ==========")
    for name, values in metrics.items():
        print(f"  {name.capitalize():<10}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print("=======================================\n")

    print("🔄 正在训练最终模型...")
    final_model = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr,
        use_label_encoder=False, eval_metric='logloss'
    )
    final_model.fit(X, y)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_path / "late_fusion_model.xgb")
    print(f"✅ 最终模型已保存至: {save_path / 'late_fusion_model.xgb'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late Fusion Training")
    parser.add_argument("--clinical_preds_dir", type=str, default="models/clinical_model/clinical_preds")
    parser.add_argument("--mri_preds_dir", type=str, default="models/mri_model/mri_preds")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="models/fusion_model")
    parser.add_argument("--clinical_target_index", type=int, default=1, help="用于融合的logit类索引")
    args = parser.parse_args()
    main(args)
