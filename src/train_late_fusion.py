# 文件名: src/train_late_fusion.py (已完善)

import pandas as pd
import numpy as np
import os
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib

def load_and_merge_predictions(clinical_preds_dir, mri_preds_dir, n_folds=5):
    """
    加载并合并所有折叠的临床和MRI预测结果。

    返回:
        一个包含 'subject_id', 'true_label', 'clinical_score', 'mri_score' 的完整DataFrame。
    """
    all_folds_data = []
    print("--- 开始加载和合并预测文件 ---")
    for i in range(n_folds):
        try:
            # 1. 加载临床预测结果
            clin_path = os.path.join(clinical_preds_dir, f"fold_{i}_predictions.csv")
            df_clin = pd.read_csv(clin_path)
            # 假设列名为 'preds', 如果不是请修改
            df_clin = df_clin.rename(columns={'preds': 'clinical_score', 'labels': 'true_label'})
            
            # 2. 加载MRI预测结果
            mri_path = os.path.join(mri_preds_dir, f"fold_{i}_predictions.csv")
            df_mri = pd.read_csv(mri_path)
            df_mri = df_mri.rename(columns={'preds': 'mri_score', 'labels': 'true_label'})
            
            # 3. 基于 'subject_id' 或索引进行合并 (这里假设行顺序和ID是一致的)
            #    如果CSV中有 'subject_id' 列，使用merge会更稳健
            #    df_merged = pd.merge(df_clin[['subject_id', 'true_label', 'clinical_score']], 
            #                         df_mri[['subject_id', 'mri_score']], on='subject_id')
            
            # 假设行顺序一致，直接拼接
            df_merged = pd.concat([df_clin[['true_label', 'clinical_score']], df_mri['mri_score']], axis=1)
            all_folds_data.append(df_merged)
            print(f"  成功加载 Fold {i} 的数据。")
        except FileNotFoundError as e:
            print(f"错误: 找不到文件 {e.filename}。请确保两个预测文件夹中的文件都存在且命名正确。")
            return None
            
    if not all_folds_data:
        print("错误: 未能加载任何预测数据。")
        return None

    # 将所有折叠的数据合并成一个大的DataFrame
    full_df = pd.concat(all_folds_data, ignore_index=True)
    print(f"--- 数据合并完成，总样本数: {len(full_df)} ---")
    return full_df

def main(args):
    """
    主函数，执行晚期融合模型的交叉验证训练和评估。
    """
    # 1. 加载并准备融合数据
    fusion_df = load_and_merge_predictions(args.clinical_preds_dir, args.mri_preds_dir, args.n_splits)
    if fusion_df is None:
        return

    # 我们的新特征就是两个模型的预测分数
    X = fusion_df[['clinical_score', 'mri_score']].values
    y = fusion_df['true_label'].values

    # 2. 使用交叉验证来训练和评估最终的融合模型
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    metrics = {'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []}

    print("\n--- 开始训练和评估晚期融合模型 ---")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 定义XGBoost融合分类器
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 在测试集上进行预测
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        
        # 计算并保存评估指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred_class))
        metrics['auc'].append(roc_auc_score(y_test, y_pred_proba))
        metrics['f1'].append(f1_score(y_test, y_pred_class))
        metrics['precision'].append(precision_score(y_test, y_pred_class))
        metrics['recall'].append(recall_score(y_test, y_pred_class))
        
        print(f"  Fold {fold+1}/{args.n_splits} | AUC: {metrics['auc'][-1]:.4f}, Accuracy: {metrics['accuracy'][-1]:.4f}")

    # --- 3. 总结并打印最终结果 ---
    print("\n========== Late Fusion CV Summary ==========")
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  Average {metric_name.capitalize()}: {mean_val:.4f} +/- {std_val:.4f}")
    print("==========================================")

    # (可选) 在所有数据上训练最终模型并保存
    print("\n正在使用全部数据训练最终的融合模型并保存...")
    final_model = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr,
        use_label_encoder=False, eval_metric='logloss'
    )
    final_model.fit(X, y)
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_path / "late_fusion_model.xgb")
    print(f"最终模型已保存至: {save_path / 'late_fusion_model.xgb'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late Fusion Model Training")
    parser.add_argument("--clinical_preds_dir", type=str, default="models/clinical_model/clinical_preds",
                        help="存放临床模型预测结果的文件夹路径")
    parser.add_argument("--mri_preds_dir", type=str, default="models/mri_model/mri_preds",
                        help="存放MRI模型预测结果的文件夹路径")
    parser.add_argument("--n_splits", type=int, default=5, help="交叉验证的折数")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="models/fusion_model")
    
    args = parser.parse_args()
    main(args)
