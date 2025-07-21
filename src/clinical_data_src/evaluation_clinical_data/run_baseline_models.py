# FILE: run_baseline_models.py (Updated to save predictions)
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def run_and_save_predictions():
    """
    Trains baseline models and saves their aggregated out-of-fold predictions.
    """
    DATA_DIR = "data/processed_clinical"
    # NEW: Define a directory to save predictions
    PRED_DIR = "results/clinical/baseline_preds"
    os.makedirs(PRED_DIR, exist_ok=True)
    
    N_SPLITS = 5
    LABEL_COLUMN = "label"
    
    if not os.path.isdir(DATA_DIR):
        print(f"❌ Error: Data directory not found at '{DATA_DIR}'.")
        return

    models_to_test = {
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42)
    }

    for model_name, model in models_to_test.items():
        all_fold_probs = []
        all_fold_labels = []
        # NEW: Store patient IDs to ensure correct alignment
        all_fold_ids = []
        
        print(f"\n===== Evaluating and Saving Predictions for {model_name} =====")

        for fold in range(N_SPLITS):
            try:
                train_path = os.path.join(DATA_DIR, f"fold_{fold}_train.csv")
                val_path = os.path.join(DATA_DIR, f"fold_{fold}_val.csv")
                
                df_train = pd.read_csv(train_path)
                df_val = pd.read_csv(val_path)
                
                id_cols = ['Patient_ID', 'patient_id', 'Patient ID']
                patient_id_col = [col for col in id_cols if col in df_val.columns][0]
                
                feat_cols = [c for c in df_train.columns if c != LABEL_COLUMN and c not in id_cols]

                X_train, y_train = df_train[feat_cols], df_train[LABEL_COLUMN]
                X_val, y_val = df_val[feat_cols], df_val[LABEL_COLUMN]

                model.fit(X_train, y_train)
                val_probs = model.predict_proba(X_val)[:, 1]
                
                all_fold_probs.extend(val_probs)
                all_fold_labels.extend(y_val)
                all_fold_ids.extend(df_val[patient_id_col])

            except Exception as e:
                print(f"⚠️ Error processing fold {fold} for {model_name}: {e}")
                continue
        
        # NEW: Save the aggregated predictions to a CSV file
        if all_fold_labels:
            pred_df = pd.DataFrame({
                'patient_id': all_fold_ids,
                'true_label': all_fold_labels,
                'prob': all_fold_probs
            })
            save_path = os.path.join(PRED_DIR, f"{model_name}_predictions.csv")
            pred_df.to_csv(save_path, index=False)
            print(f"✅ Predictions for {model_name} saved to {save_path}")

if __name__ == "__main__":
    run_and_save_predictions()