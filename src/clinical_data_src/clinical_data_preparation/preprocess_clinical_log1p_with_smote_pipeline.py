import pandas as pd
import numpy as np
import argparse
import os
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("✅✅✅ Running the Definitive All-in-One Preprocessor (v13 with SMOTE-Pipeline) ✅✅✅")

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass  # torch may not be installed; ignore if so


def create_preprocessor(numeric_features, categorical_features):
    from imblearn.pipeline import Pipeline  # ensure correct pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="passthrough")

def run_all_preprocessing(input_csv, output_dir, n_splits=5, target_disease='Ankylosing Spondylitis'):
    print(f"📥 Reading clean, raw data from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    print("✅ Raw data read successfully.")
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    id_column = "Patient_ID"
    if id_column not in df.columns:
        df[id_column] = range(len(df))
        print(f"✨ Created a new '{id_column}' column.")

    label_column = "label"
    target_disease_col_name = 'Disease'
    df[label_column] = (df[target_disease_col_name] == target_disease).astype(int)
    print(f"🎯 Created binary '{label_column}' column for target: '{target_disease}'")

    df_positive = df[df[label_column] == 1]
    df_negative = df[df[label_column] == 0].sample(n=len(df_positive), random_state=42)
    df_balanced = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"⚖️  Data balanced. Total samples: {len(df_balanced)}.")

    y = df_balanced[label_column]
    patient_ids = df_balanced[id_column]
    X = df_balanced.drop(columns=[label_column, id_column, target_disease_col_name], errors='ignore')

    log1p_features = ['CRP', 'ESR']
    for col in log1p_features:
        if col in X.columns:
            X[col] = np.log1p(X[col])
            print(f"🔁 Applied log1p transform to {col}")
        else:
            print(f"⚠️ Column '{col}' not found for log1p transform.")

    os.makedirs(output_dir, exist_ok=True)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"--- Starting {n_splits}-fold cross-validation ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"⚙️  Processing Fold {fold}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        val_ids = patient_ids.iloc[val_idx]

        pipeline = ImbPipeline(steps=[
            ("preprocessor", create_preprocessor(numeric_features, categorical_features)),
            ("smote", SMOTE(random_state=42))
        ])

        X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
        X_val_processed = pipeline.named_steps["preprocessor"].transform(X_val)

        try:
            ohe_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
            new_feature_names = numeric_features + ohe_names.tolist()
        except:
            new_feature_names = list(X.columns)

        df_train_final = pd.DataFrame(X_train_resampled, columns=new_feature_names)
        df_train_final[label_column] = y_train_resampled

        df_val_final = pd.DataFrame(X_val_processed, columns=new_feature_names)
        df_val_final[label_column] = y_val.values
        df_val_final.insert(0, id_column, val_ids.values)

        train_path = os.path.join(output_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(output_dir, f"fold_{fold}_val.csv")
        df_train_final.to_csv(train_path, index=False)
        df_val_final.to_csv(val_path, index=False)
        print(f"✅ Fold {fold} data saved.")

    print("🎉 All preprocessing complete!")

if __name__ == "__main__":
    set_seed(42)  # 🧪 固定随机种子，确保可复现性
    parser = argparse.ArgumentParser(description="Preprocessing script with SMOTE-Pipeline.")
    parser.add_argument("input_csv", help="Path to the original raw data CSV file.")
    parser.add_argument("output_dir", help="Directory to save processed fold data.")
    args = parser.parse_args()
    run_all_preprocessing(args.input_csv, args.output_dir)
