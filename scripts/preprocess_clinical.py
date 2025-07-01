import pandas as pd
import numpy as np
import argparse
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# --- This is the definitive, all-in-one preprocessing script (v12) ---
print("✅✅✅ Running the Definitive All-in-One Preprocessor (v12) ✅✅✅")

def create_preprocessor(numeric_features, categorical_features):
    """Creates a preprocessor for different feature types."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="passthrough")

def run_all_preprocessing(input_csv, output_dir, n_splits=5, target_disease='Ankylosing Spondylitis'):
    """Main function: Loads RAW data, cleans, balances, and processes it safely for CV."""
    print(f"📥 Reading clean, raw data from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading the file: {e}")
        return
        
    print("✅ Raw data read successfully.")
    
    # --- Data Cleaning and Feature Engineering ---
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    id_column = "Patient_ID"
    if id_column not in df.columns:
        df[id_column] = range(len(df))
        print(f"✨ Created a new '{id_column}' column.")

    label_column = "label"
    target_disease_col_name = 'Disease'
    df[label_column] = (df[target_disease_col_name] == target_disease).astype(int)
    print(f"🎯 Created binary '{label_column}' column for target: '{target_disease}'")

    # --- Balance the Dataset ---
    df_positive = df[df[label_column] == 1]
    df_negative = df[df[label_column] == 0].sample(n=len(df_positive), random_state=42)
    df_balanced = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"⚖️  Data balanced. Total samples: {len(df_balanced)}.")

    # --- Define Final Features (X) and Target (y) ---
    # CORE FIX: We define X and y from the SAME balanced dataframe
    y = df_balanced[label_column]
    patient_ids = df_balanced[id_column]
    
    print(f"🔥 Removing non-feature columns to create the final feature set 'X'.")
    X = df_balanced.drop(columns=[label_column, id_column, target_disease_col_name], errors='ignore')

    # --- Proceed with the trusted cross-validation pipeline ---
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n--- Starting {n_splits}-fold cross-validation data preparation ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n⚙️  Processing Fold {fold}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        val_ids = patient_ids.iloc[val_idx]

        preprocessor = create_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_train)

        try:
            ohe_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
            new_feature_names = numeric_features + ohe_names.tolist()
        except:
            new_feature_names = list(X.columns)

        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        
        df_train_final = pd.DataFrame(X_train_resampled, columns=new_feature_names)
        df_train_final[label_column] = y_train_resampled

        df_val_final = pd.DataFrame(X_val_processed, columns=new_feature_names)
        df_val_final[label_column] = y_val.values
        df_val_final.insert(0, id_column, val_ids.values)

        train_path = os.path.join(output_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(output_dir, f"fold_{fold}_val.csv")
        df_train_final.to_csv(train_path, index=False)
        df_val_final.to_csv(val_path, index=False)
        print(f"  ✅ Fold {fold} data saved.")

    print("\n🎉 All preprocessing is complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The definitive script for preprocessing the clinical data.")
    parser.add_argument("input_csv", help="Path to the ORIGINAL RAW data CSV file.")
    parser.add_argument("output_dir", help="Directory to save the processed fold data.")
    args = parser.parse_args()
    run_all_preprocessing(args.input_csv, args.output_dir)