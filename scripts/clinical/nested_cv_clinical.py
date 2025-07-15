import pandas as pd
import numpy as np
import argparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

# ✅ Create preprocessor
def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor

# ✅ Main function
def run_nested_cv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df['label'] = (df['Disease'] == 'Ankylosing Spondylitis').astype(int)
    y = df['label'].values
    X = df.drop(columns=['Patient_ID', 'Disease', 'label'], errors='ignore')

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"🔁 Fold {fold}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        preprocessor = create_preprocessor(X_train)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"✅ Fold {fold} AUROC: {auc:.4f}")
        results.append({'fold': fold, 'best_params': grid_search.best_params_, 'AUROC': auc})

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n📁 Results saved to {output_csv}")

# ✅ Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nested CV on clinical data.")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file path")
    parser.add_argument('--model_output', type=str, required=True, help="Output CSV file for results")
    args = parser.parse_args()

    run_nested_cv(args.input, args.model_output)
