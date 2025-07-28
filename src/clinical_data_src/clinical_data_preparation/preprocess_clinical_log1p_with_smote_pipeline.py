import pandas as pd
import numpy as np
import argparse
import os
import random
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("✅✅✅ Running the Robust Clinical Data Preprocessor (v14 - Fixed) ✅✅✅")

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    # Removed torch-related code as it's not necessary

def validate_input_data(df, required_columns=None):
    """Validate input data integrity"""
    if df is None or df.empty:
        raise ValueError("Input data is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def create_preprocessor(numeric_features, categorical_features):
    """Create data preprocessor"""
    if not numeric_features and not categorical_features:
        raise ValueError("At least one type of features is required")
    
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))
    
    return ColumnTransformer(transformers=transformers, remainder="drop")

def safe_log1p_transform(X, columns):
    """Safely apply log1p transformation, handling negative and zero values"""
    X_transformed = X.copy()
    
    for col in columns:
        if col in X_transformed.columns:
            # Check for negative values
            if (X_transformed[col] < 0).any():
                logger.warning(f"Column {col} contains negative values, skipping log1p transformation")
                continue
            
            # Apply log1p transformation
            X_transformed[col] = np.log1p(X_transformed[col])
            logger.info(f"Successfully applied log1p transformation to {col}")
        else:
            logger.warning(f"Column {col} does not exist, skipping log1p transformation")
    
    return X_transformed

def run_all_preprocessing(input_csv, output_dir, n_splits=5, target_disease='Ankylosing Spondylitis'):
    """Run complete preprocessing pipeline"""
    
    # Validate input file
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Cannot create output directory: {e}")
    
    # Read data
    logger.info(f"Reading data: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Successfully read data, shape: {df.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")
    
    # Validate data
    validate_input_data(df)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # Handle ID column
    id_column = "Patient_ID"
    if id_column not in df.columns:
        df[id_column] = range(len(df))
        logger.info(f"Created new ID column: {id_column}")
    
    # Validate target column
    target_disease_col_name = 'Disease'
    if target_disease_col_name not in df.columns:
        raise ValueError(f"Target column {target_disease_col_name} does not exist")
    
    # Create label column
    label_column = "label"
    df[label_column] = (df[target_disease_col_name] == target_disease).astype(int)
    logger.info(f"Created binary label column for target disease: {target_disease}")
    
    # Check class balance
    class_counts = df[label_column].value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    # Prepare features and labels
    y = df[label_column]
    patient_ids = df[id_column]
    X = df.drop(columns=[label_column, id_column, target_disease_col_name], errors='ignore')
    
    # Apply log1p transformation
    log1p_features = ['CRP', 'ESR']
    X = safe_log1p_transform(X, log1p_features)
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    logger.info(f"Number of numeric features: {len(numeric_features)}")
    logger.info(f"Number of categorical features: {len(categorical_features)}")
    
    if not numeric_features and not categorical_features:
        raise ValueError("No valid features found")
    
    # Cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger.info(f"Starting {n_splits}-fold cross-validation")
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        logger.info(f"Processing fold {fold + 1}...")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        val_ids = patient_ids.iloc[val_idx]
        
        # Validate training set
        if len(X_train) == 0:
            logger.error(f"Fold {fold + 1} training set is empty")
            continue
        
        # Create preprocessor
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        
        # Process training data
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
        except Exception as e:
            logger.error(f"Fold {fold + 1} preprocessing failed: {e}")
            continue
        
        # Get feature names
        try:
            feature_names = []
            if numeric_features:
                feature_names.extend(numeric_features)
            if categorical_features:
                ohe_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
                feature_names.extend(ohe_names)
        except Exception as e:
            logger.warning(f"Cannot get feature names: {e}")
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        # Apply SMOTE (only on training set)
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y_train))-1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            logger.info(f"SMOTE resampling completed, training set size: {len(X_train_resampled)}")
        except Exception as e:
            logger.warning(f"SMOTE failed, using original data: {e}")
            X_train_resampled, y_train_resampled = X_train_processed, y_train
        
        # Create DataFrames
        df_train_final = pd.DataFrame(X_train_resampled, columns=feature_names)
        df_train_final[label_column] = y_train_resampled
        
        df_val_final = pd.DataFrame(X_val_processed, columns=feature_names)
        df_val_final[label_column] = y_val.values
        df_val_final.insert(0, id_column, val_ids.values)
        
        # Save data
        train_path = os.path.join(output_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(output_dir, f"fold_{fold}_val.csv")
        
        try:
            df_train_final.to_csv(train_path, index=False)
            df_val_final.to_csv(val_path, index=False)
            logger.info(f"Fold {fold + 1} data saved")
            
            # Record results
            fold_results.append({
                'fold': fold,
                'train_samples': len(df_train_final),
                'val_samples': len(df_val_final),
                'train_features': len(feature_names)
            })
            
        except Exception as e:
            logger.error(f"Failed to save fold {fold + 1} data: {e}")
    
    # Summary results
    if fold_results:
        logger.info("Preprocessing completed!")
        logger.info(f"Successfully processed {len(fold_results)} folds")
        for result in fold_results:
            logger.info(f"Fold {result['fold']}: {result['train_samples']} train samples, {result['val_samples']} validation samples, {result['train_features']} features")
    else:
        raise RuntimeError("No folds were successfully processed")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clinical data preprocessing script")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_dir", help="Output directory path")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--target_disease", default='Ankylosing Spondylitis', help="Target disease name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate parameters
    if not os.path.exists(args.input_csv):
        logger.error(f"Input file does not exist: {args.input_csv}")
        return 1
    
    if args.n_splits < 2:
        logger.error("Number of cross-validation folds must be at least 2")
        return 1
    
    # Set random seed
    set_seed(args.seed)
    
    try:
        # Run preprocessing
        success = run_all_preprocessing(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            target_disease=args.target_disease
        )
        
        if success:
            logger.info("Preprocessing completed successfully!")
            return 0
        else:
            logger.error("Preprocessing failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error occurred during preprocessing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
