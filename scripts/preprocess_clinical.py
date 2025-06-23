# scripts/preprocess_clinical.py (最终修复版)
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

def create_preprocessor(numeric_features, categorical_features):
    """
    根据特征类型，创建一个scikit-learn预处理器。
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def process_and_save_folds(input_csv, output_dir, n_splits=5, id_column='patient_id', label_column='label'):
    """
    主函数：加载数据，执行K-折交叉验证划分，并在每一折中处理数据以防止泄漏，最后保存结果。
    这个版本会同时保存处理后的数据和数据划分的ID文件。
    """
    print(f"📥 正在读取临床数据: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 【新增】创建用于存放split文件的子目录
    split_dir = os.path.join(output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # 预处理：移除不合逻辑的数据
    if 'age' in df.columns:
        initial_rows = len(df)
        df = df[df['age'] >= 0]
        if initial_rows > len(df):
            print(f"✅ 已移除 {initial_rows - len(df)} 行负数年龄的数据。")

    # 分离特征(X)、标签(y)和ID
    y = df[label_column]
    # 【新增】确保patient_ids被正确提取
    patient_ids = df[id_column] if id_column in df.columns else pd.Series(np.arange(len(df)), name=id_column)
    X = df.drop(columns=[label_column, id_column], errors='ignore')

    # 识别特征类型
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"🔍 识别出的数值特征: {numeric_features}")
    print(f"🔍 识别出的分类特征: {categorical_features}")

    # 初始化K-折交叉验证
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n--- 开始处理 {n_splits}-折 交叉验证数据 ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n⚙️ 正在处理 Fold {fold}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # 【新增】获取当前验证集的患者ID
        val_ids = patient_ids.iloc[val_idx]

        preprocessor = create_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_train)
        
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
            new_feature_names = numeric_features + ohe_feature_names.tolist()
        except Exception:
            new_feature_names = numeric_features

        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        print(f"  SMOTE 过采样完成, 训练样本从 {len(X_train)} 增加到 {len(X_train_resampled)}")

        df_X_train = pd.DataFrame(X_train_resampled, columns=new_feature_names)
        df_y_train = pd.DataFrame(y_train_resampled, columns=[label_column])
        df_train_final = pd.concat([df_X_train, df_y_train], axis=1)
        
        df_X_val = pd.DataFrame(X_val_processed, columns=new_feature_names)
        df_y_val = pd.DataFrame(y_val.values, columns=[label_column])
        df_val_final = pd.concat([df_X_val, df_y_val], axis=1)
        
        # 将验证集的ID也加到保存的文件中，方便后续使用
        df_val_final.insert(0, id_column, val_ids.values)

        train_path = os.path.join(output_dir, f"fold_{fold}_train.csv")
        val_path = os.path.join(output_dir, f"fold_{fold}_val.csv")
        df_train_final.to_csv(train_path, index=False)
        df_val_final.to_csv(val_path, index=False)
        
        # 【核心修复】保存验证集的ID到txt文件
        split_file_path = os.path.join(split_dir, f"fold_{fold}.txt")
        with open(split_file_path, "w") as f:
            for pid in val_ids:
                f.write(f"{pid}\n")

        print(f"  ✅ 已保存 Fold {fold} 的处理后数据:")
        print(f"    -> 训练集: {train_path}")
        print(f"    -> 验证集: {val_path}")
        print(f"    -> 验证ID蓝图: {split_file_path}")

    print("\n🎉 全部数据处理和划分完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="临床数据预处理脚本，用于交叉验证，可防止数据泄漏。")
    parser.add_argument("input_csv", help="原始临床数据 CSV 文件路径")
    parser.add_argument("output_dir", help="保存处理后各折数据文件的目录")
    parser.add_argument("--n_splits", type=int, default=5, help="交叉验证的折数，默认为 5")
    parser.add_argument("--id_column", default="patient_id", help="患者ID列名，此列不作为特征")
    parser.add_argument("--label_column", default="label", help="标签列名")
    
    args = parser.parse_args()

    process_and_save_folds(
        args.input_csv,
        args.output_dir,
        n_splits=args.n_splits,
        id_column=args.id_column,
        label_column=args.label_column
    )
