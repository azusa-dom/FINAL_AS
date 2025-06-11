import pandas as pd

def clean_and_label_as(csv_path, out_path="cleaned_clinical_with_as.csv"):
    """
    加载临床数据，清洗缺失值和异常值，并构造伪标签 `pseudo_AS`
    保存为新的 CSV 文件
    """
    df = pd.read_csv(csv_path)

    # 标准化列名（防止空格或大小写问题）
    df.columns = [c.strip().replace(" ", "").replace("-", "_") for c in df.columns]

    # 统一字符串型值（如 Positive/Negative -> 标准化）
    binary_cols = ["HLA_B27", "ANA", "Anti_Ro", "Anti_La", "Anti_dsDNA", "Anti_Sm"]
    for col in binary_cols:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # 缺失值填充：数值字段用中位数填充
    numeric_cols = ["Age", "ESR", "CRP", "RF", "Anti_CCP", "C3", "C4"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # 二值字段填充：缺失为 Negative
    for col in binary_cols:
        df[col].fillna("Negative", inplace=True)

    # 🔍 强直性脊柱炎伪标签构建
    def label_as(row):
        if row["HLA_B27"] == "Positive":
            if row["ESR"] > 20 or row["CRP"] > 10:
                return 1  # 判定为可能AS
        return 0  # 非AS

    df["pseudo_AS"] = df.apply(label_as, axis=1)

    # 可选：只保留用于后续建模的字段
    selected_cols = numeric_cols + binary_cols + ["pseudo_AS"]
    df[selected_cols].to_csv(out_path, index=False)
    print(f"✅ 临床数据已清洗并保存: {out_path}")
    print(f"样本分布:\n{df['pseudo_AS'].value_counts()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清洗临床指标并生成 pseudo_AS 标签")
    parser.add_argument("--csv", type=str, required=True,
                        help="原始临床CSV路径（如 clinical_raw.csv）")
    parser.add_argument("--out", type=str, default="cleaned_clinical_with_as.csv",
                        help="输出清洗后的CSV文件名")
    args = parser.parse_args()

    clean_and_label_as(args.csv, args.out)
