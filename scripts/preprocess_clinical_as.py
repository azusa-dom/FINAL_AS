import pandas as pd


def clean_and_label_as(
    csv_path,
    out_path="cleaned_clinical_with_as.csv",
    only_positive=False,
    sample_size=None,
):
    """
    加载临床数据，清洗缺失值和异常值，并构造伪标签 `pseudo_AS`
    可以选取样本子集保存
    """
    df = pd.read_excel(csv_path)

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

    # 🔍 强直性脊柱炎伪标签构建（根据 HLA-B27 + 炎症指标）
    def label_as(row):
        if row["HLA_B27"] == "Positive":
            if row["ESR"] > 20 or row["CRP"] > 10:
                return 1
        return 0

    df["pseudo_AS"] = df.apply(label_as, axis=1)

    # 可选：是否只保留伪AS病人
    if only_positive:
        df = df[df["pseudo_AS"] == 1]

    # ✅ 平衡采样：每类各抽 sample_size // 2 个
    if sample_size is not None:
        pos = df[df["pseudo_AS"] == 1]
        neg = df[df["pseudo_AS"] == 0]
        per_class = sample_size // 2
        df = pd.concat(
            [
                pos.sample(n=min(len(pos), per_class), random_state=42),
                neg.sample(n=min(len(neg), per_class), random_state=42),
            ]
        )

    # 保存清洗后的文件
    selected_cols = numeric_cols + binary_cols + ["pseudo_AS"]
    df[selected_cols].to_csv(out_path, index=False)
    print(f"✅ 临床数据已清洗并保存: {out_path}")
    print(f"样本分布:\n{df['pseudo_AS'].value_counts()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="清洗临床指标并生成 pseudo_AS 标签")
    parser.add_argument("--csv", type=str, required=True, help="原始临床Excel路径")
    parser.add_argument(
        "--out",
        type=str,
        default="cleaned_clinical_with_as.csv",
        help="输出清洗后的CSV文件名",
    )
    parser.add_argument(
        "--only_positive", action="store_true", help="是否仅保留伪AS样本"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="最多保留的样本数（用于控制临床数据大小）",
    )

    args = parser.parse_args()
    clean_and_label_as(args.csv, args.out, args.only_positive, args.sample_size)
