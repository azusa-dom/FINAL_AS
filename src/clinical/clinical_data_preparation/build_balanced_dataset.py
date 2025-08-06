#!/usr/bin/env python3
"""
构建 Engine 1 所需的平衡临床数据集。

从包含多种风湿病的原始数据集中提取 Ankylosing Spondylitis（AS）患者为正样本，其余为负样本。
通过下采样或上采样控制类别平衡，输出供模型训练使用的统一数据集。

示例用法：
python scripts/build_balanced_dataset.py \
  --input data/rheumatic_autoimmune_disease.csv \
  --method undersample \
  --output data/balanced_clinical.csv
"""

import pandas as pd
import argparse
from sklearn.utils import resample
import os


def build_dataset(df, method="undersample"):
    # 检查数据集是否为空
    if df.empty:
        raise ValueError("❌ 输入数据集为空")
    
    # 检查Disease列是否存在
    if "Disease" not in df.columns:
        raise ValueError("❌ 缺少必要列：Disease")
    
    # 检查Disease列是否为空
    if df["Disease"].isna().all():
        raise ValueError("❌ Disease列全为空值")
    
    pos_df = df[df["Disease"] == "Ankylosing Spondylitis"].copy()
    neg_df = df[df["Disease"] != "Ankylosing Spondylitis"].copy()

    # 检查是否有AS病例
    if len(pos_df) == 0:
        raise ValueError("❌ 数据集中没有找到'Ankylosing Spondylitis'病例")
    
    # 检查是否有非AS病例
    if len(neg_df) == 0:
        raise ValueError("❌ 数据集中所有病例都是'Ankylosing Spondylitis'，无法构建负样本")

    pos_df["label"] = 1
    neg_df["label"] = 0

    print(f"原始正样本（AS）数量: {len(pos_df)}")
    print(f"原始负样本数量: {len(neg_df)}")

    if method == "undersample":
        neg_df = resample(neg_df,
                          replace=False,
                          n_samples=len(pos_df),
                          random_state=42)
        print(f"✅ 使用下采样：负样本缩减至 {len(neg_df)}")
    elif method == "oversample":
        pos_df = resample(pos_df,
                          replace=True,
                          n_samples=len(neg_df),
                          random_state=42)
        print(f"✅ 使用上采样：正样本扩增至 {len(pos_df)}")
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")

    full_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return full_df


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ 输入文件不存在: {args.input}")

    df = pd.read_csv(args.input)
    
    # 数据验证已在build_dataset函数中处理
    balanced_df = build_dataset(df, method=args.method)
    
    # 改进输出目录处理
    output_dir = os.path.dirname(args.output)
    if output_dir:  # 只有当输出路径包含目录时才创建
        os.makedirs(output_dir, exist_ok=True)
    
    balanced_df.to_csv(args.output, index=False)
    print(f"✅ 已保存平衡数据集至: {args.output}")
    print(f"📊 最终数据集大小: {len(balanced_df)} 行")
    print(f"📊 正样本数量: {len(balanced_df[balanced_df['label'] == 1])}")
    print(f"📊 负样本数量: {len(balanced_df[balanced_df['label'] == 0])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建平衡的AS预测训练集")
    parser.add_argument("--input", type=str, required=True, help="原始CSV路径（包含Disease列）")
    parser.add_argument("--method", type=str, default="undersample", choices=["undersample", "oversample"],
                        help="类别平衡策略")
    parser.add_argument("--output", type=str, default="data/balanced_clinical.csv",
                        help="输出路径")
    args = parser.parse_args()
    main(args)
