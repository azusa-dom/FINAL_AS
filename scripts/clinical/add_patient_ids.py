# scripts/add_patient_ids.py
# -*- coding: utf-8 -*-
"""
修复工具：为已经生成的 a-fold 预测CSV文件添加 'patient_id' 列。

工作原理:
1. 遍历指定的预测目录 (e.g., 'models/clinical_model/clinical_preds/')。
2. 对于目录中的每一个 'fold_k_predictions.csv' 文件：
   a. 读取对应的 'splits/fold_k.txt' 文件来获取患者ID列表。
   b. 读取预测CSV文件。
   c. 检查两者的行数是否匹配，这是一个重要的安全检查。
   d. 将患者ID列表作为新的一列 ('patient_id') 添加到预测数据中。
   e. 覆盖保存原有的CSV文件。
"""
import pandas as pd
import argparse
from pathlib import Path


def add_ids_to_predictions(preds_dir: str, splits_dir: str):
    """
    主函数，执行添加ID并保存的逻辑。

    Args:
        preds_dir (str): 存放 fold_k_predictions.csv 的目录。
        splits_dir (str): 存放 fold_k.txt 的目录。
    """
    preds_path = Path(preds_dir)
    splits_path = Path(splits_dir)

    print(f"\n--- 正在处理目录: {preds_path} ---")

    # 查找所有 fold_k_predictions.csv 文件
    pred_files = sorted(list(preds_path.glob("fold_*_predictions.csv")))

    if not pred_files:
        print(f"⚠️ 警告: 在 {preds_path} 中没有找到任何 'fold_*_predictions.csv' 文件。")
        return

    for pred_file in pred_files:
        try:
            # 从文件名中提取 fold 编号
            fold_num_str = pred_file.stem.split("_")[1]

            # 构建对应的split文件路径
            split_file = splits_path / f"fold_{fold_num_str}.txt"

            if not split_file.exists():
                print(f"❌ 错误: 找不到对应的 split 文件: {split_file}")
                continue

            # 1. 读取患者ID
            with open(split_file, "r") as f:
                patient_ids = [line.strip() for line in f.readlines()]

            # 2. 读取预测CSV
            df_preds = pd.read_csv(pred_file)

            # 3. 安全检查：行数是否一致
            if len(patient_ids) != len(df_preds):
                print(
                    f"❌ 错误: 行数不匹配! Split文件 '{split_file.name}' 有 {len(patient_ids)} 个ID, "
                    f"但预测文件 '{pred_file.name}' 有 {len(df_preds)} 行。"
                )
                continue

            # 4. 添加 'patient_id' 列
            # 我们把ID放在第一列，这样更容易查看
            df_preds.insert(0, "patient_id", patient_ids)

            # 5. 覆盖保存
            df_preds.to_csv(pred_file, index=False)

            print(
                f"  ✅ 成功为 '{pred_file.name}' 添加了 {len(patient_ids)} 个 patient_id。"
            )

        except Exception as e:
            print(f"❌ 处理文件 '{pred_file.name}' 时发生意外错误: {e}")

    print("--- 处理完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为一个目录下的所有 k-fold 预测CSV文件添加 patient_id。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="存放 'fold_k_predictions.csv' 文件的目录。",
    )
    parser.add_argument(
        "--splits-dir", type=str, required=True, help="存放 'fold_k.txt' 文件的目录。"
    )

    args = parser.parse_args()
    add_ids_to_predictions(args.preds_dir, args.splits_dir)
