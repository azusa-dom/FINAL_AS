import pandas as pd
from pathlib import Path
import argparse
import os

def check_data_quality(data_dir_str: str):
    """
    主检查函数：遍历所有处理后的CSV文件，检查是否存在NaN值。
    """
    print("--- 开始全面检查所有已处理的数据文件 ---")
    
    data_dir = Path(data_dir_str)

    if not data_dir.exists():
        print(f"❌ 错误: 找不到目录 '{data_dir}'。")
        print("请确保您已经成功运行了 `scripts/preprocess_clinical.py` 脚本。")
        return

    # 查找所有处理过的 fold CSV 文件
    csv_files = sorted(list(data_dir.glob("fold_*_*.csv")))

    if not csv_files:
        print(f"⚠️ 警告: 在 {data_dir} 中没有找到任何 'fold_*_*.csv' 文件。")
        return

    found_issue = False
    for file_path in csv_files:
        print(f"🔍 正在检查文件: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            
            # 检查整个DataFrame是否有任何NaN值
            if df.isnull().values.any():
                print(f"  ‼️‼️‼️ 警告: 文件 '{file_path.name}' 中发现 NaN (缺失) 值! ‼️‼️‼️")
                # 打印出具体是哪些列有多少个NaN值
                nan_info = df.isnull().sum()
                print("  缺失值统计:")
                print(nan_info[nan_info > 0])
                print("-" * 20)
                found_issue = True

        except Exception as e:
            print(f"  ❌ 读取或检查文件 '{file_path.name}' 时出错: {e}")
            found_issue = True
            
    if not found_issue:
        print("\n--- ✅ 所有文件检查完毕，没有发现明显的 NaN 问题。---")
    else:
        print("\n--- ❗ 检查发现问题，请查看上面的警告信息。问题可能源于 `preprocess_clinical.py` 的填充逻辑未能处理某些特殊情况。---")


if __name__ == "__main__":
    # 我们只保留这个脚本需要用到的参数
    parser = argparse.ArgumentParser(description="临时数据质量检查脚本")
    parser.add_argument("--data_dir", type=str, required=True, help="存放已处理好的、分折后的数据的目录。")
    
    args = parser.parse_args()
    
    check_data_quality(args.data_dir)
