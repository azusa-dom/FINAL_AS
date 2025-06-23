# scripts/debug_dataset.py
# -*- coding: utf-8 -*-
"""
一个专门用于调试数据集的脚本。

它会加载 'fold_0_train.csv' 文件，并逐一检查前几个样本，
看看哪个样本返回了 None 或者格式不正确。
"""
import sys
import os
import pandas as pd
import torch
import numpy as np

# 这是一个技巧，确保即使在 scripts/ 目录运行，也能找到 src/ 目录下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.dataset import ClinicalDataset
except ImportError:
    print("错误: 无法从 'src.dataset' 导入 ClinicalDataset。")
    print("请确认 'src/dataset.py' 文件存在且无语法错误。")
    sys.exit(1)


def debug_dataset():
    """主调试函数"""
    print("--- 开始数据集调试 ---")
    
    # 我们将直接检查上一步生成的第一个训练文件
    data_file = "data/processed_clinical_data/fold_0_train.csv"
    label_col = "Disease"
    id_col = "patient_id"

    if not os.path.exists(data_file):
        print(f"❌ 错误: 找不到数据文件 '{data_file}'")
        print("请确保您已经成功运行了 `scripts/preprocess_clinical.py` 脚本，并生成了处理后的数据。")
        return

    print(f"✅ 正在加载数据集: {data_file}")
    
    try:
        # 使用我们更新过的参数来初始化Dataset
        dataset = ClinicalDataset(csv_path=data_file, label_column=label_col, id_column=id_col)
        print(f"✅ 数据集加载成功，总共有 {len(dataset)} 个样本。")
        print("🔍 现在开始检查前 20 个样本...")

        for i in range(min(20, len(dataset))):
            sample = None # 先重置为None
            try:
                # 尝试获取第 i 个样本
                sample = dataset[i]
                
                # 检查样本是否为 None
                if sample is None:
                    print(f"\n‼️‼️‼️ 元凶找到了！索引为 {i} 的样本是 None! ‼️‼️‼️\n")
                    break
                
                # 检查样本格式是否正确
                if not isinstance(sample, tuple) or len(sample) != 3:
                    print(f"\n‼️‼️‼️ 样本 {i} 格式不正确! 期望一个包含3个元素的元组，但得到: {type(sample)} ‼️‼️‼️\n")
                    break
                
                # 检查元组内的元素类型
                features, label, pid = sample
                if not isinstance(features, torch.Tensor) or not isinstance(label, torch.Tensor):
                     print(f"\n‼️‼️‼️ 样本 {i} 内部元素类型不正确! "
                           f"特征类型: {type(features)}, 标签类型: {type(label)} ‼️‼️‼️\n")
                     break

                print(f"  样本 {i}: OK")

            except Exception as e:
                print(f"\n❌❌❌ 在获取索引为 {i} 的样本时发生严重错误: {e} ❌❌❌")
                import traceback
                traceback.print_exc()
                break
        
        print("\n--- ✅ 调试脚本运行完毕 ---")

    except Exception as e:
        print(f"❌ 初始化数据集时出现意外错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()
