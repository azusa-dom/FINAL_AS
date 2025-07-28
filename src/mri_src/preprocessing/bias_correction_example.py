#!/usr/bin/env python3
"""
Bias Field Correction 使用示例

这个脚本展示了如何使用改进后的bias_correction模块进行图像预处理。
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from mri_src.preprocessing.bias_correction import (
    bias_field_correction_single,
    bias_field_correction_batch,
    setup_logging
)


def example_single_file():
    """单个文件处理示例"""
    print("=== 单个文件处理示例 ===")
    
    # 设置输入输出路径（请根据实际情况修改）
    input_file = "data/example.nii.gz"
    output_file = "results/bias_corrected_example.nii.gz"
    
    if not os.path.exists(input_file):
        print(f"示例文件不存在: {input_file}")
        print("请确保有可用的NIfTI文件进行测试")
        return
    
    # 设置日志
    logger = setup_logging("INFO")
    
    # 处理单个文件
    success = bias_field_correction_single(
        input_path=input_file,
        output_path=output_file,
        mask_method="otsu",
        convergence_threshold=0.001,
        max_iterations=50,
        logger=logger
    )
    
    if success:
        print(f"✅ 文件处理成功: {output_file}")
    else:
        print(f"❌ 文件处理失败")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 设置输入输出目录（请根据实际情况修改）
    input_dir = "data/raw_mri"
    output_dir = "results/bias_corrected"
    
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        print("请确保有包含NIfTI文件的目录")
        return
    
    # 批量处理
    success_count, total_count = bias_field_correction_batch(
        input_folder=input_dir,
        output_folder=output_dir,
        mask_method="otsu",
        convergence_threshold=0.001,
        max_iterations=50,
        spline_order=3,
        number_of_fitting_levels=4,
        number_of_control_points=4,
        log_level="INFO"
    )
    
    print(f"批量处理结果: {success_count}/{total_count} 个文件成功处理")


def example_different_mask_methods():
    """不同mask方法示例"""
    print("\n=== 不同Mask方法示例 ===")
    
    input_file = "data/example.nii.gz"
    if not os.path.exists(input_file):
        print(f"示例文件不存在: {input_file}")
        return
    
    logger = setup_logging("INFO")
    
    # 测试不同的mask方法
    mask_methods = ["otsu", "binary", "none"]
    
    for method in mask_methods:
        output_file = f"results/bias_corrected_{method}.nii.gz"
        
        print(f"\n使用 {method} mask方法:")
        success = bias_field_correction_single(
            input_path=input_file,
            output_path=output_file,
            mask_method=method,
            logger=logger
        )
        
        if success:
            print(f"✅ {method} 方法处理成功")
        else:
            print(f"❌ {method} 方法处理失败")


def example_parameter_tuning():
    """参数调优示例"""
    print("\n=== 参数调优示例 ===")
    
    input_file = "data/example.nii.gz"
    if not os.path.exists(input_file):
        print(f"示例文件不存在: {input_file}")
        return
    
    logger = setup_logging("INFO")
    
    # 不同的参数组合
    parameter_sets = [
        {
            "name": "快速处理",
            "convergence_threshold": 0.01,
            "max_iterations": 20,
            "number_of_fitting_levels": 2
        },
        {
            "name": "高质量处理",
            "convergence_threshold": 0.0001,
            "max_iterations": 100,
            "number_of_fitting_levels": 6
        },
        {
            "name": "平衡处理",
            "convergence_threshold": 0.001,
            "max_iterations": 50,
            "number_of_fitting_levels": 4
        }
    ]
    
    for params in parameter_sets:
        output_file = f"results/bias_corrected_{params['name']}.nii.gz"
        
        print(f"\n使用 {params['name']} 参数:")
        success = bias_field_correction_single(
            input_path=input_file,
            output_path=output_file,
            convergence_threshold=params["convergence_threshold"],
            max_iterations=params["max_iterations"],
            number_of_fitting_levels=params["number_of_fitting_levels"],
            logger=logger
        )
        
        if success:
            print(f"✅ {params['name']} 参数处理成功")
        else:
            print(f"❌ {params['name']} 参数处理失败")


def main():
    """主函数"""
    print("Bias Field Correction 使用示例")
    print("=" * 50)
    
    # 检查是否有可用的数据
    data_dirs = ["data", "data/raw_mri", "data/mri_AS", "data/mri_health"]
    available_data = []
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            nifti_files = [f for f in os.listdir(data_dir) 
                          if f.endswith(('.nii', '.nii.gz'))]
            if nifti_files:
                available_data.append((data_dir, len(nifti_files)))
    
    if not available_data:
        print("⚠️  未找到可用的NIfTI文件")
        print("请确保以下目录之一包含NIfTI文件:")
        for data_dir in data_dirs:
            print(f"  - {data_dir}")
        print("\n示例将显示代码结构，但不会实际处理文件")
    
    # 运行示例
    try:
        example_single_file()
        example_batch_processing()
        example_different_mask_methods()
        example_parameter_tuning()
        
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        print("这可能是由于缺少数据文件导致的")


if __name__ == "__main__":
    main() 