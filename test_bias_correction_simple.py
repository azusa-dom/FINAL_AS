#!/usr/bin/env python3
"""
简单的Bias Correction测试脚本

这个脚本演示如何正确使用bias_correction模块。
"""

import os
import sys
import tempfile
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

def create_test_nifti_file(output_path):
    """创建一个测试用的NIfTI文件"""
    print(f"创建测试文件: {output_path}")
    
    # 创建一个简单的3D图像
    size = (64, 64, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    
    # 创建带有bias field的测试图像
    image_array = np.random.rand(*size).astype(np.float32)
    
    # 添加bias field效果
    x, y, z = np.meshgrid(
        np.linspace(0, 1, size[0]),
        np.linspace(0, 1, size[1]),
        np.linspace(0, 1, size[2]),
        indexing='ij'
    )
    
    # 创建径向bias field
    bias_field = 1.0 + 0.3 * (x**2 + y**2 + z**2)
    image_array = image_array * bias_field
    
    # 转换为SimpleITK图像
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为NIfTI文件
    sitk.WriteImage(image, output_path)
    print(f"✅ 测试文件创建成功: {output_path}")

def test_bias_correction():
    """测试bias correction功能"""
    print("=== Bias Correction 测试 ===")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    try:
        # 创建测试输入目录
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建几个测试文件
        test_files = []
        for i in range(3):
            test_file = os.path.join(input_dir, f"test_{i}.nii.gz")
            create_test_nifti_file(test_file)
            test_files.append(test_file)
        
        print(f"\n创建了 {len(test_files)} 个测试文件")
        
        # 导入bias correction模块
        try:
            from mri_src.preprocessing.bias_correction import bias_field_correction_batch
            print("✅ 成功导入bias_correction模块")
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            print("请确保在正确的目录下运行此脚本")
            return
        
        # 运行bias correction
        print("\n开始bias correction处理...")
        success_count, total_count = bias_field_correction_batch(
            input_folder=input_dir,
            output_folder=output_dir,
            mask_method="otsu",
            convergence_threshold=0.001,
            max_iterations=20,  # 减少迭代次数以加快测试
            log_level="INFO"
        )
        
        print(f"\n处理结果: {success_count}/{total_count} 个文件成功处理")
        
        # 检查输出文件
        if os.path.exists(output_dir):
            output_files = [f for f in os.listdir(output_dir) 
                           if f.endswith(('.nii', '.nii.gz'))]
            print(f"输出目录中的文件: {output_files}")
        
        if success_count == total_count:
            print("🎉 所有测试通过！")
        else:
            print("⚠️  部分文件处理失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n清理临时目录: {temp_dir}")

def show_usage_examples():
    """显示使用示例"""
    print("\n=== 使用示例 ===")
    print("1. 命令行使用:")
    print("   python src/mri_src/preprocessing/bias_correction.py input_folder output_folder")
    print()
    print("2. 如果您有NIfTI文件，可以这样运行:")
    print("   python src/mri_src/preprocessing/bias_correction.py data/mri_AS results/bias_corrected")
    print()
    print("3. 使用高级参数:")
    print("   python src/mri_src/preprocessing/bias_correction.py \\")
    print("       data/mri_AS \\")
    print("       results/bias_corrected \\")
    print("       --mask-method otsu \\")
    print("       --convergence-threshold 0.001 \\")
    print("       --max-iterations 50")
    print()
    print("注意: 输入文件夹必须包含.nii或.nii.gz格式的文件")

def main():
    """主函数"""
    print("Bias Correction 简单测试")
    print("=" * 50)
    
    # 检查当前目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查是否有可用的NIfTI文件
    data_dirs = ["data", "data/mri_AS", "data/mri_health"]
    nifti_files_found = False
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.nii', '.nii.gz')):
                        print(f"找到NIfTI文件: {os.path.join(root, file)}")
                        nifti_files_found = True
    
    if not nifti_files_found:
        print("⚠️  未找到NIfTI文件")
        print("您的数据似乎是JPG格式，需要先转换为NIfTI格式")
        print("或者运行测试脚本来验证模块功能")
    
    # 运行测试
    test_bias_correction()
    
    # 显示使用示例
    show_usage_examples()

if __name__ == "__main__":
    main() 