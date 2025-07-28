#!/usr/bin/env python3
"""
Bias Field Correction 测试脚本

这个脚本用于测试bias_correction模块的各项功能。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from mri_src.preprocessing.bias_correction import (
    bias_field_correction_single,
    bias_field_correction_batch,
    setup_logging,
    validate_input_path,
    create_mask
)


class TestBiasCorrection(unittest.TestCase):
    """Bias Correction 测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = setup_logging("WARNING")  # 减少日志输出
        
        # 创建测试用的模拟NIfTI文件
        self.test_image_path = os.path.join(self.temp_dir, "test.nii.gz")
        self._create_test_image()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_image(self):
        """创建测试用的模拟NIfTI图像"""
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
        
        # 保存为NIfTI文件
        sitk.WriteImage(image, self.test_image_path)
    
    def test_validate_input_path(self):
        """测试输入路径验证功能"""
        # 测试有效路径
        self.assertTrue(validate_input_path(self.test_image_path))
        
        # 测试无效路径
        self.assertFalse(validate_input_path("nonexistent.nii.gz"))
        self.assertFalse(validate_input_path("test.txt"))
    
    def test_create_mask(self):
        """测试mask创建功能"""
        image = sitk.ReadImage(self.test_image_path)
        
        # 测试Otsu方法
        mask_otsu = create_mask(image, "otsu")
        self.assertIsInstance(mask_otsu, sitk.Image)
        
        # 测试binary方法
        mask_binary = create_mask(image, "binary")
        self.assertIsInstance(mask_binary, sitk.Image)
        
        # 测试none方法
        mask_none = create_mask(image, "none")
        self.assertIsInstance(mask_none, sitk.Image)
        
        # 测试无效方法
        with self.assertRaises(ValueError):
            create_mask(image, "invalid_method")
    
    def test_bias_field_correction_single(self):
        """测试单个文件bias correction功能"""
        output_path = os.path.join(self.temp_dir, "corrected.nii.gz")
        
        # 测试正常处理
        success = bias_field_correction_single(
            input_path=self.test_image_path,
            output_path=output_path,
            mask_method="otsu",
            logger=self.logger
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # 验证输出图像
        corrected_image = sitk.ReadImage(output_path)
        self.assertIsInstance(corrected_image, sitk.Image)
        
        # 测试无效输入文件
        success = bias_field_correction_single(
            input_path="nonexistent.nii.gz",
            output_path=output_path,
            logger=self.logger
        )
        self.assertFalse(success)
    
    def test_bias_field_correction_batch(self):
        """测试批量处理功能"""
        # 创建多个测试文件
        test_files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"test_{i}.nii.gz")
            # 复制测试图像
            import shutil
            shutil.copy(self.test_image_path, test_file)
            test_files.append(test_file)
        
        output_dir = os.path.join(self.temp_dir, "output")
        
        # 测试批量处理
        success_count, total_count = bias_field_correction_batch(
            input_folder=self.temp_dir,
            output_folder=output_dir,
            mask_method="otsu",
            log_level="WARNING"
        )
        
        self.assertEqual(total_count, 3)
        self.assertEqual(success_count, 3)
        self.assertTrue(os.path.exists(output_dir))
        
        # 验证输出文件
        output_files = [f for f in os.listdir(output_dir) 
                       if f.endswith(('.nii', '.nii.gz'))]
        self.assertEqual(len(output_files), 3)
    
    def test_different_parameters(self):
        """测试不同参数组合"""
        output_path = os.path.join(self.temp_dir, "corrected_params.nii.gz")
        
        # 测试快速处理参数
        success = bias_field_correction_single(
            input_path=self.test_image_path,
            output_path=output_path,
            mask_method="otsu",
            convergence_threshold=0.01,
            max_iterations=10,
            number_of_fitting_levels=2,
            logger=self.logger
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试处理损坏的文件
        corrupted_path = os.path.join(self.temp_dir, "corrupted.nii.gz")
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid NIfTI file")
        
        output_path = os.path.join(self.temp_dir, "output_corrupted.nii.gz")
        
        success = bias_field_correction_single(
            input_path=corrupted_path,
            output_path=output_path,
            logger=self.logger
        )
        
        self.assertFalse(success)


def run_performance_test():
    """运行性能测试"""
    print("=== 性能测试 ===")
    
    # 创建更大的测试图像
    temp_dir = tempfile.mkdtemp()
    test_image_path = os.path.join(temp_dir, "large_test.nii.gz")
    
    try:
        # 创建128x128x64的图像
        size = (128, 128, 64)
        image_array = np.random.rand(*size).astype(np.float32)
        
        # 添加bias field
        x, y, z = np.meshgrid(
            np.linspace(0, 1, size[0]),
            np.linspace(0, 1, size[1]),
            np.linspace(0, 1, size[2]),
            indexing='ij'
        )
        bias_field = 1.0 + 0.3 * (x**2 + y**2 + z**2)
        image_array = image_array * bias_field
        
        image = sitk.GetImageFromArray(image_array)
        sitk.WriteImage(image, test_image_path)
        
        output_path = os.path.join(temp_dir, "corrected_large.nii.gz")
        logger = setup_logging("INFO")
        
        import time
        start_time = time.time()
        
        success = bias_field_correction_single(
            input_path=test_image_path,
            output_path=output_path,
            mask_method="otsu",
            logger=logger
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if success:
            print(f"✅ 大图像处理成功，耗时: {processing_time:.2f}秒")
        else:
            print(f"❌ 大图像处理失败")
            
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """主函数"""
    print("Bias Field Correction 测试")
    print("=" * 40)
    
    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    print("\n" + "=" * 40)
    run_performance_test()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main() 