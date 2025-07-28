import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple
import SimpleITK as sitk
import numpy as np


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录器"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def validate_input_path(input_path: str) -> bool:
    """验证输入文件路径是否有效"""
    if not os.path.exists(input_path):
        return False
    if not input_path.endswith(('.nii', '.nii.gz')):
        return False
    return True


def create_mask(image: sitk.Image, method: str = "otsu") -> sitk.Image:
    """
    创建用于bias correction的mask
    
    Args:
        image: SimpleITK图像对象
        method: mask创建方法 ("otsu", "binary", "none")
    
    Returns:
        SimpleITK mask图像
    """
    if method == "otsu":
        # Otsu阈值分割
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
    elif method == "binary":
        # 简单二值化
        mask = sitk.BinaryThreshold(image, 0, 1, 1, 0)
    elif method == "none":
        # 创建全1的mask
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        mask = sitk.Constant(mask, 1)
    else:
        raise ValueError(f"不支持的mask方法: {method}")
    
    return mask


def bias_field_correction_single(
    input_path: str, 
    output_path: str,
    mask_method: str = "otsu",
    convergence_threshold: float = 0.001,
    max_iterations: int = 50,
    spline_order: int = 3,
    number_of_fitting_levels: int = 4,
    number_of_control_points: int = 4,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    对单个NIfTI图像进行bias field校正
    
    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        mask_method: mask创建方法
        convergence_threshold: 收敛阈值
        max_iterations: 最大迭代次数
        spline_order: B-spline阶数
        number_of_fitting_levels: 拟合层数
        number_of_control_points: 控制点数量
        logger: 日志记录器
    
    Returns:
        bool: 处理是否成功
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # 验证输入文件
        if not validate_input_path(input_path):
            logger.error(f"无效的输入文件: {input_path}")
            return False
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"正在处理: {input_path}")
        
        # 读取图像
        image = sitk.ReadImage(input_path, sitk.sitkFloat32)
        
        # 创建mask
        mask = create_mask(image, mask_method)
        
        # 配置N4 bias field correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetConvergenceThreshold(convergence_threshold)
        corrector.SetMaximumNumberOfIterations(max_iterations)
        corrector.SetSplineOrder(spline_order)
        corrector.SetNumberOfFittingLevels(number_of_fitting_levels)
        corrector.SetNumberOfControlPoints(number_of_control_points)
        
        # 执行校正
        logger.info("开始N4 bias field校正...")
        corrected = corrector.Execute(image, mask)
        
        # 保存结果
        sitk.WriteImage(corrected, output_path)
        logger.info(f"已保存: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def bias_field_correction_batch(
    input_folder: str, 
    output_folder: str,
    mask_method: str = "otsu",
    convergence_threshold: float = 0.001,
    max_iterations: int = 50,
    spline_order: int = 3,
    number_of_fitting_levels: int = 4,
    number_of_control_points: int = 4,
    log_level: str = "INFO"
) -> Tuple[int, int]:
    """
    批量处理文件夹中的NIfTI图像进行bias field校正
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        mask_method: mask创建方法
        convergence_threshold: 收敛阈值
        max_iterations: 最大迭代次数
        spline_order: B-spline阶数
        number_of_fitting_levels: 拟合层数
        number_of_control_points: 控制点数量
        log_level: 日志级别
    
    Returns:
        Tuple[int, int]: (成功处理数量, 总文件数量)
    """
    logger = setup_logging(log_level)
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有NIfTI文件
    nifti_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith(('.nii', '.nii.gz')):
            nifti_files.append(filename)
    
    if not nifti_files:
        logger.warning(f"在 {input_folder} 中未找到NIfTI文件")
        return 0, 0
    
    logger.info(f"找到 {len(nifti_files)} 个NIfTI文件")
    
    # 批量处理
    success_count = 0
    for i, filename in enumerate(nifti_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        logger.info(f"处理进度: {i}/{len(nifti_files)} - {filename}")
        
        success = bias_field_correction_single(
            input_path=input_path,
            output_path=output_path,
            mask_method=mask_method,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            spline_order=spline_order,
            number_of_fitting_levels=number_of_fitting_levels,
            number_of_control_points=number_of_control_points,
            logger=logger
        )
        
        if success:
            success_count += 1
    
    logger.info(f"批量处理完成: {success_count}/{len(nifti_files)} 个文件成功处理")
    return success_count, len(nifti_files)


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="对NIfTI图像进行Bias Field校正")
    parser.add_argument("input_folder", help="输入NIfTI文件夹路径")
    parser.add_argument("output_folder", help="输出校正后文件夹路径")
    parser.add_argument("--mask-method", choices=["otsu", "binary", "none"], 
                       default="otsu", help="mask创建方法 (默认: otsu)")
    parser.add_argument("--convergence-threshold", type=float, default=0.001,
                       help="收敛阈值 (默认: 0.001)")
    parser.add_argument("--max-iterations", type=int, default=50,
                       help="最大迭代次数 (默认: 50)")
    parser.add_argument("--spline-order", type=int, default=3,
                       help="B-spline阶数 (默认: 3)")
    parser.add_argument("--fitting-levels", type=int, default=4,
                       help="拟合层数 (默认: 4)")
    parser.add_argument("--control-points", type=int, default=4,
                       help="控制点数量 (默认: 4)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="日志级别 (默认: INFO)")
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not os.path.exists(args.input_folder):
        print(f"错误: 输入文件夹不存在: {args.input_folder}")
        return 1
    
    # 执行批量处理
    success_count, total_count = bias_field_correction_batch(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        mask_method=args.mask_method,
        convergence_threshold=args.convergence_threshold,
        max_iterations=args.max_iterations,
        spline_order=args.spline_order,
        number_of_fitting_levels=args.fitting_levels,
        number_of_control_points=args.control_points,
        log_level=args.log_level
    )
    
    if success_count == total_count:
        print(f"✅ 所有 {total_count} 个文件处理成功")
        return 0
    else:
        print(f"⚠️  处理完成，但有 {total_count - success_count} 个文件失败")
        return 1


if __name__ == "__main__":
    exit(main())
