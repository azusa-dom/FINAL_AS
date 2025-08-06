import argparse
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio

def set_seeds(seed=42):
    """为可复现性设置所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_volume(input_path: str, output_path: str):
    """
    对单个 3D 医学图像体积执行预处理流程，并保存结果。
    该流程严格遵循论文描述。
    """
    print(f"开始处理: {input_path}")
    set_seeds(42)

    # --- 步骤 1: 使用 SimpleITK 加载图像 ---
    # 使用 float32 以便后续处理
    original_image = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # --- 步骤 2: N4 偏置场校正 ---
    # 注意：N4 校正可能需要一些时间。
    # 对于非蒙版校正，创建一个与图像大小相同的蒙版。
    print("步骤 1/5: 应用 N4 偏置场校正...")
    mask_image = sitk.OtsuThreshold(original_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    n4_corrected_image = corrector.Execute(original_image, mask_image)

    # --- 步骤 3: 3D 高斯平滑 (σ = 0.51 mm) ---
    print("步骤 2/5: 应用 3D 高斯平滑...")
    # sigma 以物理单位（mm）指定
    smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smoothing_filter.SetSigma(0.51)
    smoothed_image = smoothing_filter.Execute(n4_corrected_image)

    # --- 步骤 4: B-spline 重采样到 0.7x0.7 mm 平面内分辨率 ---
    print("步骤 3/5: 执行 B-spline 重采样...")
    original_spacing = smoothed_image.GetSpacing()
    original_size = smoothed_image.GetSize()

    # 新的平面内间距，保持 Z 轴间距不变
    new_spacing = (0.7, 0.7, original_spacing[2])

    # 根据新的间距计算新的图像尺寸
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(smoothed_image.GetDirection())
    resampler.SetOutputOrigin(smoothed_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    # 使用 B-spline 插值
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(smoothed_image)

    # --- 步骤 5: 中心裁剪或补零到 224x224 ---
    # 为了方便地使用 TorchIO 进行此操作，我们先将 sitk 图像保存到临时文件
    print("步骤 4/5: 中心裁剪或补零到 224x224...")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        sitk.WriteImage(resampled_image, tmp.name)
        tmp_path = tmp.name

    # 使用 TorchIO 加载
    tio_image = tio.ScalarImage(tmp_path)

    # 定义裁剪或填充的变换
    # Z 轴维度保持不变，仅处理平面
    target_shape = (224, 224, tio_image.shape[-1])
    crop_or_pad = tio.CropOrPad(target_shape)
    
    processed_tio_image = crop_or_pad(tio_image)
    
    # 清理临时文件
    os.remove(tmp_path)

    # --- 步骤 6: 保存最终处理过的图像 ---
    print("步骤 5/5: 保存处理后的文件...")
    final_output_path = Path(output_path) / Path(input_path).name
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_tio_image.save(final_output_path)
    print(f"处理完成。文件已保存至: {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据论文描述对 3D 医学图像进行预处理。"
    )
    parser.add_argument(
        "input_file", type=str, help="输入的 3D 图像文件路径 (例如 .nii.gz)。"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="用于保存处理后图像的输出目录。",
    )
    args = parser.parse_args()

    preprocess_volume(args.input_file, args.output_dir)