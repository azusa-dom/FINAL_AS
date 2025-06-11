# 文件名: scripts/prepare_finetune_data_final.py

import SimpleITK as sitk
import os
from PIL import Image
import numpy as np

def process_and_save_slices(image_3d, output_folder, base_filename, axis=2):
    """
    一个通用的函数，用于将任何SimpleITK的3D影像对象切片并保存为PNG。
    """
    try:
        image_np = sitk.GetArrayFromImage(image_3d) # 转换为Numpy数组

        # 根据指定的轴确定切片数量
        if axis < 0 or axis > 2:
            print(f"错误: 无效的轴向 {axis}。请选择 0, 1, 或 2。")
            return
            
        num_slices = image_np.shape[axis]
        
        # 遍历每一个切片
        for i in range(num_slices):
            if axis == 0:
                slice_np = image_np[i, :, :]
            elif axis == 1:
                slice_np = image_np[:, i, :]
            else: # axis == 2
                slice_np = image_np[:, :, i]
            
            # 将切片强度值重新缩放到0-255
            if slice_np.max() > slice_np.min():
                slice_rescaled = (255.0 * (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min())).astype(np.uint8)
            else:
                slice_rescaled = np.zeros(slice_np.shape, dtype=np.uint8)

            # 创建并保存PNG图片
            pil_image = Image.fromarray(slice_rescaled).convert('RGB')
            output_filename = f"{base_filename}_slice_{i:03d}.png"
            output_path = os.path.join(output_folder, output_filename)
            pil_image.save(output_path)
            
    except Exception as e:
        print(f"处理并保存切片时发生错误 ({base_filename}): {e}")


def main():
    """
    主函数，分别处理NIfTI格式的健康人数据和DICOM格式的AS病人数据。
    """
    # --- 配置区域 ---
    # !! 请根据您的实际路径修改下面的变量 !!
    healthy_source_dir = '/Users/hydra/Downloads/0' # 存放健康人NIfTI数据的父目录
    as_source_dir = '/Users/hydra/Downloads/1'       # 存放AS病人DICOM数据的父目录
    
    output_finetune_dir = 'AS_Finetune_Data' # 总输出文件夹
    
    # --- 执行 ---
    print("开始准备影像数据切片（最终版）...")

    # 1. 处理健康人的NIfTI数据
    healthy_output_dir = os.path.join(output_finetune_dir, '0_Healthy')
    os.makedirs(healthy_output_dir, exist_ok=True)
    print(f"\n正在处理健康人数据 (NIfTI) 从 '{healthy_source_dir}' -> '{healthy_output_dir}'")
    if os.path.isdir(healthy_source_dir):
        for subject_folder in sorted(os.listdir(healthy_source_dir)):
            subject_path = os.path.join(healthy_source_dir, subject_folder)
            if os.path.isdir(subject_path):
                for root, _, files in os.walk(subject_path):
                    for file in files:
                        if file.endswith(('.nii', '.nii.gz')):
                            nifti_path = os.path.join(root, file)
                            print(f"  - 正在处理 NIfTI 文件: {nifti_path}")
                            image_3d = sitk.ReadImage(nifti_path)
                            process_and_save_slices(image_3d, healthy_output_dir, subject_folder)
                            break # 每个受试者文件夹只处理一个NIfTI文件
    else:
        print(f"警告: 健康人数据源目录不存在: {healthy_source_dir}")

    # 2. 处理AS病人的DICOM数据
    as_output_dir = os.path.join(output_finetune_dir, '1_AS')
    os.makedirs(as_output_dir, exist_ok=True)
    print(f"\n正在处理AS病人数据 (DICOM) 从 '{as_source_dir}' -> '{as_output_dir}'")
    if os.path.isdir(as_source_dir):
        # DICOM数据通常是一个病人一个文件夹
        for subject_folder in sorted(os.listdir(as_source_dir)):
            dicom_series_path = os.path.join(as_source_dir, subject_folder)
            if os.path.isdir(dicom_series_path):
                 # 检查该文件夹是否包含DICOM文件
                if any(f.endswith('.dcm') for f in os.listdir(dicom_series_path)):
                    print(f"  - 正在处理 DICOM 序列: {dicom_series_path}")
                    # 使用ImageSeriesReader来读取一个文件夹内的DICOM序列
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
                    if not dicom_names:
                        print(f"    警告: 在 {dicom_series_path} 中找不到DICOM序列。")
                        continue
                    reader.SetFileNames(dicom_names)
                    image_3d = reader.Execute()
                    process_and_save_slices(image_3d, as_output_dir, subject_folder)
    else:
        print(f"警告: AS病人数据源目录不存在: {as_source_dir}")


    print("\n所有数据处理完成！")
    print(f"请检查 '{output_finetune_dir}' 文件夹，其中应该已经包含了所有生成的2D图片。")

if __name__ == '__main__':
    main()
