# 文件名: scripts/prepare_finetune_data.py

import SimpleITK as sitk
import os
from PIL import Image
import numpy as np

def slice_nifti_to_png(nifti_path, output_folder, subject_id, axis=2):
    """
    读取一个NIfTI文件，将其沿指定轴的每个切片都保存为PNG图片。

    Args:
        nifti_path (str): 输入的NIfT文件路径 (.nii 或 .nii.gz)。
        output_folder (str): 保存PNG图片的输出文件夹。
        subject_id (str): 受试者的ID，用于命名文件。
        axis (int): 切片的轴向 (0, 1, 或 2)。默认是2，通常对应横断面(Axial)。
    """
    try:
        # 1. 读取3D影像
        image_3d = sitk.ReadImage(nifti_path)
        image_np = sitk.GetArrayFromImage(image_3d) # 将其转换为Numpy数组

        # 调整Numpy数组的轴序，SimpleITK是(z,y,x), Numpy是(x,y,z)
        # 我们通常按 (depth, height, width) 来处理
        if axis == 0: # 矢状面 Sagittal
            num_slices = image_np.shape[2]
        elif axis == 1: # 冠状面 Coronal
            num_slices = image_np.shape[1]
        else: # 横断面 Axial
            num_slices = image_np.shape[0]

        # 2. 遍历每一个切片
        for i in range(num_slices):
            if axis == 0:
                slice_np = image_np[:, :, i]
            elif axis == 1:
                slice_np = image_np[:, i, :]
            else:
                slice_np = image_np[i, :, :]

            # 3. 将切片强度值重新缩放到0-255范围，并转换为8位无符号整数
            if slice_np.max() > slice_np.min():
                slice_rescaled = (255.0 * (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min())).astype(np.uint8)
            else:
                slice_rescaled = np.zeros(slice_np.shape, dtype=np.uint8)

            # 4. 创建并保存PNG图片
            pil_image = Image.fromarray(slice_rescaled).convert('RGB') # 转换为RGB以适配预训练模型
            
            # 定义输出文件名，例如: sub-01_slice_050.png
            output_filename = f"{subject_id}_slice_{i:03d}.png"
            output_path = os.path.join(output_folder, output_filename)
            pil_image.save(output_path)
        
        # print(f"成功处理 {subject_id}，共生成 {num_slices} 张切片。")

    except Exception as e:
        print(f"处理文件 {nifti_path} 时发生错误: {e}")


def main():
    """
    主函数，遍历所有源文件夹，处理所有NIfTI文件。
    """
    # --- 配置区域 ---
    # !! 请根据您的实际路径修改下面的变量 !!
    
    # 包含原始NIfTI文件的文件夹
    # 'path/to/SpineNerve_data' 是您14个健康人数据的父目录
    healthy_source_dir = 'path/to/SpineNerve_data' 
    # 'path/to/Mendeley_data' 是您9个AS病人数据的父目录
    as_source_dir = 'path/to/Mendeley_data' 
    
    # 我们准备好的、用于存放2D切片的文件夹
    output_finetune_dir = 'AS_Finetune_Data'
    
    # --- 执行 ---
    print("开始准备影像数据切片...")

    process_map = {
        healthy_source_dir: os.path.join(output_finetune_dir, '0_Healthy'),
        as_source_dir: os.path.join(output_finetune_dir, '1_AS')
    }

    for source_parent_dir, output_dir in process_map.items():
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n正在处理 '{source_parent_dir}' -> '{output_dir}'")
        
        if not os.path.isdir(source_parent_dir):
            print(f"警告: 源目录不存在: {source_parent_dir}，跳过处理。")
            continue

        # 遍历源目录下的所有受试者文件夹
        for subject_folder in sorted(os.listdir(source_parent_dir)):
            subject_path = os.path.join(source_parent_dir, subject_folder)
            if os.path.isdir(subject_path):
                # 寻找我们想要的NIfTI文件 (假设是T2序列)
                for root, _, files in os.walk(subject_path):
                    for file in files:
                        if file.endswith(('.nii', '.nii.gz')) and 't2' in file.lower():
                            nifti_file_path = os.path.join(root, file)
                            print(f"  - 正在切片: {nifti_file_path}")
                            # 执行切片和保存
                            slice_nifti_to_png(nifti_file_path, output_dir, subject_folder)
                            # 每个受试者只处理一个相关的NIfTI文件就跳出
                            break 
    
    print("\n所有数据处理完成！")
    print(f"请检查 '{output_finetune_dir}' 文件夹，其中应该已经包含了所有生成的2D图片。")

if __name__ == '__main__':
    main()
