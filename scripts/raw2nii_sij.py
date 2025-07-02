import numpy as np
import nibabel as nib
import os
import re

input_dir = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/mri_image/sacroiliac_joint"
output_dir = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/nii_sij"
os.makedirs(output_dir, exist_ok=True)

dtype = np.float32

for fname in os.listdir(input_dir):
    if not fname.endswith(".raw"):
        continue

    match = re.match(r"SIJ_\d+_(\d+)_(\d+)_(\d+)_2_\.raw", fname)
    if not match:
        print(f"❌ 跳过无法识别的文件名: {fname}")
        continue

    x, y, z_in_name = map(int, match.groups())
    path = os.path.join(input_dir, fname)
    arr = np.fromfile(path, dtype=dtype)
    total_size = arr.size
    # 能否被 x*y 整除
    if total_size % (x*y) != 0:
        print(f"❌ {fname} 大小不匹配：文件共 {total_size}，单层 {x}x{y}={x*y}，无法均分，跳过！")
        continue
    z = total_size // (x*y)
    if z != z_in_name:
        print(f"⚠️ {fname} 文件名Z={z_in_name}，实际推算Z={z}，将用实际Z")
    try:
        arr = arr.reshape((x, y, z))
    except Exception as e:
        print(f"❌ {fname} reshape失败: {e}，跳过！")
        continue
    nii = nib.Nifti1Image(arr, np.eye(4))
    outname = fname.replace(".raw", ".nii.gz")
    nib.save(nii, os.path.join(output_dir, outname))
    print(f"✅ {fname} → {outname} (Shape: {x},{y},{z})")

print("全部 SIJ .raw → .nii.gz 转换完成！")
