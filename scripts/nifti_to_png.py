import os
import nibabel as nib
import numpy as np
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description="批量将NIfTI切片导出为PNG")
parser.add_argument("input_folder", help="输入NIfTI文件夹")
parser.add_argument("output_folder", help="输出PNG文件夹")
parser.add_argument("--central-only", action="store_true", help="只导出中央一层（推荐）")
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

for fname in os.listdir(args.input_folder):
    if not fname.endswith('.nii.gz'):
        continue
    path = os.path.join(args.input_folder, fname)
    img = nib.load(path)
    data = img.get_fdata()
    z_count = data.shape[2]
    if args.central_only:
        zs = [z_count // 2]
    else:
        zs = range(z_count)
    for z in zs:
        slice_img = data[:, :, z]
        # 归一化到0-255
        slice_img = (slice_img - slice_img.min()) / (slice_img.ptp() + 1e-8) * 255
        slice_img = slice_img.astype(np.uint8)
        outname = fname.replace('.nii.gz', f'_z{z}.png')
        Image.fromarray(slice_img).save(os.path.join(args.output_folder, outname))
        print(f"✅ {fname} (z={z}) → {outname}")

print("全部NIfTI已导出为PNG。")
