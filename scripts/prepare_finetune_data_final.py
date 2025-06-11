#!/usr/bin/env python3
# 文件: scripts/prepare_finetune_data_final.py

import os
import re
import argparse
import SimpleITK as sitk
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_and_save_slices(image_3d, output_folder, base_filename, axis=2):
    """
    通用函数：将SimpleITK的3D影像切片并保存为PNG。
    """
    image_np = sitk.GetArrayFromImage(image_3d)  # z, y, x
    num = image_np.shape[axis]

    for i in tqdm(range(num), desc=f"Slicing {base_filename}", leave=False):
        if axis == 0:
            sl = image_np[i, :, :]
        elif axis == 1:
            sl = image_np[:, i, :]
        else:
            sl = image_np[:, :, i]

        mn, mx = sl.min(), sl.max()
        if mx > mn:
            u8 = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            u8 = np.zeros_like(sl, dtype=np.uint8)

        img = Image.fromarray(u8).convert("RGB")
        fname = f"{base_filename}_slice_{i:03d}.png"
        img.save(os.path.join(output_folder, fname))

def raw_to_png(raw_path: Path, output_folder: str):
    """
    将 .raw 体数据转中间层 PNG。
    文件名格式示例：SIJ_1_400_400_18_2_.raw
    """
    m = re.match(r".*_(\d+)_(\d+)_(\d+)_(\d+)_\.raw$", raw_path.name)
    if not m:
        print(f"    ↳ 跳过（文件名不匹配）: {raw_path.name}")
        return
    W, H, D, B = map(int, m.groups())
    dtype = np.uint16 if B == 2 else np.uint8
    data = np.fromfile(str(raw_path), dtype=dtype)
    try:
        data = data.reshape((D, H, W))
    except ValueError:
        print(f"    ↳ 跳过（尺寸不符）: {raw_path.name}")
        return

    mid = D // 2
    sl = data[mid, :, :]
    mn, mx = sl.min(), sl.max()
    if mx > mn:
        u8 = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        u8 = np.zeros_like(sl, dtype=np.uint8)

    img = Image.fromarray(u8).convert("RGB")
    out_name = raw_path.stem + ".png"
    img.save(os.path.join(output_folder, out_name))
    print(f"    ↳ Saved RAW→PNG: {out_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AS MRI finetune data.")
    parser.add_argument('--healthy_dir', required=True, help="Healthy NIfTI source dir")
    parser.add_argument('--knee_dir', required=True, help="Knee joint raw dir")
    parser.add_argument('--sacro_dir', required=True, help="Sacroiliac joint raw dir")
    parser.add_argument('--out_dir', default="AS_Finetune_Data", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    healthy_source_dir = args.healthy_dir
    knee_post_dir      = args.knee_dir
    sacro_dir          = args.sacro_dir
    output_dir         = args.out_dir

    healthy_out = os.path.join(output_dir, '0_Healthy')
    as_out      = os.path.join(output_dir, '1_AS')
    ensure_dir(healthy_out)
    ensure_dir(as_out)

    print("▶ 开始准备影像数据切片…\n")

    # 1) 健康组：NIfTI → PNG
    print(f"处理健康组 NIfTI 目录：{healthy_source_dir}")
    if os.path.isdir(healthy_source_dir):
        for sub in sorted(os.listdir(healthy_source_dir)):
            subdir = os.path.join(healthy_source_dir, sub)
            if not os.path.isdir(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(('.nii', '.nii.gz')):
                        path = os.path.join(root, f)
                        print(f"  - {sub}: {path}")
                        try:
                            img3d = sitk.ReadImage(path)
                            process_and_save_slices(img3d, healthy_out, sub)
                        except Exception as e:
                            print(f"    ↳ 错误读取: {path} → {e}")
                        break
                break
    else:
        print("⚠️ 健康组路径不存在！")

    # 2) AS组：膝关节 + 骶髂关节
    print(f"\n处理 AS 组：膝关节 Postcontrast (.raw) + 骶髂关节 (.raw)")

    # 膝关节
    if os.path.isdir(knee_post_dir):
        print(f"  • 膝关节 Postcontrast 目录: {knee_post_dir}")
        for f in sorted(os.listdir(knee_post_dir)):
            if f.lower().endswith('.raw'):
                raw_to_png(Path(knee_post_dir)/f, as_out)
    else:
        print("⚠️ 膝关节 Postcontrast 路径不存在！")

    # 骶髂关节
    if os.path.isdir(sacro_dir):
        print(f"  • 骶髂关节目录: {sacro_dir}")
        for f in sorted(os.listdir(sacro_dir)):
            if f.lower().endswith('.raw'):
                raw_to_png(Path(sacro_dir)/f, as_out)
    else:
        print("⚠️ 骶髂关节路径不存在！")

    print("\n✅ 全部切片完成！")
    print(f"输出统计:")
    print(f"  Healthy PNG 数量: {len(os.listdir(healthy_out))}")
    print(f"  AS      PNG 数量: {len(os.listdir(as_out))}")
    print(f"\n请确认 `{output_dir}` 下已有正确的二分类目录结构。")

if __name__ == '__main__':
    main()
