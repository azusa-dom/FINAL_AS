#!/usr/bin/env python3
# coding: utf-8
"""
scripts/prepare_finetune_data_balanced.py

一站式生成平衡训练集：
  • 健康组：BIDS-style NIfTI → 随机抽取切片
  • AS组  ：已有 PNG → 随机数据增强
"""

import argparse
import os
import shutil
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

def parse_args():
    p = argparse.ArgumentParser(description="构建平衡的 AS vs Healthy 数据集")
    p.add_argument("--healthy_root", type=Path, required=True,
                   help="健康组 BIDS rawdata 根目录 (sub-*/anat/*.nii.gz)")
    p.add_argument("--as_png_root", type=Path, required=True,
                   help="AS 组已有 PNG 目录 (脚本前一步生成的 AS_Finetune_Data/1_AS)")
    p.add_argument("--out_root", type=Path, default=Path("AS_Finetune_Data_balanced"),
                   help="输出根目录")
    p.add_argument("--n_slices", type=int, default=5,
                   help="从每个健康受试者抽取的最大切片数")
    p.add_argument("--as_aug_min", type=int, default=5,
                   help="AS 组每张图片最少增强次数")
    p.add_argument("--as_aug_max", type=int, default=10,
                   help="AS 组每张图片最多增强次数")
    return p.parse_args()

def ensure_clean_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def extract_random_slices(nifti_path: Path, out_dir: Path, max_slices: int):
    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    Z = arr.shape[0]
    count = min(Z, max_slices)
    idxs = sorted(random.sample(range(Z), count))
    for idx in idxs:
        sl = arr[idx]
        mn, mx = sl.min(), sl.max()
        if mx > mn:
            u8 = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            u8 = np.zeros_like(sl, dtype=np.uint8)
        im = Image.fromarray(u8).convert("RGB")
        im.save(out_dir / f"{nifti_path.parent.parent.name}_slice_{idx:03d}.png")

def build_healthy(args):
    healthy_out = args.out_root / "0_Healthy"
    ensure_clean_dir(healthy_out)
    print("📦 构建健康组子集…")
    for sub in sorted(args.healthy_root.glob("sub-*")):
        anat = sub / "anat"
        if not anat.exists(): continue
        niftis = list(anat.glob("*.nii*"))
        if not niftis: continue
        nifti = next((f for f in niftis if "T2TSE" in f.name), niftis[0])
        print(f"  - {sub.name}: {nifti.name}")
        extract_random_slices(nifti, healthy_out, args.n_slices)
    total = len(list(healthy_out.glob("*.png")))
    print(f"✅ 健康组完成，共生成切片: {total}\n")

def build_as(args):
    as_out = args.out_root / "1_AS"
    ensure_clean_dir(as_out)
    print("📈 增强 AS 组切片…")
    augmentor = T.Compose([
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
        T.ColorJitter(0.1,0.1,0.1,0.1),
        T.RandomHorizontalFlip(),
    ])
    pngs = list(args.as_png_root.glob("*.png"))
    for p in tqdm(pngs, desc="Augment AS"):
        img = Image.open(p).convert("RGB")
        base = p.stem
        n_aug = random.randint(args.as_aug_min, args.as_aug_max)
        for i in range(n_aug):
            aug = augmentor(img)
            aug.save(as_out / f"{base}_aug{i:02d}.png")
    total = len(list(as_out.glob("*.png")))
    print(f"✅ AS 组增强完成，生成切片: {total}\n")

def main():
    args = parse_args()
    print("🚀 开始构建平衡训练集 …\n")
    ensure_clean_dir(args.out_root)
    build_healthy(args)
    build_as(args)
    print("🎯 平衡数据集已完成！")
    print(f"目录: {args.out_root}")
    print(f"  Healthy: {len(list((args.out_root/'0_Healthy').glob('*.png')))} 张")
    print(f"  AS     : {len(list((args.out_root/'1_AS')    .glob('*.png')))} 张")
    print("\n✅ 即可用 torchvision.datasets.ImageFolder 加载此目录。")

if __name__ == "__main__":
    main()
