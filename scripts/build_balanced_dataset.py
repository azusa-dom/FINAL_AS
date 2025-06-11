#!/usr/bin/env python3
# coding: utf-8
"""
scripts/build_balanced_dataset.py

一站式构建平衡的 AS vs Healthy 数据集：
  • Healthy: 从 BIDS-style NIfTI 随机抽切片
  • AS     : 从 .raw 文件提取中间切片并做随机增强
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--healthy_root", type=Path, required=True,
                   help="健康组 rawdata 根目录，格式为 sub-*/anat/*.nii*")
    p.add_argument("--knee_raw",     type=Path, required=True,
                   help="AS 组膝关节 Postcontrast 文件夹，含 .raw 文件")
    p.add_argument("--sacro_raw",    type=Path, required=True,
                   help="AS 组骶髂关节 文件夹，含 .raw 文件")
    p.add_argument("--out_root",     type=Path, default=Path("AS_Finetune_Data_balanced"),
                   help="输出根目录")
    p.add_argument("--n_slices",     type=int, default=5,
                   help="从每个健康受试者抽取的切片数")
    p.add_argument("--as_aug_min",   type=int, default=5,
                   help="AS 样本最少增强次数")
    p.add_argument("--as_aug_max",   type=int, default=10,
                   help="AS 样本最多增强次数")
    return p.parse_args()

def ensure_clean_dir(d: Path):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def save_slice(arr2d: np.ndarray, out_path: Path):
    mn, mx = arr2d.min(), arr2d.max()
    if mx>mn:
        u8 = ((arr2d-mn)/(mx-mn)*255).astype(np.uint8)
    else:
        u8 = np.zeros_like(arr2d, dtype=np.uint8)
    Image.fromarray(u8).convert("RGB").save(out_path)

def build_healthy(args):
    out = args.out_root / "0_Healthy"
    ensure_clean_dir(out)
    print("📦 Healthy: 抽取切片")
    for sub in sorted(args.healthy_root.glob("sub-*")):
        anat = sub / "anat"
        if not anat.exists(): continue
        nifs = list(anat.glob("*.nii*"))
        if not nifs: continue
        nifti = next((f for f in nifs if "T2TSE" in f.name), nifs[0])
        img3d = sitk.ReadImage(str(nifti))
        arr3d = sitk.GetArrayFromImage(img3d)  # shape=(Z,Y,X)
        Z = arr3d.shape[0]
        picks = sorted(random.sample(range(Z), min(Z, args.n_slices)))
        for idx in picks:
            save_slice(arr3d[idx], out / f"{sub.name}_slice_{idx:03d}.png")
    print("✅ Healthy 完成，共", len(list(out.glob("*.png"))), "张切片\n")

def raw_to_slice(raw_f: Path, out: Path):
    # 文件名形如 XXX_W_H_D_B_.raw
    base = raw_f.stem
    nums = raw_f.stem.split("_")[-5:-1]  # ['W','H','D','B']
    W,H,D,B = map(int, nums)
    dtype = np.uint16 if B==2 else np.uint8
    data = np.fromfile(str(raw_f), dtype)
    try:
        vol = data.reshape((D,H,W))
    except:
        print("⚠️ 尺寸不符，跳过", raw_f.name); return
    mid = vol.shape[0]//2
    save_slice(vol[mid], out / f"{base}.png")

def build_as(args):
    out = args.out_root / "1_AS"
    ensure_clean_dir(out)
    print("📈 AS: 提取中间切片")
    # 膝关节 Postcontrast
    if args.knee_raw.exists():
        for f in sorted(args.knee_raw.glob("*.raw")):
            raw_to_slice(f, out)
    # 骶髂关节
    if args.sacro_raw.exists():
        for f in sorted(args.sacro_raw.glob("*.raw")):
            raw_to_slice(f, out)
    base_pngs = list(out.glob("*.png"))
    print("➡️ 提取完成，共", len(base_pngs), "张 PNG")
    # 增强
    print("📈 AS: 数据增强")
    aug = T.Compose([
        T.RandomRotation(10),
        T.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
        T.ColorJitter(0.1,0.1,0.1,0.1),
        T.RandomHorizontalFlip(),
    ])
    for p in tqdm(base_pngs, desc="Augment"):
        img = Image.open(p).convert("RGB")
        for i in range(random.randint(args.as_aug_min, args.as_aug_max)):
            aug(img).save(out / f"{p.stem}_aug{i:02d}.png")
    print("✅ AS 增强完成，共", len(list(out.glob("*.png"))), "张图片\n")

def main():
    args = parse_args()
    print("🚀 开始构建平衡数据集\n")
    ensure_clean_dir(args.out_root)
    build_healthy(args)
    build_as(args)
    print("🎯 完成!")
    print("目录：", args.out_root)
    print("  Healthy:", len(list((args.out_root/"0_Healthy").glob("*.png"))))
    print("  AS     :", len(list((args.out_root/"1_AS").glob("*.png"))))

if __name__=="__main__":
    main()
