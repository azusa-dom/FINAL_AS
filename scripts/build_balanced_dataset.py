#!/usr/bin/env python3

import os
import shutil
import random
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# ========== 参数配置 ==========
HEALTHY_BIDS_ROOT = Path("/Users/hydra/Downloads/0/rawdata")
AS_SOURCE_DIR     = Path("AS_Finetune_Data/1_AS")
OUTPUT_ROOT       = Path("AS_Finetune_Data_balanced")
HEALTHY_OUT_DIR   = OUTPUT_ROOT / "0_Healthy"
AS_OUT_DIR        = OUTPUT_ROOT / "1_AS"
N_SLICES_PER_SUB  = 5
AS_AUG_TIMES      = (5, 10)  # 每张 AS 复制 5~10 张
# =============================

# 定义数据增强器
augmentor = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.RandomHorizontalFlip(),
])

def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def extract_random_slices_from_nifti(nifti_path: Path, sub_id: str, out_dir: Path, max_slices=5):
    try:
        img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(img)  # [Z, Y, X]
        total_slices = arr.shape[0]
        indices = sorted(random.sample(range(total_slices), min(max_slices, total_slices)))
        for i, idx in enumerate(indices):
            sl = arr[idx]
            mn, mx = sl.min(), sl.max()
            if mx > mn:
                u8 = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                u8 = np.zeros_like(sl, dtype=np.uint8)
            img_pil = Image.fromarray(u8).convert("RGB")
            out_path = out_dir / f"{sub_id}_slice_{idx:03d}.png"
            img_pil.save(out_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {nifti_path.name}: {e}")

def build_healthy_subset():
    ensure_clean_dir(HEALTHY_OUT_DIR)
    print("📦 构建健康组切片子集...")
    for subdir in sorted(HEALTHY_BIDS_ROOT.glob("sub-*")):
        anat_dir = subdir / "anat"
        if not anat_dir.exists(): continue
        nifti_files = list(anat_dir.glob("*.nii*"))
        if len(nifti_files) == 0: continue
        # 优先选用 T2TSE
        nifti_file = next((f for f in nifti_files if "T2TSE" in f.name), nifti_files[0])
        extract_random_slices_from_nifti(nifti_file, subdir.name, HEALTHY_OUT_DIR, max_slices=N_SLICES_PER_SUB)
    print(f"✅ 健康组处理完成，共计保存: {len(list(HEALTHY_OUT_DIR.glob('*.png')))} 张")

def build_augmented_as():
    ensure_clean_dir(AS_OUT_DIR)
    print("📈 增强病人组切片...")
    src_pngs = list(AS_SOURCE_DIR.glob("*.png"))
    for i, src_path in enumerate(tqdm(src_pngs, desc="增强中")):
        try:
            img = Image.open(src_path).convert("RGB")
            base_name = src_path.stem
            repeat_times = random.randint(*AS_AUG_TIMES)
            for j in range(repeat_times):
                aug_img = augmentor(img)
                aug_img.save(AS_OUT_DIR / f"{base_name}_aug{j:02d}.png")
        except Exception as e:
            print(f"[ERROR] 处理失败: {src_path.name} → {e}")
    print(f"✅ AS 增强完成，最终图片数: {len(list(AS_OUT_DIR.glob('*.png')))} 张")

def main():
    print("🚀 开始构建平衡训练集...")
    build_healthy_subset()
    build_augmented_as()
    print("\n🎯 平衡数据集构建完成！输出结构如下：")
    print(f"  - 健康样本：{HEALTHY_OUT_DIR}")
    print(f"  - AS 样本：{AS_OUT_DIR}")
    print("\n✅ 可直接用 ImageFolder(root='AS_Finetune_Data_balanced') 加载！")

if __name__ == "__main__":
    main()
