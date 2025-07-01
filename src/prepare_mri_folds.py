# src/prepare_mri_folds.py

import os
import argparse
import shutil
import random

def prepare_folds(data_dir, output_dir, val_ratio=0.2, seed=42):
    """
    把 data_dir 下每个类别子文件夹里的图像，按 val_ratio 切分到 train/ 和 val/。
    """
    random.seed(seed)
    # 1) 找到所有类别
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    # 2) 创建输出目录结构
    for phase in ('train','val'):
        for cls in classes:
            os.makedirs(os.path.join(output_dir, phase, cls), exist_ok=True)

    # 3) 对每个类别切分并拷贝
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        random.shuffle(imgs)
        n_val = int(len(imgs) * val_ratio)
        for i, fname in enumerate(imgs):
            phase = 'val' if i < n_val else 'train'
            src = os.path.join(cls_dir, fname)
            dst = os.path.join(output_dir, phase, cls, fname)
            shutil.copy2(src, dst)
    print(f"✅ 完成切分: {data_dir} → {output_dir} (val_ratio={val_ratio})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare MRI train/val folds")
    p.add_argument('--data-dir',   type=str, required=True,
                   help="原始平衡数据根目录（含各类别子文件夹）")
    p.add_argument('--output-dir', type=str, required=True,
                   help="切分后输出根目录")
    p.add_argument('--val-ratio',  type=float, default=0.2,
                   help="验证集比例（默认0.2）")
    p.add_argument('--seed',       type=int,   default=42,
                   help="随机种子")
    args = p.parse_args()
    prepare_folds(args.data_dir, args.output_dir,
                  val_ratio=args.val_ratio, seed=args.seed)
