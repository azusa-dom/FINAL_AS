# scripts/h5_to_png.py
import os
import argparse
import h5py
import numpy as np
from PIL import Image

def h5_to_png(input_folder, output_folder, normalize=True, verbose=True):
    if verbose:
        print(f"🚀 Starting conversion from: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    converted = 0
    total = 0

    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".h5"):
            continue
        total += 1
        path = os.path.join(input_folder, fname)
        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                if 'reconstruction' in f:
                    arr = f['reconstruction'][()]
                elif 'kspace' in f:
                    k = f['kspace'][()]
                    arr = np.abs(np.fft.ifft2(k))
                else:
                    print(f"❌ {fname} 中未找到 'reconstruction' 或 'kspace'，字段为：{keys}")
                    continue

            if arr.ndim == 3:
                if verbose:
                    print(f"📦 {fname} 是3D，提取第1个切片")
                arr = arr[0]

            if normalize:
                arr = arr - arr.min()
                arr = arr / (arr.max() + 1e-8)
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

            out_name = os.path.splitext(fname)[0] + ".png"
            Image.fromarray(arr).save(os.path.join(output_folder, out_name))
            if verbose:
                print(f"✅ {fname} → {out_name}")
            converted += 1

        except Exception as e:
            print(f"❗ 处理 {fname} 时出错: {e}")

    if verbose:
        print(f"🎯 完成处理：共 {converted} / {total} 个文件转换成功")
    if converted == 0:
        print("⚠️ 没有任何文件被成功转换，请检查路径、数据结构或内容")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="包含 .h5 文件的输入路径")
    parser.add_argument("output_folder", help="保存 PNG 图像的输出路径")
    parser.add_argument("--no-normalize", action="store_true", help="不对图像进行归一化")
    parser.add_argument("--quiet", action="store_true", help="不打印详细日志")
    args = parser.parse_args()

    h5_to_png(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        normalize=not args.no_normalize,
        verbose=not args.quiet
    )
