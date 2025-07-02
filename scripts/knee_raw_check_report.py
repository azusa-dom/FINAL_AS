import numpy as np
import os
import re

input_dir = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/mri_image/knee_joint"
dtype = np.float32
report_path = "knee_raw_check_report.txt"

results = []
total = 0
usable = 0

with open(report_path, "w") as f:
    f.write("Knee raw 文件检查报告\n")
    f.write("="*40 + "\n")
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".raw"):
            continue
        total += 1
        match = re.match(r"KNEE_(\d+)_(\d+)_(\d+)_(\d+)_2_\.raw", fname)
        if not match:
            msg = f"❌ 跳过无法识别的文件名: {fname}"
            print(msg)
            f.write(msg + "\n")
            continue
        pid, x, y, z_in_name = match.groups()
        x = int(x)
        y = int(y)
        z_in_name = int(z_in_name)
        path = os.path.join(input_dir, fname)
        arr = np.fromfile(path, dtype=dtype)
        total_size = arr.size
        if total_size % (x*y) != 0:
            msg = (f"❌ [不能用] {fname}: 文件共 {total_size}，单层 {x}x{y}={x*y}，"
                   f"无法均分，跳过！")
            print(msg)
            f.write(msg + "\n")
            continue
        z = total_size // (x*y)
        if z != z_in_name:
            msg = (f"⚠️  [可用, 但Z不符] {fname}: 文件名Z={z_in_name}，实际推算Z={z}，"
                   f"将用实际Z")
        else:
            msg = f"✅ [完全可用] {fname}: 尺寸 {x}x{y}x{z}"
        print(msg)
        f.write(msg + "\n")
        usable += 1 if total_size % (x*y) == 0 else 0

    f.write("="*40 + "\n")
    f.write(f"总文件数: {total}\n")
    f
