import numpy as np
from collections import defaultdict
import os  # ← 加上这一句！

data = np.load("/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/outputs/mri_feats.npz", allow_pickle=True)
labels = data["labels"]
paths = data["paths"]

basenames = [os.path.basename(p) for p in paths]
patient_ids = []
for bn in basenames:
    if bn.startswith("sub-"):
        pid = bn.split("_")[0]
    elif bn.startswith(("KNEE", "SIJ")):
        pid = bn.split("_")[1]
    else:
        pid = "unknown"
    patient_ids.append(pid)

groups = defaultdict(set)
for pid, label in zip(patient_ids, labels):
    groups[pid].add(label)

for pid in sorted(groups):
    print(f"Patient {pid}: {groups[pid]}")
