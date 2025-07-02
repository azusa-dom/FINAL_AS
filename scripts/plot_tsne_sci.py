import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/outputs_sij_all/as_mri_feats.npz"
FIG_PNG = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/outputs_sij_all/as_mri_tsne_sci.png"
FIG_PDF = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/outputs_sij_all/as_mri_tsne_sci.pdf"

data = np.load(NPZ_PATH, allow_pickle=True)
tsne = data["embedding"]
patient_ids = data["patient_ids"]

unique_ids = np.unique(patient_ids)
colors = plt.cm.get_cmap('tab10', len(unique_ids))

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
for idx, pid in enumerate(unique_ids):
    mask = (patient_ids == pid)
    ax.scatter(
        tsne[mask, 0], tsne[mask, 1],
        label=f'Patient {pid}',
        s=50, alpha=0.9, linewidths=0.6, edgecolor='k',
        marker='o', c=[colors(idx)]
    )

ax.set_title("Figure 1. t-SNE Visualization of MRI Feature Embeddings in Ankylosing Spondylitis Patients",
             fontsize=16, weight='bold', pad=15)
ax.set_xlabel("t-SNE Component 1", fontsize=14, weight='bold', labelpad=10)
ax.set_ylabel("t-SNE Component 2", fontsize=14, weight='bold', labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title='Patient ID', loc='best', fontsize=11, title_fontsize=12, frameon=True)

plt.tight_layout()
fig.savefig(FIG_PNG)
fig.savefig(FIG_PDF)
print(f"✅ High-resolution t-SNE plot saved as PNG: {FIG_PNG}")
print(f"✅ High-resolution t-SNE plot saved as PDF: {FIG_PDF}")
