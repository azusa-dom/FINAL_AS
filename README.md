
# FINAL_AS – Dual-Modality AI Framework  
_Real-World Ankylosing Spondylitis (AS) Diagnosis from MRI & Clinical Data_

> **MRI pipeline × Tabular (lab + demographics) pipeline**  
> Reproducible • Calibrated • Fully interpretable • Fusion-ready

---

## 📁 Repository Layout

```

FINAL\_AS/
├── data/                  # ⚠ ignored by Git – put raw data here
│   ├── mri\_AS/            # AS patient MRI
│   ├── mri\_health/        # healthy controls
│   └── raw\_lab\_data/      # CSV / XLSX with clinical features
│
├── scripts/               # runnable pipeline scripts (entry points)
│   ├── clinical/          # tabular preprocessing / balancing
│   ├── mri/               # MRI sub-modules
│   │   ├── conversion/    # DICOM / H5 → PNG / NIfTI
│   │   ├── preprocessing/ # bias-field, ROI, slice selection
│   │   ├── analysis/      # AUC, bootstrap, permutation test
│   │   ├── gradcam/       # CAM generation (AS vs healthy)
│   │   ├── visualization/ # t-SNE / UMAP / KDE plots
│   │   └── run/           # one-click orchestration
│   ├── postprocess/       # SHAP + Decision Curve Analysis
│   └── unused/            # archived or experimental utilities
│
├── src/                   # reusable library code (importable as `final_as`)
│   ├── core/              # dataset & evaluation helpers
│   ├── models/            # CNN / MLP / fusion head definitions
│   ├── training/          # training loops for each modality
│   ├── inference/         # model inference / prediction
│   ├── preprocessing/     # fold splits etc.
│   ├── feature\_extraction/
│   ├── analysis/          # MRI feature analytics
│   ├── evaluation/        # bootstrap AUC & AP
│   └── utils/             # generic helpers
│
├── checkpoints/ 🔒        # \*.pth weights (git-ignored)
├── results/               # ready-to-publish outputs
│   ├── clinical/ …        # metrics, SHAP, calibrated curves
│   └── mri/ …             # t-SNE, Grad-CAM, etc.
│
├── requirements.txt       # Python >=3.10 dependency lock
├── LICENSE                # MIT
└── README.md              # ← you are here

````

---

## 👩‍🔬 Method Highlights

| Modality | Samples / Subjects | Core model | Calibration | Interpretability |
|----------|-------------------|------------|-------------|------------------|
| **Clinical** | 4 254 cases, 27 features | 2-layer MLP (ClinicalNet) | Temperature scaling (ECE 0.021) | SHAP + Decision Curve |
| **MRI** | 8 subjects, 39 slices | ResNet-18 frozen encoder → logistic probe | logistic probability | Grad-CAM, t-SNE |

Pipelines are **fully independent** (no paired requirement) yet emit **comparable, calibrated probabilities** – enabling late fusion.

---

## ⚙ Environment Setup

```bash
conda create -n final_as python=3.10
conda activate final_as
pip install -r requirements.txt

# install the right CUDA build of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````

---

## 📦 Data Preparation

```
data/
├── mri_AS/patient001/*.png              # or DICOM/NIfTI if you run conversion first
├── mri_health/health001/subjA/*.png
└── raw_lab_data/Raw_Lab_Dataset.csv     # 27 columns as described in the paper
```

Large or sensitive data remain local; `.gitignore` excludes the whole `data/` directory.

---

## 🚀 Quick Start

### 1 — Clinical pipeline

```bash
# preprocessing + SMOTE fold generation
python scripts/clinical/preprocess_clinical.py \
       --csv data/raw_lab_data/Raw_Lab_Dataset.csv

# training + calibration
python src/training/train.py \
       --folds results/clinical/clinical_data_fold
```

Key outputs appear in `results/clinical/` (metrics CSV, SHAP plots, calibrated curves).

---

### 2 — MRI pipeline

```bash
# optional DICOM → PNG conversion
python scripts/mri/conversion/mri_convert_dicom_to_png.py \
       --input data/mri_AS \
       --output data/mri_images_png

# slice-level embedding + bootstrap CI
python scripts/mri/analysis/mri_eval_auc_bootstrap.py \
       --png_dir data/mri_images_png
```

Grad-CAM heat-maps:

```bash
python scripts/mri/gradcam/As_run_sij_gradcam_analysis.py \
       --png_dir data/mri_images_png
```

All figures land in `results/mri/`.

---

## 🔍 How to Inspect Results

```bash
# Clinical ROC curve
open results/clinical/data_results/sci_roc_curve.png

# MRI t-SNE embedding
open results/mri/embedding_viz/tsne_slice_level.png

# Example Grad-CAM overlay
open results/mri/grad_cam/as/_slice03_gradcam.png
```

---

## 🔬 Interpretability Modules

| Script                                        | Output                  |
| --------------------------------------------- | ----------------------- |
| `scripts/postprocess/shap_compute_summary.py` | global SHAP values      |
| `.../shap_compute_dca.py`                     | decision curve analysis |
| `scripts/mri/gradcam/*.py`                    | attention heat-maps     |

---

## 🔗 Fusion (optional)

`src/training/train_late_fusion.py` already implements:

* Probability weighted average
* Meta-learner stacking

Simply point it to the calibrated CSVs from both modalities.

---

## 🛠 Tips & Troubleshooting

| Issue                    | Fix                                                      |
| ------------------------ | -------------------------------------------------------- |
| CUDA OOM                 | decrease `--batch_size` in preprocessing & train scripts |
| Results slightly vary    | set `--seed 42` everywhere                               |
| Missing ImageNet weights | first `python -m torch.hub` or allow auto-download       |

---

## 🤝 Contributing

1. Fork → branch `feature/<name>`
2. Run `black . && isort .` before PR
3. Include minimal working example & docstring

---

## 📜 License

Released under the MIT License.

---

### Contact

Open an issue or drop an e-mail: **[zczqzh9@ucl.ac.uk](mailto:zczqzh9@ucl.ac.uk)** 🙌


```

# REFERENCES

  - Ai, F., Zhang, W., Liu, H., Song, W., Wu, H., Han, Y., et al. (2012) Value of diffusion-weighted quantification for MRI assessment of sacroiliac joints in early diagnosis of ankylosing spondylitis. *Rheumatology International*, **32**(12), pp.4009–4015. [https://doi.org/10.1007/s00296-011-2253-0(https://doi.org/10.1007/s00296-011-2253-0)
  - Bennani, S., Ohayon, S., Laleye, F., Bauvin, P., Messas, E., et al. (2025) Is multimodal better? A systematic review of multimodal versus unimodal machine learning in clinical decision-making. *medRxiv [Preprint.* [https://doi.org/10.1101/2025.03.12.25322656(https://doi.org/10.1101/2025.03.12.25322656)
  - Bradbury, L.A., Hollis, K.A., Gazer, B., Gollow, I., Shankar, A., Cope, N., et al. (2018) Diffusion-weighted imaging as a sensitive and specific MRI sequence in the diagnosis of chronic nonbacterial osteomyelitis of the sacroiliac joints in children. *The Journal of Rheumatology*, **45**(5), pp.690–697. [https://doi.org/10.3899/jrheum.170871(https://doi.org/10.3899/jrheum.170871)
  - Dubey, S., Chan, A., Adebajo, A.O., Walker, D. and Treglia, G. (2024) Artificial intelligence and machine learning in rheumatology: A systematic literature review. *Rheumatology*, **63**(8), pp.2040–2053. [https://doi.org/10.1093/rheumatology/kead190(https://doi.org/10.1093/rheumatology/kead190)
  - Hosny, A., Parmar, C., Quackenbush, J., Schwartz, L.H. and Aerts, H.J.W.L. (2018) Artificial intelligence in radiology. *Nature Reviews Cancer*, **18**, pp.500–510. [https://doi.org/10.1038/s41568-018-0016-5(https://doi.org/10.1038/s41568-018-0016-5)
  - Jamaludin, A., Kadir, T. and Zisserman, A. (2017) Automated analysis of spinal MRI using deep learning. *Medical Image Analysis*, **40**, pp.67–77. [https://doi.org/10.1016/j.media.2017.06.003(https://doi.org/10.1016/j.media.2017.06.003)
  - Li, H., Zhou, Y., Zhang, Q., Tao, X., Liang, T., Jiang, J., et al. (2023) A multicentre artificial intelligence tool for ankylosing spondylitis supervised by human experts. *Frontiers in Public Health*, **11**, 1063633. [https://doi.org/10.3389/fpubh.2023.1063633(https://doi.org/10.3389/fpubh.2023.1063633)
  - Liao, W., Matsumoto, T., Tanaka, M., Kakehi, T., Nakajima, K., Imagawa, T., et al. (2021) Machine learning in rheumatoid arthritis: applications and challenges. *Modern Rheumatology*, **31**(1), pp.48–55. [https://doi.org/10.1080/14397595.2020.1766343(https://doi.org/10.1080/14397595.2020.1766343)
  - Liu, H., Yang, C., Zhao, M., Ni, L., Chen, R., Zheng, Z., et al. (2020) IgG galactosylation status combined with MYOM2-rs2294066 precisely predicts anti-TNF response in ankylosing spondylitis. *Frontiers in Immunology*, **11**, 600019. [https://doi.org/10.3389/fimmu.2020.600019(https://doi.org/10.3389/fimmu.2020.600019)
  - Maksymowych, W.P., Wichuk, S., Chiowchanwisawakit, P., Lambert, R.G.W. and Pedersen, S.J. (2023) Resolution of MRI inflammation and its association with long-term outcomes in patients with axial spondyloarthritis treated with etanercept. *RMD Open*, **9**(3), e003123. [https://doi.org/10.1136/rmdopen-2023-003123(https://doi.org/10.1136/rmdopen-2023-003123)
  - Pons, M., Georgiadis, S., Hetland, M.L., et al. (2025) Predictors of secukinumab treatment response and continuation in axial spondyloarthritis: Results from the EuroSpA research collaboration network. *The Journal of Rheumatology [Epub ahead of print.* [https://doi.org/10.3899/jrheum.2024-0920(https://doi.org/10.3899/jrheum.2024-0920)
  - Tas, N.P., Kaya, O., Macin, G., Tasci, B., Dogan, S. and Tuncer, T. (2023) ASNET: A novel AI framework for accurate ankylosing spondylitis diagnosis from MRI. *Biomedicines*, **11**(9), 2441. [https://doi.org/10.3390/biomedicines11092441(https://doi.org/10.3390/biomedicines11092441)
  - Tas, S., Siemons, M., Yilmaz, E., Karabulut, E., Ozkan, E., Algin, O. and Cetin, P. (2024) Performance of different classification algorithms in differentiating sacroiliitis grades in patients with axial spondyloarthritis using an MRI-based radiomics model. *Biomedicines*, **12**(1), 200. [https://doi.org/10.3390/biomedicines12010200(https://doi.org/10.3390/biomedicines12010200)
  - Tenório, A.P.M., Cunha, L.P., Almeida, D.A., Ferreira-Junior, J.R., Appenzeller, S. and Rittner, L. (2021) Radiomic diagnosis of sacroiliitis on MRI. *Physics in Medicine & Biology*, **66**(20), 205002. [https://doi.org/10.1088/1361-6560/ac2502(https://doi.org/10.1088/1361-6560/ac2502)
  - Shenavarmasouleh, A., Wahab, H.A., Khaled, M., Sonawane, R., Henry, R. and Iyer, R.K. (2025) Algorithmic foundations for AI in imaging: Dataset design and benchmarking practices. *Data in Brief*, **50**, 109784. [https://doi.org/10.1016/j.dib.2024.109784(https://doi.org/10.1016/j.dib.2024.109784)
  - van der Heijde, D., Landewé, R., Rudwaleit, M., et al. (2018) MRI inflammation at the vertebral unit level and clinical progression in patients with early axial spondyloarthritis: data from the DESIR cohort. *Rheumatology*, **57**(6), pp.1037–1044. [https://doi.org/10.1093/rheumatology/key021(https://doi.org/10.1093/rheumatology/key021)
  - Venerito, V., Brusi, V., Spinelli, F.R., et al. (2023) Beyond the horizon: Innovations and future directions in axial spondyloarthritis. *Archives of Rheumatology*, **38**(4), pp.491–498. [https://doi.org/10.46497/ArchRheumatol.2023.9535(https://doi.org/10.46497/ArchRheumatol.2023.9535)
  - Groza, A., Popescu, D., Ionescu, R., et al. (2021) Multimodal deep learning for clinical prognosis from medical imaging and electronic health records. *Scientific Reports*, **11**, 13594. [https://doi.org/10.1038/s41598-021-93010-0(https://doi.org/10.1038/s41598-021-93010-0)
  - Lee, J., Laouar, Y., Tsoi, L.C. and Zhou, X. (2025) Community series in towards precision medicine for immune-mediated disorders: Advances in using big data and artificial intelligence to understand heterogeneity in disease pathogenesis. *Frontiers in Immunology*, **15**, 1553004. [https://doi.org/10.3389/fimmu.2025.1553004(https://doi.org/10.3389/fimmu.2025.1553004)
  - Vastesaeger, N., van der Heijde, D., Inman, R.D., et al. (2011) Predicting the outcome of ankylosing spondylitis therapy based on baseline characteristics: Data from the ASSERT trial. *The Journal of Rheumatology*, **38**(6), pp.1250–1257. [https://doi.org/10.3899/jrheum.100345(https://doi.org/10.3899/jrheum.100345)
  - Thorley, A., Jensen, M., Brown, S., et al. (2023) Imaging biomarkers for treatment prediction in axial spondyloarthritis: A review. *Current Rheumatology Reports*, **25**(2), pp.123–135. [https://doi.org/10.1007/s11926-023-01078-5(https://doi.org/10.1007/s11926-023-01078-5)
