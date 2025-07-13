
# Dual‐Modality AI Framework for Ankylosing Spondylitis Diagnosis Independent Validation on Clinical and Imaging Cohorts

---

## 📖 Overview

This repository implements a **dual‐pathway**, modular AI framework for diagnosing axial spondyloarthritis (AxSpA, or Ankylosing Spondylitis, AS) under real‐world, unpaired data constraints. It comprises:

1. **Clinical Pipeline**  
   - End-to-end preprocessing, balancing, and fold‐wise data splits for a large structured cohort (_N_ = 4 254).  
   - Prepares stratified train/validation CSVs with imputation, encoding, scaling, and SMOTE oversampling.

2. **Imaging Pipeline**  
   - Conversion of MRI volumes (DICOM/NIfTI/H5) to 2D PNG slices.  
   - ResNet-18 feature extraction + subject-level aggregation.  
   - Logistic Regression with GroupKFold/LOOCV, bootstrap CIs & permutation testing.  
   - Grad-CAM and t-SNE for interpretability and visualization.

3. **Visualization & Calibration**  
   - Publication-quality ROC, PR, calibration & probability‐distribution plots.  
   - Calibration metrics (ECE, Brier score) and temperature-scaling.

![Pipeline Overview](./docs/pipeline.png)

---

## 📂 Repository Structure

```

.
├── scripts/
│   ├── preprocess\_clinical.py        # Clinical data cleaning, feature‐engineering & fold CSVs
│   ├── create\_balanced\_data.py       # Excel → balanced CSV (legacy / alternative)
│   ├── nifti\_to\_png.py               # Batch export NIfTI → PNG slices
│   ├── h5\_to\_png.py                  # Batch export HDF5 → PNG slices
│   ├── mri\_bootstrap\_auc\_group.py    # MRI GroupKFold + bootstrap AUC & 95% CI
│   ├── mri\_subject\_level\_auc.py      # Subject-level AUC (LOOCV/KFold + bootstrap)
│   ├── mri\_permutation\_test\_full.py  # MRI LOOCV + permutation‐test p-value
│   ├── linear\_probe\_sij.py           # SIJ “linear‐probe” LOOCV + permutation test
│   ├── plot\_tsne\_sci.py              # High-res t-SNE visualization of MRI embeddings
│   └── generate\_publication\_plots.py # Publication-grade ROC/PR/Calib/ConfMat & hist plots
│
├── data/                             # (not committed) place raw clinical CSV & MRI volumes here
│   ├── clinical\_raw\.csv
│   ├── mri\_niftis/                   # DICOM/NIfTI files
│   └── mri\_h5/                       # optional HDF5 files
│
├── results/                          # Outputs: fold CSVs, model predictions, plots…
│   ├── clinical\_folds/
│   ├── clinical\_preds/
│   └── mri\_outputs/
│
├── docs/
│   └── pipeline.png                  # Diagram: “Data Processing Pipeline for Multimodal AS Diagnostics”
│
├── requirements.txt                  # `pip install -r requirements.txt`
└── README.md

````

---

## 🚀 Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/azusa-dom/FINAL_AS.git
   cd FINAL_AS
````

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

> **Requirements snapshot**:
> `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`,
> `nibabel`, `h5py`, `Pillow`, `tqdm`, `imbalanced-learn`, `shap`

---

## 🛠️ Usage

### 1. Clinical Pipeline

1. **Preprocess & split**

   ```bash
   python scripts/preprocess_clinical.py \
     path/to/clinical_raw.csv \
     results/clinical_folds \
     --n_splits 5
   ```

   * Generates `fold_{0..4}_train.csv` and `fold_{0..4}_val.csv`.

2. **Train & evaluate your ClinicalNet model**

   > Use your preferred training script (e.g., `train_clinical.py`) on each `fold_*_train.csv`,
   > then save validation predictions to `results/clinical_preds/fold_{i}_predictions.csv`.

3. **Generate publication-quality plots**

   ```bash
   python scripts/generate_publication_plots.py \
     --preds-dir results/clinical_preds
   ```

   * Outputs ROC, PR, calibration, confusion matrix & probability‐distribution figures under `results/clinical_preds/publication_plots/`.

### 2. Imaging Pipeline

1. **Convert volumes to PNG slices**

   * **NIfTI → PNG** (central slice only):

     ```bash
     python scripts/nifti_to_png.py \
       data/mri_niftis \
       data/mri_slices_png \
       --central-only
     ```
   * **HDF5 → PNG**:

     ```bash
     python scripts/h5_to_png.py \
       data/mri_h5 \
       data/mri_slices_png
     ```

2. **Run MRI linear-probe / bootstrap / permutation tests**

   * **GroupKFold + bootstrap AUC**

     ```bash
     python scripts/mri_bootstrap_auc_group.py \
       --data-dir data/mri_slices_png \
       --n-splits 5 \
       --n-bootstrap 2000 \
       --batch-size 16 \
       --seed 42
     ```
   * **Subject-level LOOCV/KFold AUC**

     ```bash
     python scripts/mri_subject_level_auc.py \
       --data-dir data/mri_slices_png \
       --n-splits 8 \
       --n-bootstrap 2000
     ```
   * **Permutation test (LOOCV)**

     ```bash
     python scripts/mri_permutation_test_full.py \
       --data-dir data/mri_slices_png \
       --n-perm 5000
     ```
   * **SIJ linear probe (6 AS vs 2 healthy)**

     ```bash
     python scripts/linear_probe_sij.py \
       --sij-as-dir data/sij_as_png \
       --sij-healthy-dir data/sij_healthy_png \
       --n-perm 5000
     ```

3. **Visualize embeddings & attention**

   * **t-SNE plot**

     ```bash
     python scripts/plot_tsne_sci.py
     ```
   * **Grad-CAM**

     * (Implemented inline in the fine-tuning notebook or script—see comments.)

---

## 🧪 Evaluation & Metrics

* **Discrimination**: AUROC, AUPRC
* **Calibration**: Brier Score, Expected Calibration Error (ECE), Temperature Scaling
* **Statistical Tests**: Bootstrap 95% CI, Permutation p-values
* **Interpretability**: SHAP (clinical), Grad-CAM (imaging)

---

## 📄 Citation

If you use this framework, please cite our manuscript:

> **A Dual‐Modality AI Framework for Ankylosing Spondylitis Diagnosis Under Real‐World Data Constraints: Independent Validation on Clinical and Imaging Cohorts**
> *\[Authors et al.], Journal/Preprint (Year).*

---

## 🤝 Contributing

* ✅ Fork & branch under `feature/*`
* ✅ Add tests or examples for new functionality
* ✅ Update `README.md` & docs
* ✅ Submit a pull request!

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```

**Is there any code or asset you’re missing before running these scripts?**  
- Place your **raw clinical CSV** in `data/clinical_raw.csv`.  
- Place your **MRI volumes** in `data/mri_niftis/` (or H5 files in `data/mri_h5/`).  
- Ensure the pipeline diagram (`docs/pipeline.png`) is available or adjust the path in this README.
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
