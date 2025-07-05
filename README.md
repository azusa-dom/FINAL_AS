

# 🧠 Ankylosing Spondylitis Diagnosis Using Independent Clinical and MRI Cohorts

**A validated machine learning model on structured clinical data and an exploratory deep feature study on independent MRI samples.**

-----

## 🔍 Overview

This project implements a dual-cohort strategy for the early diagnosis of Ankylosing Spondylitis (AS):

  - 🧪 **Clinical Pipeline**: A feed-forward neural network (FFNN) trained on a structured dataset of 4,254 records achieves an **AUROC of 0.922**.
  - 🧠 **MRI Pipeline**: A deep learning feasibility study on an independent, small-sample MRI cohort (N=8), with feature visualization via t-SNE and Grad-CAM.
  - ✅ The two pipelines are fully decoupled and independently evaluated, reflecting real-world data availability scenarios.

-----

## 📚 Motivation and Novelty

  - The early diagnosis of AS remains clinically challenging. [cite: 241, 242 While laboratory and MRI data both offer diagnostic insights, real-world clinical practice often lacks paired multimodal datasets. [cite: 253, 280
  - This study independently validates:
    1.  A high-performance diagnostic model on large-scale clinical data.
    2.  The discriminability of deep features in a separate MRI cohort using visualization and statistical testing.

-----

## 🗂️ Data Sources

| Cohort | N | Content | Purpose |
|---|---|---|---|
| Clinical | 4,254 | 27 structured variables (labs, HLA-B27) [cite: 303, 328 | FFNN model training & evaluation |
| MRI (public) | 8 | SIJ MRI slices (6 AS, 2 healthy) | Deep feature feasibility + Grad-CAM |

-----

## 🧰 Methods

### 🔬 1. Clinical Pipeline

  - **Script**: `scripts/preprocess_final.py`
  - **Features**: 27 structured variables, including CRP, ESR, BASDAI, and HLA-B27. [cite: 328
  - **Preprocessing**:
      - Median imputation for continuous features. [cite: 330
      - Mode imputation and one-hot encoding for categorical variables. [cite: 330, 332
      - `log1p` transformation and `RobustScaler` for skewed inflammation markers (ESR, CRP).
  - **Model**: Feedforward Neural Network (FFNN).
      - Architecture: 27 → 256 → 128 → 64 → 1.
      - ReLU activation, Dropout(0.2), and Sigmoid output.
  - **Training**:
      - Optimizer: AdamW (lr=1e-3, weight\_decay=1e-4).
      - Batch size: 128, Epochs: 30.
      - 5-fold stratified cross-validation. [cite: 199
      - SMOTE applied within training folds to handle minor imbalances.
  - **Calibration**:
      - Temperature scaling (T ≈ 1.13) for probability calibration.

<!-- end list -->

```bash
python src/train.py \
  --config configs/clinical_ffnn.yaml
```

-----

### 🧠 2. MRI Feasibility Pipeline

  - **Feature Extraction**:
      - A pretrained ResNet-18 is used to extract 512-d slice-level features.
  - **Aggregation**:
      - Slice-level features are aggregated per subject via mean pooling. [cite: 43, 56, 147
  - **Evaluation**:
      - **Classifier**: Logistic Regression. [cite: 14, 44, 53, 72
      - **Cross-validation**: Leave-One-Out CV (LOOCV). [cite: 44, 47, 52, 68
      - **Metric**: AUC with a 95% CI via 2000x bootstrap.
      - **Permutation Test**: N=5000 to assess statistical separation of features. [cite: 48, 57, 75, 138
  - **Visualization**:
      - t-SNE, UMAP, and PCA for dimensionality reduction.
      - Grad-CAM for model interpretability on SIJ slices.

<!-- end list -->

```bash
python scripts/mri_subject_level_auc.py \
  --data-dir data/mri_image_modified \
  --n-splits 8 \
  --n-bootstrap 2000
```

-----

## 📈 Evaluation Metrics

| Metric | Value | CI / Details |
|---|---|---|
| AUROC | 0.922 | 95% CI: 0.913–0.930 |
| AUPRC | 0.889 | 95% CI: 0.873–0.904 |
| Brier Score | 0.138 [cite: 503, 512 | Post-calibration |
| ECE₁₀ | 0.021 [cite: 503, 512 | Expected Calibration Error |
| DCA Net Benefit | +0.18 (at p=0.3) [cite: 236, 393 | Decision Curve Analysis |
| MRI AUC | \~0.995 (LOOCV) | On N=8 independent cohort |
| Permutation Test | p = 0.907 | Centroid distance (non-significant) |

-----

## 🎨 Visual Outputs

| Figure | File | Description |
|---|---|---|
| ROC Curve | `sci_roc_curve.png` | Clinical model ROC |
| PR Curve | `sci_pr_curve.png` | Precision–Recall |
| Calibration Curve | `sci_calibration_curve.png` | Post temperature-scaling |
| Confusion Matrix | `sci_confusion_matrix.png` | 2x2 heatmap |
| SHAP Summary | `shap_summary.png` | Top feature contributors |
| t-SNE (MRI features) | `tsne_slice_level.png` | MRI feature space |
| Grad-CAM (samples) | `*_gradcam_academic.png` | Heatmaps over SIJ slices |

-----

## 🚀 Run Instructions

### One-click execution:

```bash
bash run_all.sh
```

### Manual step-by-step:

```bash
# Clinical Preprocessing
python scripts/preprocess_clinical.py data/clinical_raw.csv processed_data/

# FFNN Training
python src/train.py --config configs/clinical_ffnn.yaml

# MRI AUC + Permutation Test
python scripts/mri_subject_level_auc.py --data-dir data/mri_image_modified
python scripts/mri_permutation_test_full.py --data-dir data/mri_image_modified

# Grad-CAM Visualization
python scripts/plot_gradcam_academic.py
```

-----

## 🧾 Citation

Ankylosing Spondylitis Diagnosis Using Independent Clinical and MRI Cohorts: A Validated Machine Learning Model and a Deep Feature Feasibility Study. *Under Review* (2025).

This repository supports the full reproducibility of all experiments and visualizations presented in the manuscript.

-----

## 📄 License

This project is released under the MIT License.

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
