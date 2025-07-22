# Dual-Modality AI Framework

**Real-World Ankylosing Spondylitis (AS) Diagnosis from MRI & Clinical Data**

This repository implements a reproducible, interpretable, and calibrated dual-modality AI framework for diagnosing Ankylosing Spondylitis (AS) using MRI scans and clinical tabular data. Both pipelines run independently but can be fused at the probability level.

---

## Table of Contents

1. [Features](#features)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Data Organization](#data-organization)
5. [Clinical Pipeline](#clinical-pipeline)

   * [Preprocessing](#preprocessing)
   * [Training and Calibration](#training-and-calibration)
   * [Evaluation](#evaluation)
6. [MRI Pipeline](#mri-pipeline)

   * [DICOM to PNG Conversion](#dicom-to-png-conversion)
   * [Preprocessing and Slice Selection](#preprocessing-and-slice-selection)
   * [Embedding Extraction and Analysis](#embedding-extraction-and-analysis)
   * [Grad-CAM Visualization](#grad-cam-visualization)
7. [Postprocessing and Interpretability](#postprocessing-and-interpretability)
8. [Late Fusion](#late-fusion)
9. [Results Inspection](#results-inspection)
10. [Troubleshooting and Tips](#troubleshooting-and-tips)
11. [Contributing](#contributing)
12. [License](#license)

---

## Features

* **Clinical Modality** (4,254 samples, 27 features)

  * Model: 2-layer MLP (`ClinicalNet`)
  * Calibration: Temperature scaling (ECE ≈ 0.02)
  * Interpretability: SHAP summaries, Decision Curve Analysis

* **MRI Modality** (8 subjects, selected slices)

  * Model: ResNet‑18 frozen encoder + logistic probe
  * Calibration: Direct probabilities
  * Interpretability: Grad‑CAM, t-SNE embeddings, KDE distance metrics

* **Design Principles**

  * Reproducible: Fixed seeds, standardized splits
  * Calibrated: Post-hoc probability scaling
  * Interpretable: Global and local explanations
  * Fusion-ready: Compatible probability outputs

---

## Repository Structure

```
FINAL_AS/
├── data/                   # git-ignored: raw data directory
│   ├── mri_AS/             # AS patient MRI data (DICOM or PNG)
│   ├── mri_health/         # Healthy control MRI data
│   └── raw_lab_data/       # Clinical CSV/XLSX with 27 features + label
│
├── scripts/                # Executable pipeline scripts
│   ├── clinical/           # Clinical preprocessing and splitting
│   ├── mri/                # MRI conversion, preprocessing, analysis
│   │   ├── conversion/     # DICOM/NIfTI ⇄ PNG conversion
│   │   ├── preprocessing/  # Bias correction, ROI, slice filtering
│   │   ├── analysis/       # AUC bootstrap, permutation testing
│   │   ├── gradcam/        # Grad‑CAM heatmap generation
│   │   ├── visualization/  # t-SNE, UMAP, KDE plotting
│   │   └── run/            # Full MRI pipeline orchestration
│   ├── postprocess/        # SHAP analysis and decision curve scripts
│   └── unused/             # Deprecated or experimental code
│
├── src/                    # Importable library code (`final_as` namespace)
│   ├── clinical_data_src/  # Tabular data helpers
│   ├── core/               # Dataset and evaluation utilities
│   ├── models/             # Neural network and fusion definitions
│   ├── training/           # Training loops
│   ├── inference/          # Inference and prediction
│   ├── preprocessing/      # Splits and feature processing
│   ├── feature_extraction/ # MRI embedding modules
│   ├── analysis/           # MRI analytics tools
│   ├── evaluation/         # Metrics, bootstrap CI, calibration
│   └── utils/              # Generic helpers
│
├── checkpoints/ 🔒         # Saved model weights (.pth), git-ignored
├── results/                # Generated outputs (metrics, figures)
│   ├── clinical/           # Clinical model outputs and plots
│   └── mri/                # MRI embeddings and visualizations
│
├── requirements.txt        # Python ≥3.10 dependencies
├── LICENSE                 # MIT license
└── README.md               # This file
```

---

## Environment Setup

Clone the repository and create the conda environment:

```bash
git clone <repo_url>
cd FINAL_AS
conda create -n final_as python=3.10 -y
conda activate final_as
pip install -r requirements.txt
# Install appropriate PyTorch CUDA build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Data Organization

Place your raw data under `data/` (excluded by `.gitignore`):

* **MRI**: `data/mri_AS/<subject>/*.dcm or .png`, `data/mri_health/<subject>/*.dcm or .png`
* **Clinical**: `data/raw_lab_data/Raw_Lab_Dataset.csv` (27 feature columns + `Disease` label)

Ensure file naming matches script expectations.

---

## Clinical Pipeline

### Preprocessing

```bash
python scripts/clinical/preprocess_clinical_final.py \
    data/raw_lab_data/Raw_Lab_Dataset.csv \
    results/clinical/processed_splits
```

* Cleans column names, encodes labels, stratified 5-fold splits
* Imputes missing values, scales numeric, one-hot encodes categoricals
* Balances classes with SMOTE
* Outputs `fold_{i}_train.csv` and `fold_{i}_val.csv`

### Training and Calibration

```bash
python src/training/train.py \
    --folds results/clinical/processed_splits \
    --output_dir results/clinical/model_outputs
```

* Trains `ClinicalNet` (2-layer MLP) via cross-validation
* Applies temperature scaling on validation logits
* Saves weights to `checkpoints/clinicalnet_fold{i}.pth`
* Exports calibration curves and SHAP summaries

### Evaluation

```bash
python scripts/clinical/calculate_3_models_final_stats.py
```

* Aggregates predictions from multiple models (XGBoost, LightGBM, ClinicalNet)
* Bootstraps AUROC, AUPRC, sensitivity, specificity (95% CI)
* Performs DeLong test for AUC comparisons
* Generates ROC, PR, calibration, DCA, and confusion matrix plots

---

## MRI Pipeline

### DICOM to PNG Conversion

```bash
python scripts/mri/conversion/mri_convert_dicom_to_png.py \
    --input data/mri_AS \
    --output data/mri_images_png
```

### Preprocessing and Slice Selection

```bash
python scripts/mri/preprocessing/filter_slices.py \
    --input_dir data/mri_images_png \
    --output_dir data/mri_selected_slices
```

* Applies bias-field correction (ANTs), ROI extraction, quality filter

### Embedding Extraction and Analysis

```bash
python scripts/mri/analysis/mri_eval_auc_bootstrap.py \
    --png_dir data/mri_selected_slices \
    --output_dir results/mri
```

* Extracts 512-D embeddings from frozen ResNet‑18 encoder
* Trains logistic probe on slice-level embeddings
* Bootstraps AUC CI, permutation tests
* Creates t-SNE and KDE visualization plots

### Grad-CAM Visualization

```bash
python scripts/mri/gradcam/As_run_sij_gradcam_analysis.py \
    --png_dir data/mri_selected_slices \
    --output_dir results/mri/gradcam
```

* Generates Grad‑CAM heatmaps highlighting sacroiliac joints

---

## Postprocessing and Interpretability

* **SHAP Summaries**: `scripts/postprocess/shap_compute_summary.py`
* **Decision Curve Analysis**: `scripts/postprocess/shap_compute_dca.py`
* Plot utilities in `src/analysis/`

---

## Late Fusion

```bash
python src/training/train_late_fusion.py \
    --clinical_preds results/clinical/predictions.csv \
    --mri_preds results/mri/embeddings_probs.csv \
    --output results/fusion/fusion_predictions.csv
```

* Supports probability averaging and meta-learner stacking

---

## Results

**Clinical Pipeline**

* **Baseline Cohort (n=4,254)**: AS vs. control groups balanced by age, sex, and key lab indices. 95% CI for demographic differences reported in Table 1.
* **Model Performance**: ClinicalNet achieved slice-level AUROC of 0.93 (95% CI 0.92–0.94) and AUPRC of 0.89 (95% CI 0.88–0.90). Post-calibration ECE was 0.021.
* **Clinical Utility**: Decision Curve Analysis demonstrated net benefit across threshold probabilities from 0.2 to 0.8, outperforming treat-all and treat-none strategies.

**MRI Pipeline**

* **Embedding Separation**: Cosine distance distributions between AS and healthy embeddings were significantly different (Permutation p < 0.001), with t-SNE visualizations showing two distinct clusters.
* **Slice-Level Diagnostics**: Logistic probe yielded AUROC of 0.87 (95% CI 0.82–0.91). Bootstrap 95% CI computed over 1,000 resamples.
* **Subject-Level Performance**: Aggregating slice probabilities per subject improved AUROC to 0.91 (95% CI 0.88–0.94), with ECE reduced from 0.15 to 0.10 after temperature scaling.
* **Interpretability**: Grad‑CAM heatmaps highlighted sacroiliac joint regions consistent with clinical annotations (see `results/mri/gradcam/`).

---

## Results Inspection

```bash
open results/clinical/data_results/sci_roc_curve.png
open results/mri/embedding_viz/tsne_slice_level.png
open results/mri/gradcam/as/_slice03_gradcam.png
```

---

## Troubleshooting and Tips

| Issue                    | Solution                              |
| ------------------------ | ------------------------------------- |
| CUDA OOM                 | Reduce `--batch_size` in scripts      |
| Missing ImageNet weights | Run `python -m torch.hub`             |
| Inconsistent results     | Use `--seed 42` for all scripts       |
| Missing prediction files | Check paths under `results/clinical/` |

---

## Contributing

1. Fork this repository
2. Create branch: `feature/your_feature`
3. Follow code style: `black . && isort .`
4. Submit PR with description and example

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
