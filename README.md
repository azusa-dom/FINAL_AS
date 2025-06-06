## Multimodal AS Diagnosis Pipeline

This project aims to implement a reproducible multimodal deep learning pipeline combining sacroiliac joint MRI and clinical features for early detection of axial spondyloarthritis (axSpA), including ankylosing spondylitis (AS). Features include:

* Patient-level stratified split (70%/15%/15%) to avoid data leakage
* Fine-tuning ResNet-50 backbone on MRI data
* Training clinical branch using fully connected neural networks
* Training early-fusion (Transformer) and late-fusion (XGBoost) models
* 5-fold stratified cross-validation + 10 random hyperparameter searches per fold
* Test set evaluation with 200 bootstraps to calculate 95% CI of AUROC, AUPRC, and Brier score
* SHAP-based interpretability analysis (global and local)
* Decision Curve Analysis (DCA) to assess net clinical benefit

## Directory Structure

```bash
.
├── README.md                     # This document
├── run_all.sh                   # One-click script for the entire pipeline
├── data_splits/                 # Patient-level train/val/test CSVs
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── data/                        # Raw and processed data
│   ├── raw/                     # Original clinical CSVs, MRI images
│   └── processed/              # Output from preprocess.py
│       ├── clinical.csv
│       └── mri_images/         # Resampled and normalized MRI images
├── src/                         # Core source code
│   ├── preprocess.py
│   ├── split_data.py
│   ├── train_mri.py
│   ├── train_clin.py
│   ├── train_fusion.py
│   ├── evaluate.py
│   ├── datasets/
│   │   ├── mri_dataset.py
│   │   ├── clin_dataset.py
│   │   └── multisource_dataset.py
│   ├── models/
│   │   ├── mri_model.py
│   │   ├── clin_model.py
│   │   └── fusion_model.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── shap_utils.py
│   │   └── dca_utils.py
│   └── config.py
├── checkpoints/                # Model weights
│   ├── mri_model/
│   ├── clin_model/
│   └── xgb_model/
└── results/                    # Evaluation results and visualizations
    ├── roc_curves/
    ├── pr_curves/
    ├── calib_curves/
    ├── shap_summary.png
    ├── shap_bar.png
    └── dca_curve.png
```

## Environment & Dependencies

1. **Python Version**: >=3.8
2. **Recommended Setup**:

   * Use `conda` or `virtualenv` to create a new environment
   * Example `requirements.txt`:

     ```txt
     torch>=1.10.0
     torchvision>=0.11.0
     scikit-learn>=1.0.0
     xgboost>=1.5.0
     pandas>=1.3.0
     numpy>=1.19.0
     shap>=0.40.0
     matplotlib>=3.4.0
     seaborn>=0.11.0
     tqdm>=4.60.0
     pydicom>=2.2.0
     ```
   * Installation:

     ```bash
     pip install -r requirements.txt
     ```

## Data Preparation

1. **Raw Clinical Data** (`data/raw/clinical.csv`):

   * Includes `patient_id`, binary label (0/1), and clinical indicators (e.g., CRP, HLA-B27, BASDAI).
   * Each row represents a single MRI-associated record or one patient linking to multiple MRIs.

2. **Raw MRI Images** (`data/raw/mri_images/`):

   * All sacroiliac MRIs, ideally single-channel NIfTI/DICOM/PNG.
   * Filenames or folders should match `patient_id` in the clinical CSV.

3. **Optional: Pretraining Dataset**:

   * For MRI backbone pretraining on fastMRI / SPIDER datasets. Specify path in `pretrain_mri_backbone.py`.

## Usage

Example for Linux/MacOS (adapt for Windows as needed):

### 1. Run All Scripts

```bash
bash run_all.sh
```

This will:

1. Create stratified train/val/test splits in `data_splits/`
2. (Optional) Pretrain ResNet-50 on public MRI data
3. Fine-tune ResNet-50 on sacroiliac MRI
4. Train clinical MLP on structured data
5. Train XGBoost using out-of-fold MRI & clinical predictions
6. Evaluate models on test set and save all visualizations

You can also run any step individually.

### 2. Script Argument Examples

See documentation inside each script or below for detailed CLI usage.

## Results Format

Metrics (printed and saved in `results/metrics.txt`):

```
[MRI Only] AUROC: 0.78, AUPRC: 0.70, Brier: 0.049
[Clinical Only] AUROC: 0.82, AUPRC: 0.75, Brier: 0.036
[Early Fusion] AUROC: 0.89, AUPRC: 0.85, Brier: 0.021
[Late Fusion] AUROC: 0.86, AUPRC: 0.80, Brier: 0.034
```

SHAP and DCA visualizations will be saved in the `results/` directory.

## References

* Ai, F. et al. (2012). *Rheumatology International*, 32(12), 4009–4015.
* Bennani, S. et al. (2025). *medRxiv*.
* Jamaludin, A. et al. (2017). *Medical Image Analysis*, 40, 67–77.
* Hosny, A. et al. (2018). *Nature Reviews Cancer*, 18, 500–510.
* Li, H. et al. (2023). *Frontiers in Public Health*, 11, 1063633.
* Tas, N.P. et al. (2023). *Biomedicines*, 11(9), 2441.
* Tas, S. et al. (2024). *Biomedicines*, 12(1), 200.

## Contact

For questions or feedback:

* Email: [your\_email@example.com](mailto:your_email@example.com)
* GitHub: [https://github.com/azusa-dom/MRI-AS/issues](https://github.com/azusa-dom/MRI-AS/issues)
