
# Multimodal AS Diagnosis Pipeline

This project implements a reproducible deep-learning pipeline for early detection and treatment-response prediction in axial spondyloarthritis (axSpA), including ankylosing spondylitis (AS). Models are developed separately for sacroiliac‐joint MRI and clinical features; multimodal fusion is planned once matched data are available.

## Pipeline Overview

1. **Data splitting**  
   - Patient-level stratified split (70% train / 15% validation / 15% test)  
   - Ensures no data leakage  

2. **MRI branch**  
   - Fine-tune ResNet-50 backbone on preprocessed MRI slices (or volumes)  
   - 5-fold stratified cross-validation  
   - 10 random hyperparameter trials per fold  
   - Test-set evaluation with 200 bootstraps to compute 95% CI for AUROC, AUPRC, Brier score  

3. **Clinical branch**  
   - Fully connected neural network (FCNN) on structured features (e.g. CRP, ESR, BASDAI)  
   - Same CV and bootstrap evaluation strategy as MRI branch  

4. **Fusion strategies (future work)**  
   - **Early fusion**: transformer encoder on combined embeddings  
   - **Late fusion**: XGBoost on concatenated MRI + clinical predictions  
   - Fusion scripts are included but require patient-matched MRI & clinical data  

5. **Interpretability & clinical utility**  
   - SHAP (global and local explanations)  
   - Decision Curve Analysis (DCA) for net benefit assessment  

## Directory Structure


FINAL_AS/
├── README.md
├── runall.sh                   # One-click pipeline driver (see “Usage”)
├── data_splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── data/
│   ├── raw/
│   │   ├── clinical.csv
│   │   └── mri_images/          # NIfTI: patient001.nii.gz, …
│   └── processed/
│       ├── clinical_clean.csv
│       └── mri_preprocessed/    # Resampled, normalized volumes
├── src/
│   ├── preprocess.py
│   ├── split_data.py
│   ├── train_mri.py
│   ├── train_clinical.py
│   ├── train_fusion.py         # Requires matched data
│   ├── evaluate.py
│   ├── datasets/
│   │   ├── mri_dataset.py
│   │   ├── clin_dataset.py
│   │   └── fusion_dataset.py    # Expects paired records
│   ├── models/
│   │   ├── mri_model.py
│   │   ├── clin_model.py
│   │   └── fusion_model.py
│   └── utils/
│       ├── metrics.py
│       ├── shap_utils.py
│       └── dca_utils.py
├── checkpoints/
│   ├── mri_model/
│   ├── clin_model/
│   └── fusion_model/
└── results/
    ├── metrics.txt
    ├── roc_curves/
    ├── pr_curves/
    ├── shap_summary.png
    └── dca_curve.png
````

## Environment & Dependencies

* **Python** ≥ 3.8
* Create a fresh virtual environment (conda or venv) and install:

```txt
torch>=1.10.0
torchvision>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
pandas>=1.3.0
numpy>=1.19.0
shap>=0.40.0
matplotlib>=3.4.0
tqdm>=4.60.0
pydicom>=2.2.0
```

```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Raw clinical data** (`data/raw/clinical.csv`):

   * Columns: `patient_id`, `label` (0/1), CRP, ESR, BASDAI, …
   * Note: HLA-B27 is excluded due to missing values.

2. **Raw MRI images** (`data/raw/mri_images/`):

   * Format: NIfTI (`.nii` / `.nii.gz`)
   * Filenames must encode `patient_id` for future pairing.

3. **Processing scripts**:

   * `preprocess.py` handles normalization, resampling.
   * `split_data.py` generates `data_splits/*.csv`.

## Usage

### 1. One-click execution

```bash
bash runall.sh
```

This sequentially:

1. Splits data (train/val/test).
2. Preprocesses MRI volumes.
3. Fine-tunes ResNet-50 on MRI branch.
4. Trains clinical FCNN on structured data.
5. (Optional) Runs fusion model if matched data are present.
6. Evaluates all models and saves metrics & visualizations.

### 2. Individual steps

```bash
python src/preprocess.py
python src/split_data.py
python src/train_mri.py   --config configs/mri.yaml
python src/train_clinical.py --config configs/clin.yaml
python src/train_fusion.py  --config configs/fusion.yaml  # only if paired data
python src/evaluate.py      --splits test
```

## Results Format

Metrics are saved in `results/metrics.txt` and plots under `results/`. Example:

```
[MRI-only]     AUROC: 0.78 (95% CI 0.72–0.84)
[Clinical-only] AUROC: 0.82 (95% CI 0.77–0.87)
[Early-fusion] AUROC: 0.89 (95% CI 0.85–0.93)
[Late-fusion]  AUROC: 0.86 (95% CI 0.81–0.90)
```

## Limitations

* MRI and clinical datasets are not yet patient-matched; fusion is not currently executable.
* HLA-B27 status excluded due to missingness.
* Sample size may limit generalisability.

## Future Work

* Acquire matched MRI + clinical cohort for full multimodal fusion.
* Incorporate SHAP saliency maps for spatial interpretability.
* Perform Decision Curve Analysis on prospective data.
* Develop a web-based clinical decision-support prototype.

## References

1. Jamaludin A, et al. *Medical Image Analysis*, 2017;40:67–77.
2. Ai F, et al. *Rheumatology International*, 2012;32(12):4009–4015.
3. Bennani S, et al. medRxiv, 2025.
4. Hosny A, et al. *Nature Reviews Cancer*, 2018;18:500–510.
5. Li H, et al. *Frontiers in Public Health*, 2023;11:1063633.


```
