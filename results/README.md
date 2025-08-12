# DDI-AS Framework Results

This directory contains all the results from the DDI-AS (Dual Diagnostic Intelligence for Ankylosing Spondylitis) framework analysis.

## Directory Structure

### `/performance/`
- `clinical_performance_summary.csv` - Clinical model performance metrics
- `mri_performance_summary.csv` - MRI model performance metrics  
- `ensemble_performance_summary.csv` - Ensemble model performance metrics
- `statistical_tests.csv` - Statistical test results and significance

### `/clinical/`
- `feature_importance_summary.csv` - Feature importance rankings
- `cross_validation_results.csv` - 5-fold CV results
- `dataset_characteristics.csv` - Clinical dataset demographics

### `/mri/`
- `mri_validation_results.csv` - Leave-Two-Out CV results
- `mri_dataset_info.csv` - MRI subject demographics

### `/ensemble/`
- `ensemble_comparison.csv` - Model comparison metrics
- `clinical_utility_analysis.csv` - Decision curve analysis results

### `/models/`
- `model_info.txt` - Model architecture and training details

### `/figures/`
- All publication-quality figures and visualizations

## Key Results Summary

### Clinical Model (ClinicalNet)
- **Best Model**: Gradient Boosting
- **AUROC**: 0.938 ± 0.003
- **Calibration**: ECE = 0.155
- **Top Features**: HLA-B27 (0.231), ESR (0.203), CRP (0.167)

### MRI Model (ImagingNet)  
- **Architecture**: ResNet-18 + Logistic Regression
- **AUROC**: 0.833 ± 0.021 (p=0.017)
- **Validation**: Leave-Two-Out CV (12 folds)
- **Sample Size**: 8 subjects (6 AS, 2 HC)

### Ensemble Model
- **Fusion Method**: Late fusion (simple averaging)
- **AUROC**: 0.941 (95% CI: 0.924-0.959)
- **Improvement**: ΔAUROC = 0.003 over ClinicalNet
- **Calibration**: ECE = 0.168 (95% CI: 0.154-0.188)

## Data Sources
- **Clinical**: 4,254 balanced records from Mahdi et al. (2025)
- **MRI**: 8 subjects from Radiopaedia.org public repository
- **Features**: 20 engineered features from 14 original variables

## Validation Methods
- **Clinical**: Stratified 5-fold CV + hold-out test set
- **MRI**: Leave-Two-Out CV + permutation testing
- **Ensemble**: Cross-validation stability assessment

## Clinical Impact
- **Optimal Threshold**: 0.62 (calibrated by Youden's index / ROC proximity)
- **Net Benefit**: Consistent improvement over Treat-All baseline across 0.4–0.7
- **Potential Impact**: Up to 18–24 months earlier diagnosis based on referral triage improvements