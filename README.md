# AI-Based Prediction of Treatment Response in Ankylosing Spondylitis

This project examines the application of artificial intelligence to predict treatment response in patients with Ankylosing Spondylitis (AS). Due to current data constraints, separate models are developed for MRI imaging and clinical indicators; multimodal integration is planned once matched data are available.

## Project Objectives

1. Develop a deep-learning model based on MRI scans to predict AS diagnosis or treatment response.  
2. Construct a machine-learning model using clinical features (e.g. CRP, ESR, BASDAI scores).  
3. Design a multimodal fusion strategy (late fusion or transformer-based), contingent on acquisition of paired MRI + clinical data.

## Data Description

- **MRI data**  
  - Format: NIfTI (`.nii` / `.nii.gz`)  
  - Labels: Binary (AS vs. non-AS)  
  - Status: Ready for preprocessing and training  

- **Clinical data**  
  - Format: CSV  
  - Features: CRP, ESR, BASDAI, etc.  
  - Limitation: Not currently matched to MRI on a per-patient basis  

- **HLA-B27 status**  
  - Variable excluded due to missing values in most cases  

## Methodology

1. **MRI Model**  
   - Architecture: ResNet-50 on 2D slices (or 3D CNN if volumetric data are used)  
   - Input: Preprocessed MRI images  
   - Output: Binary classification (AS vs. non-AS)  

2. **Clinical Model**  
   - Algorithm: XGBoost  
   - Input: Selected clinical indicators  
   - Output: Predicted treatment response or diagnostic label  

3. **Multimodal Fusion (Future)**  
   - Strategy: Late fusion of MRI and clinical model outputs via ensemble voting or transformer encoder  
   - Requirement: Fully paired MRI and clinical datasets  

## Repository Structure

- **mri_data/**  
  - `patient001.nii.gz`, `patient002.nii.gz`, …  
- **clinical_data.csv**  
- **models/**  
  - `train_mri.py`  
  - `train_clinical.py`  
- **results/**  
- **README.md**  
- **requirements.txt**  

## Current Status

- MRI data have been preprocessed.  
- Baseline MRI classification model implemented.  
- Clinical data cleaning is in progress.  
- Fusion strategy design is pending matched data.

## Known Limitations

- MRI and clinical datasets are from different cohorts and cannot be paired at present.  
- Exclusion of HLA-B27 may affect model performance.  
- Limited sample size may restrict generalizability.

## Future Work

- Secure a dataset with patient-matched MRI and clinical records.  
- Enhance interpretability with SHAP or saliency maps.  
- Evaluate performance using decision-curve analysis.  
- Prototype a clinical decision-support interface.

---

**Author**  
Zoya Huo  
MSc Global Healthcare Management, University College London  

**Supervisor**  
Dr. Mohammed Mirza  
