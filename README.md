Multimodal Learning Pipeline for Early Diagnosis of Ankylosing Spondylitis (AS)
This project provides a reproducible machine learning pipeline for the early diagnosis of Axial Spondyloarthritis (AxSpA), including Ankylosing Spondylitis. The pipeline features separate, unimodal diagnostic models for sacroiliac joint MRI and clinical data, respectively. It further implements a Late Fusion model that integrates outputs from both branches to improve diagnostic performance.

Core Features
Unimodal Diagnostic Branches:
MRI Branch: A 3D Convolutional Neural Network based on a ResNet-50 backbone, fine-tuned to identify pathological features from MRI scans.
Clinical Branch: A a Fully-Connected Neural Network (FCNN) trained on structured clinical data.
Implemented Fusion Strategy:
Late Fusion: An XGBoost model that combines the prediction probability from the MRI branch with structured clinical features to yield a final, integrated diagnosis.
Interpretability and Clinical Utility Analysis:
SHAP (SHapley Additive exPlanations): Used to analyze the contribution of each clinical feature to the model's predictions.
Decision Curve Analysis (DCA): Implemented to assess the net benefit and clinical utility of the models.
Reproducible Workflow:
The entire pipeline, from data preprocessing and splitting to model training and evaluation, is automated via the scripts/runall.sh script.
Pipeline Overview
Data Preprocessing and Splitting

scripts/preprocess_clinical_as.py: Cleans the raw clinical data (e.g., handles missing values).
scripts/build_balanced_dataset.py: Performs a patient-level stratified split of the data into training (70%), validation (15%), and test (15%) sets to prevent data leakage.
MRI Branch (src/train_mri.py)

Fine-tunes a ResNet-50 backbone pre-trained on ImageNet.
Employs a 5-fold stratified cross-validation strategy for robust training.
Uses a fixed set of hyperparameters (e.g., lr=1e-4); does not include a hyperparameter search.
Clinical Branch (src/train.py)

Trains a multi-layer Fully-Connected Neural Network (FCNN).
Follows the same 5-fold cross-validation strategy as the MRI branch.
Late Fusion (src/train_late_fusion.py)

Uses the prediction probability from the trained MRI model as a new, high-level feature.
Concatenates this single "imaging feature" with the original clinical features.
Trains an XGBoost classifier on this augmented feature set to produce the final fused prediction.
Evaluation and Analysis (src/evaluate.py, scripts/plot_shap_dca.py)

Evaluates the performance of all models (MRI-only, Clinical-only, Late Fusion) on the hold-out test set.
Calculates 95% confidence intervals for metrics like AUROC and AUPRC using 200 bootstrap iterations.
Generates and saves SHAP feature importance plots and Decision Curves.
Directory Structure
FINAL_AS/
├── README.md
├── requirements.txt
├── environment.yml             # Conda environment definition file
├── data/
│   ├── rheumatic_autoimmune_disease.csv # Raw clinical data CSV
│   └── (User must provide MRI NIfTI files separately)
├── models/                     # Stores trained model weights and predictions
│   ├── mri_model/
│   ├── clinical_model/
│   └── late_fusion_model/
├── results/                    # Stores evaluation metrics and plots
│   ├── metrics.txt
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── shap_summary.png
│   └── dca_curve.png
├── scripts/
│   ├── runall.sh               # One-click script to run the full pipeline
│   ├── preprocess_clinical_as.py
│   ├── build_balanced_dataset.py
│   └── plot_shap_dca.py
└── src/
    ├── dataset.py              # PyTorch Dataset definitions
    ├── models.py               # Model architectures (ResNet3D, FCNN)
    ├── train.py                # Script to train the clinical model
    ├── train_mri.py            # Script to train the MRI model
    ├── train_late_fusion.py    # Script to train the late fusion XGBoost model
    ├── evaluate.py             # Script to evaluate model performance
    └── utils.py                # Utility functions
Environment Setup
It is recommended to use Conda to create an isolated Python environment.

Bash

# 1. Create and activate the Conda environment from the .yml file
conda env create -f environment.yml
conda activate axspa_env

# 2. (Alternative) If not using Conda, install dependencies via pip
pip install -r requirements.txt
Key Dependencies: torch, xgboost, pandas, scikit-learn, numpy, shap, matplotlib. See requirements.txt for the full list.

Data Preparation
Clinical Data: Place your raw clinical data file, named rheumatic_autoimmune_disease.csv, in the data/ directory. This file must contain a patient_id column, a label column (0/1), and other clinical features.
MRI Data: This pipeline expects MRI scans in NIfTI (.nii or .nii.gz) format.
Important Note: Upstream preprocessing steps, such as DICOM-to-NIfTI conversion and N4 bias field correction, must be performed offline before running this pipeline. The repository does not include integrated scripts for these initial steps.
Place the preprocessed NIfTI files in a directory of your choice and ensure the path is correctly configured in src/train_mri.py.
Usage
One-Click Execution (Recommended)
The runall.sh script automates the entire experimental workflow in the correct sequence.

Bash

bash scripts/runall.sh
This script will sequentially execute data preprocessing, data splitting, MRI model training, clinical model training, late fusion model training, and final evaluation on the test set.

Step-by-Step Execution
You can also run each step of the pipeline manually, which is useful for debugging.

Bash

# 1. Preprocess clinical data and split the dataset
python scripts/preprocess_clinical_as.py
python scripts/build_balanced_dataset.py

# 2. Train the unimodal models (using 5-fold CV)
python src/train_mri.py
python src/train.py

# 3. Train the late fusion model
python src/train_late_fusion.py

# 4. Run final evaluation on the test set and generate plots
python src/evaluate.py
python scripts/plot_shap_dca.py
Expected Output
Evaluation metrics will be saved to results/metrics.txt, and visualization plots will be saved in the results/ directory. An example metrics.txt format is shown below:

[MRI-only]      AUROC: 0.78 (95% CI 0.72–0.84)
[Clinical-only] AUROC: 0.82 (95% CI 0.77–0.87)
[Late-fusion]   AUROC: 0.86 (95% CI 0.81–0.90)
(Note: These values are for illustration purposes only.)

Known Limitations
Incomplete MRI Preprocessing Integration: As noted, key upstream preprocessing steps (DICOM conversion, N4 correction) are not integrated into the main pipeline and must be run offline.
Single-Center Data: The models were developed using data from a single institution, and their generalizability to external, multi-center datasets has not yet been validated.
No Domain-Specific Pre-training: The MRI model was fine-tuned from ImageNet weights without an intermediate pre-training step on a large medical imaging dataset.
Future Work
Implement Early Fusion Models: Develop and integrate an early fusion strategy (e.g., using a Transformer architecture) to compare against the current late fusion model.
Enhance MRI Interpretability: Implement Grad-CAM visualizations to create saliency maps that highlight the regions of the MRI the model focuses on.
Multi-Center Validation: Validate the performance and robustness of the models on external datasets from different hospitals and scanners.
Develop a Decision-Support Prototype: Build a web-based Clinical Decision Support System (CDSS) prototype for clinical trial and feedback.
