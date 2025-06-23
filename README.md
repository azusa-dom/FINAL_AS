# Multimodal Learning Pipeline for Early Diagnosis of Ankylosing Spondylitis (AS)

This project provides a reproducible machine learning pipeline for the early diagnosis of Axial Spondyloarthritis (AxSpA), including Ankylosing Spondylitis. The pipeline features separate, unimodal diagnostic models for sacroiliac joint MRI and clinical data, respectively. It further implements a **Late Fusion** model that integrates outputs from both branches to improve diagnostic performance.

## Core Features

  * **Unimodal Diagnostic Branches:**
      * **MRI Branch:** A 3D Convolutional Neural Network based on a ResNet-50 backbone, fine-tuned to identify pathological features from MRI scans.
      * **Clinical Branch:** A a Fully-Connected Neural Network (FCNN) trained on structured clinical data.
  * **Implemented Fusion Strategy:**
      * **Late Fusion:** An **XGBoost** model that combines the prediction probability from the MRI branch with structured clinical features to yield a final, integrated diagnosis.
  * **Interpretability and Clinical Utility Analysis:**
      * **SHAP (SHapley Additive exPlanations):** Used to analyze the contribution of each clinical feature to the model's predictions.
      * **Decision Curve Analysis (DCA):** Implemented to assess the net benefit and clinical utility of the models.
  * **Reproducible Workflow:**
      * The entire pipeline, from data preprocessing and splitting to model training and evaluation, is automated via the `scripts/runall.sh` script.

## Pipeline Overview

1.  **Data Preprocessing and Splitting**

      * `scripts/preprocess_clinical_as.py`: Cleans the raw clinical data (e.g., handles missing values).
      * `scripts/build_balanced_dataset.py`: Performs a patient-level stratified split of the data into training (70%), validation (15%), and test (15%) sets to prevent data leakage.

2.  **MRI Branch (`src/train_mri.py`)**

      * Fine-tunes a ResNet-50 backbone pre-trained on ImageNet.
      * Employs a 5-fold stratified cross-validation strategy for robust training.
      * Uses a fixed set of hyperparameters (e.g., `lr=1e-4`); **does not include** a hyperparameter search.

3.  **Clinical Branch (`src/train.py`)**

      * Trains a multi-layer Fully-Connected Neural Network (FCNN).
      * Follows the same 5-fold cross-validation strategy as the MRI branch.

4.  **Late Fusion (`src/train_late_fusion.py`)**

      * Uses the prediction probability from the trained MRI model as a new, high-level feature.
      * Concatenates this single "imaging feature" with the original clinical features.
      * Trains an XGBoost classifier on this augmented feature set to produce the final fused prediction.

5.  **Evaluation and Analysis (`src/evaluate.py`, `scripts/plot_shap_dca.py`)**

      * Evaluates the performance of all models (MRI-only, Clinical-only, Late Fusion) on the hold-out test set.
      * Calculates 95% confidence intervals for metrics like AUROC and AUPRC using 200 bootstrap iterations.
      * Generates and saves SHAP feature importance plots and Decision Curves.

## Directory Structure

```
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
```

## Environment Setup

It is recommended to use Conda to create an isolated Python environment.

```bash
# 1. Create and activate the Conda environment from the .yml file
conda env create -f environment.yml
conda activate axspa_env

# 2. (Alternative) If not using Conda, install dependencies via pip
pip install -r requirements.txt
```

**Key Dependencies:** `torch`, `xgboost`, `pandas`, `scikit-learn`, `numpy`, `shap`, `matplotlib`. See `requirements.txt` for the full list.

## Data Preparation

1.  **Clinical Data**: Place your raw clinical data file, named `rheumatic_autoimmune_disease.csv`, in the `data/` directory. This file must contain a `patient_id` column, a `label` column (0/1), and other clinical features.
2.  **MRI Data**: This pipeline expects MRI scans in **NIfTI (`.nii` or `.nii.gz`)** format.
      * **Important Note**: Upstream preprocessing steps, such as DICOM-to-NIfTI conversion and N4 bias field correction, must be performed offline **before** running this pipeline. The repository does not include integrated scripts for these initial steps.
      * Place the preprocessed NIfTI files in a directory of your choice and ensure the path is correctly configured in `src/train_mri.py`.

## Usage

### One-Click Execution (Recommended)

The `runall.sh` script automates the entire experimental workflow in the correct sequence.

```bash
bash scripts/runall.sh
```

This script will sequentially execute data preprocessing, data splitting, MRI model training, clinical model training, late fusion model training, and final evaluation on the test set.

### Step-by-Step Execution

You can also run each step of the pipeline manually, which is useful for debugging.

```bash
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
```

## Expected Output

Evaluation metrics will be saved to `results/metrics.txt`, and visualization plots will be saved in the `results/` directory. An example `metrics.txt` format is shown below:

```
[MRI-only]      AUROC: 0.78 (95% CI 0.72–0.84)
[Clinical-only] AUROC: 0.82 (95% CI 0.77–0.87)
[Late-fusion]   AUROC: 0.86 (95% CI 0.81–0.90)
```

*(Note: These values are for illustration purposes only.)*

## Known Limitations

  * **Incomplete MRI Preprocessing Integration**: As noted, key upstream preprocessing steps (DICOM conversion, N4 correction) are not integrated into the main pipeline and must be run offline.
  * **Single-Center Data**: The models were developed using data from a single institution, and their generalizability to external, multi-center datasets has not yet been validated.
  * **No Domain-Specific Pre-training**: The MRI model was fine-tuned from ImageNet weights without an intermediate pre-training step on a large medical imaging dataset.

## Future Work

  * **Implement Early Fusion Models**: Develop and integrate an early fusion strategy (e.g., using a Transformer architecture) to compare against the current late fusion model.
  * **Enhance MRI Interpretability**: Implement Grad-CAM visualizations to create saliency maps that highlight the regions of the MRI the model focuses on.
  * **Multi-Center Validation**: Validate the performance and robustness of the models on external datasets from different hospitals and scanners.
  * **Develop a Decision-Support Prototype**: Build a web-based Clinical Decision Support System (CDSS) prototype for clinical trial and feedback.

# REFERENCES

- Ai, F., Zhang, W., Liu, H., Song, W., Wu, H., Han, Y., et al. (2012) Value of diffusion-weighted quantification for MRI assessment of sacroiliac joints in early diagnosis of ankylosing spondylitis. *Rheumatology International*, **32**(12), pp.4009–4015. [https://doi.org/10.1007/s00296-011-2253-0](https://doi.org/10.1007/s00296-011-2253-0)

- Bennani, S., Ohayon, S., Laleye, F., Bauvin, P., Messas, E., et al. (2025) Is multimodal better? A systematic review of multimodal versus unimodal machine learning in clinical decision-making. *medRxiv [Preprint].* [https://doi.org/10.1101/2025.03.12.25322656](https://doi.org/10.1101/2025.03.12.25322656)

- Bradbury, L.A., Hollis, K.A., Gazer, B., Gollow, I., Shankar, A., Cope, N., et al. (2018) Diffusion-weighted imaging as a sensitive and specific MRI sequence in the diagnosis of chronic nonbacterial osteomyelitis of the sacroiliac joints in children. *The Journal of Rheumatology*, **45**(5), pp.690–697. [https://doi.org/10.3899/jrheum.170871](https://doi.org/10.3899/jrheum.170871)

- Dubey, S., Chan, A., Adebajo, A.O., Walker, D. and Treglia, G. (2024) Artificial intelligence and machine learning in rheumatology: A systematic literature review. *Rheumatology*, **63**(8), pp.2040–2053. [https://doi.org/10.1093/rheumatology/kead190](https://doi.org/10.1093/rheumatology/kead190)

- Hosny, A., Parmar, C., Quackenbush, J., Schwartz, L.H. and Aerts, H.J.W.L. (2018) Artificial intelligence in radiology. *Nature Reviews Cancer*, **18**, pp.500–510. [https://doi.org/10.1038/s41568-018-0016-5](https://doi.org/10.1038/s41568-018-0016-5)

- Jamaludin, A., Kadir, T. and Zisserman, A. (2017) Automated analysis of spinal MRI using deep learning. *Medical Image Analysis*, **40**, pp.67–77. [https://doi.org/10.1016/j.media.2017.06.003](https://doi.org/10.1016/j.media.2017.06.003)

- Li, H., Zhou, Y., Zhang, Q., Tao, X., Liang, T., Jiang, J., et al. (2023) A multicentre artificial intelligence tool for ankylosing spondylitis supervised by human experts. *Frontiers in Public Health*, **11**, 1063633. [https://doi.org/10.3389/fpubh.2023.1063633](https://doi.org/10.3389/fpubh.2023.1063633)

- Liao, W., Matsumoto, T., Tanaka, M., Kakehi, T., Nakajima, K., Imagawa, T., et al. (2021) Machine learning in rheumatoid arthritis: applications and challenges. *Modern Rheumatology*, **31**(1), pp.48–55. [https://doi.org/10.1080/14397595.2020.1766343](https://doi.org/10.1080/14397595.2020.1766343)

- Liu, H., Yang, C., Zhao, M., Ni, L., Chen, R., Zheng, Z., et al. (2020) IgG galactosylation status combined with MYOM2-rs2294066 precisely predicts anti-TNF response in ankylosing spondylitis. *Frontiers in Immunology*, **11**, 600019. [https://doi.org/10.3389/fimmu.2020.600019](https://doi.org/10.3389/fimmu.2020.600019)

- Maksymowych, W.P., Wichuk, S., Chiowchanwisawakit, P., Lambert, R.G.W. and Pedersen, S.J. (2023) Resolution of MRI inflammation and its association with long-term outcomes in patients with axial spondyloarthritis treated with etanercept. *RMD Open*, **9**(3), e003123. [https://doi.org/10.1136/rmdopen-2023-003123](https://doi.org/10.1136/rmdopen-2023-003123)

- Pons, M., Georgiadis, S., Hetland, M.L., et al. (2025) Predictors of secukinumab treatment response and continuation in axial spondyloarthritis: Results from the EuroSpA research collaboration network. *The Journal of Rheumatology [Epub ahead of print].* [https://doi.org/10.3899/jrheum.2024-0920](https://doi.org/10.3899/jrheum.2024-0920)

- Tas, N.P., Kaya, O., Macin, G., Tasci, B., Dogan, S. and Tuncer, T. (2023) ASNET: A novel AI framework for accurate ankylosing spondylitis diagnosis from MRI. *Biomedicines*, **11**(9), 2441. [https://doi.org/10.3390/biomedicines11092441](https://doi.org/10.3390/biomedicines11092441)

- Tas, S., Siemons, M., Yilmaz, E., Karabulut, E., Ozkan, E., Algin, O. and Cetin, P. (2024) Performance of different classification algorithms in differentiating sacroiliitis grades in patients with axial spondyloarthritis using an MRI-based radiomics model. *Biomedicines*, **12**(1), 200. [https://doi.org/10.3390/biomedicines12010200](https://doi.org/10.3390/biomedicines12010200)

- Tenório, A.P.M., Cunha, L.P., Almeida, D.A., Ferreira-Junior, J.R., Appenzeller, S. and Rittner, L. (2021) Radiomic diagnosis of sacroiliitis on MRI. *Physics in Medicine & Biology*, **66**(20), 205002. [https://doi.org/10.1088/1361-6560/ac2502](https://doi.org/10.1088/1361-6560/ac2502)

- Shenavarmasouleh, A., Wahab, H.A., Khaled, M., Sonawane, R., Henry, R. and Iyer, R.K. (2025) Algorithmic foundations for AI in imaging: Dataset design and benchmarking practices. *Data in Brief*, **50**, 109784. [https://doi.org/10.1016/j.dib.2024.109784](https://doi.org/10.1016/j.dib.2024.109784)

- van der Heijde, D., Landewé, R., Rudwaleit, M., et al. (2018) MRI inflammation at the vertebral unit level and clinical progression in patients with early axial spondyloarthritis: data from the DESIR cohort. *Rheumatology*, **57**(6), pp.1037–1044. [https://doi.org/10.1093/rheumatology/key021](https://doi.org/10.1093/rheumatology/key021)

- Venerito, V., Brusi, V., Spinelli, F.R., et al. (2023) Beyond the horizon: Innovations and future directions in axial spondyloarthritis. *Archives of Rheumatology*, **38**(4), pp.491–498. [https://doi.org/10.46497/ArchRheumatol.2023.9535](https://doi.org/10.46497/ArchRheumatol.2023.9535)

- Groza, A., Popescu, D., Ionescu, R., et al. (2021) Multimodal deep learning for clinical prognosis from medical imaging and electronic health records. *Scientific Reports*, **11**, 13594. [https://doi.org/10.1038/s41598-021-93010-0](https://doi.org/10.1038/s41598-021-93010-0)

- Lee, J., Laouar, Y., Tsoi, L.C. and Zhou, X. (2025) Community series in towards precision medicine for immune-mediated disorders: Advances in using big data and artificial intelligence to understand heterogeneity in disease pathogenesis. *Frontiers in Immunology*, **15**, 1553004. [https://doi.org/10.3389/fimmu.2025.1553004](https://doi.org/10.3389/fimmu.2025.1553004)

- Vastesaeger, N., van der Heijde, D., Inman, R.D., et al. (2011) Predicting the outcome of ankylosing spondylitis therapy based on baseline characteristics: Data from the ASSERT trial. *The Journal of Rheumatology*, **38**(6), pp.1250–1257. [https://doi.org/10.3899/jrheum.100345](https://doi.org/10.3899/jrheum.100345)

- Thorley, A., Jensen, M., Brown, S., et al. (2023) Imaging biomarkers for treatment prediction in axial spondyloarthritis: A review. *Current Rheumatology Reports*, **25**(2), pp.123–135. [https://doi.org/10.1007/s11926-023-01078-5](https://doi.org/10.1007/s11926-023-01078-5)


