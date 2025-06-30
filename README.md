# Dual-Engine Diagnostic Workflow for Ankylosing Spondylitis (AS)

This repository contains the source code and documentation for a two-stage diagnostic workflow for Ankylosing Spondylitis (AS). The methodology is designed to emulate a real-world clinical pathway by first employing a robust pre-screening model on tabular clinical data, followed by an exploratory feature analysis on MRI data. This dual-engine approach aims to demonstrate a feasible AI-powered diagnostic framework, acknowledging the current limitations in public data availability for this specific disease.

## Core Features

  * **Unimodal Engines:**
      * **Engine 1 – Clinical Model:** A fully-connected neural network (FCNN) trained on structured clinical features, validated using a 5-fold cross-validation strategy.
      * **Engine 2 – MRI Feature Explorer:** A feature extraction pipeline that utilizes a ResNet50 model, pre-trained on ImageNet, to generate quantitative feature vectors from sacroiliac joint MRI slices. Feature distributions are visualized using t-SNE.
  * **Interpretability and Clinical Utility Analysis:**
      * **SHAP (SHapley Additive exPlanations):** Integrated to analyze the contribution of each clinical feature to the predictions of the clinical model.
      * **Decision Curve Analysis (DCA):** Implemented to assess the net benefit and clinical utility of the clinical model, providing insights beyond traditional accuracy metrics.
  * **Reproducible Workflow:**
      * The entire pipeline is automated via the `scripts/runall.sh` script, ensuring full reproducibility from data preparation to final evaluation.

## Pipeline Overview

1.  **Data Preprocessing (`scripts/preprocess_clinical_as.py`)**

      * This script cleans the raw clinical data and generates stratified 5-fold splits for robust, leakage-free cross-validation.

2.  **Engine 1 – Clinical Model Training (`src/train.py`)**

      * This script trains the multi-layer FCNN on the preprocessed clinical data folds.

3.  **Engine 2 – MRI Feature Exploration (`src/explore_mri_features.py`)**

      * This script executes the feature extraction pipeline on the small-sample MRI dataset, generating and saving a t-SNE visualization of the deep features.

4.  **Optional Late Fusion (`src/train_late_fusion.py`)**

      * This script is provided as a template for future experiments. It outlines how predictions from an MRI model and clinical features could be combined using an XGBoost model, contingent on the availability of a large, paired dataset.

5.  **Evaluation and Analysis (`src/evaluate.py`, `scripts/plot_shap_dca.py`)**

      * The evaluation script assesses the performance of the trained clinical model using predictions from the hold-out validation sets across all folds.
      * It calculates 95% confidence intervals for key metrics like AUROC and AUPRC via bootstrapping.
      * The plotting script generates and saves the SHAP and DCA plots for model interpretation.

## Directory Structure

```
FINAL_AS/
├── README.md
├── requirements.txt
├── data/
│   ├── rheumatic_autoimmune_disease.csv # Raw clinical data CSV
│   └── (MRI NIfTI files must be provided by the user)
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
│   └── plot_shap_dca.py
└── src/
    ├── dataset.py              # PyTorch Dataset definitions
    ├── models.py               # Model architectures (ResNet, FCNN)
    ├── train.py                # Script to train the clinical model
    ├── explore_mri_features.py # Script to extract and visualize MRI features
    ├── train_late_fusion.py    # Script for the late fusion model (Future Work)
    ├── evaluate.py             # Script to evaluate model performance
    └── utils.py                # Utility functions
```

## Environment Setup

```bash
pip install -r requirements.txt
```

**Key Dependencies:** `torch`, `xgboost`, `pandas`, `scikit-learn`, `numpy`, `shap`, `matplotlib`. Please refer to `requirements.txt` for the full list of dependencies and their versions.

## Data Preparation

1.  **Clinical Data**: The primary clinical data file, expected to be named `rheumatic_autoimmune_disease.csv`, must be placed in the `data/` directory. This CSV file should contain a unique `patient_id` column, the binary `label` column (1 for AS, 0 for control), and all other relevant clinical features.
2.  **MRI Data**: The pipeline is designed to process MRI scans in the **NIfTI (`.nii` or `.nii.gz`)** format.
      * **Note on Preprocessing**: It is assumed that necessary upstream preprocessing steps, such as DICOM-to-NIfTI conversion and N4 bias field correction, have been performed offline. The repository does not include integrated scripts for these initial steps.
      * The preprocessed NIfTI files should be placed in a dedicated directory. The path to this directory must be correctly configured as an argument when running `src/explore_mri_features.py`.

## Usage

### Automated Execution

The `runall.sh` script provides a method for automating the entire experimental workflow in the correct sequence.

```bash
bash scripts/runall.sh
```

This script will sequentially execute clinical data preprocessing, clinical model training, MRI feature extraction, and final model evaluation.

### Manual Step-by-Step Execution

For debugging or modular execution, each step of the pipeline can be run manually.

```bash
# 1. Preprocess the clinical data and generate 5-fold splits
python scripts/preprocess_clinical_as.py

# 2. Train the clinical model across all folds
python src/train.py --data_dir data/processed_clinical_data --model_dir models/clinical_model

# 3. Perform exploratory feature extraction on MRI data
python src/explore_mri_features.py --data_dir path/to/your/mri_data --out_dir results/mri_features

# 4. Evaluate the predictions from the clinical model
python src/evaluate.py --preds_dir models/clinical_model/clinical_preds
python scripts/plot_shap_dca.py --csv data/processed_clinical_data/fold_0_val.csv --save_dir results
```

## Expected Output

Aggregated evaluation metrics for the clinical model will be saved to `results/metrics.txt`. All generated plots (e.g., ROC/PR curves, SHAP summary, DCA curves, t-SNE visualization) will be saved in the `results/` directory. An example `metrics.txt` format is shown below:

```
[Clinical-only] AUROC: 0.82 (95% CI 0.77–0.87)
```

*(Note: These values are for illustration purposes only.)*

## Known Limitations

  * **MRI Preprocessing**: Key upstream preprocessing steps (e.g., DICOM conversion, N4 bias field correction) are not integrated into the main pipeline and must be conducted offline.
  * **Data Source**: The models were developed using data from a single institutional source; therefore, generalizability to external, multi-center datasets has not been validated.
  * **Pre-training**: The feature extractor model was fine-tuned from ImageNet weights without an intermediate pre-training step on a large-scale medical imaging dataset (e.g., RadImageNet), which could potentially improve feature quality.

## Future Work

  * **Multi-Modal Fusion**: Upon acquisition of a large, paired dataset, implement and evaluate various early and late fusion strategies.
  * **MRI Interpretability**: Integrate methods such as Grad-CAM to generate saliency maps, providing visual explanations for the features learned by the model.
  * **Multi-Center Validation**: Validate the performance and robustness of the developed models on external datasets from different institutions and scanners.
  * **Clinical Decision Support Prototype**: Develop a web-based Clinical Decision Support System (CDSS) prototype to demonstrate the clinical applicability of the models for trial and feedback.

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
