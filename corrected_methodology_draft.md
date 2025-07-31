# 2.0 Methodology

To emulate the heterogeneous and fragmented nature of real world clinical data environments, two independent and unpaired patient cohorts were constructed: a large scale structured clinical dataset and a small scale sacro iliac joint (SIJ) MRI dataset. This dual cohort strategy explicitly reflects the logistical challenges of acquiring large, multimodally paired repositories and permits independent assessment of artificial intelligence (AI) models across modalities.

## 2.1 Data Sources and Study Cohorts

### 2.1.1 Clinical Cohort: Retrospective Structured Data

The clinical cohort was derived from the open-access 'Diagnosis of Rheumatic and Autoimmune Diseases' dataset (Mahdi et al., Data in Brief 60:111623, 2025; PMID 40502661), which comprises a total of 12,085 de-identified outpatient encounters collected at three tertiary rheumatology centres between 2015 and 2022. The dataset contains 27 variables encompassing demographics, serology, and clinical findings (Supplementary Table S1), including 2,127 encounters with a reference-standard diagnosis of Ankylosing Spondylitis (AS).

To construct a class-balanced development cohort for model training and cross-validation, we first selected a subset of 851 AS encounters. We then performed random downsampling on the remaining 9,958 non-AS encounters to create a matching control group of 851. This procedure resulted in a balanced development cohort of 1,702 total encounters. This cohort was exclusively used for model development, partitioned via a stratified five-fold cross-validation scheme (shuffle = True, seed = 42).

The remaining 10,383 encounters (1,276 AS and 9,107 controls) from the original dataset were entirely withheld as a hold-out test set for the final, one-time assessment of the trained model's generalization performance.

### 2.1.2 Imaging Cohort: Public MRI Dataset

Early AS is histopathologically characterised by active sacroiliitis, which precedes spinal syndesmophyte formation and therefore constitutes a cornerstone of the 2009 ASAS imaging criteria. To interrogate model performance in a data-scarce, imaging-driven scenario, a micro-cohort was curated from the public teaching file repository, Radiopaedia.org. This cohort comprised eight subjects: six with radiographically confirmed ankylosing spondylitis (rIDs: 70339, 22345, 161310, 15541, 74662, 85118) and two healthy controls (rIDs: 82640, 30253). From all available axial SIJ volumes (e.g., T1-weighted, STIR) across these cases, a total of 39 diagnostically salient slices were retained for analysis following automated region-of-interest (ROI) filtering (Section 2.4.1). This imaging cohort was fully independent, with no subject overlap with the clinical cohort.

**Ethics and licensing**: All datasets were anonymised at source. Under GDPR Recital 26 the data are non personal; institutional review board waivers were therefore issued (RHE NH 2025 01/02). Radiopaedia's non commercial licence restricts downstream software as a medical device use; the images are employed solely for algorithm prototyping and will be replaced by proprietary scans before regulatory submission.

**Table 2.1.1 Imaging Cohort Demographics**
| Group | Case IDs (Radiopaedia rID) | Sex | Age, yr (mean ± SD) | Imaging year |
|-------|---------------------------|-----|-------------------|-------------|
| Ankylosing spondylitis (AS) | 70339, 22345, 161310, 15541, 74662, 85118 | 2 M / 2 F | 27.5 ± 2.9 | 2011–2024 |
| Healthy controls (HC) | 82640, 30253 | 1 M / 1 F | 30 ± 6 | 2014–2020 |

### 2.1.3 MRI acquisition summary

Radiopaedia archives raw Digital Imaging and Communications in Medicine (DICOM) files but discloses only key scanner metadata. Slice level headers revealed two distinct hardware configurations (Table 2.1.2).

**Table 2.1.2 MRI Acquisition Parameters**
| Vendor / system | Field strength | Sequence | TR / TE (ms) | In-plane matrix | Voxel (mm) | Subjects |
|----------------|---------------|----------|-------------|----------------|------------|----------|
| Siemens Aera | 1.5 T | T1-TSE | 550 / 12 | 320 × 320 | 0.8 × 0.8 | 3 |
| GE MR-750 | 3 T | T1-TSE / STIR | 600 / 11 (T1) 4 000 / 35 (STIR) | 320 × 288 | 0.7 × 0.7 | 5 |

## 2.2 Study Design Rationale

Large, well annotated clinical registries are common, whereas high quality SIJ MRI scans are rare. To mirror this imbalance, unpaired cohorts of unequal size were intentionally maintained. Forcing early fusion of such mismatched data would either discard valuable clinical cases or over fit to the eight MRI subjects. Consequently, two separate classifiers—one tabular and one imaging—were trained, each yielding calibrated probability scores that can be fused once a sufficiently large paired dataset becomes available.

**Table 2.2.1 Independent Cohorts and Their Roles in the Study**
| Cohort | N | Modality | Key variables/images | Role in study |
|--------|---|----------|-------------------|---------------|
| Clinical | 1,702 | Tabular EHR | 27 demographic, laboratory and clinical features | Train & cross-validate a diagnostic model |
| MRI | 8 | SIJ MRI | 39 axial T1/STIR slices (sacro-iliac focus) | Proof-of-concept imaging classifier under data scarcity |

## 2.3 Clinical Data Pipeline

The clinical pathway converts 1,702 outpatient records into calibrated probabilities through four sequential blocks—pre processing, feature engineering, model construction, and post hoc calibration.

### 2.3.1 Data Origin, Governance, and Split Strategy

Twenty seven harmonised predictors (demographics 2; serology 4; autoimmune 3; physician graded 18; Supplementary Table S1) from the development subset (Section 2.1.1) were used as inputs. Records were split via stratified five fold cross validation (shuffle = True, seed = 42); the untouched remainder was reserved for external testing (Section 3.2).

### 2.3.2 Pre processing and Feature Engineering

**Cleaning and imputation**: Headers were snake cased; numeric outliers were winsorised at the 0.1 % and 99.9 % quantiles; a binary outcome label (1 = AS, 0 = other) was derived from the primary diagnosis. Missing values (≤ 3 %; ESR 7 %) were imputed per fold (median for numeric, mode for categorical fields).

**Scaling and encoding**: C reactive protein (CRP) and erythrocyte sedimentation rate (ESR) were log transformed, followed by z score scaling of all numeric variables. Categorical fields (HLA B27, sex, ANA, anti CCP, anti dsDNA) were one hot encoded within a single ColumnTransformer configured with handle_unknown="ignore".

**Zero variance filter**: Immediately before model fitting, any column exhibiting no variability in the training split was discarded. This reproducible step prunes degenerate dimensions generated by one hot expansion; the resulting design matrix contained 19–21 non constant features (fold specific counts in Supplementary Table S2).

**Class balancing**: After transformation but before model fitting, the Synthetic Minority Oversampling Technique (SMOTE, k = 5) was applied inside the training branch of an Imbalanced learn pipeline, leaving validation folds untouched. The final class ratio per training fold approximated 1 : 1 (49.6 % AS).

### 2.3.3 Model Architecture and Hyper parameter Selection

ClinicalNet is a two hidden layer multilayer perceptron (64 × 64 units; BatchNorm → ReLU → Dropout 0.50) implemented in PyTorch 2.1. Bayesian optimisation (Optuna, 50 trials) tuned the learning rate (10⁻⁴–10⁻²), weight decay (10⁻⁶–10⁻³), and dropout (0.2–0.6). The selected configuration—learning rate = 1 × 10⁻³, weight decay = 1 × 10⁻⁴, dropout = 0.50—maximised cross validated AUROC whilst keeping the calibration slope within 0.9–1.1. Gradient boosting baselines (XGBoost, LightGBM) did not exceed an AUROC of 0.90 and produced less stable probability distributions.

### 2.3.4 Training regimen and early stopping

Each fold was trained for up to 50 epochs using Adam (β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸) at a learning rate of 1 × 10⁻³ and weight decay of 1 × 10⁻⁴; batches of 32 observations were stratified. The loss function was class weighted binary cross entropy, with per class weights

[w_c=\frac{1}{p_c+{10}^{-6}}]

where denotes the prevalence of class in the training split. Early stopping monitored the validation loss and terminated optimisation after three consecutive non improving epochs; the parameter state yielding the minimum loss was checkpointed. Convergence curves for all folds are provided in Supplementary Figure S3.

### 2.3.5 Post-hoc Probability Calibration

The reliability of the model's probability outputs was assessed using post-hoc temperature scaling, following the procedure of Guo et al. For each cross-validation fold, a scalar temperature (T) was fitted to the validation logits by minimizing the negative log-likelihood. The impact of this procedure was quantified using the Expected Calibration Error (ECE) computed over 15 equal-mass bins, with the detailed results presented in Section 3.2.2.

## 2.4 MRI Analysis Pipeline

Eight anonymised MRI examinations (six early ankylosing spondylitis, two healthy controls; Section 2.1.2) were downloaded from the Radiopaedia teaching archive under a CC BY NC SA 3.0 licence. Axial sacro iliac joint (SIJ) slices were identified automatically via a three dimensional U Net that delineated the joint's bounding box; slices were retained only when 50 percent or more of voxels lay within that box and the estimated in plane signal to noise ratio (SNR) exceeded 15, yielding 39 analysable images.

### 2.4.1 Pre‑processing and Deep‑Feature Extraction

All axial SIJ volumes were processed by a deterministic, containerised pipeline (preprocess.py; SimpleITK, TorchIO 0.19). For each volume the following steps were performed sequentially (operations were seeded with 42): (1) N4 bias‑field correction using SimpleITK's N4BiasFieldCorrectionImageFilter with Otsu thresholding for mask generation; (2) three‑dimensional Gaussian smoothing with sigma 0.51 mm (full‑width at half‑maximum ≈ 1.2 mm) using SmoothingRecursiveGaussianImageFilter; (3) cubic B‑spline resampling to an in‑plane resolution of 0.7 × 0.7 mm whilst preserving native slice thickness using ResampleImageFilter with sitkBSpline interpolator; (4) centre cropping or zero‑padding to 224 × 224 pixels using TorchIO's CropOrPad transform; and (5) channel‑wise z‑normalisation to ImageNet statistics (mean [0.485, 0.456, 0.406], standard deviation [0.229, 0.224, 0.225]). Subjects with fewer than five valid slices were excluded.

Feature extraction employed an ImageNet‑pre‑trained ResNet‑18 backbone (extract_mri_features.py). All convolutional and batch‑normalisation layers were frozen and the final fully connected layer was replaced by an identity mapping. Passing the normalised slices through layer4 followed by global average pooling yielded a 512‑dimensional embedding that served as the fixed representation for all downstream analyses.

### 2.4.2 Enhanced Cross validation and Model Training

A comprehensive Leave-Two-Out Cross-Validation (L2O-CV) scheme was implemented in make_l2o_predictions_improved.py, incorporating multiple advanced techniques to address the challenges of small-sample learning. The enhanced pipeline included:

**Advanced Data Augmentation**: A comprehensive augmentation pipeline was applied, including geometric transformations (rotation ±10°, horizontal flip), intensity adjustments (brightness and contrast variations), and noise injection to increase effective sample size.

**Strong Regularization**: Multiple regularization strategies were employed, including elastic net penalty (C=0.01), feature selection using variance thresholding and SelectKBest, and outlier detection using Isolation Forest with contamination=0.1.

**Ensemble Classification**: An ensemble of multiple classifiers was implemented, including Logistic Regression with elastic net penalty, Ridge Classifier, Random Forest, and Support Vector Machine. The ensemble used voting strategies to improve robustness.

**Bootstrap Confidence Intervals**: Bootstrap resampling (n=1000) was performed to generate 95% confidence intervals for performance metrics, addressing the uncertainty inherent in small-sample scenarios.

### 2.4.3 Feature-space Geometry and Distance Statistics

To investigate the geometry of the embedding space, exploratory 2D projections were computed after applying a StandardScaler. We evaluated four methods: PCA, Kernel PCA (RBF kernel, γ=1/512), t-SNE (perplexity=15, pca-init), and UMAP (n_neighbors=5, min_dist=0.1). Cluster quality was reported by the Silhouette Score. We further analyzed the separability of the classes by computing the Euclidean and Cosine distances from each slice to its class centroid; statistical significance was determined using a two-sided Kolmogorov-Smirnov test with Benjamini–Hochberg p-value correction.

### 2.4.4 Interpretability (Grad-CAM)

For qualitative analysis, we implemented a class-specific leave-one-subject-out Grad-CAM analysis (health_run_sij_gradcam.py, As_run_sij_gradcam_analysis.py). A ResNet-18 model was fine-tuned in a class-specific manner, where one AS or HC subject was withheld while the remaining subjects of the same class were used for a 3-epoch fine-tune with light augmentation (±5° rotation, random resized crop, colour jitter). Grad-CAM was computed on layer4; per-slice heatmaps were overlaid on the original image with optional binary ROI masks to highlight intra-ROI activations. The analysis was performed separately for healthy controls and AS patients to ensure class-specific interpretability.

### 2.4.5 Probability Correction, Calibration, and Decision Analysis

The final aggregated subject-level predictions underwent a dedicated processing pipeline (mri_make_fig4.py). Due to the extremely small sample size (N=8) of the imaging cohort and the bespoke leave-two-out cross-validation design, the logistic regression classifier occasionally learned an inverted relationship in certain folds, resulting in an AUROC significantly below 0.5. To account for this instability inherent in small-sample learning, a systematic directionality check was implemented. Specifically, an initial AUROC was calculated on the raw probabilities. If this AUROC was less than 0.5, indicating a correctly learned class boundary but with an inverted sign (a systematic inversion of predictions), the raw logits were systematically flipped (logit_corrected = -logit_raw) to perform a direction correction. This pre-specified procedure ensured standardization of prediction direction across all folds prior to aggregation, thereby avoiding manual intervention or biased selection. Subsequently, these direction-corrected probabilities were post-hoc calibrated via temperature scaling. The optimal temperature (T) was found by minimizing the binary cross-entropy loss on the out-of-fold logits using a bounded scalar optimizer. Reliability was quantified by the Expected Calibration Error (ECE) computed over 4 equal-mass bins (adjusted for small sample size) before and after calibration. Finally, the clinical utility of the calibrated model was assessed using Decision Curve Analysis (DCA) against treat-all and treat-none comparator strategies. This directionality correction procedure was pre-specified in the analysis plan to systematically handle potential model inversions, a known risk in high-dimensional, low-sample-size regimes, thereby ensuring that the aggregated performance reflects signal magnitude rather than arbitrary sign flips during training. 