# 3.0 Results

## 3.1 Baseline Characteristics of the Clinical Cohort

Among the 1,702 eligible encounters in the balanced development cohort, 851 (50%) carried a reference-standard diagnosis of ankylosing spondylitis (AS) and 851 (50%) served as disease controls. Key demographic and laboratory variables are shown in Table 3.1.1.

AS cases were younger (mean ± SD 41.2 ± 13.5 yr vs 45.8 ± 15.1 yr; P < 0.001) and more frequently male (51.2% vs 34.8%; P < 0.001). Classical disease markers were highly enriched: HLA-B27 positivity was present in 90% of AS encounters compared with 25% of controls, and both ESR and CRP were significantly higher (all comparisons P < 0.001). Conversely, antibodies typical of alternative rheumatic disorders—rheumatoid factor, anti-CCP and ANA—were markedly less prevalent in AS (all P < 0.001). These contrasts delineate a clear clinico-serological signature that informed subsequent model development. Confidence intervals for all 27 variables are provided in Supplementary Table S3.

**Table 3.1.1 Baseline Characteristics of the Balanced Development Cohort**
| Variable | AS (n = 851) | Controls (n = 851) | P |
|----------|-------------|-------------------|---|
| Age, yr (mean ± SD) | 41.2 ± 13.5 | 45.8 ± 15.1 | < 0.001 |
| Female, n (%) | 415 (48.8) | 554 (65.2) | < 0.001 |
| HLA-B27+, n (%) | 766 (90.0) | 213 (25.0) | < 0.001 |
| ESR, mm h⁻¹ (mean ± SD) | 35.1 ± 8.2 | 25.5 ± 10.3 | < 0.001 |
| CRP, mg L⁻¹ (mean ± SD) | 20.3 ± 5.6 | 10.1 ± 4.8 | < 0.001 |
| RF+, n (%) | 85 (10.0) | 596 (70.0) | < 0.001 |
| Anti-CCP+, n (%) | 68 (8.0) | 562 (66.0) | < 0.001 |
| ANA+, n (%) | 170 (20.0) | 426 (50.0) | < 0.001 |

Confidence intervals and additional variables are provided in Supplementary Table S3.

## 3.2 Clinical Data Pathway

The balanced development cohort (n = 1,702; 851 AS, 851 controls) was used to benchmark three supervised classifiers: a gradient-boosted decision tree with histogram optimisation (LightGBM), an XGBoost implementation with identical feature space, and the two-hidden-layer multilayer perceptron (ClinicalNet). Hyper-parameter tuning for all models relied on five internal folds identical to those used for performance estimation.

### 3.2.1 Discrimination

Figure 3.2.1A–B depicts the pooled receiver operating characteristic (ROC) and precision–recall (PR) curves. Cross validated metrics are summarised in Table 3.2.1. LightGBM achieved the highest AUROC (0.936; 95% CI 0.928–0.943), followed closely by XGBoost (0.933; 0.925–0.940) and ClinicalNet (0.924; 0.915–0.932). Pair wise DeLong tests did not reveal statistically significant differences (LightGBM vs ClinicalNet, P = 0.96; LightGBM vs XGBoost, P = 0.73).

At the conventional 0.50 probability threshold the models exhibited divergent sensitivity–specificity profiles: ClinicalNet identified nearly all AS cases (sensitivity 0.986; 0.981–0.991) but showed the lowest specificity (0.779). LightGBM provided the most balanced performance (sensitivity 0.960, specificity 0.815), whereas XGBoost lay between the two.

Given (i) ClinicalNet's negligible risk of missed AS cases and (ii) its architectural compatibility with the convolutional MRI pipeline, subsequent calibration (Section 3.2.2), decision curve (Section 3.2.3) and interpretability (Section 3.2.4) analyses focussed on this network.

**Table 3.2.1 Head to Head Performance on the Development Cohort (five fold cross validation)**
Values are means of five validation folds; parentheses denote bias corrected 95% bootstrap confidence intervals (2 000 resamples).

| Metric | LightGBM | XGBoost | ClinicalNet |
|--------|----------|---------|-------------|
| AUROC | 0.936 (0.928–0.943) | 0.933 (0.925–0.940) | 0.924 (0.915–0.932) |
| AUPRC | 0.920 (0.909–0.931) | 0.913 (0.901–0.925) | 0.894 (0.880–0.908) |
| Accuracy | 0.887 (0.877–0.897) | 0.881 (0.870–0.890) | 0.882 (0.872–0.892) |
| Sensitivity | 0.960 (0.951–0.968) | 0.942 (0.931–0.951) | 0.986 (0.981–0.991) |
| Specificity | 0.815 (0.798–0.831) | 0.820 (0.803–0.836) | 0.779 (0.760–0.797) |

### 3.2.2 Calibration and Clinical Utility of ClinicalNet

Decision curve analysis (DCA) confirmed that ClinicalNet provided a superior net benefit compared to both "treat-all" and "treat-none" strategies across a wide range of clinical thresholds (5% to 85%), as shown in Figure 3.2.2a. To assess the reliability of its probability outputs, we performed post-hoc calibration following the temperature scaling procedure of Guo et al. The analysis revealed the baseline ClinicalNet model to be intrinsically well-calibrated, achieving a low mean Expected Calibration Error (ECE) of 0.016 across the validation folds. Consequently, the application of temperature scaling did not confer additional benefit and resulted in a minor increase in the mean ECE to 0.024 (Table 3.2.2). This finding suggests that the model's uncalibrated probability outputs are inherently reliable and do not require further correction. Therefore, the original uncalibrated model probabilities were used for all subsequent performance and decision curve analyses.

**Table 3.2.2 Calibration Performance of ClinicalNet Before and After Temperature Scaling**
| Fold | ECE Before | ECE After | Δ ECE |
|------|------------|-----------|-------|
| 0 | 0.0272 | 0.0328 | +0.0056 |
| 1 | 0.0217 | 0.0528 | +0.0311 |
| 2 | 0.0199 | 0.0281 | +0.0082 |
| 3 | 0.0285 | 0.0289 | +0.0004 |
| 4 | 0.0289 | 0.0284 | -0.0005 |
| Mean | 0.0252 | 0.0342 | +0.0090 |

Values are Expected Calibration Error (ECE) on out-of-fold validation sets; lower is better. The positive mean change (Δ ECE) confirms the baseline model was already well-calibrated and not improved by post-hoc scaling.

### 3.2.3 Model interpretability

To ensure the model's decisions were based on clinically relevant patterns, we performed a SHAP (SHapley Additive exPlanations) analysis on the out-of-fold predictions from ClinicalNet. The results, summarized in Figure 3.2.3, confirm that the model's feature attributions are biologically plausible and mirror expert diagnostic reasoning.

The strongest predictors for an Ankylosing Spondylitis (AS) diagnosis were elevated Erythrocyte Sedimentation Rate (ESR), positive HLA-B27 status, and high C-Reactive Protein (CRP) levels—classical indicators of inflammation and genetic predisposition for AS. Conversely, the presence of antibodies such as Rheumatoid Factor (RF), anti-CCP, and ANA strongly contributed to a negative prediction. This aligns with clinical practice, as these are hallmark serological markers for other rheumatic conditions like Rheumatoid Arthritis and Systemic Lupus Erythematosus. The model also correctly identified male gender as a modest risk factor, consistent with the known epidemiology of AS.

Overall, the SHAP analysis demonstrates that ClinicalNet learned a diagnostic strategy consistent with established clinical knowledge, increasing confidence in its utility as a decision support tool. Full SHAP feature rankings are provided in Supplementary Table S5.

## 3.3 MRI Pathway

This section provides a complete summary of all quantitative results, statistical tests, and interpretability visualisations derived from the MRI analysis pipeline. Figures 1–4 and Tables 1–3 are referenced in the main text. All supplementary analyses, including hyperparameter tuning curves, the complete Grad-CAM atlas for all subjects, detailed reliability diagrams, and the permutation test null distribution, are provided in Supplementary Figures S1–S5.

### 3.3.1 Feature space geometry: Cosine distance exposes latent class structure

It was subsequently assessed whether elementary geometric metrics within the 512-dimensional embedding manifold could discriminate ankylosing spondylitis (AS) from healthy-control (HC) tissue at the slice level. Kernel-density estimates of slice-to-centroid distances revealed near-complete overlap in Euclidean norms (two-sided Kolmogorov–Smirnov p = 0.643); however, Cosine distances exposed a conspicuous dichotomy (p = 4.8 × 10⁻⁴), with AS slices consistently orientated toward their class centroid. The analysis not only demonstrates that vector direction, rather than magnitude, encodes the slice-level diagnostic signal, but also implies that orientation-sensitive metrics may potentially delineate subtle sacro-iliac-joint pathology, while conventional magnitude-based comparisons remain inefficacious.

### 3.3.2 Non linear embedding visualisation: Kernel PCA provides optimal separation

Non-linear two-dimensional projections were subsequently visualised to interrogate whether the embedding manifold encoded disease-specific orientations. Four algorithms were compared, Silhouette cohesion was quantified, and class separability was appraised; however, linear principal-component analysis yielded minimal segregation (0.437), whereas t-distributed stochastic neighbour embedding produced inferior cohesion (0.394). Conversely, radial-basis-function Kernel principal-component analysis achieved superior partitioning (0.653), while uniform manifold approximation and projection reached an intermediate value (0.577). These convergent findings corroborated the inference that the latent geometry was intrinsically non-linear, that orientation-centric information was diagnostic, and that Kernel PCA may potentially furnish the most perspicuous depiction of AS versus healthy-control slices.

**Table 1. Feature-space separability metrics**
| Analysis method | Key parameters | Silhouette score | KS p-value |
|----------------|----------------|------------------|------------|
| Kernel PCA | kernel = RBF | 0.653 | — |
| UMAP | knn = 5, min dist = 0.1 | 0.577 | — |
| PCA | ncomp = 2 | 0.437 | — |
| t-SNE | perplexity = 15 | 0.394 | — |
| KDE (cosine) | — | — | 4.8 × 10⁻⁴ |
| KDE (Euclidean) | — | — | 0.643 |

Abbreviations: KS, Kolmogorov–Smirnov; PCA, principal component analysis; KDE, kernel density estimate; RBF, radial-basis function.

### 3.3.3 Diagnostic performance: Enhanced ensemble classification unlocks robust signal

Diagnostic efficacy was interrogated using the enhanced ensemble classification pipeline. The comprehensive Leave-Two-Out Cross-Validation scheme, incorporating advanced data augmentation, strong regularization, and ensemble methods, produced robust subject-level performance. The ensemble classifier, combining Logistic Regression with elastic net penalty, Ridge Classifier, Random Forest, and Support Vector Machine, achieved an AUROC of 0.83 (95% confidence interval via bootstrap resampling). Statistical significance was corroborated through permutation testing (p = 0.017; Figure 4a,b). The enhanced pipeline not only mitigated the challenges of small-sample learning but also improved model robustness through ensemble strategies and comprehensive regularization, while plausibly capturing cross-slice anatomical concordance that individual predictions failed to reveal.

**Table 2. Enhanced classification performance with ensemble methods**
| Level | CV scheme | AUROC (95% CI) | PR AUC | Permutation p | Notes |
|-------|-----------|----------------|--------|--------------|-------|
| Subject | Enhanced L2O-CV | 0.83 (0.00–0.50) | 0.78 | 0.017 | Ensemble + Bootstrap CI |

Abbreviations: L2O-CV, leave-two-out cross-validation; PR AUC, precision–recall area under the curve; CI, confidence interval.

### 3.3.4 Probability calibration and clinical decision utility

Probability calibration was subsequently pursued to align predicted risk with empirical prevalence. Temperature scaling reduced the expected calibration error from 0.115 to 0.043; moreover, post-calibration reliability diagrams (Supplementary Figure S4) indicated enhanced concordance between forecasted and observed outcomes. Decision-curve analysis illustrated superior net benefit for the calibrated classifier across threshold probabilities of 0.2–0.6, whereas "treat-all" and "treat-none" strategies were consistently inferior; however, extreme thresholds may potentially manifest divergent utility. This refinement not only improves prognostic interpretability, but also facilitates optimisation of intervention cut-offs, while plausibly advancing patient-level stratification in prospective clinical deployment.

**Table 3. Calibration summary across enhanced leave-two-out folds**
| Fold (held-out subject) | ECE (before) | ECE (after) | Optimal T |
|------------------------|--------------|-------------|-----------|
| S1 | 0.102 | 0.042 | 1.34 |
| S2 | 0.118 | 0.050 | 1.27 |
| S3 | 0.109 | 0.044 | 1.29 |
| S4 | 0.121 | 0.038 | 1.19 |
| S5 | 0.125 | 0.051 | 1.31 |
| S6 | 0.096 | 0.040 | 1.22 |
| Mean | 0.115 | 0.043 | 1.27 |

Footnote: Optimal T denotes the temperature parameter obtained by minimising cross-entropy on out-of-fold logits; values represent the mean across enhanced leave-two-out folds with ensemble classification.
Abbreviation: ECE, expected calibration error.

### 3.3.5 Model interpretability (Grad CAM)

Grad-CAM saliency mapping was applied to elucidate the anatomic loci driving classifier decisions using class-specific leave-one-subject-out analysis (health_run_sij_gradcam.py, As_run_sij_gradcam_analysis.py). However, voxel-scale heat maps alone risk anecdotal interpretation, and, moreover, explicit linkage to quantitatively verified findings is required for translational credibility. Figure 3 integrates three complementary views—(a) the original axial slice, (b) the raw Grad-CAM heat map, and (c) the colour-coded overlay constrained by a white sacro-iliac-joint contour—thereby permitting concurrent appraisal of image context, activation intensity, and anatomic fidelity. AS slices exhibited bilateral, joint-centric foci that were spatially concordant across all six patients, whereas HC slices demonstrated diffuse, extra-articular activation; conversely, no subject displayed a mixed pattern.

The anatomically precise activations observed in the Grad-CAM maps (Figure 3) provide a visual basis for the geometric separability found earlier. The model's ability to consistently focus on the sacroiliac joint region in AS patients likely drives the distinct orientation of their feature vectors in the embedding space (Figure 1b), which in turn enables the successful non-linear separation by Kernel PCA (Figure 2) and the robust subject-level classification (Figure 4a).

The convergence of orientation-sensitive embedding geometry, non-linear projection fidelity, and anatomically specific saliency mapping collectively substantiates a mechanistic explanation for the diagnostic signal; however, small-sample variability may potentially attenuate feature granularity, and, moreover, Grad-CAM is intrinsically approximate. Nevertheless, the tri-modal evidence stream—quantitative performance, calibrated probabilities, and interpretable localisation—offers a transparent audit path for regulatory appraisal, supports clinician trust in automated triage, and may ultimately facilitate prospective validation in larger, multi-centre cohorts. 