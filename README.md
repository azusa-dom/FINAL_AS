Absolutely! Here’s the complete, professional-grade, academic-style README.md in pure English (Markdown format), fully aligned with your Final Revised Outline and deeply integrated with your codebase and methodology 👇

⸻


# 🧠 Ankylosing Spondylitis Diagnosis Using Independent Clinical and MRI Cohorts

**A validated machine learning model on structured clinical data and an exploratory deep feature study on independent MRI samples.**

---

## 🔍 Overview

This project implements a dual-cohort strategy for early Ankylosing Spondylitis (AS) diagnosis:

- 🧪 **Clinical Pipeline**: A feed-forward neural network (FFNN) trained on a structured dataset (N=4,254) achieves **AUROC = 0.922**.
- 🧠 **MRI Pipeline**: A deep learning feasibility study on an independent small-sample MRI cohort (N=8), with feature visualization via t-SNE and Grad-CAM.
- ✅ The two pipelines are fully decoupled and independently evaluated.

---

## 📚 Motivation and Novelty

- Early AS diagnosis remains clinically challenging. While laboratory and MRI data both offer diagnostic insights, real-world clinical practice often lacks paired multimodal datasets.
- This study independently validates:
  1. A high-performance diagnostic model on large-scale clinical data.
  2. Deep feature discriminability in a separate MRI cohort using visualization and statistical testing.

---

## 🗂️ Data Sources

| Cohort         | N     | Content                                 | Purpose                             |
|----------------|-------|-----------------------------------------|-------------------------------------|
| Clinical       | 4,254 | 27 structured variables (labs, HLA-B27) | FFNN model training & evaluation    |
| MRI (public)   | 8     | SIJ MRI slices (6 AS, 2 healthy)        | Deep feature feasibility + Grad-CAM |

---

## 🧰 Methods

### 🔬 1. Clinical Pipeline

- **Script**: `scripts/preprocess_clinical.py`
- **Features**: 27 structured variables, including CRP, ESR, BASDAI, HLA-B27
- **Preprocessing**:
  - Median imputation for continuous features
  - Mode imputation + one-hot encoding for categorical
  - `log1p` + `RobustScaler` for skewed inflammation markers (ESR, CRP)
- **Model**: Feedforward Neural Network (FFNN)
  - 27 → 256 → 128 → 64 → 1
  - ReLU activation, Dropout(0.2), Sigmoid output
- **Training**:
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - Batch size: 128, Epochs: 30
  - 5-fold stratified cross-validation
  - SMOTE applied within training folds
- **Calibration**:
  - Temperature scaling (T ≈ 1.13) for probability calibration

```bash
python src/train.py --config configs/clinical_ffnn.yaml


⸻

🧠 2. MRI Feasibility Pipeline
	•	Feature Extraction:
	•	Pretrained ResNet-18 used to extract 512-d slice-level features
	•	Aggregation:
	•	Slice-level features aggregated per subject (mean pooling)
	•	Evaluation:
	•	Classifier: Logistic Regression
	•	Cross-validation: Leave-One-Out CV
	•	Metric: AUC with 95% CI via 2000x bootstrap
	•	Permutation Test: N=5000 to assess statistical separation
	•	Visualization:
	•	t-SNE, UMAP, PCA for dimensionality reduction
	•	Grad-CAM for model interpretability on SIJ slices

python scripts/mri_subject_level_auc.py \
  --data-dir data/mri_image_modified \
  --n-splits 8 \
  --n-bootstrap 2000


⸻

📈 Evaluation Metrics

Metric	Value	CI / Details
AUROC	0.922	95% CI: 0.913–0.930
AUPRC	0.889	95% CI: 0.873–0.904
Brier Score	0.138	Post-calibration
ECE₁₀	0.021	Expected Calibration Error
DCA Net Gain	+0.18 (at p=0.3)	Decision Curve Analysis
MRI AUC	~0.995 (LOOCV)	On N=8 independent cohort
Perm. Test	p = 0.907	Centroid distance (non-sig)


⸻

🎨 Visual Outputs

Figure	File	Description
ROC Curve	sci_roc_curve.png	Clinical model ROC
PR Curve	sci_pr_curve.png	Precision–Recall
Calibration Curve	sci_calibration_curve.png	Post temperature-scaling
Confusion Matrix	sci_confusion_matrix.png	2x2 heatmap
SHAP Summary	shap_summary.png	Top feature contributors
t-SNE (MRI features)	tsne_slice_level.png	MRI feature space
Grad-CAM (samples)	*_gradcam_academic.png	Heatmaps over SIJ slices


⸻

🚀 Run Instructions

One-click execution:

bash run_all.sh

Manual step-by-step:

# Clinical Preprocessing
python scripts/preprocess_clinical.py data/clinical_raw.csv processed_data/

# FFNN Training
python src/train.py --config configs/clinical_ffnn.yaml

# MRI AUC + Permutation Test
python scripts/mri_subject_level_auc.py --data-dir data/mri_image_modified
python scripts/mri_permutation_test_full.py --data-dir data/mri_image_modified

# Grad-CAM Visualization
python scripts/plot_gradcam_academic.py


⸻

🧾 Citation

Ankylosing Spondylitis Diagnosis Using Independent Clinical and MRI Cohorts: A Validated Machine Learning Model and a Deep Feature Feasibility Study
Under Review (2025)

This repository supports the full reproducibility of all experiments and visualizations presented in the manuscript.

⸻

📄 License

This project is released under the MIT License.

---

## 📌 Notes

- This `README.md` adheres to academic clarity, GitHub markdown compatibility, and reproducibility principles.
- If you'd like, I can **automatically generate a `requirements.txt`** based on the Python code analysis.
- Want this converted to **LaTeX abstract** or **journal submission template**? Just ask!

Ready for upload! 🚀