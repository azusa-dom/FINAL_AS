# Dual-Pathway AI Framework for Ankylosing Spondylitis Diagnosis

A novel artificial intelligence system addressing the multimodal data asynchrony problem in early ankylosing spondylitis (AS) diagnosis through independent clinical and imaging pathways.

## 📋 Abstract

This study introduces a decoupled dual-pathway framework that directly addresses the non-paired data barrier in ankylosing spondylitis (AS) diagnostics, whereby electronic health records (EHR) and imaging are rarely co-registered in routine practice. The architecture pursues independent, modality-specific optimization with late-fusion readiness for subsequent integration when paired cohorts become feasible.

## 🏗️ System Architecture

### Dual-Pathway Design
- **Clinical Data Pipeline**: ClinicalNet model trained on 12,085 outpatient encounters
- **MRI Analysis Pipeline**: ImagingNet model built from 8 MRI subjects (39 slices)
- **Fusion-Ready Architecture**: Modular design supporting future multimodal integration

### Technical Features
- ✅ Containerized deployment (Docker)
- ✅ HL7 FHIR-compliant API interfaces
- ✅ Data version control (DVC)
- ✅ Interpretability analysis (Grad-CAM, SHAP)
- ✅ Probability calibration and decision analysis
- ✅ TRIPOD-AI, SPIRIT-AI/CONSORT-AI compliance

## 📁 Project Structure

```
FINAL_AS/
├── src/
│   ├── clinical_data_src/          # Clinical data processing
│   │   ├── clinical_data_preparation/
│   │   ├── training_clinical_data/
│   │   └── evaluation_clinical_data/
│   ├── mri_src/                    # MRI analysis
│   │   ├── preprocessing/
│   │   ├── feature_extraction/
│   │   ├── analysis/
│   │   ├── mri_feature_analysis/
│   │   └── gradcam/
│   ├── utils/                      # Utility functions
│   ├── visualization/              # Visualization modules
│   └── api/                        # FHIR API interface
├── data/                           # Data directory
├── models/                         # Model files
├── results/                        # Output results
├── Dockerfile                      # Container configuration
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd FINAL_AS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Clinical data preprocessing
python src/clinical_data_src/clinical_data_preparation/preprocess_clinical_final.py \
    data/raw/clinical_data.csv \
    data/processed/clinical/

# MRI data preprocessing
python src/mri_src/preprocessing/preprocess.py \
    data/raw/mri/ \
    data/processed/mri/
```

### 3. Model Training

```bash
# Train clinical model
python src/clinical_data_src/training_clinical_data/train_clinical_mondrian.py \
    --data_dir data/processed/clinical/ \
    --model_dir models/clinical/ \
    --epochs 50

# MRI feature extraction and analysis
python src/mri_src/analysis/make_l2o_predictions.py \
    --data-root data/processed/mri/ \
    --out-csv results/mri/l2o_predictions.csv
```

### 4. Launch API Service

```bash
# Using Docker
docker build -t as-diagnosis-ai .
docker run -p 8080:8080 as-diagnosis-ai

# Or direct execution
python src/api/fhir_server.py
```

## 📊 Performance Metrics

### Clinical Model (ClinicalNet)
- **AUROC**: 0.924 (95% CI: 0.915-0.932)
- **Sensitivity**: 98.6%
- **Specificity**: 77.9%
- **Calibration Error (ECE)**: 0.016
- **Net Benefit**: Positive across 5-85% decision thresholds

### MRI Model (ImagingNet)
- **AUROC**: 0.83 (permutation p = 0.017)
- **Sample Size**: 8 subjects (39 slices)
- **Feature Dimension**: 512-dimensional embeddings
- **Cross-validation**: Leave-Two-Out (L2O-CV)

## 🔬 Methodological Features

### 1. Clinical Data Pipeline
- **Data Source**: 4,254 outpatient records (851 AS, 3,403 controls)
- **Feature Engineering**: 27 harmonized predictors
- **Model Architecture**: Two-hidden-layer MLP (64×64 units)
- **Training**: Adam optimizer, class-weighted cross-entropy
- **Calibration**: Temperature scaling with ECE evaluation

### 2. MRI Analysis Pipeline
- **Data Source**: Radiopaedia archive (6 AS, 2 healthy controls)
- **Preprocessing**: N4 bias-field correction, Gaussian smoothing (σ=0.51mm)
- **Feature Extraction**: ImageNet-pretrained ResNet-18 backbone
- **Validation**: Leave-Two-Out cross-validation
- **Direction Correction**: Systematic logit inversion for AUROC < 0.5

### 3. Interpretability Analysis
- **SHAP Analysis**: Feature importance for clinical model
- **Grad-CAM**: Anatomical attention mapping for MRI model
- **Feature Space Geometry**: Cosine distance and KS-test statistics
- **Embedding Projections**: PCA, Kernel PCA, t-SNE, UMAP

### 4. Small-Sample Learning
- **Pre-trained Features**: ImageNet-initialized ResNet-18
- **Subject-Level Aggregation**: Mean-pooled embeddings
- **Directionality Correction**: Pre-specified systematic inversion
- **Temperature Scaling**: Post-hoc calibration

## 📈 Results and Visualizations

### Clinical Pathway Results
```python
# Calibration curves and decision analysis
python src/clinical_data_src/evaluation_clinical_data/plot_overall_metrics.py

# SHAP feature importance
python src/clinical_data_src/evaluation_clinical_data/shap_plot_interactions.py
```

### MRI Pathway Results
```python
# Grad-CAM attention maps
python src/mri_src/gradcam/As_run_sij_gradcam_analysis.py

# Feature space geometry analysis
python src/mri_src/mri_feature_analysis/feature_space_geometry.py

# Direction correction and calibration
python src/mri_src/analysis/mri_direction_correction.py
```

## 🔧 API Usage

### Clinical Data Diagnosis
```bash
curl -X POST "http://localhost:8080/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "request_type": "clinical",
    "clinical_data": {
      "patient_id": "P001",
      "age": 35.0,
      "sex": "M",
      "hla_b27": "positive",
      "esr": 45.2,
      "crp": 18.5,
      "rf": "negative",
      "anti_ccp": "negative",
      "ana": "negative"
    }
  }'
```

### MRI Diagnosis
```bash
curl -X POST "http://localhost:8080/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "request_type": "mri",
    "mri_data": {
      "patient_id": "P001",
      "image_path": "/path/to/mri/image.nii.gz",
      "sequence_type": "T1"
    }
  }'
```

### Fusion Diagnosis
```bash
curl -X POST "http://localhost:8080/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "request_type": "fusion",
    "clinical_data": {...},
    "mri_data": {...}
  }'
```

## 📚 Key Contributions

### 1. Novel Dual-Pathway Architecture
- Addresses the multimodal data asynchrony problem (MDAP)
- Independent optimization of clinical and imaging pathways
- Late-fusion readiness for future paired datasets

### 2. Clinical Model Performance
- AUROC 0.924 with 98.6% sensitivity
- Intrinsic calibration (ECE = 0.016)
- Positive net benefit across clinical thresholds
- SHAP-based interpretability

### 3. Small-Sample MRI Analysis
- Proof-of-concept on 8 subjects (39 slices)
- Direction-sensitive feature space geometry
- Kernel PCA optimal separation (silhouette 0.653)
- Anatomically precise Grad-CAM attention

### 4. Regulatory Compliance
- TRIPOD-AI reporting standards
- SPIRIT-AI/CONSORT-AI trial guidance
- DECIDE-AI early clinical evaluation
- FDA PCCP principles for change control

## 🔬 Experimental Design

### Clinical Cohort
- **Sample Size**: 4,254 encounters (851 AS, 3,403 controls)
- **Data Source**: Retrospective structured EHR data
- **Cross-validation**: 5-fold stratified (shuffle=True, seed=42)
- **Feature Engineering**: 27 harmonized predictors

### MRI Cohort
- **Sample Size**: 8 subjects (6 AS, 2 healthy controls)
- **Data Source**: Radiopaedia teaching archive
- **Validation**: Leave-Two-Out cross-validation
- **Preprocessing**: Containerized pipeline (ANTs 2.4, TorchIO 0.19)

## 📊 Comparison with Literature

| Model | AUROC | Calibration (ECE) | Sample Size | Reference |
|-------|-------|-------------------|-------------|-----------|
| Kennedy et al. (2023) | 0.90 | Not reported | >10,000 EHR | - |
| Liu et al. (2024) | 0.87 (CT-based) | 0.05 | >800 MRI | - |
| ClinicalNet (Ours) | 0.924 | 0.016 | 12,085 EHR | This study |
| ImagingNet (Ours) | 0.83 | 0.043 | 8 MRI | This study |

## 🚀 Future Directions

### 1. Multi-center Validation
- Federated learning across multiple institutions
- External validation on representative cohorts
- Real-world deployment studies

### 2. Cross-modal Integration
- Late-fusion meta-learning on paired datasets
- Contrastive self-supervised learning
- Temporal integration of longitudinal data

### 3. Regulatory Pathway
- Prospective clinical trials (SPIRIT-AI)
- Early deployment evaluation (DECIDE-AI)
- FDA submission and approval

## 📄 Citation

If you use this system in your research, please cite:

```bibtex
@article{as_dual_pathway_2024,
  title={Dual-Pathway AI Framework for Ankylosing Spondylitis Diagnosis: 
         Addressing the Multimodal Data Asynchrony Problem},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or suggestions, please contact:
- Email: [zczqzh9@ucl.ac.uk]


## 🙏 Acknowledgments

We thank all researchers and developers who contributed to this project. Special thanks to the Radiopaedia community for providing the MRI teaching cases under CC BY-NC-SA 3.0 license.

## ⚠️ Limitations

- Clinical model requires external validation on representative cohorts
- MRI model performance based on small convenience sample
- Multimodal fusion not yet tested on paired data
- Need for larger multi-center validation studies

---

**Note**: This system is designed for research purposes. Clinical deployment requires additional validation and regulatory approval.
