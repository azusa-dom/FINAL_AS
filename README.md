# DDI-AS: Dual Diagnostic Intelligence Framework for Ankylosing Spondylitis

## 📋 Project Overview

**Dual Diagnostic Intelligence: An AI Framework for Ankylosing Spondylitis Diagnosis Under Real-World Data Constraints - Independent Validation on Clinical and Imaging Cohorts**

This project presents a groundbreaking dual-pathway artificial intelligence framework designed to address the critical challenge of diagnosing Ankylosing Spondylitis (AS) in real-world clinical settings where perfectly paired multimodal datasets are rare. The framework specifically tackles the **Multimodal Data Asynchrony Problem (MDAP)** by implementing independent ClinicalNet and ImagingNet streams with sophisticated late-fusion ensemble integration.

### 🎯 Key Innovation

Unlike traditional AI approaches that require perfectly aligned clinical and imaging data, DDI-AS operates effectively in fragmented healthcare environments where electronic health records (EHR) and MRI scans are rarely temporally synchronized. This makes it particularly valuable for resource-constrained clinical settings where comprehensive datasets are the exception rather than the rule.

## 🏗️ Project Architecture

```
DDI-AS/
├── README.md                    # Project overview and setup
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── run_ddi_as.py               # Main execution script
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code
│   ├── clinical/               # Clinical data processing
│   │   ├── training_clinical_data/
│   │   └── feature_engineering/
│   ├── mri/                    # MRI data processing
│   │   ├── analysis/
│   │   └── preprocessing/
│   └── ensemble/               # Ensemble model integration
│
├── scripts/                    # Utility scripts
│   ├── unified_figure_style_guide.py
│   ├── supplementary_visualizations.py
│   └── data_analysis_tools/
│
├── data/                       # Data files
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── accurate_data_results/  # Accurate results data
│
├── results/                    # Output results
│   ├── figures/                # Generated figures
│   ├── models/                 # Trained models
│   ├── performance/            # Performance metrics
│   ├── ensemble/               # Ensemble results
│   └── reports/                # Result reports
│
├── docs/                       # Documentation
│   ├── paper/                  # Research paper and appendices
│   ├── technical/              # Technical documentation
│   ├── academic/               # Academic documentation
│   └── guides/                 # User guides
│
└── project_summary/            # Project summary documents
    ├── PROJECT_CLEANUP_SUMMARY.md
    ├── CODE_CORRECTIONS_SUMMARY.md
    ├── DATA_CONSISTENCY_ANALYSIS.md
    └── ACCURATE_RESULTS_RECOVERY_SUMMARY.md
```

## 🔬 Methodology

### ClinicalNet Pathway
- **Data Source**: 4,254 balanced EHR records (2,127 AS cases, 2,127 controls)
- **Model Architecture**: Gradient Boosting Classifier with advanced feature engineering
- **Performance Metrics**: AUROC 0.938 ± 0.003, ECE 0.155, Log Loss 0.225
- **Validation Strategy**: 5-fold stratified cross-validation
- **Key Features**: 20 engineered features from 14 original clinical variables

### ImagingNet Pathway
- **Data Source**: 8 subjects (6 AS, 2 HC) yielding 39 MRI slices
- **Model Architecture**: ResNet-18 (frozen) + Global Average Pooling + Logistic Regression
- **Performance Metrics**: AUROC 0.833 ± 0.021 (p=0.017)
- **Validation Strategy**: Leave-Two-Out cross-validation
- **Preprocessing**: N4 bias correction, Gaussian smoothing, resampling

### Ensemble Integration
- **Fusion Method**: Late fusion with calibrated probability averaging (0.5 each)
- **Performance Metrics**: AUROC 0.941 (95% CI: 0.924–0.959)
- **Improvement**: ΔAUROC = 0.003 over ClinicalNet alone
- **Calibration**: ECE 0.168 (95% CI: 0.154-0.188)

## 📊 Performance Results

| Model | AUROC | ECE | Log Loss | Accuracy | Validation Method |
|-------|-------|-----|----------|----------|-------------------|
| ClinicalNet | 0.938 ± 0.003 | 0.155 | 0.225 | 0.906 | 5-fold CV |
| ImagingNet | 0.833 ± 0.021 | - | - | - | L2O-CV |
| **Ensemble** | **0.941 ± 0.009** | **0.168** | **0.420** | **0.912** | **Fusion** |

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/azusa-dom/FINAL_AS
cd FINAL_AS

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# (Optional) Prepare directory scaffolding (data/, results/)
python config.py
```

### 3. Model Training
```bash
# Train ClinicalNet only
python run_ddi_as.py --mode clinical --clinical_data data/clinical --output_dir results

# Train ImagingNet only
python run_ddi_as.py --mode imaging --as_data data/mri/as --healthy_data data/mri/healthy --output_dir results

# Train ensemble (requires predictions produced by the above steps)
python run_ddi_as.py --mode ensemble --output_dir results
```

### 4. Build the paper (release)
```bash
# Quick local build (requires TeXLive)
make build-paper

# Or build the synchronized release file
cd docs/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error final_release.tex
```

## 🔬 Technical Details

### Clinical Data Processing
- **Feature Engineering**: 20 engineered features from 14 original variables
- **Preprocessing**: Log1p transformation, z-score standardization
- **Encoding**: One-hot encoding for categorical variables
- **Validation**: Comprehensive cross-validation with calibration analysis

### MRI Data Processing
- **Preprocessing Pipeline**: N4 bias correction, Gaussian smoothing, resampling
- **Feature Extraction**: ResNet-18 (frozen) + Global Average Pooling
- **Validation**: 12-fold Leave-Two-Out cross-validation
- **Reproducibility**: Fixed random seeds and deterministic processing

### Model Interpretability
- **ClinicalNet**: SHAP analysis for feature importance ranking
- **ImagingNet**: Grad-CAM for anatomical focus visualization
- **Ensemble**: Decision curve analysis for clinical utility assessment

## 📚 Documentation

### Research Paper
- **Main Paper**: `docs/paper/paper_overall.md` - Complete research manuscript
- **Appendices**: `docs/paper/APPENDIX_CORRECTED.md` - Technical appendices
- **Methodology**: `docs/paper/PAPER_METHODOLOGY.md` - Detailed methodology
- **LaTeX Version**: `docs/paper/essay.latex` - Academic paper format

### Technical Documentation
- **Technical Docs**: `docs/technical/` - Implementation details
- **User Guides**: `docs/guides/` - Usage instructions
- **API Reference**: `docs/api/` - Code documentation

## 🛠️ Dependencies

### Core Requirements
- Python 3.10+
- PyTorch 2.2.0
- scikit-learn 1.3.0
- matplotlib 3.7.2
- seaborn 0.12.2
- SHAP 0.42.1

### Additional Libraries
- SimpleITK (for MRI processing)
- nibabel (for neuroimaging)
- pandas, numpy (for data manipulation)
- scipy (for statistical analysis)

## 🎯 Clinical Impact

### Problem Addressed
- **Diagnostic Delay**: Average 6.7 years in AS diagnosis
- **Data Fragmentation**: Only 12.3% of patients have contemporaneous multimodal data
- **Resource Constraints**: Limited MRI access in developing regions

### Solution Benefits
- **Reduced False Positives**: 5-10% improvement in low-prevalence settings
- **Resource Efficiency**: Works with fragmented, non-paired data
- **Clinical Utility**: Calibrated probabilities for informed decision-making

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you use this work in your research, please cite:

```bibtex
@article{ddi_as_2025,
  title={Dual Diagnostic Intelligence: An AI Framework for Ankylosing Spondylitis Diagnosis Under Real-World Data Constraints - Independent Validation on Clinical and Imaging Cohorts},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]},
  url={https://github.com/azusa-dom/FINAL_AS}
}
```

## 📞 Contact

For questions, support, or collaboration opportunities, please contact:
- **Email**: [your.email@institution.edu]
- **Institution**: [Your Institution]
- **Research Group**: [Your Research Group]

## 🙏 Acknowledgments

We thank the medical professionals and patients who contributed to this research, as well as the open-source community for providing the foundational tools that made this work possible.

---

**Note**: This project represents a significant advancement in addressing the Multimodal Data Asynchrony Problem (MDAP) in clinical AI, providing a robust framework for real-world deployment where perfectly paired datasets are rare. The framework's ability to work with fragmented healthcare data makes it particularly valuable for resource-constrained clinical environments.
