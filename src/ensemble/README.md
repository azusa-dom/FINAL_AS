# DDI-AS Ensemble Model

This module implements the ensemble model for the DDI-AS (Dual Diagnostic Intelligence for Ankylosing Spondylitis) framework, combining ClinicalNet and ImagingNet predictions using late fusion.

## Architecture

The ensemble model follows the architecture described in the paper:

- **ClinicalNet**: Gradient Boosting model for clinical data analysis
- **ImagingNet**: ResNet-18 + Logistic Regression for MRI analysis
- **Ensemble**: Simple averaging fusion with equal weights (0.5 each)

## Formula

The ensemble probability is calculated as:

```
P_Ensemble = 0.5 × P_ClinicalNet + 0.5 × P_ImagingNet
```

## Files

- `ensemble_model.py`: Main ensemble model implementation
- `train_ensemble.py`: Training script for the ensemble model
- `README.md`: This documentation

## Usage

### Basic Usage

```python
from src.ensemble.ensemble_model import create_ensemble_model
from src.clinical.training_clinical_data.train_clinical_ensemble import ClinicalNet
from src.mri.models.imaging_net import ImagingNet

# Create ensemble model
ensemble = create_ensemble_model(clinical_weight=0.5, imaging_weight=0.5)

# Set individual models
clinical_model = ClinicalNet()
imaging_model = ImagingNet()

# Train models (example)
clinical_model.fit(clinical_data, clinical_labels)
imaging_model.fit(imaging_data, imaging_labels)

# Set models in ensemble
ensemble.set_models(clinical_model, imaging_model)

# Make predictions
ensemble_probs = ensemble.predict_proba(clinical_data, imaging_data)
```

### Training Script

```bash
python src/ensemble/train_ensemble.py \
    --clinical_data path/to/clinical_data.csv \
    --imaging_data path/to/imaging_data \
    --output_dir results/ensemble \
    --n_folds_clinical 5 \
    --n_folds_imaging 8
```

## Model Components

### ClinicalNet

- **Type**: Gradient Boosting Classifier
- **Features**: Clinical EHR data (demographics, lab results, etc.)
- **Cross-validation**: 5-fold stratified
- **Performance**: AUROC ~0.938

### ImagingNet

- **Type**: ResNet-18 + Logistic Regression
- **Features**: MRI images (sacroiliac joint)
- **Cross-validation**: Leave-Two-Out (L2O-CV)
- **Performance**: AUROC ~0.833

### Ensemble

- **Fusion Method**: Late fusion with simple averaging
- **Weights**: Equal (0.5 each)
- **Performance**: AUROC ~0.941 (improvement of 0.003 over ClinicalNet)

## Performance Metrics

The ensemble model provides the following metrics:

- **AUROC**: Area Under the Receiver Operating Characteristic curve
- **Accuracy**: Overall classification accuracy
- **Log Loss**: Logarithmic loss for probability calibration
- **Improvement**: Performance gain over individual models

## Integration with Paper

This implementation matches the paper description:

1. **ClinicalNet as Gradient Boosting**: Modified from MLP to Gradient Boosting
2. **ImagingNet as ResNet-18 + LR**: Implemented as described
3. **Simple averaging fusion**: Equal weights (0.5 each)
4. **Cross-validation strategies**: 5-fold for clinical, L2O for imaging

## Dependencies

- scikit-learn
- torch
- torchvision
- numpy
- pandas
- joblib

## Notes

- The ClinicalNet implementation has been updated to use Gradient Boosting instead of MLP to match the paper description
- ImagingNet uses frozen ResNet-18 for feature extraction and Logistic Regression for classification
- The ensemble uses simple averaging as described in the paper formula
- All models include proper cross-validation and evaluation metrics 