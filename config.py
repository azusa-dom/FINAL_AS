#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

DDI-AS project configuration.
Defines core parameters, paths, and model configurations.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
CLINICAL_DATA_DIR = DATA_DIR / "clinical"
MRI_DATA_DIR = DATA_DIR / "mri"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
CLINICAL_RESULTS_DIR = RESULTS_DIR / "clinical"
MRI_RESULTS_DIR = RESULTS_DIR / "mri"
ENSEMBLE_RESULTS_DIR = RESULTS_DIR / "ensemble"

# Model configuration
MODEL_CONFIG = {
    # ClinicalNet configuration
    "clinical": {
        "model_type": "gradient_boosting",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "cv_folds": 5
    },
    
    # ImagingNet configuration
    "imaging": {
        "model_type": "resnet18_lr",
        "pretrained": True,
        "num_classes": 2,
        "random_state": 42,
        "cv_folds": 12,  # Leave-Two-Out CV with 8 subjects -> 12 folds (1 AS + 1 HC held out)
        "feature_dim": 512
    },
    
    # Ensemble configuration
    "ensemble": {
        "clinical_weight": 0.5,
        "imaging_weight": 0.5,
        "fusion_method": "simple_average"
    }
}

# Data configuration
DATA_CONFIG = {
    "clinical": {
        "sample_size": 4254,
        "as_cases": 2127,
        "controls": 2127,
        "feature_count": 20,
        "prevalence": 0.5  # balanced dataset
    },
    
    "imaging": {
        "total_subjects": 8,
        "as_subjects": 6,
        "healthy_subjects": 2,
        "total_slices": 39,
        "image_size": (224, 224)
    }
}

# Performance targets
PERFORMANCE_TARGETS = {
    "clinical": {
        "auroc": 0.938,
        "auroc_std": 0.003,
        "ece": 0.155,
        "log_loss": 0.225
    },
    
    "imaging": {
        "auroc": 0.833,
        "auroc_std": 0.021,
        "p_value": 0.017
    },
    
    "ensemble": {
        "auroc": 0.941,
        "auroc_ci": (0.924, 0.959),
        "improvement": 0.003,
        "ece": 0.168,
        "log_loss": 0.420
    }
}

# File path mapping
FILE_PATHS = {
    # Training scripts
    "train_clinical": "src/clinical/training_clinical_data/train_clinical_ensemble.py",
    "train_imaging": "src/mri/analysis/mri_subject_level_auc.py",
    "train_ensemble": "src/ensemble/train_ensemble.py",
    
    # Evaluation scripts
    "evaluate_clinical": "src/clinical/evaluation_clinical_data/calculate_3_models_final_stats.py",
    "shap_analysis": "src/clinical/evaluation_clinical_data/shap_plot_interactions.py",
    
    # Data preprocessing
    "preprocess_clinical": "src/clinical/clinical_data_preparation/preprocess_clinical_final.py",
    "build_dataset": "src/clinical/clinical_data_preparation/build_balanced_dataset.py",
    
    # Model files
    "clinical_model": "src/clinical/training_clinical_data/train_clinical_ensemble.py",
    "imaging_model": "src/mri/models/imaging_net.py",
    "ensemble_model": "src/ensemble/ensemble_model.py"
}

# Create necessary directories
def create_directories():
    """Create required project directory structure."""
    directories = [
        DATA_DIR,
        CLINICAL_DATA_DIR,
        MRI_DATA_DIR,
        RESULTS_DIR,
        CLINICAL_RESULTS_DIR,
        MRI_RESULTS_DIR,
        ENSEMBLE_RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

# Validate configuration
def validate_config():
    """Validate configuration correctness."""
    print("🔍 Validating project configuration...")
    
    # 检查模型配置
    assert MODEL_CONFIG["ensemble"]["clinical_weight"] + MODEL_CONFIG["ensemble"]["imaging_weight"] == 1.0, \
        "Ensemble weights must sum to 1.0"
    
    # 检查数据配置
    assert DATA_CONFIG["clinical"]["as_cases"] + DATA_CONFIG["clinical"]["controls"] == DATA_CONFIG["clinical"]["sample_size"], \
        "Clinical data sample sizes mismatch"
    
    assert DATA_CONFIG["imaging"]["as_subjects"] + DATA_CONFIG["imaging"]["healthy_subjects"] == DATA_CONFIG["imaging"]["total_subjects"], \
        "Imaging data subject counts mismatch"
    
    print("✅ Configuration validated")

# Get model parameters
def get_model_params(model_type):
    """Get parameters for the given model type."""
    return MODEL_CONFIG.get(model_type, {})

# Get performance targets
def get_performance_targets(model_type):
    """Get performance targets for the given model type."""
    return PERFORMANCE_TARGETS.get(model_type, {})

# Get file path
def get_file_path(file_key):
    """Get path for a configured file key."""
    return PROJECT_ROOT / FILE_PATHS.get(file_key, "")

if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Validate configuration
    validate_config()
    
    print("\n📋 Project configuration summary:")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"ClinicalNet AUROC target: {PERFORMANCE_TARGETS['clinical']['auroc']}")
    print(f"ImagingNet AUROC target: {PERFORMANCE_TARGETS['imaging']['auroc']}")
    print(f"Ensemble AUROC target: {PERFORMANCE_TARGETS['ensemble']['auroc']}")