#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

DDI-AS项目配置文件
定义核心参数、路径和模型配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
CLINICAL_DATA_DIR = DATA_DIR / "clinical"
MRI_DATA_DIR = DATA_DIR / "mri"

# 结果路径
RESULTS_DIR = PROJECT_ROOT / "results"
CLINICAL_RESULTS_DIR = RESULTS_DIR / "clinical"
MRI_RESULTS_DIR = RESULTS_DIR / "mri"
ENSEMBLE_RESULTS_DIR = RESULTS_DIR / "ensemble"

# 模型配置
MODEL_CONFIG = {
    # ClinicalNet配置
    "clinical": {
        "model_type": "gradient_boosting",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "cv_folds": 5
    },
    
    # ImagingNet配置
    "imaging": {
        "model_type": "resnet18_lr",
        "pretrained": True,
        "num_classes": 2,
        "random_state": 42,
        "cv_folds": 12,  # Leave-Two-Out CV with 8 subjects -> 12 folds (1 AS + 1 HC held out)
        "feature_dim": 512
    },
    
    # 集成配置
    "ensemble": {
        "clinical_weight": 0.5,
        "imaging_weight": 0.5,
        "fusion_method": "simple_average"
    }
}

# 数据配置
DATA_CONFIG = {
    "clinical": {
        "sample_size": 4254,
        "as_cases": 2127,
        "controls": 2127,
        "feature_count": 20,
        "prevalence": 0.5  # 平衡数据集
    },
    
    "imaging": {
        "total_subjects": 8,
        "as_subjects": 6,
        "healthy_subjects": 2,
        "total_slices": 39,
        "image_size": (224, 224)
    }
}

# 性能指标
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

# 文件路径配置
FILE_PATHS = {
    # 训练脚本
    "train_clinical": "src/clinical/training_clinical_data/train_clinical_ensemble.py",
    "train_imaging": "src/mri/analysis/mri_subject_level_auc.py",
    "train_ensemble": "src/ensemble/train_ensemble.py",
    
    # 评估脚本
    "evaluate_clinical": "src/clinical/evaluation_clinical_data/calculate_3_models_final_stats.py",
    "shap_analysis": "src/clinical/evaluation_clinical_data/shap_plot_interactions.py",
    
    # 数据预处理
    "preprocess_clinical": "src/clinical/clinical_data_preparation/preprocess_clinical_final.py",
    "build_dataset": "src/clinical/clinical_data_preparation/build_balanced_dataset.py",
    
    # 模型文件
    "clinical_model": "src/clinical/training_clinical_data/train_clinical_ensemble.py",
    "imaging_model": "src/mri/models/imaging_net.py",
    "ensemble_model": "src/ensemble/ensemble_model.py"
}

# 创建必要的目录
def create_directories():
    """创建项目所需的目录结构"""
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
        print(f"✅ 创建目录: {directory}")

# 验证配置
def validate_config():
    """验证配置的正确性"""
    print("🔍 验证项目配置...")
    
    # 检查模型配置
    assert MODEL_CONFIG["ensemble"]["clinical_weight"] + MODEL_CONFIG["ensemble"]["imaging_weight"] == 1.0, \
        "集成权重必须等于1.0"
    
    # 检查数据配置
    assert DATA_CONFIG["clinical"]["as_cases"] + DATA_CONFIG["clinical"]["controls"] == DATA_CONFIG["clinical"]["sample_size"], \
        "临床数据样本量不匹配"
    
    assert DATA_CONFIG["imaging"]["as_subjects"] + DATA_CONFIG["imaging"]["healthy_subjects"] == DATA_CONFIG["imaging"]["total_subjects"], \
        "影像数据样本量不匹配"
    
    print("✅ 配置验证通过")

# 获取模型参数
def get_model_params(model_type):
    """获取指定模型的参数"""
    return MODEL_CONFIG.get(model_type, {})

# 获取性能目标
def get_performance_targets(model_type):
    """获取指定模型的性能目标"""
    return PERFORMANCE_TARGETS.get(model_type, {})

# 获取文件路径
def get_file_path(file_key):
    """获取指定文件的路径"""
    return PROJECT_ROOT / FILE_PATHS.get(file_key, "")

if __name__ == "__main__":
    # 创建目录结构
    create_directories()
    
    # 验证配置
    validate_config()
    
    print("\n📋 项目配置摘要:")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"ClinicalNet AUROC目标: {PERFORMANCE_TARGETS['clinical']['auroc']}")
    print(f"ImagingNet AUROC目标: {PERFORMANCE_TARGETS['imaging']['auroc']}")
    print(f"集成模型 AUROC目标: {PERFORMANCE_TARGETS['ensemble']['auroc']}") 