#!/usr/bin/env python3
"""
AS诊断AI系统配置文件
统一管理所有参数和配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 结果目录配置
RESULTS_DIR = PROJECT_ROOT / "results"
CLINICAL_RESULTS_DIR = RESULTS_DIR / "clinical"
MRI_RESULTS_DIR = RESULTS_DIR / "mri"
FUSION_RESULTS_DIR = RESULTS_DIR / "fusion"

# 模型目录配置
MODELS_DIR = PROJECT_ROOT / "models"
CLINICAL_MODELS_DIR = MODELS_DIR / "clinical"
MRI_MODELS_DIR = MODELS_DIR / "mri"

# 临床数据配置
class ClinicalConfig:
    """临床数据配置"""
    
    # 数据文件
    RAW_CLINICAL_CSV = RAW_DATA_DIR / "clinical_data.csv"
    PROCESSED_CLINICAL_DIR = PROCESSED_DATA_DIR / "clinical"
    
    # 特征配置
    LABEL_COLUMN = "label"
    ID_COLUMN = "Patient_ID"
    TARGET_DISEASE = "Ankylosing Spondylitis"
    
    # 模型配置
    HIDDEN_SIZE = 64
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 3
    
    # 交叉验证配置
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # 校准配置
    N_BINS = 15  # ECE计算用的bin数量

# MRI数据配置
class MRIConfig:
    """MRI数据配置"""
    
    # 数据目录
    RAW_MRI_DIR = RAW_DATA_DIR / "mri"
    PROCESSED_MRI_DIR = PROCESSED_DATA_DIR / "mri"
    
    # 预处理配置
    TARGET_SHAPE = (224, 224)  # 目标图像尺寸
    GAUSSIAN_SIGMA = 0.51  # 高斯平滑参数 (mm)
    RESAMPLE_SPACING = (0.7, 0.7)  # 重采样间距 (mm)
    
    # ImageNet标准化参数
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # 特征提取配置
    FEATURE_DIM = 512  # ResNet-18特征维度
    BACKBONE = "resnet18"
    
    # 交叉验证配置
    LEAVE_TWO_OUT = True  # 使用Leave-Two-Out交叉验证
    
    # 校准配置
    TEMPERATURE_SCALING = True
    DIRECTION_CORRECTION = True

# 融合配置
class FusionConfig:
    """融合配置"""
    
    # 融合方法
    FUSION_METHOD = "average"  # "average", "weighted", "meta_learner"
    
    # 权重配置 (如果使用加权融合)
    CLINICAL_WEIGHT = 0.6
    MRI_WEIGHT = 0.4

# API配置
class APIConfig:
    """API配置"""
    
    HOST = "0.0.0.0"
    PORT = 8080
    DEBUG = True
    
    # FHIR配置
    FHIR_BASE_URL = "http://localhost:8080/api/fhir"
    
    # 模型加载配置
    MODEL_LOAD_TIMEOUT = 30  # 秒

# 可视化配置
class VisualizationConfig:
    """可视化配置"""
    
    # 图表样式
    FIGURE_SIZE = (12, 8)
    DPI = 300
    
    # 颜色配置
    COLORS = {
        'as': '#D55E00',      # 橙色 (AS)
        'control': '#0072B2', # 蓝色 (对照)
        'fusion': '#009E73'   # 绿色 (融合)
    }
    
    # 字体配置
    FONT_SIZE = 12
    TITLE_SIZE = 16

# 日志配置
class LogConfig:
    """日志配置"""
    
    LEVEL = "INFO"
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "pipeline.log"

# 实验配置
class ExperimentConfig:
    """实验配置"""
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 实验名称
    EXPERIMENT_NAME = "AS_Diagnosis_Dual_Pathway"
    
    # 版本控制
    VERSION = "1.0.0"

# 创建必要的目录
def create_directories():
    """创建必要的目录结构"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        CLINICAL_RESULTS_DIR,
        MRI_RESULTS_DIR,
        FUSION_RESULTS_DIR,
        MODELS_DIR,
        CLINICAL_MODELS_DIR,
        MRI_MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 验证配置
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查必要的目录
    if not DATA_DIR.exists():
        errors.append(f"数据目录不存在: {DATA_DIR}")
    
    # 检查临床数据文件
    if not ClinicalConfig.RAW_CLINICAL_CSV.exists():
        errors.append(f"临床数据文件不存在: {ClinicalConfig.RAW_CLINICAL_CSV}")
    
    # 检查MRI数据目录
    if not MRIConfig.RAW_MRI_DIR.exists():
        errors.append(f"MRI数据目录不存在: {MRIConfig.RAW_MRI_DIR}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过")
    return True

# 获取配置摘要
def get_config_summary():
    """获取配置摘要"""
    summary = {
        "project_root": str(PROJECT_ROOT),
        "data_directories": {
            "raw": str(RAW_DATA_DIR),
            "processed": str(PROCESSED_DATA_DIR)
        },
        "results_directories": {
            "clinical": str(CLINICAL_RESULTS_DIR),
            "mri": str(MRI_RESULTS_DIR),
            "fusion": str(FUSION_RESULTS_DIR)
        },
        "clinical_config": {
            "hidden_size": ClinicalConfig.HIDDEN_SIZE,
            "learning_rate": ClinicalConfig.LEARNING_RATE,
            "n_splits": ClinicalConfig.N_SPLITS
        },
        "mri_config": {
            "target_shape": MRIConfig.TARGET_SHAPE,
            "feature_dim": MRIConfig.FEATURE_DIM,
            "backbone": MRIConfig.BACKBONE
        },
        "api_config": {
            "host": APIConfig.HOST,
            "port": APIConfig.PORT
        }
    }
    return summary

if __name__ == "__main__":
    # 创建目录
    create_directories()
    
    # 验证配置
    validate_config()
    
    # 打印配置摘要
    print("\n配置摘要:")
    import json
    print(json.dumps(get_config_summary(), indent=2, ensure_ascii=False)) 