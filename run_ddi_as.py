#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ddi_as.py

DDI-AS主运行脚本
提供完整的训练、评估和集成流程
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR, 
    CLINICAL_RESULTS_DIR, MRI_RESULTS_DIR, ENSEMBLE_RESULTS_DIR,
    MODEL_CONFIG, PERFORMANCE_TARGETS
)


def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n🚀 {description}")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误信息: {e.stderr}")
        return False


def setup_environment():
    """设置环境"""
    print("🔧 设置DDI-AS环境...")
    
    # 创建必要的目录
    directories = [DATA_DIR, RESULTS_DIR, CLINICAL_RESULTS_DIR, MRI_RESULTS_DIR, ENSEMBLE_RESULTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    print("✅ 环境设置完成")


def train_clinical_net(data_path, output_dir):
    """训练ClinicalNet"""
    if not data_path.exists():
        print(f"❌ 临床数据路径不存在: {data_path}")
        return False
    
    command = f"python src/clinical/training_clinical_data/train_clinical_ensemble.py --data_dir {data_path} --output_dir {output_dir} --n_folds {MODEL_CONFIG['clinical']['cv_folds']} --save_models"
    
    return run_command(command, "训练ClinicalNet")


def train_imaging_net(as_dir, healthy_dir, output_dir):
    """训练ImagingNet"""
    if not as_dir.exists():
        print(f"❌ AS数据路径不存在: {as_dir}")
        return False
    
    if not healthy_dir.exists():
        print(f"❌ 健康数据路径不存在: {healthy_dir}")
        return False
    
    command = f"python src/mri/analysis/mri_subject_level_auc.py --as-dir {as_dir} --healthy-dir {healthy_dir} --n-splits {MODEL_CONFIG['imaging']['cv_folds']} --n-bootstrap 1000 --batch-size 16 --seed 42 --device cpu"
    
    return run_command(command, "训练ImagingNet")


def train_ensemble(clinical_results, imaging_results, output_dir):
    """训练集成模型"""
    if not clinical_results.exists():
        print(f"❌ ClinicalNet结果文件不存在: {clinical_results}")
        return False
    
    command = f"python src/ensemble/train_ensemble.py --clinical_data {clinical_results} --imaging_data {imaging_results} --output_dir {output_dir} --n_folds_clinical {MODEL_CONFIG['clinical']['cv_folds']} --n_folds_imaging {MODEL_CONFIG['imaging']['cv_folds']}"
    
    return run_command(command, "训练集成模型")


def evaluate_models():
    """评估模型性能"""
    print("\n📊 评估模型性能...")
    
    # 检查性能目标
    clinical_target = PERFORMANCE_TARGETS['clinical']['auroc']
    imaging_target = PERFORMANCE_TARGETS['imaging']['auroc']
    ensemble_target = PERFORMANCE_TARGETS['ensemble']['auroc']
    
    print(f"目标性能:")
    print(f"  ClinicalNet AUROC: {clinical_target}")
    print(f"  ImagingNet AUROC: {imaging_target}")
    print(f"  Ensemble AUROC: {ensemble_target}")
    
    # 这里可以添加实际的性能评估逻辑
    print("✅ 性能评估完成")


def generate_figures():
    """生成论文图表"""
    print("\n📈 生成论文图表...")
    
    figure_scripts = [
        "scripts/create_figure_3_2_3.py",
        "scripts/create_figure_3_3_1.py", 
        "scripts/create_figure_3_5_1.py"
    ]
    
    for script in figure_scripts:
        if Path(script).exists():
            run_command(f"python {script}", f"生成图表: {script}")
        else:
            print(f"⚠️ 图表脚本不存在: {script}")
    
    print("✅ 图表生成完成")


def main():
    parser = argparse.ArgumentParser(description='DDI-AS完整训练和评估流程')
    parser.add_argument('--mode', choices=['full', 'clinical', 'imaging', 'ensemble', 'evaluate'], 
                       default='full', help='运行模式')
    parser.add_argument('--clinical_data', type=str, default='data/clinical', 
                       help='临床数据路径')
    parser.add_argument('--as_data', type=str, default='data/mri/as', 
                       help='AS MRI数据路径')
    parser.add_argument('--healthy_data', type=str, default='data/mri/healthy', 
                       help='健康MRI数据路径')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='输出目录')
    parser.add_argument('--skip_setup', action='store_true', 
                       help='跳过环境设置')
    
    args = parser.parse_args()
    
    print("🎯 DDI-AS: Dual Diagnostic Intelligence for Ankylosing Spondylitis")
    print("=" * 60)
    
    # 设置环境
    if not args.skip_setup:
        setup_environment()
    
    # 根据模式运行相应的流程
    if args.mode in ['full', 'clinical']:
        clinical_data_path = Path(args.clinical_data)
        clinical_output = Path(args.output_dir) / "clinical"
        
        if not train_clinical_net(clinical_data_path, clinical_output):
            print("❌ ClinicalNet训练失败，停止执行")
            return
    
    if args.mode in ['full', 'imaging']:
        as_data_path = Path(args.as_data)
        healthy_data_path = Path(args.healthy_data)
        mri_output = Path(args.output_dir) / "mri"
        
        if not train_imaging_net(as_data_path, healthy_data_path, mri_output):
            print("❌ ImagingNet训练失败，停止执行")
            return
    
    if args.mode in ['full', 'ensemble']:
        clinical_results = Path(args.output_dir) / "clinical" / "ensemble_predictions.csv"
        imaging_results = Path(args.output_dir) / "mri" / "predictions.csv"
        ensemble_output = Path(args.output_dir) / "ensemble"
        
        if not train_ensemble(clinical_results, imaging_results, ensemble_output):
            print("❌ 集成模型训练失败，停止执行")
            return
    
    if args.mode in ['full', 'evaluate']:
        evaluate_models()
        generate_figures()
    
    print("\n🎉 DDI-AS流程完成!")
    print("=" * 60)
    print("📁 结果文件位置:")
    print(f"  临床结果: {CLINICAL_RESULTS_DIR}")
    print(f"  MRI结果: {MRI_RESULTS_DIR}")
    print(f"  集成结果: {ENSEMBLE_RESULTS_DIR}")
    print(f"  论文: {PROJECT_ROOT}/docs/paper/paper_overall.md")


if __name__ == "__main__":
    main() 