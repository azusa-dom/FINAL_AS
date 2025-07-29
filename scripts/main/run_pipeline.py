#!/usr/bin/env python3
"""
AS诊断AI系统主运行脚本
整合临床和MRI双通路分析流程
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_clinical_pipeline(data_dir: str, output_dir: str, logger):
    """运行临床数据管道"""
    logger.info("开始临床数据管道...")
    
    try:
        # 1. 数据预处理
        logger.info("步骤1: 临床数据预处理")
        from src.clinical_data_src.clinical_data_preparation.preprocess_clinical_final import run_final_preprocessing
        
        input_csv = os.path.join(data_dir, "raw", "clinical_data.csv")
        clinical_output = os.path.join(output_dir, "clinical", "processed")
        
        if os.path.exists(input_csv):
            run_final_preprocessing(input_csv, clinical_output)
            logger.info("临床数据预处理完成")
        else:
            logger.warning(f"临床数据文件不存在: {input_csv}")
            return False
        
        # 2. 模型训练
        logger.info("步骤2: 临床模型训练")
        import subprocess
        
        train_script = os.path.join(project_root, "src/clinical_data_src/training_clinical_data/train_clinical_mondrian.py")
        cmd = [
            sys.executable, train_script,
            "--data_dir", clinical_output,
            "--model_dir", os.path.join(output_dir, "clinical", "models"),
            "--epochs", "50",
            "--n_splits", "5"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("临床模型训练完成")
        else:
            logger.error(f"临床模型训练失败: {result.stderr}")
            return False
        
        # 3. 模型评估
        logger.info("步骤3: 临床模型评估")
        eval_script = os.path.join(project_root, "src/clinical_data_src/evaluation_clinical_data/calculate_3_models_final_stats.py")
        
        result = subprocess.run([sys.executable, eval_script], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("临床模型评估完成")
        else:
            logger.error(f"临床模型评估失败: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"临床管道运行失败: {e}")
        return False

def run_mri_pipeline(data_dir: str, output_dir: str, logger):
    """运行MRI数据管道"""
    logger.info("开始MRI数据管道...")
    
    try:
        # 1. MRI数据预处理
        logger.info("步骤1: MRI数据预处理")
        mri_input = os.path.join(data_dir, "raw", "mri")
        mri_output = os.path.join(output_dir, "mri", "processed")
        
        if os.path.exists(mri_input):
            # 运行预处理脚本
            from src.mri_src.preprocessing.preprocess import preprocess_volume
            
            # 这里需要根据实际数据结构调整
            for root, dirs, files in os.walk(mri_input):
                for file in files:
                    if file.endswith(('.nii.gz', '.nii')):
                        input_path = os.path.join(root, file)
                        preprocess_volume(input_path, mri_output)
            
            logger.info("MRI数据预处理完成")
        else:
            logger.warning(f"MRI数据目录不存在: {mri_input}")
            return False
        
        # 2. 特征提取
        logger.info("步骤2: MRI特征提取")
        feature_script = os.path.join(project_root, "src/mri_src/feature_extraction/extract_mri_features.py")
        
        import subprocess
        cmd = [
            sys.executable, feature_script,
            "--input-dir", mri_output,
            "--output-dir", os.path.join(output_dir, "mri", "features")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("MRI特征提取完成")
        else:
            logger.error(f"MRI特征提取失败: {result.stderr}")
            return False
        
        # 3. 分析预测
        logger.info("步骤3: MRI分析预测")
        analysis_script = os.path.join(project_root, "src/mri_src/analysis/make_l2o_predictions.py")
        
        cmd = [
            sys.executable, analysis_script,
            "--data-root", mri_output,
            "--out-csv", os.path.join(output_dir, "mri", "predictions.csv")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("MRI分析预测完成")
        else:
            logger.error(f"MRI分析预测失败: {result.stderr}")
            return False
        
        # 4. 方向性校正和校准
        logger.info("步骤4: MRI方向性校正和校准")
        from src.mri_src.analysis.mri_direction_correction import apply_direction_correction, temperature_scaling_calibration
        
        predictions_path = os.path.join(output_dir, "mri", "predictions.csv")
        if os.path.exists(predictions_path):
            import pandas as pd
            predictions_df = pd.read_csv(predictions_path)
            
            # 应用方向性校正
            corrected_df = apply_direction_correction(predictions_df)
            
            # 应用温度缩放校准
            calibration_results = temperature_scaling_calibration(corrected_df)
            
            # 保存校正后的结果
            corrected_df.to_csv(os.path.join(output_dir, "mri", "corrected_predictions.csv"), index=False)
            
            logger.info(f"MRI校准完成，ECE: {calibration_results['ece']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"MRI管道运行失败: {e}")
        return False

def run_feature_analysis(data_dir: str, output_dir: str, logger):
    """运行特征空间分析"""
    logger.info("开始特征空间分析...")
    
    try:
        from src.mri_src.mri_feature_analysis.feature_space_geometry import (
            compute_distance_statistics,
            compute_embedding_projections,
            generate_geometry_report
        )
        
        # 加载特征数据
        features_path = os.path.join(output_dir, "mri", "features", "embeddings.npy")
        if os.path.exists(features_path):
            import numpy as np
            embeddings = np.load(features_path)
            
            # 这里需要加载对应的标签
            # labels = np.load(os.path.join(output_dir, "mri", "features", "labels.npy"))
            
            # 计算距离统计
            distance_stats = compute_distance_statistics(embeddings, labels)
            
            # 计算嵌入投影
            projection_results = compute_embedding_projections(embeddings, labels)
            
            # 生成报告
            report_path = os.path.join(output_dir, "mri", "geometry_report.txt")
            generate_geometry_report(distance_stats, projection_results, report_path)
            
            logger.info("特征空间分析完成")
            return True
        else:
            logger.warning(f"特征文件不存在: {features_path}")
            return False
            
    except Exception as e:
        logger.error(f"特征空间分析失败: {e}")
        return False

def run_gradcam_analysis(data_dir: str, output_dir: str, logger):
    """运行Grad-CAM分析"""
    logger.info("开始Grad-CAM分析...")
    
    try:
        gradcam_script = os.path.join(project_root, "src/mri_src/gradcam/As_run_sij_gradcam_analysis.py")
        
        import subprocess
        cmd = [
            sys.executable, gradcam_script
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Grad-CAM分析完成")
            return True
        else:
            logger.error(f"Grad-CAM分析失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Grad-CAM分析失败: {e}")
        return False

def run_fusion_analysis(output_dir: str, logger):
    """运行融合分析"""
    logger.info("开始融合分析...")
    
    try:
        fusion_script = os.path.join(project_root, "src/clinical_data_src/training_clinical_data/train_late_fusion.py")
        
        clinical_preds = os.path.join(output_dir, "clinical", "predictions.csv")
        mri_preds = os.path.join(output_dir, "mri", "corrected_predictions.csv")
        
        if os.path.exists(clinical_preds) and os.path.exists(mri_preds):
            import subprocess
            cmd = [
                sys.executable, fusion_script,
                "--clinical_preds", clinical_preds,
                "--mri_preds", mri_preds,
                "--output", os.path.join(output_dir, "fusion", "fusion_results.csv")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("融合分析完成")
                return True
            else:
                logger.error(f"融合分析失败: {result.stderr}")
                return False
        else:
            logger.warning("预测文件不存在，跳过融合分析")
            return False
            
    except Exception as e:
        logger.error(f"融合分析失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AS诊断AI系统完整流程")
    parser.add_argument("--data_dir", default="data", help="数据目录路径")
    parser.add_argument("--output_dir", default="results", help="输出目录路径")
    parser.add_argument("--skip_clinical", action="store_true", help="跳过临床管道")
    parser.add_argument("--skip_mri", action="store_true", help="跳过MRI管道")
    parser.add_argument("--skip_fusion", action="store_true", help="跳过融合分析")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始AS诊断AI系统完整流程")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "clinical"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mri"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "fusion"), exist_ok=True)
    
    success = True
    
    # 运行临床管道
    if not args.skip_clinical:
        if not run_clinical_pipeline(args.data_dir, args.output_dir, logger):
            success = False
            logger.error("临床管道失败")
    
    # 运行MRI管道
    if not args.skip_mri:
        if not run_mri_pipeline(args.data_dir, args.output_dir, logger):
            success = False
            logger.error("MRI管道失败")
    
    # 运行特征空间分析
    if not args.skip_mri:
        if not run_feature_analysis(args.data_dir, args.output_dir, logger):
            logger.warning("特征空间分析失败，但继续执行")
    
    # 运行Grad-CAM分析
    if not args.skip_mri:
        if not run_gradcam_analysis(args.data_dir, args.output_dir, logger):
            logger.warning("Grad-CAM分析失败，但继续执行")
    
    # 运行融合分析
    if not args.skip_fusion:
        if not run_fusion_analysis(args.output_dir, logger):
            logger.warning("融合分析失败")
    
    if success:
        logger.info("AS诊断AI系统完整流程执行成功！")
        logger.info(f"结果保存在: {args.output_dir}")
    else:
        logger.error("AS诊断AI系统完整流程执行失败！")
        sys.exit(1)

if __name__ == "__main__":
    main() 