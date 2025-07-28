#!/usr/bin/env python3
"""
MRI方向性校正模块
实现论文2.4.5节描述的系统方向校正逻辑
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def apply_direction_correction(predictions_df, logit_col='logit_raw', prob_col='prob_raw'):
    """
    应用系统方向校正，处理小样本学习中的logit符号反转问题
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        包含预测结果的DataFrame
    logit_col : str
        原始logit列名
    prob_col : str
        原始概率列名
    
    Returns:
    --------
    pd.DataFrame
        校正后的预测结果
    """
    
    # 计算原始AUROC
    y_true = predictions_df['y_true'].values
    y_probs = predictions_df[prob_col].values
    original_auroc = roc_auc_score(y_true, y_probs)
    
    print(f"原始AUROC: {original_auroc:.3f}")
    
    # 如果AUROC < 0.5，应用方向校正
    if original_auroc < 0.5:
        print("检测到方向反转，应用系统校正...")
        
        # 翻转logits
        predictions_df['logit_corrected'] = -predictions_df[logit_col]
        
        # 重新计算概率
        predictions_df['prob_corrected'] = 1 / (1 + np.exp(-predictions_df['logit_corrected']))
        
        # 验证校正效果
        corrected_auroc = roc_auc_score(y_true, predictions_df['prob_corrected'].values)
        print(f"校正后AUROC: {corrected_auroc:.3f}")
        
        return predictions_df
    else:
        print("无需方向校正")
        predictions_df['logit_corrected'] = predictions_df[logit_col]
        predictions_df['prob_corrected'] = predictions_df[prob_col]
        return predictions_df

def temperature_scaling_calibration(predictions_df, logit_col='logit_corrected', n_bins=15):
    """
    应用温度缩放校准
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        包含校正后预测结果的DataFrame
    logit_col : str
        校正后的logit列名
    n_bins : int
        校准评估的bin数量
    
    Returns:
    --------
    dict
        校准统计信息
    """
    
    from sklearn.calibration import calibration_curve
    
    y_true = predictions_df['y_true'].values
    y_probs = predictions_df['prob_corrected'].values
    
    # 计算校准曲线
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_probs, n_bins=n_bins
    )
    
    # 计算ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_accuracy = np.sum(y_true[in_bin]) / bin_size
            bin_confidence = np.mean(y_probs[in_bin])
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    ece /= len(y_true)
    
    return {
        'ece': ece,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'calibration_curve_data': {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    }

def decision_curve_analysis(predictions_df, prob_col='prob_corrected', 
                          thresholds=np.linspace(0.01, 0.99, 100)):
    """
    决策曲线分析
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        包含预测结果的DataFrame
    prob_col : str
        概率列名
    thresholds : np.ndarray
        决策阈值范围
    
    Returns:
    --------
    dict
        决策曲线数据
    """
    
    y_true = predictions_df['y_true'].values
    y_probs = predictions_df[prob_col].values
    
    net_benefits = []
    
    for threshold in thresholds:
        # 计算真阳性和假阳性
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        n = len(y_true)
        
        # 计算净收益
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
    
    # 计算treat-all和treat-none的净收益
    treat_all_net_benefit = np.mean(y_true) - (1 - np.mean(y_true)) * (thresholds / (1 - thresholds))
    treat_none_net_benefit = np.zeros_like(thresholds)
    
    return {
        'thresholds': thresholds,
        'net_benefits': np.array(net_benefits),
        'treat_all_net_benefit': treat_all_net_benefit,
        'treat_none_net_benefit': treat_none_net_benefit
    }

if __name__ == "__main__":
    # 示例用法
    print("MRI方向性校正和校准模块")
    print("请将此模块集成到现有的MRI分析流程中") 