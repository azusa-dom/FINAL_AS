#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_enhanced_mri.py

Quick test script to validate the enhanced MRI analysis improvements.
This script runs a simplified version to quickly check if the improvements work.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def run_enhanced_analysis(data_root, output_dir):
    """Run the enhanced MRI analysis"""
    print("🔬 Running Enhanced MRI Analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test basic enhanced analysis
    cmd = [
        sys.executable, "src/mri_src/analysis/make_l2o_predictions_improved.py",
        "--data-root", data_root,
        "--out-csv", f"{output_dir}/test_enhanced.csv",
        "--seed", "42",
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Basic enhanced analysis completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running enhanced analysis: {e}")
        print(f"Error output: {e.stderr}")
        return False

def analyze_results(output_dir):
    """Analyze the results"""
    results_file = f"{output_dir}/test_enhanced.csv"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Calculate metrics
    total_subjects = len(df)
    as_subjects = sum(df['y_true'] == 1)
    hc_subjects = sum(df['y_true'] == 0)
    
    # Classification results
    as_correct = sum((df['y_true'] == 1) & (df['prob_raw'] > 0.5))
    hc_correct = sum((df['y_true'] == 0) & (df['prob_raw'] < 0.5))
    total_correct = as_correct + hc_correct
    
    # Calculate metrics
    accuracy = total_correct / total_subjects if total_subjects > 0 else 0
    sensitivity = as_correct / as_subjects if as_subjects > 0 else 0
    specificity = hc_correct / hc_subjects if hc_subjects > 0 else 0
    
    # Probability analysis
    as_prob_mean = df[df['y_true'] == 1]['prob_raw'].mean()
    hc_prob_mean = df[df['y_true'] == 0]['prob_raw'].mean()
    
    # Overfitting indicators
    hc_high_prob = sum((df['y_true'] == 0) & (df['prob_raw'] > 0.7))
    as_low_prob = sum((df['y_true'] == 1) & (df['prob_raw'] < 0.3))
    
    return {
        'total_subjects': total_subjects,
        'as_subjects': as_subjects,
        'hc_subjects': hc_subjects,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'as_prob_mean': as_prob_mean,
        'hc_prob_mean': hc_prob_mean,
        'hc_high_prob': hc_high_prob,
        'as_low_prob': as_low_prob,
        'overfitting_score': hc_high_prob + as_low_prob
    }

def compare_with_original(enhanced_metrics, original_file):
    """Compare with original results"""
    if not os.path.exists(original_file):
        print(f"⚠️  Original results file not found: {original_file}")
        return None
    
    original_df = pd.read_csv(original_file)
    
    # Calculate original metrics
    total_subjects = len(original_df)
    as_subjects = sum(original_df['y_true'] == 1)
    hc_subjects = sum(original_df['y_true'] == 0)
    
    as_correct = sum((original_df['y_true'] == 1) & (original_df['prob_raw'] > 0.5))
    hc_correct = sum((original_df['y_true'] == 0) & (original_df['prob_raw'] < 0.5))
    total_correct = as_correct + hc_correct
    
    accuracy = total_correct / total_subjects if total_subjects > 0 else 0
    sensitivity = as_correct / as_subjects if as_subjects > 0 else 0
    specificity = hc_correct / hc_subjects if hc_subjects > 0 else 0
    
    as_prob_mean = original_df[original_df['y_true'] == 1]['prob_raw'].mean()
    hc_prob_mean = original_df[original_df['y_true'] == 0]['prob_raw'].mean()
    
    hc_high_prob = sum((original_df['y_true'] == 0) & (original_df['prob_raw'] > 0.7))
    as_low_prob = sum((original_df['y_true'] == 1) & (original_df['prob_raw'] < 0.3))
    
    original_metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'as_prob_mean': as_prob_mean,
        'hc_prob_mean': hc_prob_mean,
        'hc_high_prob': hc_high_prob,
        'as_low_prob': as_low_prob,
        'overfitting_score': hc_high_prob + as_low_prob
    }
    
    return original_metrics

def print_comparison(enhanced_metrics, original_metrics):
    """Print comparison results"""
    print("\n📊 Results Comparison")
    print("=" * 40)
    
    print(f"{'Metric':<20} {'Original':<10} {'Enhanced':<10} {'Improvement':<12}")
    print("-" * 52)
    
    metrics = ['accuracy', 'sensitivity', 'specificity', 'as_prob_mean', 'hc_prob_mean', 'overfitting_score']
    
    for metric in metrics:
        orig_val = original_metrics[metric]
        enh_val = enhanced_metrics[metric]
        
        if metric == 'overfitting_score':
            improvement = orig_val - enh_val  # Lower is better
            improvement_str = f"{improvement:+.0f}"
        else:
            improvement = enh_val - orig_val
            improvement_str = f"{improvement:+.3f}"
        
        print(f"{metric:<20} {orig_val:<10.3f} {enh_val:<10.3f} {improvement_str:<12}")
    
    print("\n🎯 Key Improvements:")
    
    # Specificity improvement
    spec_improvement = enhanced_metrics['specificity'] - original_metrics['specificity']
    if spec_improvement > 0:
        print(f"✅ Specificity improved by {spec_improvement:.3f}")
    else:
        print(f"⚠️  Specificity decreased by {abs(spec_improvement):.3f}")
    
    # Overfitting reduction
    overfitting_reduction = original_metrics['overfitting_score'] - enhanced_metrics['overfitting_score']
    if overfitting_reduction > 0:
        print(f"✅ Overfitting reduced by {overfitting_reduction:.0f} cases")
    else:
        print(f"⚠️  Overfitting increased by {abs(overfitting_reduction):.0f} cases")
    
    # HC probability reduction
    hc_prob_reduction = original_metrics['hc_prob_mean'] - enhanced_metrics['hc_prob_mean']
    if hc_prob_reduction > 0:
        print(f"✅ HC probability reduced by {hc_prob_reduction:.3f}")
    else:
        print(f"⚠️  HC probability increased by {abs(hc_prob_reduction):.3f}")

def main():
    """Main test function"""
    print("🚀 Enhanced MRI Analysis Test")
    print("=" * 30)
    
    # Configuration
    data_root = "data"
    output_dir = "results/consolidated/mri/test_enhanced"
    original_file = "results/consolidated/mri/l2o_predictions_improved.csv"
    
    # Check if data directory exists
    if not os.path.exists(data_root):
        print(f"❌ Data directory not found: {data_root}")
        print("Please ensure the data directory exists with mri_AS/ and mri_health/ subdirectories")
        return
    
    # Run enhanced analysis
    success = run_enhanced_analysis(data_root, output_dir)
    if not success:
        return
    
    # Analyze results
    enhanced_metrics = analyze_results(output_dir)
    if enhanced_metrics is None:
        return
    
    print(f"\n📈 Enhanced Analysis Results:")
    print(f"  Accuracy: {enhanced_metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {enhanced_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {enhanced_metrics['specificity']:.3f}")
    print(f"  AS Probability: {enhanced_metrics['as_prob_mean']:.3f}")
    print(f"  HC Probability: {enhanced_metrics['hc_prob_mean']:.3f}")
    print(f"  Overfitting Score: {enhanced_metrics['overfitting_score']}")
    
    # Compare with original
    original_metrics = compare_with_original(enhanced_metrics, original_file)
    if original_metrics:
        print_comparison(enhanced_metrics, original_metrics)
    
    print(f"\n✅ Test completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 