#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_enhancement_results.py

Analyze and compare the results of enhanced MRI analysis with the original results.
This script provides comprehensive analysis of the improvements made to address overfitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os

def load_results(file_path):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return None

def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics"""
    if df is None or len(df) == 0:
        return {}
    
    # Basic metrics
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
    as_prob_std = df[df['y_true'] == 1]['prob_raw'].std()
    hc_prob_std = df[df['y_true'] == 0]['prob_raw'].std()
    
    # Overfitting indicators
    hc_high_prob = sum((df['y_true'] == 0) & (df['prob_raw'] > 0.7))
    as_low_prob = sum((df['y_true'] == 1) & (df['prob_raw'] < 0.3))
    
    # Confidence in predictions
    confidence_as = abs(df[df['y_true'] == 1]['prob_raw'] - 0.5).mean()
    confidence_hc = abs(df[df['y_true'] == 0]['prob_raw'] - 0.5).mean()
    
    return {
        'total_subjects': total_subjects,
        'as_subjects': as_subjects,
        'hc_subjects': hc_subjects,
        'as_correct': as_correct,
        'hc_correct': hc_correct,
        'total_correct': total_correct,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'as_prob_mean': as_prob_mean,
        'hc_prob_mean': hc_prob_mean,
        'as_prob_std': as_prob_std,
        'hc_prob_std': hc_prob_std,
        'hc_high_prob': hc_high_prob,
        'as_low_prob': as_low_prob,
        'confidence_as': confidence_as,
        'confidence_hc': confidence_hc,
        'overfitting_score': hc_high_prob + as_low_prob  # Lower is better
    }

def compare_results(original_file, enhanced_files):
    """Compare original results with enhanced results"""
    print("🔍 Comparing Original vs Enhanced Results")
    print("=" * 50)
    
    # Load original results
    original_df = load_results(original_file)
    original_metrics = calculate_performance_metrics(original_df)
    
    print(f"📊 Original Results:")
    print(f"  Accuracy: {original_metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {original_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {original_metrics['specificity']:.3f}")
    print(f"  AS Probability: {original_metrics['as_prob_mean']:.3f} ± {original_metrics['as_prob_std']:.3f}")
    print(f"  HC Probability: {original_metrics['hc_prob_mean']:.3f} ± {original_metrics['hc_prob_std']:.3f}")
    print(f"  Overfitting Score: {original_metrics['overfitting_score']}")
    print()
    
    # Load and compare enhanced results
    enhanced_results = {}
    for file_path in enhanced_files:
        if os.path.exists(file_path):
            name = Path(file_path).stem.replace('enhanced_', '')
            df = load_results(file_path)
            metrics = calculate_performance_metrics(df)
            enhanced_results[name] = metrics
    
    # Create comparison dataframe
    comparison_data = {'original': original_metrics}
    comparison_data.update(enhanced_results)
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    # Calculate improvements
    improvements = {}
    for method, metrics in enhanced_results.items():
        improvements[method] = {
            'accuracy_improvement': metrics['accuracy'] - original_metrics['accuracy'],
            'specificity_improvement': metrics['specificity'] - original_metrics['specificity'],
            'overfitting_reduction': original_metrics['overfitting_score'] - metrics['overfitting_score'],
            'hc_prob_reduction': original_metrics['hc_prob_mean'] - metrics['hc_prob_mean'],
            'confidence_improvement': (metrics['confidence_as'] + metrics['confidence_hc']) / 2 - 
                                   (original_metrics['confidence_as'] + original_metrics['confidence_hc']) / 2
        }
    
    improvements_df = pd.DataFrame(improvements).T
    
    return comparison_df, improvements_df

def create_visualizations(comparison_df, improvements_df, output_dir):
    """Create visualization plots"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance metrics comparison
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity']
    comparison_df[metrics_to_plot].plot(kind='bar', ax=axes[0,0], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0,0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend()
    
    # 2. Probability distributions
    prob_metrics = ['as_prob_mean', 'hc_prob_mean']
    comparison_df[prob_metrics].plot(kind='bar', ax=axes[0,1], color=['#2E86AB', '#A23B72'])
    axes[0,1].set_title('Average Prediction Probabilities', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Probability')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].legend(['AS Subjects', 'HC Subjects'])
    
    # 3. Improvements summary
    improvement_metrics = ['accuracy_improvement', 'specificity_improvement', 'overfitting_reduction']
    improvements_df[improvement_metrics].plot(kind='bar', ax=axes[1,0], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1,0].set_title('Improvements vs Original', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Improvement')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend()
    
    # 4. Overfitting analysis
    overfitting_metrics = ['overfitting_score', 'hc_high_prob', 'as_low_prob']
    comparison_df[overfitting_metrics].plot(kind='bar', ax=axes[1,1], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1,1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Count')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend(['Total Overfitting', 'HC High Prob', 'AS Low Prob'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhancement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(comparison_df, improvements_df, output_dir):
    """Generate a detailed analysis report"""
    report = []
    report.append("# Enhanced MRI Analysis Results Report")
    report.append("=" * 50)
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append("| Method | Accuracy | Sensitivity | Specificity | Overfitting Score |")
    report.append("|--------|----------|-------------|-------------|-------------------|")
    
    for method, row in comparison_df.iterrows():
        report.append(f"| {method} | {row['accuracy']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | {row['overfitting_score']} |")
    
    report.append("")
    
    # Best performing method
    best_method = comparison_df['specificity'].idxmax()
    report.append(f"## Best Performing Method: {best_method}")
    report.append("")
    report.append(f"- **Accuracy**: {comparison_df.loc[best_method, 'accuracy']:.3f}")
    report.append(f"- **Sensitivity**: {comparison_df.loc[best_method, 'sensitivity']:.3f}")
    report.append(f"- **Specificity**: {comparison_df.loc[best_method, 'specificity']:.3f}")
    report.append(f"- **Overfitting Score**: {comparison_df.loc[best_method, 'overfitting_score']}")
    report.append("")
    
    # Improvements analysis
    report.append("## Improvements Analysis")
    report.append("")
    for method, improvements in improvements_df.iterrows():
        report.append(f"### {method}")
        report.append("")
        report.append(f"- **Accuracy Improvement**: {improvements['accuracy_improvement']:+.3f}")
        report.append(f"- **Specificity Improvement**: {improvements['specificity_improvement']:+.3f}")
        report.append(f"- **Overfitting Reduction**: {improvements['overfitting_reduction']:+.0f}")
        report.append(f"- **HC Probability Reduction**: {improvements['hc_prob_reduction']:+.3f}")
        report.append(f"- **Confidence Improvement**: {improvements['confidence_improvement']:+.3f}")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("Based on the analysis:")
    report.append("")
    
    # Find best methods for different criteria
    best_specificity = improvements_df['specificity_improvement'].idxmax()
    best_overfitting = improvements_df['overfitting_reduction'].idxmax()
    best_overall = improvements_df['accuracy_improvement'].idxmax()
    
    report.append(f"1. **Best for Specificity**: {best_specificity} (+{improvements_df.loc[best_specificity, 'specificity_improvement']:.3f})")
    report.append(f"2. **Best for Overfitting Reduction**: {best_overfitting} (-{improvements_df.loc[best_overfitting, 'overfitting_reduction']:.0f} cases)")
    report.append(f"3. **Best Overall Performance**: {best_overall} (+{improvements_df.loc[best_overall, 'accuracy_improvement']:.3f})")
    report.append("")
    
    # Save report
    with open(f'{output_dir}/enhancement_analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("📄 Detailed report saved to: enhancement_analysis_report.md")

def main():
    """Main analysis function"""
    # Configuration
    original_file = "results/consolidated/mri/l2o_predictions_improved.csv"
    enhanced_dir = "results/consolidated/mri"
    output_dir = "results/consolidated/mri/enhanced_analysis"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find enhanced result files
    enhanced_files = glob.glob(f"{enhanced_dir}/enhanced_*.csv")
    enhanced_files = [f for f in enhanced_files if 'fold_performances' not in f and 'test_enhanced' not in f]
    
    if not enhanced_files:
        print("❌ No enhanced result files found!")
        print(f"Expected files in: {enhanced_dir}")
        return
    
    print(f"📁 Found {len(enhanced_files)} enhanced result files")
    
    # Compare results
    comparison_df, improvements_df = compare_results(original_file, enhanced_files)
    
    # Save comparison data
    comparison_df.to_csv(f'{output_dir}/comparison_results.csv')
    improvements_df.to_csv(f'{output_dir}/improvements_analysis.csv')
    
    print("\n📊 Comparison Results:")
    print(comparison_df.round(3))
    
    print("\n🚀 Improvements Analysis:")
    print(improvements_df.round(3))
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(comparison_df, improvements_df, output_dir)
    
    # Generate detailed report
    print("\n📄 Generating detailed report...")
    generate_detailed_report(comparison_df, improvements_df, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 