#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_improvements.py

Run improved MRI and clinical model training to address overfitting and calibration issues.

This script will:
1. Retrain MRI model with stronger regularization and ensemble methods
2. Retrain clinical model with ensemble methods and better calibration
3. Generate comparison reports
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        print(f"Error output: {e.stderr}")
        return False

def compare_results(original_file, improved_file, output_file):
    """Compare original and improved results"""
    print(f"\nComparing results...")
    
    try:
        # Load original results
        orig_df = pd.read_csv(original_file)
        print(f"Original results: {len(orig_df)} samples")
        
        # Load improved results
        improved_df = pd.read_csv(improved_file)
        print(f"Improved results: {len(improved_df)} samples")
        
        # Compare metrics
        comparison = {
            'metric': [],
            'original': [],
            'improved': [],
            'improvement': []
        }
        
        # Check for overfitting in MRI results
        if 'y_true' in orig_df.columns and 'prob_raw' in orig_df.columns:
            # Count misclassified healthy controls
            orig_hc_misclassified = len(orig_df[(orig_df['y_true'] == 0) & (orig_df['prob_raw'] > 0.9)])
            improved_hc_misclassified = len(improved_df[(improved_df['y_true'] == 0) & (improved_df['prob_raw'] > 0.9)])
            
            comparison['metric'].extend(['HC Misclassified (>90%)', 'HC Misclassified (>90%)'])
            comparison['original'].extend([orig_hc_misclassified, orig_hc_misclassified])
            comparison['improved'].extend([improved_hc_misclassified, improved_hc_misclassified])
            comparison['improvement'].extend([
                f"{orig_hc_misclassified - improved_hc_misclassified}",
                f"{(orig_hc_misclassified - improved_hc_misclassified) / max(orig_hc_misclassified, 1) * 100:.1f}%"
            ])
        
        # Save comparison
        comp_df = pd.DataFrame(comparison)
        comp_df.to_csv(output_file, index=False)
        print(f"Comparison saved to {output_file}")
        
        return True
    except Exception as e:
        print(f"Error comparing results: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run improved model training')
    parser.add_argument('--mri_data_root', required=True, help='Path to MRI data root')
    parser.add_argument('--clinical_data_dir', required=True, help='Path to clinical processed data')
    parser.add_argument('--output_dir', default='results/improvements', help='Output directory')
    parser.add_argument('--skip_mri', action='store_true', help='Skip MRI model retraining')
    parser.add_argument('--skip_clinical', action='store_true', help='Skip clinical model retraining')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting Model Improvements")
    print(f"MRI Data: {args.mri_data_root}")
    print(f"Clinical Data: {args.clinical_data_dir}")
    print(f"Output: {args.output_dir}")
    
    # 1. Retrain MRI model with improvements
    if not args.skip_mri:
        print("\n📊 Retraining MRI Model with Improvements...")
        
        # Run improved MRI training
        mri_cmd = f"""python {os.path.join(project_root, 'src/mri_src/analysis/make_l2o_predictions_improved.py')} \
            --data-root {args.mri_data_root} \
            --out-csv {args.output_dir}/l2o_predictions_improved.csv \
            --use-ensemble \
            --save-models"""
        
        if run_command(mri_cmd, "MRI Model Retraining"):
            # Compare MRI results
            original_mri = "results/mri_analysis/l2o_predictions.csv"
            improved_mri = f"{args.output_dir}/l2o_predictions_improved.csv"
            
            if os.path.exists(original_mri):
                compare_results(
                    original_mri, 
                    improved_mri, 
                    f"{args.output_dir}/mri_comparison.csv"
                )
        else:
            print("❌ MRI model retraining failed")
    
    # 2. Retrain clinical model with ensemble
    if not args.skip_clinical:
        print("\n🏥 Retraining Clinical Model with Ensemble...")
        
        # Run improved clinical training
        clinical_cmd = f"""python {os.path.join(project_root, 'src/clinical_data_src/training_clinical_data/train_clinical_ensemble.py')} \
            --data_dir {args.clinical_data_dir} \
            --output_dir {args.output_dir}/clinical_ensemble \
            --save_models"""
        
        if run_command(clinical_cmd, "Clinical Model Ensemble Training"):
            print("✅ Clinical ensemble training completed")
        else:
            print("❌ Clinical model retraining failed")
    
    # 3. Generate summary report
    print("\n📋 Generating Summary Report...")
    
    summary = {
        'improvement': [],
        'original_performance': [],
        'improved_performance': [],
        'notes': []
    }
    
    # MRI improvements summary
    if not args.skip_mri and os.path.exists(f"{args.output_dir}/l2o_predictions_improved.csv"):
        try:
            improved_mri_df = pd.read_csv(f"{args.output_dir}/l2o_predictions_improved.csv")
            
            # Check for overfitting
            hc_misclassified = len(improved_mri_df[(improved_mri_df['y_true'] == 0) & (improved_mri_df['prob_raw'] > 0.9)])
            
            summary['improvement'].append('MRI Overfitting Reduction')
            summary['original_performance'].append('2/2 HC misclassified (>99%)')
            summary['improved_performance'].append(f'{hc_misclassified}/2 HC misclassified')
            summary['notes'].append('Stronger regularization and ensemble methods')
            
        except Exception as e:
            print(f"Error analyzing MRI improvements: {e}")
    
    # Clinical improvements summary
    if not args.skip_clinical and os.path.exists(f"{args.output_dir}/clinical_ensemble/ensemble_metrics.csv"):
        try:
            clinical_metrics = pd.read_csv(f"{args.output_dir}/clinical_ensemble/ensemble_metrics.csv")
            
            if 'ensemble' in clinical_metrics.index:
                ensemble_auc = clinical_metrics.loc['ensemble', 'auc']
                summary['improvement'].append('Clinical Model Ensemble')
                summary['original_performance'].append('AUROC: 0.924')
                summary['improved_performance'].append(f'AUROC: {ensemble_auc:.3f}')
                summary['notes'].append('Ensemble of LightGBM, XGBoost, Neural Network, and Logistic Regression')
                
        except Exception as e:
            print(f"Error analyzing clinical improvements: {e}")
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{args.output_dir}/improvements_summary.csv", index=False)
    
    print(f"\n📊 Summary Report:")
    print(summary_df.to_string(index=False))
    print(f"\n📁 All results saved to: {args.output_dir}")
    
    # 4. Recommendations
    print("\n💡 Recommendations:")
    print("1. For MRI model: Consider collecting more data (at least 50-100 subjects)")
    print("2. For clinical model: The ensemble approach should provide better calibration")
    print("3. Monitor performance on external validation sets")
    print("4. Consider using uncertainty quantification methods")

if __name__ == "__main__":
    main() 