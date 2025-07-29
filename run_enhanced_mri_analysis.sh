#!/bin/bash

# Enhanced MRI Analysis Script
# This script runs the improved MRI analysis with all enhancements

echo "🚀 Starting Enhanced MRI Analysis with Comprehensive Improvements"
echo "================================================================"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
cd "$(dirname "$0")"

# Create output directories
mkdir -p results/consolidated/mri/models
mkdir -p results/consolidated/mri/enhanced_analysis

# Configuration
DATA_ROOT="data"
OUTPUT_DIR="results/consolidated/mri/enhanced_analysis"
SEED=42

echo "📊 Running Enhanced MRI Analysis..."
echo "Configuration:"
echo "  - Data root: $DATA_ROOT"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Random seed: $SEED"
echo ""

# Test 1: Basic enhanced analysis with strong regularization
echo "🔬 Test 1: Enhanced Analysis with Strong Regularization"
python src/mri_src/analysis/make_l2o_predictions_improved.py \
    --data-root "$DATA_ROOT" \
    --out-csv "$OUTPUT_DIR/enhanced_basic.csv" \
    --seed $SEED \
    --device cpu \
    --save-models

echo "✅ Test 1 completed"
echo ""

# Test 2: Ensemble analysis
echo "🔬 Test 2: Ensemble Analysis with Multiple Classifiers"
python src/mri_src/analysis/make_l2o_predictions_improved.py \
    --data-root "$DATA_ROOT" \
    --out-csv "$OUTPUT_DIR/enhanced_ensemble.csv" \
    --seed $SEED \
    --device cpu \
    --use-ensemble \
    --save-models

echo "✅ Test 2 completed"
echo ""

# Test 3: Data augmentation
echo "🔬 Test 3: Data Augmentation Analysis"
python src/mri_src/analysis/make_l2o_predictions_improved.py \
    --data-root "$DATA_ROOT" \
    --out-csv "$OUTPUT_DIR/enhanced_augmentation.csv" \
    --seed $SEED \
    --device cpu \
    --use-augmentation \
    --save-models

echo "✅ Test 3 completed"
echo ""

# Test 4: Full enhanced analysis with all improvements
echo "🔬 Test 4: Full Enhanced Analysis (All Improvements)"
python src/mri_src/analysis/make_l2o_predictions_improved.py \
    --data-root "$DATA_ROOT" \
    --out-csv "$OUTPUT_DIR/enhanced_full.csv" \
    --seed $SEED \
    --device cpu \
    --use-ensemble \
    --use-augmentation \
    --outlier-detection \
    --bootstrap-ci \
    --feature-selection variance \
    --n-features 100 \
    --save-models

echo "✅ Test 4 completed"
echo ""

# Test 5: Feature selection comparison
echo "🔬 Test 5: Feature Selection Comparison"
for method in variance kbest pca; do
    echo "  Testing $method feature selection..."
    python src/mri_src/analysis/make_l2o_predictions_improved.py \
        --data-root "$DATA_ROOT" \
        --out-csv "$OUTPUT_DIR/enhanced_${method}_features.csv" \
        --seed $SEED \
        --device cpu \
        --feature-selection $method \
        --n-features 100
done

echo "✅ Test 5 completed"
echo ""

# Generate comparison report
echo "📈 Generating Comparison Report..."
python -c "
import pandas as pd
import glob
import os

# Load all results
results_files = glob.glob('$OUTPUT_DIR/enhanced_*.csv')
results_data = {}

for file in results_files:
    if 'fold_performances' not in file:
        name = os.path.basename(file).replace('.csv', '').replace('enhanced_', '')
        df = pd.read_csv(file)
        
        # Calculate metrics
        as_correct = sum((df['y_true'] == 1) & (df['prob_raw'] > 0.5))
        hc_correct = sum((df['y_true'] == 0) & (df['prob_raw'] < 0.5))
        total_correct = as_correct + hc_correct
        
        results_data[name] = {
            'total_subjects': len(df),
            'as_correct': as_correct,
            'hc_correct': hc_correct,
            'total_correct': total_correct,
            'accuracy': total_correct / len(df),
            'as_accuracy': as_correct / sum(df['y_true'] == 1),
            'hc_accuracy': hc_correct / sum(df['y_true'] == 0),
            'avg_prob_as': df[df['y_true'] == 1]['prob_raw'].mean(),
            'avg_prob_hc': df[df['y_true'] == 0]['prob_raw'].mean()
        }

# Create comparison dataframe
comparison_df = pd.DataFrame(results_data).T
comparison_df.to_csv('$OUTPUT_DIR/enhancement_comparison.csv')

print('Comparison Report:')
print(comparison_df.round(3))
print(f'\\nDetailed results saved to: $OUTPUT_DIR/enhancement_comparison.csv')
"

echo ""
echo "🎉 Enhanced MRI Analysis Complete!"
echo "=================================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key improvements implemented:"
echo "✅ Strong regularization (C=0.01, elasticnet penalty)"
echo "✅ Ensemble of multiple classifiers"
echo "✅ Comprehensive data augmentation"
echo "✅ Outlier detection and removal"
echo "✅ Feature selection and dimensionality reduction"
echo "✅ Bootstrap confidence intervals"
echo "✅ Robust scaling"
echo "✅ Enhanced performance metrics"
echo ""
echo "Check the comparison report for detailed results!" 