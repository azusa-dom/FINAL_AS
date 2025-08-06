# Accurate Data Summary Report

Generated on: 2025-08-01T21:40:55.569046

## Data Sources
- **Clinical Data**: Raw_Lab_Dataset.csv (Mahdi et al., 2025)
- **MRI Data**: Radiopaedia.org public repository
- **Processing Script**: create_real_performance_tables.py

## Key Findings
- **Best Clinical Model**: Gradient Boosting (AUROC: 0.938 ± 0.003)
- **MRI Performance**: AUROC: 0.830 (p=0.017)
- **Optimal Threshold**: 0.62 for MRI model
- **Feature Engineering**: 20 features from 14 original variables

## Validation Methods
- **Clinical**: Stratified 5-fold CV
- **MRI**: Leave-Two-Out CV with permutation testing

## Files Generated
- clinical_performance_data.json/csv
- dataset_characteristics.json/csv
- feature_engineering_details.json/csv
- cross_validation_results.json/csv
- feature_importance.json/csv
- summary_report.json

This data represents the final, accurate results after complete feature engineering and script processing.
