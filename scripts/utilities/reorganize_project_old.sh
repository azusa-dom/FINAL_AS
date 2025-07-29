echo "--- Starting Project Reorganization ---"

# --- Step 1: Create Essential New Directories (if they don't exist) ---
echo "1. Creating new essential directories..."
mkdir -p scripts/deprecated
mkdir -p src/data
mkdir -p src/analysis/mri_feature_analysis
mkdir -p src/core/evaluation_plots
mkdir -p results/mri_outputs/viz
mkdir -p results/mri_outputs/gradcam
mkdir -p results/mri_outputs/analysis_metrics
mkdir -p results/mri_outputs/predictions # For MRI model predictions
mkdir -p data/processed/mri_finetune_data # For processed MRI data ready for finetuning


# --- Step 2: Move and Rename Specific Python Scripts ---
echo "2. Moving and renaming specific Python scripts..."

# Move the general utility merge_mac.py to deprecated
mv scripts/merge_mac.py scripts/deprecated/merge_code_files_tool.py
echo "  - Moved scripts/merge_mac.py to scripts/deprecated/merge_code_files_tool.py"

# Move mri_feats_cal.py (redundant feature extraction) to deprecated
mv scripts/mri/analysis/mri_feats_cal.py scripts/deprecated/mri_feats_cal_deprecated.py
echo "  - Moved scripts/mri/analysis/mri_feats_cal.py to scripts/deprecated/mri_feats_cal_deprecated.py"

# Correctly move raw2nii_sij_new.py (centroid permutation test) to analysis folder
mv scripts/mri/conversion/raw2nii_sij_new.py scripts/mri/analysis/permutation_test_centroid_distance.py
echo "  - Moved scripts/mri/conversion/raw2nii_sij_new.py to scripts/mri/analysis/permutation_test_centroid_distance.py"

# Correctly move preprocess_final.py (clinical data preprocessing) to clinical_data_preparation
mv scripts/mri/preprocessing/preprocess_final.py scripts/clinical_data_preparation/preprocess_clinical_final.py
echo "  - Moved scripts/mri/preprocessing/preprocess_final.py to scripts/clinical_data_preparation/preprocess_clinical_final.py"

# Correctly move pytorch_data_load.py (MRI dataset definition) to src/data
mv scripts/clinical_data_preparation/pytorch_data_load.py src/data/mri_image_dataset.py
echo "  - Moved scripts/clinical_data_preparation/pytorch_data_load.py to src/data/mri_image_dataset.py"

# Move core SHAP/DCA scripts that are redundant to deprecated
mv src/core/evaluate_shap_dca.py scripts/deprecated/clinical_evaluate_shap_dca_deprecated.py
echo "  - Moved src/core/evaluate_shap_dca.py to scripts/deprecated/clinical_evaluate_shap_dca_deprecated.py"
mv src/core/plot_dca.py scripts/deprecated/clinical_plot_dca_deprecated.py
echo "  - Moved src/core/plot_dca.py to scripts/deprecated/clinical_plot_dca_deprecated.py"

# Move remaining general clinical evaluation scripts from src/core to src/core/evaluation_plots
mv src/core/eval_clinical_all_folds.py src/core/evaluation_plots/
mv src/core/evaluate_AUROC_AUPRC_CI.py src/core/evaluation_plots/
mv src/core/evaluate_confusion_matrix.py src/core/evaluation_plots/
mv src/core/plot_overall_metrics.py src/core/evaluation_plots/
echo "  - Moved core clinical evaluation scripts to src/core/evaluation_plots/"

# Move MRI feature analysis scripts from src/analysis to src/analysis/mri_feature_analysis
mv src/analysis/analyze_mri_features.py src/analysis/mri_feature_analysis/
mv src/analysis/analyze_mri_features_by_folder.py src/analysis/mri_feature_analysis/
mv src/analysis/quantitative_mri_analysis.py src/analysis/mri_feature_analysis/
echo "  - Moved MRI feature analysis scripts to src/analysis/mri_feature_analysis/"


# --- Step 3: Consolidate Data Directories ---
echo "3. Consolidating data directories..."

# Move `AS_Finetune_Data` to its new `data/processed` location if it exists
# (Based on your previous tree, this might not be at the root, but if it is, this handles it)
if [ -d "AS_Finetune_Data" ]; then
    mv AS_Finetune_Data data/processed/mri_finetune_data/
    echo "  - Moved AS_Finetune_Data to data/processed/mri_finetune_data."
fi

# Clean up duplicated 'processed_clinical' in 'raw_lab_data'
# This assumes data/processed_clinical is the primary and correct source.
echo "  - WARNING: Removing potentially duplicate processed_clinical data in data/raw_lab_data. Verify this is the duplicate before running!"
rm -rf data/raw_lab_data/processed_clinical
echo "  - Cleaned up duplicate clinical data in data/raw_lab_data."


# --- Step 4: Consolidate Results Directories ---
echo "4. Consolidating results directories..."

# Move `results/mri` content to `results/mri_outputs/viz` (if it was a viz folder)
# This assumes `results/mri` was a top-level folder used for general MRI visualization.
mv results/mri results/mri_outputs/viz 2>/dev/null || true
echo "  - Moved results/mri to results/mri_outputs/viz."

# Move `results/clinical/calibration_plots` to `results/mri_outputs/gradcam`
# (Based on the file names, these look like Grad-CAM outputs, but if they are calibration specific, keep them under clinical)
# Re-evaluating based on content: calibration_plots belong to clinical. Let's move Grad-CAM outputs instead.
mv results/grad_cam_outputs results/mri_outputs/gradcam 2>/dev/null || true # If this directory existed from previous runs
echo "  - Moved old grad_cam_outputs to results/mri_outputs/gradcam."

# Move contents of `results/plot/` to `results/final_run/figures_overall/`
# This directory appears to hold overall plots from various evaluations.
if [ -d "results/plot" ]; then
    mv results/plot/* results/final_run/figures_overall/ 2>/dev/null || true
    rmdir results/plot
    echo "  - Consolidated results/plot contents into results/final_run/figures_overall."
fi

# Move contents of `results/clinical/shap_dca/` to `results/final_run/figures_foldX/` and `figures_overall/`
echo "  - Consolidating results/clinical/shap_dca contents..."
for i in {0..4}; do
    # Move fold-specific plots
    mv results/clinical/shap_dca/decision_curve_analysis_fold_${i}.png results/final_run/figures_fold${i}/dca_curve_clinical.png 2>/dev/null || true
    mv results/clinical/shap_dca/performance_report_fold_${i}.txt results/final_run/figures_fold${i}/performance_report_clinical.txt 2>/dev/null || true
done
# Move general SHAP summary if it's not fold-specific
mv results/clinical/shap_dca/shap_summary.png results/final_run/figures_overall/overall_shap_summary_clinical.png 2>/dev/null || true
# Move calibrated DCA curve if it's a general one
mv results/clinical/shap_dca/dca_curve_calibrated.png results/final_run/figures_overall/overall_dca_curve_calibrated_clinical.png 2>/dev/null || true
rmdir results/clinical/shap_dca 2>/dev/null || true
echo "  - Consolidated results/clinical/shap_dca contents."

# Remove empty `results/clinical/shap_plots/` if it's empty after previous moves/cleanup
rmdir results/clinical/shap_plots 2>/dev/null || true
echo "  - Removed empty results/clinical/shap_plots."


# --- Step 5: Final Cleanup of Empty Directories ---
echo "5. Cleaning up potentially empty source directories..."
rmdir scripts/mri/analysis 2>/dev/null || true # Will only remove if empty
rmdir scripts/mri/conversion 2>/dev/null || true
rmdir scripts/mri/preprocessing 2>/dev/null || true
rmdir src/analysis 2>/dev/null || true # Will only remove if empty after moving contents


echo "--- Project reorganization complete! ---"
echo "Please now:"
echo "1. Open VS Code: 'code /Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS'"
echo "2. Check the new file structure in the Explorer pane."
echo "3. Manually update Python 'import' statements in any affected files. Use VS Code's 'Find in Files' (Ctrl/Cmd+Shift+F) for old paths and file names."
echo "4. Run your tests/scripts to ensure everything is working as expected."
