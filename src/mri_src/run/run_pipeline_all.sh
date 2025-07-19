#!/usr/bin/env bash
# rushall.sh: End-to-end pipeline for AS project
# Usage: bash rushall.sh [options]
# Options:
#   --data-dir      Path to data directory (default: ./data)
#   --checkpoints   Path to checkpoints directory (default: ./checkpoints)
#   --results-dir   Path to results directory (default: ./results)
#   --scripts-dir   Path to scripts directory (default: ./scripts)
#   --src-dir       Path to src directory (default: ./src)
#   --help          Show this help message

# Default directories
DATA_DIR="./data"
CHECKPOINTS_DIR="./checkpoints"
RESULTS_DIR="./results"
SCRIPTS_DIR="./scripts"
SRC_DIR="./src"

# Parse arguments
temp_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --checkpoints)
      CHECKPOINTS_DIR="$2"; shift 2;;
    --results-dir)
      RESULTS_DIR="$2"; shift 2;;
    --scripts-dir)
      SCRIPTS_DIR="$2"; shift 2;;
    --src-dir)
      SRC_DIR="$2"; shift 2;;
    --help)
      sed -n '2,8p' "$0"; exit 0;;
    *)
      temp_args+=("$1"); shift;;
  esac
done
set -- "${temp_args[@]}"

echo "Starting end-to-end pipeline"

# 1. Clinical preprocessing
echo "[Clinical] Preprocessing clinical data"
python "$SCRIPTS_DIR/clinical/preprocess_clinical.py" --input "$DATA_DIR/raw_lab_data/Raw_Lab_Dataset.csv" --output "$RESULTS_DIR/clinical/data_results/clinical_processed.csv"

# 2. Build balanced clinical dataset
echo "[Clinical] Building balanced clinical dataset"
python "$SCRIPTS_DIR/clinical/build_balanced_dataset.py" --input "$RESULTS_DIR/clinical/data_results/clinical_processed.csv" --output-dir "$RESULTS_DIR/clinical/clinical_data_fold"

# 3. Clinical training and evaluation (5-fold)
echo "[Clinical] Training and evaluating clinical models"
python "$SCRIPTS_DIR/clinical/check_clinical_quality.py" --data-dir "$RESULTS_DIR/clinical/clinical_data_fold"

# 4. MRI conversion
echo "[MRI] Converting DICOM to PNG"
bash "$SCRIPTS_DIR/mri/conversion/raw2nii_sij_new.py" --dicom-dir "$DATA_DIR/288_dicom" --output-dir "$DATA_DIR/288_images_png"

# 5. MRI preprocessing
echo "[MRI] Preprocessing MRI data"
python "$SCRIPTS_DIR/mri/preprocessing/preprocess_final.py" --input-dir "$DATA_DIR/288_images_png" --output-dir "$RESULTS_DIR/mri/processed_images"

# 6. Prepare MRI folds
echo "[MRI] Preparing MRI folds"
python "$SRC_DIR/preprocessing/prepare_mri_folds.py" --data-dir "$RESULTS_DIR/mri/processed_images" --folds 5 --output-dir "$RESULTS_DIR/mri/mri_folds"

# 7. MRI training
echo "[MRI] Training MRI model"
python "$SRC_DIR/training/train_mri.py" --folds-dir "$RESULTS_DIR/mri/mri_folds" --checkpoints-dir "$CHECKPOINTS_DIR" --results-dir "$RESULTS_DIR/mri"

# 8. MRI inference
echo "[MRI] Running inference on MRI data"
python "$SRC_DIR/inference/predict_mri.py" --model-dir "$CHECKPOINTS_DIR" --data-dir "$RESULTS_DIR/mri/processed_images" --output "$RESULTS_DIR/mri/predictions.csv"

# 9. Post-process and generate reports
echo "[Postprocess] Computing SHAP and DCA"
python "$SCRIPTS_DIR/postprocess/shap_compute_summary.py" --input "$RESULTS_DIR/mri/predictions.csv" --output-dir "$RESULTS_DIR/clinical/shap_summary"

# 10. Visualizations
echo "[Visualization] Generating MRI embedding and Grad-CAMs"
python "$SCRIPTS_DIR/mri/visualization/viz_tsne_as_vs_healthy.py" --input-dir "$RESULTS_DIR/mri/processed_images" --output "$RESULTS_DIR/mri/embedding_viz/tsne_as_vs_healthy.png"
python "$SCRIPTS_DIR/mri/gradcam/As_run_sij_gradcam_analysis.py" --input-dir "$RESULTS_DIR/mri/processed_images" --output-dir "$RESULTS_DIR/mri/grad_cam/as"

echo "Pipeline completed successfully!"
