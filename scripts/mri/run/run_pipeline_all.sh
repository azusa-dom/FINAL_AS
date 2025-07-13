#!/bin/bash
# This script automates the entire experimental workflow for the Dual-Engine AS Diagnosis Framework.
# It ensures that all steps are executed in the correct order, from data preparation to model training and evaluation.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DATA_DIR="data"
PROCESSED_DIR="${DATA_DIR}/processed_clinical_data"
MODEL_DIR="models/clinical_model"
RESULT_DIR="results"

# --- Workflow ---

echo "================================================="
echo "=== STAGE 1: Preprocessing Clinical Data      ==="
echo "================================================="
python scripts/preprocess_clinical.py --input "${DATA_DIR}/rheumatic_autoimmune_disease.csv" --out_dir "${PROCESSED_DIR}"

echo "================================================="
echo "=== STAGE 2: Training Clinical Model          ==="
echo "================================================="
python src/train.py --data_dir "${PROCESSED_DIR}" --model_dir "${MODEL_DIR}"

echo "================================================="
echo "=== STAGE 3: Exploring MRI Features           ==="
echo "================================================="
# Note: The --data_dir for this script points to the small sample of MRI images.
python src/explore_mri_features.py --data_dir "AS_Finetune_Data_balanced" --out_dir "${RESULT_DIR}/mri_features"

echo "================================================="
echo "=== STAGE 4: Evaluating Clinical Predictions  ==="
echo "================================================="
# Note: This evaluates the predictions generated during the training stage.
python src/evaluate.py --preds_dir "${MODEL_DIR}/clinical_preds"

echo "================================================="
echo "=== Workflow finished successfully.           ==="
echo "=== Check the '${RESULT_DIR}' and '${MODEL_DIR}' directories. ==="
echo "================================================="
