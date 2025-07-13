# FILE: scripts/new_mri/prepare_sij_dataset.py
# PURPOSE: To create a clean, analysis-ready dataset containing only Sacroiliac Joint (SIJ) images.
# VERSION: 3.0 - Definitive Path Logic

import os
import shutil
import glob

print("--- Starting Clean Sacroiliac Joint (SIJ) Dataset Preparation (v3.0) ---")

# --- Configuration ---
# DEFINITIVE PATH LOGIC: Assumes this script is in a subfolder of 'scripts'
# The project root is two levels up from this file.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"Project root identified as: {project_root}")

# Source directory where all modified PNGs/JPGs are stored
source_dir = os.path.join(project_root, 'data', 'mri_image_modified')

# Target directory for the new, clean dataset
target_dir = os.path.join(project_root, 'data', 'sij_analysis')

# Define source sub-folders based on your directory tree
as_sij_source_folder = 'png_AS_sij'
healthy_sij_source_folder = 'png_healthy_sij'

# Define target sub-folders for the clean dataset
as_sij_target_folder = os.path.join(target_dir, 'AS_SIJ')
healthy_sij_target_folder = os.path.join(target_dir, 'Healthy_SIJ')

# --- 1. Clean and Create Target Directory ---
if os.path.exists(target_dir):
    print(f"Target directory '{target_dir}' already exists. Removing it for a clean start.")
    shutil.rmtree(target_dir)
os.makedirs(as_sij_target_folder)
os.makedirs(healthy_sij_target_folder)
print(f"Successfully created clean target directory structure at: '{target_dir}'")

# --- 2. Copy AS Sacroiliac Joint (SIJ) Images ---
source_as_path = os.path.join(source_dir, as_sij_source_folder)
if not os.path.isdir(source_as_path):
    raise FileNotFoundError(f"CRITICAL ERROR: Source folder for AS SIJ not found at: {source_as_path}")

as_files = glob.glob(os.path.join(source_as_path, 'SIJ_*.png'))
if not as_files:
    print(f"WARNING: No AS SIJ .png files were found in {source_as_path}. Please check the folder content.")
else:
    for f in as_files:
        shutil.copy(f, as_sij_target_folder)
    print(f"Successfully copied {len(as_files)} AS SIJ images to '{as_sij_target_folder}'")

# --- 3. Copy Healthy Sacroiliac Joint (SIJ) Images ---
source_healthy_path = os.path.join(source_dir, healthy_sij_source_folder)
if not os.path.isdir(source_healthy_path):
    raise FileNotFoundError(f"CRITICAL ERROR: Source folder for Healthy SIJ not found at: {source_healthy_path}")

healthy_files = glob.glob(os.path.join(source_healthy_path, '*.jpg'))
if not healthy_files:
    print(f"WARNING: No Healthy SIJ .jpg files were found in {source_healthy_path}. Please check the folder content.")
else:
    for f in healthy_files:
        shutil.copy(f, healthy_sij_target_folder)
    print(f"Successfully copied {len(healthy_files)} Healthy SIJ images to '{healthy_sij_target_folder}'")

print("\n--- Dataset Preparation Complete! ---")
print(f"Your clean, analysis-ready dataset is now located at: '{target_dir}'")
print("You may now proceed to run the analysis script.")