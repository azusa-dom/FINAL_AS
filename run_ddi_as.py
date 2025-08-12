#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ddi_as.py

DDI-AS main orchestration script.
Provides end-to-end training, evaluation and ensemble fusion with
reproducible folder scaffolding and structured logging.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Add project root to sys.path for imports
sys.path.append(str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR, 
    CLINICAL_RESULTS_DIR, MRI_RESULTS_DIR, ENSEMBLE_RESULTS_DIR,
    MODEL_CONFIG, PERFORMANCE_TARGETS
)
from src.common.logging_utils import create_logger


def run_command(command: str, description: str, logger: logging.Logger) -> bool:
    """Run a shell command with structured logging."""
    logger.info("%s", description)
    logger.debug("Executing: %s", command)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.debug(result.stdout)
        logger.info("%s - done", description)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("%s - failed", description)
        logger.error("stderr: %s", e.stderr)
        return False


def setup_environment(logger: logging.Logger):
    """Create directory scaffolding."""
    for directory in [DATA_DIR, RESULTS_DIR, CLINICAL_RESULTS_DIR, MRI_RESULTS_DIR, ENSEMBLE_RESULTS_DIR, RESULTS_DIR / "logs"]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", directory)
    logger.info("Environment ready")


def train_clinical_net(data_path: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """Train ClinicalNet pipeline."""
    if not data_path.exists():
        logger.error("Clinical data path not found: %s", data_path)
        return False
    command = (
        f"python src/clinical/training_clinical_data/train_clinical_ensemble.py "
        f"--data_dir {data_path} --output_dir {output_dir} "
        f"--n_folds {MODEL_CONFIG['clinical']['cv_folds']} --save_models"
    )
    return run_command(command, "Train ClinicalNet", logger)


def train_imaging_net(as_dir: Path, healthy_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """Train ImagingNet pipeline."""
    if not as_dir.exists():
        logger.error("AS MRI path not found: %s", as_dir)
        return False
    if not healthy_dir.exists():
        logger.error("Healthy MRI path not found: %s", healthy_dir)
        return False
    command = (
        f"python src/mri/analysis/mri_subject_level_auc.py "
        f"--as-dir {as_dir} --healthy-dir {healthy_dir} "
        f"--n-splits {MODEL_CONFIG['imaging']['cv_folds']} --n-bootstrap 1000 "
        f"--batch-size 16 --seed 42 --device cpu"
    )
    return run_command(command, "Train ImagingNet", logger)


def train_ensemble(clinical_results: Path, imaging_results: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """Train ensemble (late fusion)."""
    if not clinical_results.exists():
        logger.error("ClinicalNet oof predictions not found: %s", clinical_results)
        return False
    command = (
        f"python src/ensemble/train_ensemble.py --clinical_data {clinical_results} "
        f"--imaging_data {imaging_results} --output_dir {output_dir} "
        f"--n_folds_clinical {MODEL_CONFIG['clinical']['cv_folds']} "
        f"--n_folds_imaging {MODEL_CONFIG['imaging']['cv_folds']}"
    )
    return run_command(command, "Train Ensemble", logger)


def evaluate_models(logger: logging.Logger):
    """Report performance targets (placeholder for full eval)."""
    clinical_target = PERFORMANCE_TARGETS['clinical']['auroc']
    imaging_target = PERFORMANCE_TARGETS['imaging']['auroc']
    ensemble_target = PERFORMANCE_TARGETS['ensemble']['auroc']
    logger.info("Performance targets → clinical: %.3f | imaging: %.3f | ensemble: %.3f",
                clinical_target, imaging_target, ensemble_target)


def generate_figures(logger: logging.Logger):
    """Optional figure generation hooks (no-op by default)."""
    figure_scripts = []  # add custom scripts if needed
    for script in figure_scripts:
        if Path(script).exists():
            run_command(f"python {script}", f"Generate figure: {script}", logger)
        else:
            logger.warning("Figure script not found: %s", script)
    logger.info("Figure generation completed")


def main():
    parser = argparse.ArgumentParser(description='DDI-AS end-to-end pipeline')
    parser.add_argument('--mode', choices=['full', 'clinical', 'imaging', 'ensemble', 'evaluate'], 
                       default='full', help='运行模式')
    parser.add_argument('--clinical_data', type=str, default='data/clinical', 
                        help='Path to clinical data directory')
    parser.add_argument('--as_data', type=str, default='data/mri/as', 
                        help='Path to AS MRI directory')
    parser.add_argument('--healthy_data', type=str, default='data/mri/healthy', 
                        help='Path to healthy MRI directory')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Output directory')
    parser.add_argument('--skip_setup', action='store_true', 
                        help='Skip environment setup')
    
    args = parser.parse_args()
    
    logs_dir = RESULTS_DIR / "logs"
    logger = create_logger("ddi_as", logs_dir)
    logger.info("DDI-AS: Dual Diagnostic Intelligence for Ankylosing Spondylitis")
    
    # 设置环境
    if not args.skip_setup:
        setup_environment(logger)
    
    # 根据模式运行相应的流程
    if args.mode in ['full', 'clinical']:
        clinical_data_path = Path(args.clinical_data)
        clinical_output = Path(args.output_dir) / "clinical"
        
        if not train_clinical_net(clinical_data_path, clinical_output, logger):
            logger.error("Abort: ClinicalNet failed")
            return
    
    if args.mode in ['full', 'imaging']:
        as_data_path = Path(args.as_data)
        healthy_data_path = Path(args.healthy_data)
        mri_output = Path(args.output_dir) / "mri"
        
        if not train_imaging_net(as_data_path, healthy_data_path, mri_output, logger):
            logger.error("Abort: ImagingNet failed")
            return
    
    if args.mode in ['full', 'ensemble']:
        clinical_results = Path(args.output_dir) / "clinical" / "ensemble_predictions.csv"
        imaging_results = Path(args.output_dir) / "mri" / "predictions.csv"
        ensemble_output = Path(args.output_dir) / "ensemble"
        
        if not train_ensemble(clinical_results, imaging_results, ensemble_output, logger):
            logger.error("Abort: Ensemble failed")
            return
    
    if args.mode in ['full', 'evaluate']:
        evaluate_models(logger)
        generate_figures(logger)
    
    logger.info("Pipeline finished")
    logger.info("Results → clinical: %s", CLINICAL_RESULTS_DIR)
    logger.info("Results → mri: %s", MRI_RESULTS_DIR)
    logger.info("Results → ensemble: %s", ENSEMBLE_RESULTS_DIR)
    logger.info("Paper (release): %s", PROJECT_ROOT / "docs/paper/final_release.tex")


if __name__ == "__main__":
    main() 