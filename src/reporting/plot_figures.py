#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_figures.py

Generate publication-ready figures from authoritative processed results under
`data/processed/` and save them into `results/figures_generated/`.

No heuristics or synthetic values are created; only processed summary files are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="whitegrid", font_scale=1.1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_model_metrics(processed_dir: Path, out_dir: Path) -> None:
    perf = read_json(processed_dir / "ensemble_performance_data.json")
    # Clinical metrics (from clinical_performance_data.json)
    clin_json = read_json(processed_dir / "clinical_performance_data.json")
    clin = clin_json.get("Gradient Boosting", {})
    clinical_ece = clin.get("ECE", None)

    records = [
        {
            "model": "ClinicalNet (GB)",
            "AUROC": perf.get("clinical_auroc", None),
            "AUROC_std": perf.get("clinical_auroc_std", None),
            "ECE": clinical_ece,
        },
        {
            "model": "ImagingNet",
            "AUROC": perf.get("mri_auroc", None),
            "AUROC_std": perf.get("mri_auroc_std", None),
            # No authoritative MRI ECE in processed files; keep None to avoid fabrication
            "ECE": None,
        },
        {
            "model": "Ensemble",
            "AUROC": perf.get("ensemble_auroc", None),
            "AUROC_std": perf.get("ensemble_auroc_std", None),
            "ECE": perf.get("ensemble_ece", None),
        },
    ]
    df = pd.DataFrame(records)

    # AUROC bar with error bars
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="model", y="AUROC", color="#4C72B0")
    # add error bars manually
    if df["AUROC_std"].notna().any():
        plt.errorbar(
            x=range(len(df)),
            y=df["AUROC"],
            yerr=df["AUROC_std"],
            fmt="none",
            ecolor="black",
            capsize=3,
            capthick=1,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC (mean ± SD)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_model_auroc.png", dpi=300)
    plt.close()

    # ECE bar (if available)
    plt.figure(figsize=(6, 4))
    # Only plot rows with valid ECE to avoid fabricated values
    df_ece = df.dropna(subset=["ECE"])  # ImagingNet likely removed here if ECE is None
    ax = sns.barplot(data=df_ece, x="model", y="ECE", color="#55A868")
    if not df_ece.empty:
        ax.set_ylim(0.0, max(0.25, float(df_ece["ECE"].max())))
    ax.set_xlabel("")
    ax.set_ylabel("ECE (lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_model_ece.png", dpi=300)
    plt.close()


def plot_mri_l2o(processed_dir: Path, out_dir: Path) -> None:
    csv_path = processed_dir / "mri_cv_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    # Histogram of AUROC across folds
    plt.figure(figsize=(6, 4))
    sns.histplot(df["AUROC"], bins=6, color="#C44E52", edgecolor="white")
    plt.xlabel("MRI AUROC (per L2O fold)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_mri_l2o_hist.png", dpi=300)
    plt.close()

    # Line plot across folds
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(df) + 1), df["AUROC"], marker="o", color="#8172B3")
    plt.xticks(range(1, len(df) + 1))
    plt.xlabel("Fold")
    plt.ylabel("AUROC")
    plt.ylim(0.6, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_mri_l2o_line.png", dpi=300)
    plt.close()


def plot_feature_importance(processed_dir: Path, out_dir: Path) -> None:
    rf = pd.read_csv(processed_dir / "rf_feature_importance.csv")
    gb = pd.read_csv(processed_dir / "gb_feature_importance.csv")

    for name, d in [("rf", rf), ("gb", gb)]:
        top = d.head(10).copy()
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=top, x="Importance", y="Feature", color="#4C72B0")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_feature_importance_{name}.png", dpi=300)
        plt.close()


def plot_dataset_characteristics(processed_dir: Path, out_dir: Path) -> None:
    ds = read_json(processed_dir / "dataset_characteristics.json")
    clinical = ds.get("clinical_data", {})
    mri = ds.get("mri_data", {})

    # Simple comparative bar chart of counts
    labels = ["AS", "Controls", "MRI Subjects", "MRI AS", "MRI HC"]
    values = [
        clinical.get("as_cases", 0),
        clinical.get("controls", 0),
        mri.get("total_subjects", 0),
        mri.get("as_subjects", 0),
        mri.get("healthy_subjects", 0),
    ]
    plt.figure(figsize=(7, 4))
    sns.barplot(x=labels, y=values, color="#64B5CD")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_dataset_counts.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from processed results")
    parser.add_argument("--input", dest="input_dir", type=str, default="data/processed", help="Processed results directory")
    parser.add_argument("--outdir", dest="out_dir", type=str, default="results/figures_generated", help="Output directory for figures")
    args = parser.parse_args()

    processed_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    plot_model_metrics(processed_dir, out_dir)
    plot_mri_l2o(processed_dir, out_dir)
    plot_feature_importance(processed_dir, out_dir)
    plot_dataset_characteristics(processed_dir, out_dir)

    print(f"✅ Generated figures in {out_dir}")


if __name__ == "__main__":
    main()