#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_figures_pro.py

SCI-grade, fully data-driven visualizations:
- Reads summary metrics from data/processed
- Reads predictions from results/clinical/ensemble_run and results/mri/l2o_predictions.csv
- Produces publication-quality figures with consistent styling

Outputs: results/figures_generated_pro/
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set(context="paper", style="whitegrid", font_scale=1.2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ind = np.digitize(y_prob, bins) - 1
    ind = np.clip(ind, 0, n_bins - 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = ind == b
        if not np.any(mask):
            rows.append((bins[b], bins[b+1], np.nan, np.nan, 0))
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
        rows.append((bins[b], bins[b+1], acc, conf, mask.sum()))
    df = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "acc", "conf", "count"])
    return float(ece), df


def plot_roc(ax, y_true: np.ndarray, y_prob: np.ndarray, label: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")
    return float(roc_auc)


def plot_clinical_figures(processed_dir: Path, clinical_results_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    preds_path = clinical_results_dir / "ensemble_predictions.csv"
    if not preds_path.exists():
        return
    df = pd.read_csv(preds_path)
    y_true = df["true_label"].to_numpy()

    # ROC curves: ClinicalNet vs Ensemble
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], ls="--", c="#888888", lw=1)
    roc_ens = plot_roc(ax, y_true, df["ensemble_prob"].to_numpy(), label="Ensemble")
    if "clinical_net_prob" in df:
        _ = plot_roc(ax, y_true, df["clinical_net_prob"].to_numpy(), label="ClinicalNet (GB)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Clinical ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "pro_clinical_roc.png", dpi=300)
    plt.close()

    # Calibration: reliability diagram for Ensemble and ClinicalNet
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], ls=":", c="#444444", lw=1)
    ece_ens, bins_ens = compute_ece(y_true, df["ensemble_prob"].to_numpy())
    ax.plot(bins_ens["conf"], bins_ens["acc"], marker="o", lw=2, label=f"Ensemble (ECE={ece_ens:.3f})")
    if "clinical_net_prob" in df:
        ece_gb, bins_gb = compute_ece(y_true, df["clinical_net_prob"].to_numpy())
        ax.plot(bins_gb["conf"], bins_gb["acc"], marker="s", lw=2, label=f"ClinicalNet (ECE={ece_gb:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Clinical Reliability Diagram")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "pro_clinical_calibration.png", dpi=300)
    plt.close()


def plot_mri_figures(mri_pred_csv: Path, out_dir: Path) -> None:
    if not mri_pred_csv.exists():
        return
    df = pd.read_csv(mri_pred_csv)
    y_true = df["y_true"].to_numpy()
    y_prob = df["prob_raw"].to_numpy()

    # ROC
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], ls="--", c="#888888", lw=1)
    _ = plot_roc(ax, y_true, y_prob, label="ImagingNet")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("MRI ROC Curve (Subject-Level)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "pro_mri_roc.png", dpi=300)
    plt.close()

    # Probability distributions by class
    plt.figure(figsize=(6, 4))
    plot_df = pd.DataFrame({"y": y_true, "p": y_prob})
    plot_df["class"] = plot_df["y"].map({0: "Healthy", 1: "AS"})
    sns.violinplot(data=plot_df, x="class", y="p", inner="box", palette=["#4C72B0", "#C44E52"])
    plt.xlabel("")
    plt.ylabel("Predicted probability")
    plt.title("MRI Predicted Probability Distributions")
    plt.tight_layout()
    plt.savefig(out_dir / "pro_mri_prob_distributions.png", dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SCI-grade figures from raw and processed results")
    ap.add_argument("--processed", default="data/processed", help="Processed results dir")
    ap.add_argument("--clinres", default="results/clinical/ensemble_run", help="Clinical results dir")
    ap.add_argument("--mri_preds", default="results/mri/l2o_predictions.csv", help="MRI subject-level predictions CSV")
    ap.add_argument("--out", default="results/figures_generated_pro", help="Output directory")
    args = ap.parse_args()

    processed_dir = Path(args.processed)
    clinical_dir = Path(args.clinres)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    plot_clinical_figures(processed_dir, clinical_dir, out_dir)
    plot_mri_figures(Path(args.mri_preds), out_dir)

    print(f"✅ Generated SCI-grade figures in {out_dir}")


if __name__ == "__main__":
    main()


