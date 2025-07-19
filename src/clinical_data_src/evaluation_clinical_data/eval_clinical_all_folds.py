# src/evaluation/eval_clinical_all_folds.py
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import Custom Theme ---
import sys
PROJECT_ROOT_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, 'src'))

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into eval_clinical_all_folds.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. Plots will use default Matplotlib style.")
    plt.rcParams.update({
        "figure.dpi":      300, "savefig.dpi":     300, "figure.figsize":  (8, 6),
        "font.family":     "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":       12, "axes.titlesize":  16, "axes.labelsize":  14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })


def load_predictions(pred_dir):
    all_logits, all_labels, all_probs = [], [], []
    found_folds = 0
    for i in range(5): # Assuming 5 folds (0 to 4)
        fold_path = os.path.join(pred_dir, f"fold_{i}_predictions.csv")
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            found_folds += 1

            true_label_col = 'true_label' if 'true_label' in df.columns else 'label'
            if true_label_col not in df.columns:
                raise KeyError(f"Error: Neither 'true_label' nor 'label' column found in {fold_path}")
            labels = df[true_label_col].values

            prob_col_name = None
            for col in df.columns:
                if col.lower() == 'prob':
                    prob_col_name = col
                    break
            
            if prob_col_name:
                probs = df[prob_col_name].values
            else:
                logit_cols = [c for c in df.columns if c.startswith("logit_")]
                if len(logit_cols) < 2:
                    raise KeyError(f"Error: Neither 'prob' column nor enough 'logit_' columns found in {fold_path}")
                logits = df[[f'logit_{k}' for k in range(len(logit_cols))] if all(f'logit_{k}' in logit_cols for k in range(len(logit_cols))) else sorted(logit_cols)].values
                
                probs = torch.nn.Softmax(dim=1)(torch.tensor(logits)).numpy()[:, 1]
            
            all_labels.append(labels)
            all_probs.append(probs)
            # all_logits.append(logits) # Logits not strictly needed after prob calculation for overall metrics

    if found_folds == 0:
        raise FileNotFoundError(f"No fold_X_predictions.csv files found in {pred_dir}")
        
    final_labels = np.hstack(all_labels)
    final_probs = np.hstack(all_probs)
    
    final_preds = (final_probs >= 0.5).astype(int)

    return final_labels, final_probs, final_preds


def evaluate_and_plot_overall(pred_dir):
    if _theme_loaded:
        configure_cns_style()
        print("🎉 Overall CNSStyle theme applied for plotting in eval_clinical_all_folds.py.")

    try:
        labels, probs, preds = load_predictions(pred_dir)
    except (KeyError, FileNotFoundError) as e:
        print(f"Evaluation aborted: {e}")
        return

    print("\n=== Overall Cross-Validation Evaluation ===")
    
    overall_acc = accuracy_score(labels, preds)
    print(f"🎯 Overall Accuracy: {overall_acc:.4f}")

    overall_auc = roc_auc_score(labels, probs)
    print(f"🎯 Overall AUROC: {overall_auc:.4f}")
    
    overall_auprc = average_precision_score(labels, probs)
    print(f"🎯 Overall AUPRC: {overall_auprc:.4f}")

    print("\n📋 Overall Classification Report:")
    target_names = [str(label) for label in sorted(np.unique(labels))]
    print(classification_report(labels, preds, target_names=target_names))

    # --- Plot Overall Confusion Matrix ---
    cm = confusion_matrix(labels, preds)
    labels_unique = sorted(list(np.unique(labels)))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels_unique, 
                yticklabels=labels_unique)
    plt.title("Overall Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Output directory for overall plots
    output_dir = os.path.join(pred_dir, "overall_plots") 
    os.makedirs(output_dir, exist_ok=True)
    
    cm_path = os.path.join(output_dir, "overall_confusion_matrix.png")
    
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Overall Confusion Matrix saved to: {cm_path}")

    # Save overall metrics to CSV
    metrics = {
        "Accuracy": overall_acc, 
        "AUROC": overall_auc, 
        "AUPRC": overall_auprc
    }
    metrics_path = os.path.join(output_dir, "overall_metrics_summary.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"📊 Overall metrics summary saved to: {metrics_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate overall clinical model results across all folds and generate plots.")
    parser.add_argument("--preds_dir", required=True, help="Path to directory with fold_X_predictions.csv files (e.g., results/clinical/clinical_model/clinical_preds)")
    args = parser.parse_args()
    evaluate_and_plot_overall(args.pred_dir)