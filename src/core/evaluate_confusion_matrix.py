import torch
import pandas as pd
import numpy as np
import os
import argparse
from glob import glob
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager # Used for font checks, but will be simplified

# --- Import Custom Theme ---
import sys
# Hardcode project root to ensure theme file is found
PROJECT_ROOT_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, 'src'))

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into evaluate_confusion_matrix.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. Plots will use default Matplotlib style.")
    # Fallback Matplotlib configuration if theme fails to load (though CNSStyle overrides many of these)
    plt.rcParams.update({
        "figure.dpi":      300,
        "savefig.dpi":     300,
        "figure.figsize":  (8, 6),
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":       12,
        "axes.titlesize":  16,
        "axes.labelsize":  14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })


def evaluate(args):
    if _theme_loaded:
        configure_cns_style()
        print("🎉 Custom CNSStyle theme applied in evaluate_confusion_matrix.py.")
    
    # --- Font settings for English text (ensure no Chinese characters are attempted) ---
    # Your CNSStyle already sets 'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans']
    # This ensures English characters are used. No need for specific Chinese font handling here.
    # We will remove the problematic font_manager.findfont() block.

    # --- Data Loading: Load only the specified prediction file ---
    pred_file = args.preds_file_path
    
    if not os.path.exists(pred_file):
        print(f"❌ Error: Specified prediction file not found: {pred_file}")
        return

    df_all = pd.read_csv(pred_file)
    print(f"✅ Loaded prediction file: {pred_file} (Total records: {len(df_all)})")

    # Handle 'true_label' or 'label' column
    if 'true_label' in df_all.columns:
        true_labels = df_all["true_label"]
    elif 'label' in df_all.columns:
        true_labels = df_all["label"]
    else:
        print("❌ Error: Neither 'true_label' nor 'label' column found. Please check the prediction CSV file.")
        return

    logit_cols = [c for c in df_all.columns if "logit_" in c]
    prob_col = [c for c in df_all.columns if c.lower() == "prob"] # Check for 'prob' column

    if prob_col:
        # If 'prob' column exists, use it directly
        probs = df_all[prob_col[0]].values
        # For predicted labels, typically use a 0.5 threshold for binary classification
        pred_labels = (probs >= 0.5).astype(int)
        print("Using 'prob' column for scores.")
    elif logit_cols and len(logit_cols) >= 2:
        # If 'logit_' columns exist, convert using softmax
        logits = df_all[logit_cols].values
        # Assuming logit_0 is for negative class, logit_1 for positive class.
        # Adjust if your logit columns are ordered differently.
        probs = torch.nn.Softmax(dim=1)(torch.tensor(logits)).numpy()[:, 1] # Get probability for positive class
        pred_labels = np.argmax(torch.tensor(logits).numpy(), axis=1) # Get class with highest logit
        print("Using 'logit_' columns for scores and predictions.")
    else:
        print("❌ Error: Neither 'prob' column found, nor at least two 'logit_' columns. Cannot perform evaluation.")
        return


    # --- Overall Metrics ---
    print("\n--- Evaluation Results ---")
    acc = accuracy_score(true_labels, pred_labels)
    print(f"🎯 Accuracy: {acc:.4f}")

    try:
        # For binary classification, use positive class probabilities for AUC
        auc_score = roc_auc_score(true_labels, probs)
        print(f"🎯 AUC Score: {auc_score:.4f}")
    except Exception as e:
        auc_score = -1
        print(f"⚠️ Could not compute AUC: {e}. Error message: {e}")

    print("\n📋 Classification Report:")
    # Ensure target_names correspond to actual label values (e.g., 0 and 1)
    target_names = [str(label) for label in sorted(np.unique(true_labels))]
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    # --- Save Plot and Metrics ---
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Get unique label values for tick labels
    labels_unique = sorted(list(np.unique(true_labels)))
    
    plt.figure(figsize=(8, 6)) # figsize from your CNSStyle
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels_unique, 
                yticklabels=labels_unique)
    plt.title("Confusion Matrix") # Simplified English title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Ensure save path is in the same directory as the input file
    output_dir = os.path.dirname(args.preds_file_path)
    # Generate filename dynamically to include source file name, avoiding overwrite
    base_file_name = os.path.basename(args.preds_file_path).replace(".csv", "")
    cm_path = os.path.join(output_dir, f"confusion_matrix_{base_file_name}.png") 
    
    # Use save_for_publication from your CNSStyle's CNSTools if you want advanced saving options
    # For simplicity and direct use with matplotlib, we'll keep plt.savefig here.
    plt.savefig(cm_path, bbox_inches="tight") # Use bbox_inches="tight" for better saving
    plt.close() # Close figure to free up memory
    print(f"\n✅ Confusion Matrix saved to: {cm_path}")

    metrics = {"accuracy": acc, "auc": auc_score}
    metrics_path = os.path.join(output_dir, f"metrics_summary_{base_file_name}.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"📊 Overall metrics saved to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clinical model results and generate a confusion matrix.")
    # Parameter now accepts a precise file path, not a directory
    parser.add_argument("--preds_file_path", required=True, help="Full path to the prediction CSV file.")
    args = parser.parse_args()
    evaluate(args)