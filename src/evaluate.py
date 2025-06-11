import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import argparse
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(args):
    """
    Evaluates the model's predictions from K-fold cross-validation.
    """
    all_pred_files = glob(os.path.join(args.preds_dir, "fold_*_predictions.csv"))
    if not all_pred_files:
        print(f"Error: No prediction files found in {args.preds_dir}")
        return

    all_dfs = [pd.read_csv(f) for f in all_pred_files]
    df_all_preds = pd.concat(all_dfs, ignore_index=True)

    true_labels = df_all_preds['true_label']
    
    # Get logit columns
    logit_cols = [col for col in df_all_preds.columns if 'logit_' in col]
    pred_logits = df_all_preds[logit_cols].values
    
    # Convert logits to probabilities using softmax
    softmax = torch.nn.Softmax(dim=1)
    pred_probs = softmax(torch.tensor(pred_logits)).numpy()
    
    # Get predicted labels by taking the argmax
    pred_labels = np.argmax(pred_probs, axis=1)

    # --- Calculate and Print Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # For multi-class AUC, we use One-vs-Rest
    try:
        auc_score = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='weighted')
        print(f"\nOverall Weighted AUC (OvR): {auc_score:.4f}")
    except ValueError as e:
        print(f"\nCould not compute AUC: {e}")

    print(f"Overall Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    # Optional: Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Overall Confusion Matrix')
    
    # Save the plot
    plot_save_path = os.path.join(args.preds_dir, 'confusion_matrix.png')
    plt.savefig(plot_save_path)
    print(f"\nConfusion matrix plot saved to {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate K-fold cross-validation predictions.")
    parser.add_argument("--preds_dir", type=str, required=True, help="Directory containing the prediction CSV files.")
    
    # A small hack to make the script find torch if it's not in the environment
    try:
        import torch
    except ImportError:
        print("PyTorch not found, which is required for softmax. Please install PyTorch.")
        exit()
        
    args = parser.parse_args()
    evaluate(args)
