#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shap_overall_summary.py
Generates a single, overall SHAP summary plot by aggregating SHAP values
from all cross-validation folds.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # Added for ClinicalNet
import shap
import matplotlib.pyplot as plt

# ─────────── Import Custom CNSStyle Theme ───────────
PROJECT_ROOT_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, 'src'))

_theme_loaded = False
try:
    from visualization.theme import configure_cns_style
    _theme_loaded = True
    print("✅ Custom theme 'configure_cns_style' successfully imported into shap_overall_summary.py.")
except ImportError:
    _theme_loaded = False
    print("❌ Could not import theme file 'src/visualization/theme.py'. Will use default Matplotlib style.")
    plt.rcParams.update({
        "figure.dpi":      300, "savefig.dpi":     300, "figure.figsize":  (8, 6),
        "font.family":     "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":       12, "axes.titlesize":  16, "axes.labelsize":  14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

# ─────────── ClinicalNet Model Definition (EXACT MATCH from your training script) ───────────
class ClinicalNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(ClinicalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ─────────── Main Logic for Overall SHAP ───────────
def generate_overall_shap(n_folds=5):
    if _theme_loaded:
        configure_cns_style()
        print("🎉 Custom CNSStyle theme applied for overall SHAP plot.")

    # --- Path Configuration ---
    BASE_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
    data_processed_dir = os.path.join(BASE_DIR, "data", "processed_clinical") 
    models_dir = os.path.join(BASE_DIR, "results", "clinical", "clinical_model")
    
    # Save directory for overall figures - ENSURE THIS IS CREATED HERE
    save_dir = os.path.join(BASE_DIR, "results", "final_run", "figures_overall")
    os.makedirs(save_dir, exist_ok=True) # <--- MAKE SURE THIS LINE IS HERE, AND EXECUTED EARLY

    all_shap_values = []
    all_X_data = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ... (rest of the script) ...

    # Feature column selection logic (consistent across folds)
    # Load one df to determine feature columns (assuming they are consistent)
    sample_df_tr = pd.read_csv(os.path.join(data_processed_dir, 'fold_0_train.csv'))
    drop_cols_candidates = ["label", "true_label", "Patient_ID", "patient_id", "Patient ID"]
    all_tr_cols_sample = [c for c in sample_df_tr.columns if c not in drop_cols_candidates]
    numeric_feat_cols = []
    for col in all_tr_cols_sample:
        # Check if column is numeric and has variance (not all same value)
        # Assuming NaN filling happens within loop for each dataframe
        if pd.api.types.is_numeric_dtype(sample_df_tr[col]) and sample_df_tr[col].nunique() > 1:
             numeric_feat_cols.append(col)
    feat_names = numeric_feat_cols
    model_input_dim = len(feat_names) # Input dimension based on selected features

    if model_input_dim == 0:
        print("❌ No valid numeric features found in the dataset. Cannot proceed.")
        return

    # Instantiate model once outside the loop if it's meant to be the same architecture
    # but loaded with different weights per fold.
    num_classes = 2 # Assuming binary classification

    print("\n--- Computing SHAP values across all folds ---")
    for fold in range(n_folds):
        print(f"📦 Processing Fold {fold}...")
        tr_csv = os.path.join(data_processed_dir, f"fold_{fold}_train.csv")
        va_csv = os.path.join(data_processed_dir, f"fold_{fold}_val.csv")
        model_path = os.path.join(models_dir, f"best_model_fold_{fold}.pth")

        if not os.path.exists(tr_csv) or not os.path.exists(va_csv) or not os.path.exists(model_path):
            print(f"⚠️ Skipping Fold {fold}: Missing data or model file.")
            continue

        df_tr = pd.read_csv(tr_csv)
        df_va = pd.read_csv(va_csv)

        # Handle NaNs and select features for current fold's data
        for col in feat_names: # Use consistent feat_names determined from fold 0
            df_tr[col] = df_tr[col].fillna(0)
            df_va[col] = df_va[col].fillna(0)
        
        X_tr = df_tr[feat_names]
        X_va = df_va[feat_names]

        # Convert to Tensors
        # X_tr_t is used for background, X_va_t for calculating SHAP values
        X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32, device=device)
        X_va_t = torch.tensor(X_va.values, dtype=torch.float32, device=device)

        # Load model for current fold
        model = ClinicalNet(input_size=model_input_dim, hidden_size=64, output_size=num_classes).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Define prediction function for KernelExplainer
        def predict_fn(arr):
            # Ensure input tensor is on the same device as the model
            with torch.no_grad():
                logits = model(torch.tensor(arr, dtype=torch.float32).to(device))
                # Return probabilities for positive class (assuming binary classification)
                return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # Compute SHAP values using KernelExplainer (more robust for varied model outputs)
        # Choose a background sample from the training data of this fold
        # Reduce nsamples for faster computation during debugging if needed, but 100-500 is common.
        background_for_kernel = shap.sample(X_tr.values, min(100, len(X_tr)), random_state=42)
        kernel_explainer = shap.KernelExplainer(predict_fn, background_for_kernel)
        
        # Get SHAP values for the validation set (out-of-fold data) of the current fold
        # KernelExplainer typically returns a single array for binary classification probabilities.
        shap_values_for_fold = kernel_explainer.shap_values(X_va.values) # Pass numpy array directly

        # KernelExplainer output might be a list if predict_fn returns two outputs
        # or a single array (samples, features) if predict_fn returns 1 output (like positive prob)
        if isinstance(shap_values_for_fold, list) and len(shap_values_for_fold) > 0:
            # If predict_fn returns a single probability ([:,1]), it might wrap it in a list.
            # Or if it returns both class probabilities ([:,0], [:,1]), we pick [1].
            shap_vals_positive_class = shap_values_for_fold[0] if len(shap_values_for_fold) == 1 else shap_values_for_fold[1]
        else:
            shap_vals_positive_class = shap_values_for_fold # Assume it's already the array

        # Basic shape check for KernelExplainer output (should be (samples, features))
        if shap_vals_positive_class.shape != (X_va.shape[0], X_va.shape[1]):
            print(f"⚠️ WARNING: KernelExplainer output for Fold {fold} has unexpected shape: {shap_vals_positive_class.shape}. Expected ({X_va.shape[0]}, {X_va.shape[1]})")
            # If it's still wrong, we might need to debug this specific output, but Kernel is usually reliable.
            # If it comes as (features, samples) by mistake, transpose it:
            if shap_vals_positive_class.shape == (X_va.shape[1], X_va.shape[0]):
                print("   Attempting transpose for (features, samples) to (samples, features).")
                shap_vals_positive_class = shap_vals_positive_class.T
            else:
                print("   SHAP values shape remains incorrect. This fold might lead to error.")
                # Consider skipping this fold or exiting if critical
                continue


        all_shap_values.append(shap_vals_positive_class)
        all_X_data.append(X_va) # Append the DataFrame (for consistent feature names)

    if not all_shap_values:
        print("❌ No SHAP values computed across any fold. Check data/model paths or explainer behavior.")
        return

    # Concatenate SHAP values and data from all folds
    combined_shap_values = np.vstack(all_shap_values)
    combined_X_data = pd.concat(all_X_data, ignore_index=True)

    print(f"\n✅ Aggregated SHAP values shape: {combined_shap_values.shape}")
    print(f"✅ Aggregated data shape: {combined_X_data.shape}")

    # --- FINAL SHAPE ASSERTION BEFORE PLOT ---
    if combined_shap_values.shape[1] != combined_X_data.shape[1]:
        print(f"FATAL ERROR: Mismatch in feature count for summary_plot after aggregation!")
        print(f"  Combined SHAP values feature count: {combined_shap_values.shape[1]}")
        print(f"  Combined Data feature count: {combined_X_data.shape[1]}")
        sys.exit(1) # Exit because this is a guaranteed plot failure.


    # --- Generate Overall SHAP Summary Plot ---
    plt.figure(figsize=(10, 8)) # Adjusted size for potentially more features/samples
    shap.summary_plot(
        combined_shap_values, combined_X_data,
        plot_type="dot", # This is the "beeswarm" or "dot" plot
        max_display=min(20, len(feat_names)), # Display up to 20 features or fewer if available
        show=False # Don't show immediately, save to file
    )
    plt.title("Overall SHAP Summary Plot (Aggregated Out-of-Fold)")
    plt.tight_layout()

    overall_shap_png = os.path.join(save_dir, "overall_shap_summary.png")
    plt.savefig(overall_shap_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n🎉 Overall SHAP Summary Plot saved to → {overall_shap_png}")

if __name__ == "__main__":
    generate_overall_shap()