# FILE: calculate_3_models_final_stats.py
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import scipy.stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.calibration import calibration_curve # Correct import for calibration_curve
import matplotlib.pyplot as plt
import os
import glob
import warnings
import seaborn as sns # NEW IMPORT for confusion matrix plotting

warnings.filterwarnings("ignore", category=UserWarning)

# --- CORRECTED: Robust Path Definition ---
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..')) 
print(f"✅ Project Root identified as: {PROJECT_ROOT}")


# --- DeLong test implementation ---
def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(np.sum(ground_truth))
    return order, label_1_count

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    
    tx = np.dot(positive_examples, np.ones((m, 1)))
    ty = np.dot(negative_examples, np.ones((n, 1)))
    tz = np.dot(positive_examples, negative_examples.T)

    aucs = (np.sum(tz, axis=1) / (m * n))
    v01 = (tz / n)
    v10 = 1 - (tz.T / m)
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count)
    
    auc_diff = aucs[0] - aucs[1]
    cov_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = auc_diff / np.sqrt(cov_diff)
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    return p_value

# --- Bootstrap CI function ---
def bootstrap_metrics(y_true, y_probs, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    bootstrapped_scores = {"AUROC": [], "AUPRC": [], "Accuracy": [], "Sensitivity": [], "Specificity": []}
    
    y_preds = (y_probs >= 0.5).astype(int)
    point_estimates = {
        "AUROC": roc_auc_score(y_true, y_probs), "AUPRC": average_precision_score(y_true, y_probs),
        "Accuracy": accuracy_score(y_true, y_preds), "Sensitivity": recall_score(y_true, y_preds),
    }
    cm_point = confusion_matrix(y_true, y_preds)
    tn_point, fp_point, fn_point, tp_point = cm_point.ravel()
    point_estimates["Specificity"] = tn_point / (tn_point + fp_point)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot, y_probs_boot = y_true[indices], y_probs[indices]
        if len(np.unique(y_true_boot)) < 2: continue
        y_preds_boot = (y_probs_boot >= 0.5).astype(int)
        bootstrapped_scores["AUROC"].append(roc_auc_score(y_true_boot, y_probs_boot))
        bootstrapped_scores["AUPRC"].append(average_precision_score(y_true_boot, y_probs_boot))
        bootstrapped_scores["Accuracy"].append(accuracy_score(y_true_boot, y_preds_boot))
        bootstrapped_scores["Sensitivity"].append(recall_score(y_true_boot, y_preds_boot))
        cm = confusion_matrix(y_true_boot, y_preds_boot)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            bootstrapped_scores["Specificity"].append(spec)
        
    results = {}
    for metric, scores in bootstrapped_scores.items():
        if scores:
            ci_lower, ci_upper = np.percentile(scores, [2.5, 97.5])
            results[metric] = f"{point_estimates[metric]:.3f} ({ci_lower:.3f}–{ci_upper:.3f})"
        else:
            results[metric] = f"{point_estimates[metric]:.3f} (CI calculation failed)"
    return results

# --- Plot Calibration Curve ---
def plot_calibration_curve(model_data, plot_save_path):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Ideal Calibration')

    colors = plt.cm.get_cmap('tab10', len(model_data))
    
    for i, (model_name, data) in enumerate(model_data.items()):
        y_true = data['y_true']
        y_probs = data['y_probs']
        
        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Warning: Cannot plot calibration for {model_name}. Requires both classes in y_true.")
            continue

        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=10)
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=f'{model_name}', color=colors(i))

    ax.set_xlabel('Mean Predicted Value')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Diagram')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"✅ Calibration plot saved to {plot_save_path}")
    plt.close()

# --- Plot Precision-Recall Curve ---
def plot_pr_curve(model_data, plot_save_path):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    y_true_any = None
    for model_name, data in model_data.items():
        if len(np.unique(data['y_true'])) >= 2:
            y_true_any = data['y_true']
            break
    if y_true_any is not None:
        pos_ratio = np.sum(y_true_any) / len(y_true_any)
        ax.plot([0, 1], [pos_ratio, pos_ratio], linestyle='--', color='gray', label='Random Classifier (AP = {:.2f})'.format(pos_ratio))
    
    colors = plt.cm.get_cmap('tab10', len(model_data))

    for i, (model_name, data) in enumerate(model_data.items()):
        y_true = data['y_true']
        y_probs = data['y_probs']

        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Warning: Cannot plot PR curve for {model_name}. Requires both classes in y_true.")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)
        
        ax.plot(recall, precision, color=colors(i), 
                label=f'{model_name} (AP = {ap_score:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"✅ Precision-Recall curve saved to {plot_save_path}")
    plt.close()

# --- Plot ROC Curve ---
def plot_roc_curve(model_data, plot_save_path):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier (AUC = 0.50)')
    
    colors = plt.cm.get_cmap('tab10', len(model_data))

    for i, (model_name, data) in enumerate(model_data.items()):
        y_true = data['y_true']
        y_probs = data['y_probs']

        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Warning: Cannot plot ROC curve for {model_name}. Requires both classes in y_true.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors(i), 
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"✅ ROC curve saved to {plot_save_path}")
    plt.close()

# --- Plot Probability Distribution ---
def plot_probability_distribution(model_data, plot_save_path, bins=50):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    colors = plt.cm.get_cmap('Dark2', len(model_data) * 2)
    
    model_counter = 0
    for model_name, data in model_data.items():
        y_true = data['y_true']
        y_probs = data['y_probs']

        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Warning: Cannot plot probability distribution for {model_name}. Requires both classes in y_true.")
            continue
        
        probs_neg = y_probs[y_true == 0]
        probs_pos = y_probs[y_true == 1]

        ax.hist(probs_neg, bins=bins, alpha=0.6, color=colors(model_counter), 
                label=f'{model_name} (Label 0)', density=True)
        ax.hist(probs_pos, bins=bins, alpha=0.6, color=colors(model_counter + 1), 
                label=f'{model_name} (Label 1)', density=True)
        model_counter += 2

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Predicted Probabilities')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"✅ Probability distribution plot saved to {plot_save_path}")
    plt.close()

# --- Plot Decision Curve Analysis (DCA) ---
def plot_dca_curve(model_data, plot_save_path, thresholds=np.linspace(0.01, 0.99, 100)):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    y_true_any = None
    for model_name, data in model_data.items():
        if len(np.unique(data['y_true'])) >= 2:
            y_true_any = data['y_true']
            break

    if y_true_any is not None:
        n_pos = np.sum(y_true_any)
        n_neg = len(y_true_any) - n_pos
        n_total = len(y_true_any)

        treat_all_net_benefit = []
        for t in thresholds:
            if (1 - t) == 0: 
                treat_all_net_benefit.append(0) 
            else:
                treat_all_net_benefit.append((n_pos / n_total) - (n_neg / n_total) * (t / (1 - t)))
        ax.plot(thresholds, treat_all_net_benefit, linestyle='-', color='red', label='Treat All')

    ax.plot(thresholds, [0] * len(thresholds), linestyle='-', color='green', label='Treat None')

    colors = plt.cm.get_cmap('tab10', len(model_data))

    for i, (model_name, data) in enumerate(model_data.items()):
        y_true = data['y_true']
        y_probs = data['y_probs']

        if len(np.unique(y_true)) < 2:
            print(f"⚠️ Warning: Cannot plot DCA for {model_name}. Requires both classes in y_true.")
            continue
        
        net_benefits = []
        for t in thresholds:
            if (1 - t) == 0:
                 net_benefits.append(0) 
                 continue
            
            y_pred_at_t = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_at_t).ravel()
            
            net_benefit = (tp / n_total) - (fp / n_total) * (t / (1 - t))
            net_benefits.append(net_benefit)
            
        ax.plot(thresholds, net_benefits, linestyle='-', color=colors(i), label=f'{model_name}')

    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend(loc='upper right')
    ax.set_xlim([0.0, 1.0])
    
    # Dynamically adjust y-axis limits for DCA
    all_net_benefits = []
    if y_true_any is not None:
        all_net_benefits.extend(treat_all_net_benefit)
    for model_name, data in model_data.items():
        if len(np.unique(data['y_true'])) >= 2:
            # Re-calculate net_benefits to get the list for max/min finding
            net_benefits_model = []
            for t in thresholds:
                if (1 - t) == 0:
                    net_benefits_model.append(0)
                    continue
                y_pred_at_t = (data['y_probs'] >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(data['y_true'], y_pred_at_t).ravel()
                net_benefits_model.append((tp / n_total) - (fp / n_total) * (t / (1 - t)))
            all_net_benefits.extend(net_benefits_model)

    if all_net_benefits:
        min_nb = min(all_net_benefits)
        max_nb = max(all_net_benefits)
        # Add some padding
        ax.set_ylim([min_nb - 0.02, max_nb + 0.02])
    else:
        ax.set_ylim([-0.05, 0.15]) # Default reasonable range if no models loaded
    
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"✅ Decision Curve Analysis plot saved to {plot_save_path}")
    plt.close()

# --- NEW FUNCTION: Plot Confusion Matrix ---
def plot_confusion_matrix(model_name, y_true, y_probs, plot_save_path, threshold=0.5, class_names=['Control', 'AS']):
    """
    Plots a confusion matrix for a given model.
    Args:
        model_name (str): Name of the model.
        y_true (np.array): True labels.
        y_probs (np.array): Predicted probabilities.
        plot_save_path (str): Path to save the plot.
        threshold (float): Probability threshold for classification.
        class_names (list): List of class names (e.g., ['Negative', 'Positive']).
    """
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix for {model_name} (Threshold={threshold:.2f})', fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"✅ Confusion Matrix for {model_name} saved to {plot_save_path}")
    plt.close()


# --- Main Analysis Workflow ---
def main_workflow():
    # Define standard locations for data and results using the corrected PROJECT_ROOT
    DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed_clinical")
    PRED_DIR = os.path.join(PROJECT_ROOT, "results/consolidated/predictions")
    # Corrected path for ClinicalNet predictions
    CLINICAL_NET_PRED_SOURCE_DIR = os.path.join(PROJECT_ROOT, "results/consolidated/clinical/clinical_model_clinical_preds")
    
    os.makedirs(PRED_DIR, exist_ok=True)
    
    # === PART 1: Generate predictions for baseline models ===
    baseline_models = {
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42)
    }

    for model_name, model in baseline_models.items():
        print(f"\n===== Running Baseline: {model_name} =====")
        all_fold_probs, all_fold_labels, all_fold_ids = [], [], []
        for fold in range(5):
            val_path = os.path.join(DATA_DIR, f"fold_{fold}_val.csv")
            train_path = os.path.join(DATA_DIR, f"fold_{fold}_train.csv")
            df_val, df_train = pd.read_csv(val_path), pd.read_csv(train_path)
            id_cols = ['Patient_ID', 'patient_id', 'Patient ID']
            patient_id_col = next((col for col in id_cols if col in df_val.columns), None)
            if patient_id_col is None:
                raise ValueError(f"No suitable patient ID column found in {val_path}. Looked for: {id_cols}")

            feat_cols = [c for c in df_train.columns if c != 'label' and c not in id_cols] 
            X_train, y_train = df_train[feat_cols], df_train['label']
            X_val, y_val = df_val[feat_cols], df_val['label']
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            all_fold_probs.extend(val_probs); all_fold_labels.extend(y_val); all_fold_ids.extend(df_val[patient_id_col])
        pred_df = pd.DataFrame({'patient_id': all_fold_ids, 'true_label': all_fold_labels, 'prob': all_fold_probs})
        save_path = os.path.join(PRED_DIR, f"{model_name}_predictions.csv")
        pred_df.to_csv(save_path, index=False)
        print(f"✅ Predictions for {model_name} saved to {save_path}")

    # === PART 2: Aggregate predictions for ClinicalNet ===
    print("\n===== Aggregating ClinicalNet Predictions =====")
    fold_files = glob.glob(os.path.join(CLINICAL_NET_PRED_SOURCE_DIR, "fold_*_predictions.csv"))
    if not fold_files:
        print(f"❌ FATAL ERROR: No fold prediction files found for ClinicalNet in {CLINICAL_NET_PRED_SOURCE_DIR}")
        pass 
    else:
        df_list = [pd.read_csv(f) for f in fold_files]
        agg_df = pd.concat(df_list, ignore_index=True)
        clinical_net_save_path = os.path.join(PRED_DIR, "ClinicalNet_predictions.csv")
        agg_df.to_csv(clinical_net_save_path, index=False)
        print(f"✅ Aggregated ClinicalNet predictions saved to {clinical_net_save_path}")

    # === PART 3: Run final statistics on all predictions ===
    print("\n===== Running Final Statistical Analysis =====")
    pred_paths = {
        "LightGBM": os.path.join(PRED_DIR, "LightGBM_predictions.csv"),
        "XGBoost": os.path.join(PRED_DIR, "XGBoost_predictions.csv"),
        "ClinicalNet": os.path.join(PRED_DIR, "ClinicalNet_predictions.csv")
    }
    model_data = {}
    for name, path in pred_paths.items():
        try:
            df = pd.read_csv(path)
            y_true = df['true_label'].values
            if 'prob' in df.columns: y_probs = df['prob'].values
            elif 'logit_0' in df.columns and 'logit_1' in df.columns:
                l0, l1 = df['logit_0'].values, df['logit_1'].values
                y_probs = np.exp(l1) / (np.exp(l0) + np.exp(l1))
            else: raise ValueError(f"No 'prob' or 'logit' columns found in {path}. Skipping.")
            model_data[name] = {"y_true": y_true, "y_probs": y_probs}
        except FileNotFoundError:
            print(f"⚠️ Warning: Prediction file not found for {name} at {path}. Skipping final stats for this model.")
        except ValueError as e:
            print(f"⚠️ Warning: Error processing {name} predictions from {path}: {e}. Skipping final stats for this model.")

    final_results = {}
    for name, data in model_data.items():
        print(f"Calculating CIs for {name}...")
        final_results[name] = bootstrap_metrics(data['y_true'], data['y_probs'])
    results_df = pd.DataFrame(final_results)
    
    print("\n\n" + "="*70)
    print("      >>> FINAL PERFORMANCE TABLE (WITH 95% CIs) <<<")
    print("="*70)
    print(results_df)
    print("="*70 + "\n")
    
    print("Calculating DeLong's Test p-values for AUROC comparison...")
    
    if "LightGBM" in model_data and "ClinicalNet" in model_data:
        y_true_lgbm = model_data['LightGBM']['y_true']
        p_lgbm = model_data['LightGBM']['y_probs']
        p_clin = model_data['ClinicalNet']['y_probs']
        p1 = delong_roc_test(y_true_lgbm, p_lgbm, p_clin)
        print(f"P₁ (LightGBM vs. ClinicalNet): p = {p1:.4f}")
    else:
        print("Skipping P₁ (LightGBM vs. ClinicalNet) due to missing data.")


    if "LightGBM" in model_data and "XGBoost" in model_data:
        y_true_lgbm = model_data['LightGBM']['y_true']
        p_lgbm = model_data['LightGBM']['y_probs']
        p_xgb = model_data['XGBoost']['y_probs']
        p2 = delong_roc_test(y_true_lgbm, p_lgbm, p_xgb)
        print(f"P₂ (LightGBM vs. XGBoost): p = {p2:.4f}")
    else:
        print("Skipping P₂ (LightGBM vs. XGBoost) due to missing data.")

    # === Generate and save Plots ===
    print("\n===== Generating Plots =====")
    
    # Calibration Plot
    calibration_plot_path = os.path.join(PROJECT_ROOT, "results", "calibration_plot.png")
    plot_calibration_curve(model_data, calibration_plot_path)

    # PR Curve Plot
    pr_curve_plot_path = os.path.join(PROJECT_ROOT, "results", "pr_curve_plot.png")
    plot_pr_curve(model_data, pr_curve_plot_path)

    # ROC Curve Plot
    roc_curve_plot_path = os.path.join(PROJECT_ROOT, "results", "roc_curve_plot.png")
    plot_roc_curve(model_data, roc_curve_plot_path)

    # Probability Distribution Plot
    prob_dist_plot_path = os.path.join(PROJECT_ROOT, "results", "probability_distribution_plot.png")
    plot_probability_distribution(model_data, prob_dist_plot_path)

    # DCA Plot
    dca_plot_path = os.path.join(PROJECT_ROOT, "results", "dca_plot.png")
    plot_dca_curve(model_data, dca_plot_path)

    # Confusion Matrix Plot(s)
    for model_name, data in model_data.items():
        cm_plot_path = os.path.join(PROJECT_ROOT, "results", f"confusion_matrix_{model_name}.png")
        plot_confusion_matrix(model_name, data['y_true'], data['y_probs'], cm_plot_path)


if __name__ == "__main__":
    main_workflow()