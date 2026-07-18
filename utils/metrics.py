import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    cohen_kappa_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

from config import Config


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes the Expected Calibration Error (ECE) of a classification model.
    ECE measures the correspondence between predicted probabilities and actual accuracy.
    """
    valid_mask = labels != -100
    if not np.any(valid_mask):
        return 0.0
    probs = probs[valid_mask]
    labels = labels[valid_mask]

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    # Get maximum predicted probability and corresponding class prediction
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Identify samples falling into the current probability bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Computes the multi-class Brier Score.
    Brier Score measures the mean squared difference between predicted probabilities
    and the actual one-hot encoded targets. Lower is better.
    """
    valid_mask = labels != -100
    if not np.any(valid_mask):
        return 0.0
    probs = probs[valid_mask]
    labels = labels[valid_mask]

    # One-hot encode the target labels
    targets_onehot = np.eye(num_classes)[labels]
    
    # Compute mean squared difference
    brier = np.mean(np.sum((probs - targets_onehot) ** 2, axis=1))
    return float(brier)


def compute_multitask_metrics(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """
    Computes medical-grade classification metrics for DR and DME predictions,
    including calibration metrics and clinical referral analysis.

    Args:
        df (pd.DataFrame): DataFrame containing columns:
                           - 'dr_target', 'dr_pred'
                           - 'dme_target', 'dme_pred'
                           - Probabilities 'dr_prob_class_x', 'dme_prob_class_y'
                           - Evidential uncertainties: 'dr_uncertainty', 'dme_uncertainty' (optional)
        config (Config): Configuration parameters.

    Returns:
        dict: Calculated metrics dictionary.
    """
    metrics = {}

    # --- 1. DR Grading Metrics ---
    dr_y_true = df["dr_target"].values
    dr_y_pred = df["dr_pred"].values
    dr_num_classes = config.heads.dr_num_classes
    
    dr_valid_mask = dr_y_true != -100
    if np.any(dr_valid_mask):
        dr_y_true_valid = dr_y_true[dr_valid_mask]
        dr_y_pred_valid = dr_y_pred[dr_valid_mask]
        
        metrics["dr_accuracy"] = accuracy_score(dr_y_true_valid, dr_y_pred_valid)
        metrics["dr_precision"] = precision_score(dr_y_true_valid, dr_y_pred_valid, average="macro", zero_division=0)
        metrics["dr_recall"] = recall_score(dr_y_true_valid, dr_y_pred_valid, average="macro", zero_division=0)
        metrics["dr_f1_macro"] = f1_score(dr_y_true_valid, dr_y_pred_valid, average="macro", zero_division=0)
        metrics["dr_qwk"] = cohen_kappa_score(dr_y_true_valid, dr_y_pred_valid, weights="quadratic")

        # Extract DR probabilities
        dr_prob_cols = sorted([c for c in df.columns if c.startswith("dr_prob_class_")])
        if len(dr_prob_cols) == dr_num_classes:
            dr_probs = df[dr_prob_cols].values[dr_valid_mask]
            metrics["dr_ece"] = compute_ece(dr_probs, dr_y_true_valid)
            metrics["dr_brier"] = compute_brier_score(dr_probs, dr_y_true_valid, dr_num_classes)
            try:
                metrics["dr_auc"] = roc_auc_score(dr_y_true_valid, dr_probs, multi_class="ovr", average="macro")
            except Exception:
                metrics["dr_auc"] = np.nan
    else:
        for k in ["dr_accuracy", "dr_precision", "dr_recall", "dr_f1_macro", "dr_qwk", "dr_ece", "dr_brier", "dr_auc"]:
            metrics[k] = np.nan

    # --- 2. DME Staging Metrics ---
    dme_y_true = df["dme_target"].values
    dme_y_pred = df["dme_pred"].values
    dme_num_classes = config.heads.dme_num_classes
    
    dme_valid_mask = dme_y_true != -100
    if np.any(dme_valid_mask):
        dme_y_true_valid = dme_y_true[dme_valid_mask]
        dme_y_pred_valid = dme_y_pred[dme_valid_mask]
        
        metrics["dme_accuracy"] = accuracy_score(dme_y_true_valid, dme_y_pred_valid)
        metrics["dme_precision"] = precision_score(dme_y_true_valid, dme_y_pred_valid, average="macro", zero_division=0)
        metrics["dme_recall"] = recall_score(dme_y_true_valid, dme_y_pred_valid, average="macro", zero_division=0)
        metrics["dme_f1_macro"] = f1_score(dme_y_true_valid, dme_y_pred_valid, average="macro", zero_division=0)
        metrics["dme_qwk"] = cohen_kappa_score(dme_y_true_valid, dme_y_pred_valid, weights="quadratic")

        # Extract DME probabilities
        dme_prob_cols = sorted([c for c in df.columns if c.startswith("dme_prob_class_")])
        if len(dme_prob_cols) == dme_num_classes:
            dme_probs = df[dme_prob_cols].values[dme_valid_mask]
            metrics["dme_ece"] = compute_ece(dme_probs, dme_y_true_valid)
            metrics["dme_brier"] = compute_brier_score(dme_probs, dme_y_true_valid, dme_num_classes)
            try:
                metrics["dme_auc"] = roc_auc_score(dme_y_true_valid, dme_probs, multi_class="ovr", average="macro")
            except Exception:
                metrics["dme_auc"] = np.nan
    else:
        for k in ["dme_accuracy", "dme_precision", "dme_recall", "dme_f1_macro", "dme_qwk", "dme_ece", "dme_brier", "dme_auc"]:
            metrics[k] = np.nan

    # --- 3. Combined Multi-task Metrics ---
    metrics["multitask_avg_accuracy"] = np.nanmean([metrics["dr_accuracy"], metrics["dme_accuracy"]])
    metrics["multitask_avg_qwk"] = np.nanmean([metrics["dr_qwk"], metrics["dme_qwk"]])

    # --- 4. Clinical Referral Module Performance ---
    # Determine referral flags
    if "referral_recommended" in df.columns:
        referred = df["referral_recommended"].values.astype(bool)
    elif "dr_uncertainty" in df.columns and "dme_uncertainty" in df.columns:
        # Compute dynamically based on configuration thresholds
        dr_u = df["dr_uncertainty"].values
        dme_u = df["dme_uncertainty"].values
        
        dr_sev_threshold = config.referral.dr_severity_threshold
        dme_sev_threshold = config.referral.dme_severity_threshold
        u_threshold = config.referral.uncertainty_threshold
        
        high_severity_dr = dr_y_pred >= dr_sev_threshold
        high_severity_dme = dme_y_pred >= dme_sev_threshold
        high_uncertainty_dr = dr_u >= u_threshold
        high_uncertainty_dme = dme_u >= u_threshold
        
        referred = high_severity_dr | high_severity_dme | high_uncertainty_dr | high_uncertainty_dme
    else:
        # Fall back to severity threshold only
        dr_sev_threshold = config.referral.dr_severity_threshold
        dme_sev_threshold = config.referral.dme_severity_threshold
        referred = (dr_y_pred >= dr_sev_threshold) | (dme_y_pred >= dme_sev_threshold)

    # Calculate referral metrics
    referral_rate = np.mean(referred)
    coverage = 1.0 - referral_rate
    
    metrics["referral_rate"] = float(referral_rate)
    metrics["coverage"] = float(coverage)

    # Compute performance on accepted (non-referred) predictions
    accepted_mask = ~referred
    num_accepted = np.sum(accepted_mask)
    metrics["accepted_count"] = int(num_accepted)

    if num_accepted > 0:
        dr_y_true_acc = dr_y_true[accepted_mask]
        dr_y_pred_acc = dr_y_pred[accepted_mask]
        dme_y_true_acc = dme_y_true[accepted_mask]
        dme_y_pred_acc = dme_y_pred[accepted_mask]

        dr_acc_mask = dr_y_true_acc != -100
        if np.any(dr_acc_mask):
            metrics["accepted_dr_accuracy"] = accuracy_score(dr_y_true_acc[dr_acc_mask], dr_y_pred_acc[dr_acc_mask])
            metrics["accepted_dr_f1_macro"] = f1_score(dr_y_true_acc[dr_acc_mask], dr_y_pred_acc[dr_acc_mask], average="macro", zero_division=0)
            metrics["accepted_dr_qwk"] = cohen_kappa_score(dr_y_true_acc[dr_acc_mask], dr_y_pred_acc[dr_acc_mask], weights="quadratic")
        else:
            for k in ["accepted_dr_accuracy", "accepted_dr_f1_macro", "accepted_dr_qwk"]:
                metrics[k] = np.nan

        dme_acc_mask = dme_y_true_acc != -100
        if np.any(dme_acc_mask):
            metrics["accepted_dme_accuracy"] = accuracy_score(dme_y_true_acc[dme_acc_mask], dme_y_pred_acc[dme_acc_mask])
            metrics["accepted_dme_f1_macro"] = f1_score(dme_y_true_acc[dme_acc_mask], dme_y_pred_acc[dme_acc_mask], average="macro", zero_division=0)
            metrics["accepted_dme_qwk"] = cohen_kappa_score(dme_y_true_acc[dme_acc_mask], dme_y_pred_acc[dme_acc_mask], weights="quadratic")
        else:
            for k in ["accepted_dme_accuracy", "accepted_dme_f1_macro", "accepted_dme_qwk"]:
                metrics[k] = np.nan
        
        metrics["accepted_multitask_avg_accuracy"] = np.nanmean([
            metrics["accepted_dr_accuracy"],
            metrics["accepted_dme_accuracy"]
        ])
    else:
        for k in ["accepted_dr_accuracy", "accepted_dr_f1_macro", "accepted_dr_qwk", 
                  "accepted_dme_accuracy", "accepted_dme_f1_macro", "accepted_dme_qwk", 
                  "accepted_multitask_avg_accuracy"]:
            metrics[k] = np.nan

    return metrics
