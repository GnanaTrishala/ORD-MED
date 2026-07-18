import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)
from typing import Dict, Any


def set_publication_style() -> None:
    """Sets a clean, publication-quality style for Matplotlib plots."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#2b2b2b"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = "#e0e0e0"


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    title: str,
    save_path: str
) -> None:
    """Generates and saves a clean, publication-quality confusion matrix."""
    set_publication_style()
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate row-wise normalized percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5), dpi=300)
    
    # We display normalized percentages in colors, but annotate with both counts and percentages
    labels = np.asarray([
        f"{count}\n({percent * 100:.1f}%)" 
        for count, percent in zip(cm.flatten(), cm_norm.flatten())
    ]).reshape(cm.shape)

    sns.heatmap(
        cm_norm, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        square=True,
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"shrink": 0.8}
    )
    
    plt.ylabel('True Class', fontsize=10, fontweight='bold', labelpad=10)
    plt.xlabel('Predicted Class', fontsize=10, fontweight='bold', labelpad=10)
    plt.title(title, fontsize=11, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list,
    title: str,
    save_path: str
) -> None:
    """Plots multi-class One-vs-Rest ROC curves with AUC values."""
    set_publication_style()
    plt.figure(figsize=(6, 5), dpi=300)
    
    num_classes = len(classes)
    colors = sns.color_palette("plasma", n_colors=num_classes)
    
    # Plot individual class curves
    for i in range(num_classes):
        # Convert target labels to binary for class i
        y_binary = (y_true == i).astype(int)
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_binary, probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr, 
            color=colors[i], 
            lw=1.5,
            label=f'{classes[i]} (AUC = {roc_auc:.3f})'
        )
        
    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.0)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10, fontweight='bold')
    plt.title(title, fontsize=11, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list,
    title: str,
    save_path: str
) -> None:
    """Plots multi-class One-vs-Rest Precision-Recall curves with Average Precision (AP)."""
    set_publication_style()
    plt.figure(figsize=(6, 5), dpi=300)
    
    num_classes = len(classes)
    colors = sns.color_palette("viridis", n_colors=num_classes)
    
    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, probs[:, i])
        ap = average_precision_score(y_binary, probs[:, i])
        
        plt.plot(
            recall, precision, 
            color=colors[i], 
            lw=1.5,
            label=f'{classes[i]} (AP = {ap:.3f})'
        )
        
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall (Sensitivity)', fontsize=10, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=10, fontweight='bold')
    plt.title(title, fontsize=11, fontweight='bold', pad=15)
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    title: str,
    save_path: str,
    n_bins: int = 10
) -> None:
    """Generates and saves a reliability diagram (calibration curve)."""
    set_publication_style()
    
    # Get highest predicted confidence and predicted class
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Identify samples falling in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        
        if count > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(np.nan)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
        bin_counts.append(count)
        
    # Calculate Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(y_true)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * np.abs(bin_accs[i] - bin_confs[i])
            
    fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Plot ideal calibration line y = x
    ax1.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.2, label='Ideal')
    
    # Plot empirical calibration points
    # Filter out bins with no samples for the line plot
    plot_confs = [c for c, count in zip(bin_confs, bin_counts) if count > 0]
    plot_accs = [a for a, count in zip(bin_accs, bin_counts) if count > 0]
    
    ax1.plot(plot_confs, plot_accs, marker='o', color='teal', lw=1.5, label=f'Model (ECE = {ece:.3f})')
    
    # Formatting left axis
    ax1.set_xlabel('Mean Predicted Confidence', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Observed Accuracy', fontsize=10, fontweight='bold')
    ax1.set_title(title, fontsize=11, fontweight='bold', pad=15)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Overlay histogram of bin counts on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(
        (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0, 
        bin_counts, 
        width=1.0/n_bins, 
        color='teal', 
        alpha=0.1, 
        edgecolor='teal', 
        lw=0.5
    )
    ax2.set_ylabel('Bin Sample Count', color='teal', fontsize=9, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='teal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uncertainty_vs_accuracy(
    df: pd.DataFrame,
    task_name: str,
    save_path: str,
    n_bins: int = 5
) -> None:
    """
    Plots predictive accuracy as a function of the model's epistemic uncertainty.
    Ideally, accuracy should drop as the model expresses higher uncertainty.
    """
    set_publication_style()
    uncertainty_col = f"{task_name}_uncertainty"
    target_col = f"{task_name}_target"
    pred_col = f"{task_name}_pred"
    
    if uncertainty_col not in df.columns or target_col not in df.columns:
        return

    # Bin the uncertainty values
    df_sorted = df.sort_values(by=uncertainty_col).copy()
    df_sorted["bin"] = pd.qcut(df_sorted[uncertainty_col], q=n_bins, labels=False, duplicates='drop')

    bin_accs = []
    bin_uncs = []
    
    for b in range(n_bins):
        sub_df = df_sorted[df_sorted["bin"] == b]
        if len(sub_df) == 0:
            continue
        acc = (sub_df[pred_col] == sub_df[target_col]).mean()
        unc = sub_df[uncertainty_col].mean()
        bin_accs.append(acc)
        bin_uncs.append(unc)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(bin_uncs, bin_accs, marker='o', linestyle='-', color='#7c5295', linewidth=2)
    plt.xlabel('Mean Epistemic Uncertainty', fontsize=10, fontweight='bold')
    plt.ylabel('Classification Accuracy', fontsize=10, fontweight='bold')
    plt.title(f'{task_name.upper()} Uncertainty vs. Accuracy Calibration', fontsize=11, fontweight='bold', pad=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_evaluation_results(
    df_predictions: pd.DataFrame,
    metrics: Dict[str, Any],
    save_dir: str
) -> None:
    """
    Orchestrates all output visualization plotting, creating publication-quality figures.

    Args:
        df_predictions (pd.DataFrame): DataFrame containing targets and predictions.
        metrics (dict): Computed evaluation metrics.
        save_dir (str): Folder destination to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. DR Analysis Plots
    if "dr_target" in df_predictions.columns:
        dr_df = df_predictions[df_predictions["dr_target"] != -100]
        if len(dr_df) > 0:
            dr_y_true = dr_df["dr_target"].values
            dr_y_pred = dr_df["dr_pred"].values
            dr_classes = [f"Grade {i}" for i in range(5)]
            
            # Confusion Matrix
            dr_qwk_val = metrics.get('dr_qwk', 0.0)
            dr_qwk_display = dr_qwk_val if dr_qwk_val is not None and not (isinstance(dr_qwk_val, float) and np.isnan(dr_qwk_val)) else 0.0
            plot_confusion_matrix(
                y_true=dr_y_true,
                y_pred=dr_y_pred,
                classes=dr_classes,
                title=f"DR Confusion Matrix (QWK = {dr_qwk_display:.3f})",
                save_path=os.path.join(save_dir, "dr_confusion_matrix.png")
            )

            # Retrieve probabilities columns
            dr_prob_cols = sorted([c for c in dr_df.columns if c.startswith("dr_prob_class_")])
            if len(dr_prob_cols) == 5:
                dr_probs = dr_df[dr_prob_cols].values
                
                # ROC Curves
                plot_roc_curves(
                    y_true=dr_y_true,
                    probs=dr_probs,
                    classes=dr_classes,
                    title="DR Severity Grading ROC Curves (OVR)",
                    save_path=os.path.join(save_dir, "dr_roc_curves.png")
                )
                
                # PR Curves
                plot_precision_recall_curves(
                    y_true=dr_y_true,
                    probs=dr_probs,
                    classes=dr_classes,
                    title="DR Severity Grading Precision-Recall Curves (OVR)",
                    save_path=os.path.join(save_dir, "dr_pr_curves.png")
                )
                
                # Calibration Curves
                plot_calibration_curve(
                    y_true=dr_y_true,
                    probs=dr_probs,
                    title="DR Calibration Reliability Diagram",
                    save_path=os.path.join(save_dir, "dr_calibration_curve.png")
                )

    # 2. DME Analysis Plots
    if "dme_target" in df_predictions.columns:
        dme_df = df_predictions[df_predictions["dme_target"] != -100]
        if len(dme_df) > 0:
            dme_y_true = dme_df["dme_target"].values
            dme_y_pred = dme_df["dme_pred"].values
            dme_classes = [f"Stage {i}" for i in range(3)]
            
            # Confusion Matrix
            dme_qwk_val = metrics.get('dme_qwk', 0.0)
            dme_qwk_display = dme_qwk_val if dme_qwk_val is not None and not (isinstance(dme_qwk_val, float) and np.isnan(dme_qwk_val)) else 0.0
            plot_confusion_matrix(
                y_true=dme_y_true,
                y_pred=dme_y_pred,
                classes=dme_classes,
                title=f"DME Confusion Matrix (QWK = {dme_qwk_display:.3f})",
                save_path=os.path.join(save_dir, "dme_confusion_matrix.png")
            )

            # Retrieve probabilities columns
            dme_prob_cols = sorted([c for c in dme_df.columns if c.startswith("dme_prob_class_")])
            if len(dme_prob_cols) == 3:
                dme_probs = dme_df[dme_prob_cols].values
                
                # ROC Curves
                plot_roc_curves(
                    y_true=dme_y_true,
                    probs=dme_probs,
                    classes=dme_classes,
                    title="DME Staging ROC Curves (OVR)",
                    save_path=os.path.join(save_dir, "dme_roc_curves.png")
                )
                
                # PR Curves
                plot_precision_recall_curves(
                    y_true=dme_y_true,
                    probs=dme_probs,
                    classes=dme_classes,
                    title="DME Staging Precision-Recall Curves (OVR)",
                    save_path=os.path.join(save_dir, "dme_pr_curves.png")
                )
                
                # Calibration Curves
                plot_calibration_curve(
                    y_true=dme_y_true,
                    probs=dme_probs,
                    title="DME Calibration Reliability Diagram",
                    save_path=os.path.join(save_dir, "dme_calibration_curve.png")
                )

    # 3. Evidential Uncertainty vs. Accuracy
    if "dr_uncertainty" in df_predictions.columns:
        plot_uncertainty_vs_accuracy(
            df=df_predictions,
            task_name="dr",
            save_path=os.path.join(save_dir, "dr_uncertainty_vs_accuracy.png")
        )
        plot_uncertainty_vs_accuracy(
            df=df_predictions,
            task_name="dme",
            save_path=os.path.join(save_dir, "dme_uncertainty_vs_accuracy.png")
        )
