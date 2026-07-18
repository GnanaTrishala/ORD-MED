#!/usr/bin/env python
"""
Two-Stage Training Orchestrator for the ORD-MED framework.
  - Stage 1: Pre-train DR grading representation on APTOS dataset.
  - Stage 2: Fine-tune joint DR and DME diagnosis on IDRiD dataset.
  - Incorporates automatic recovery from CUDA OOM or disk errors.
  - Generates full evaluation statistics, plots, and a training summary.
"""

import os
import sys
import shutil
import json
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint
from datasets.dataset import get_dataloaders, verify_and_report_dataset
from models import build_model
from models.losses.multitask_loss import MultiTaskLoss
from trainers import MultiTaskTrainer
from evaluators import MultiTaskEvaluator
from visualization.plots import plot_evaluation_results


def create_stage_datasets():
    """Splits the integrated CSV files into Stage 1 (APTOS) and Stage 2 (IDRiD) sub-metadata."""
    print("Preparing Stage 1 (APTOS) and Stage 2 (IDRiD) partition files...")
    
    os.makedirs("dataset", exist_ok=True)
    
    splits = ["train", "val", "test"]
    for split in splits:
        integrated_path = f"dataset/integrated_{split}.csv"
        if not os.path.exists(integrated_path):
            raise FileNotFoundError(f"Missing integrated dataset mapping: {integrated_path}")
            
        df = pd.read_csv(integrated_path)
        
        # Filter for Stage 1 (APTOS only)
        df_aptos = df[df["dataset_source"] == "aptos"]
        df_aptos.to_csv(f"dataset/aptos_{split}_stage1.csv", index=False)
        
        # Filter for Stage 2 (IDRiD only)
        df_idrid = df[df["dataset_source"] == "idrid"]
        df_idrid.to_csv(f"dataset/idrid_{split}_stage2.csv", index=False)
        
    print("Partition metadata files successfully written to dataset/")


def run_training_stage(
    stage_num: int,
    config: Config,
    train_csv: str,
    val_csv: str,
    epochs: int,
    checkpoint_load_path: str = None,
    checkpoint_save_path: str = None,
    logger = None
) -> str:
    """Runs a single training stage with automatic error recovery capabilities."""
    logger.info(f"\n=======================================================")
    logger.info(f"STARTING TRAINING STAGE {stage_num}")
    logger.info(f"=======================================================")
    
    # 1. Update config parameters for this stage
    config.dataset.train_csv = train_csv
    config.dataset.val_csv = val_csv
    config.trainer.epochs = epochs
    config.trainer.checkpoint_path = checkpoint_load_path
    
    # 2. Setup training run parameters
    max_retries = 3
    retry_count = 0
    start_epoch = 0
    device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")

    # Load parameters and setup dataloaders
    verify_and_report_dataset(config, logger)
    train_loader, val_loader = get_dataloaders(config)

    model = build_model(config)
    model = model.to(device)
    criterion = MultiTaskLoss(config)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.trainer.lr, weight_decay=config.trainer.weight_decay)

    # Load starting checkpoint weights if provided
    if checkpoint_load_path and os.path.exists(checkpoint_load_path):
        logger.info(f"Loading checkpoint weights to start Stage {stage_num}: {checkpoint_load_path}")
        model, optimizer, start_epoch = load_checkpoint(model, checkpoint_load_path, optimizer)
        logger.info(f"Resuming stage from epoch {start_epoch + 1}")

    while retry_count < max_retries:
        try:
            # Re-initialize scheduler and dataloaders in case batch size changed during recovery
            train_loader, val_loader = get_dataloaders(config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.trainer.epochs,
                last_epoch=start_epoch - 1 if start_epoch > 0 else -1
            )

            trainer = MultiTaskTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                logger=logger,
                start_epoch=start_epoch
            )

            # Run training loop
            trainer.fit()
            break  # Success! Exit retry loop

        except RuntimeError as e:
            retry_count += 1
            logger.error(f"[FAILURE] Error encountered during training: {str(e)}")
            
            # Check for CUDA Out Of Memory (OOM) error
            is_oom = "out of memory" in str(e).lower()
            
            if is_oom:
                logger.warning("[RECOVERY] CUDA Out of Memory detected. Scaling down batch size...")
                # Scale down batch size by half
                new_bs = max(1, config.dataset.batch_size // 2)
                if new_bs == config.dataset.batch_size:
                    logger.error("[RECOVERY] Batch size is already 1. Cannot downscale further.")
                    raise e
                config.dataset.batch_size = new_bs
                logger.info(f"[RECOVERY] New batch size set to: {config.dataset.batch_size}")
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Identify last saved epoch checkpoint to resume
            latest_path = os.path.join(config.trainer.save_dir, f"{config.trainer.experiment_name}_latest.pth")
            if os.path.exists(latest_path):
                logger.info(f"[RECOVERY] Resuming training from last saved checkpoint: {latest_path}")
                model, optimizer, start_epoch = load_checkpoint(model, latest_path, optimizer)
            else:
                logger.warning("[RECOVERY] No checkpoint file found to resume from. Re-starting from epoch 0.")
                start_epoch = 0

            logger.info(f"[RECOVERY] Retry attempt {retry_count}/{max_retries}...")
            
    if retry_count >= max_retries:
        logger.error(f"Stage {stage_num} training failed after reaching max retries.")
        raise RuntimeError(f"Training failed in Stage {stage_num}")

    # Copy the best model checkpoint to the target save path
    best_temp_path = os.path.join(config.trainer.save_dir, f"{config.trainer.experiment_name}_best.pth")
    if os.path.exists(best_temp_path) and checkpoint_save_path:
        os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
        shutil.copy(best_temp_path, checkpoint_save_path)
        logger.info(f"Stage {stage_num} completed successfully! Best model saved to: {checkpoint_save_path}")
        return checkpoint_save_path
    
    return best_temp_path


def run_evaluation_and_reports(config: Config, checkpoint_path: str, logger) -> Dict[str, Any]:
    """Runs evaluation on the test dataset split and generates metric logs and plots."""
    logger.info("\n=======================================================")
    logger.info("RUNNING FINAL MULTI-TASK EVALUATION")
    logger.info("=======================================================")

    # 1. Update config test path
    config.dataset.test_csv = "dataset/integrated_test.csv"
    
    # 2. Get loader
    _, test_loader = get_dataloaders(config, splits=["train", "test"])
    
    # 3. Load model
    model = build_model(config)
    model, _, epoch = load_checkpoint(model, checkpoint_path)
    device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 4. Evaluate
    evaluator = MultiTaskEvaluator(
        model=model,
        dataloader=test_loader,
        device=device,
        config=config,
        logger=logger
    )
    metrics, predictions = evaluator.evaluate()

    # Save prediction CSV and reports
    predictions_dir = "outputs/predictions"
    reports_dir = "outputs/reports"
    figures_dir = "outputs/figures"
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    predictions.to_csv(os.path.join(predictions_dir, "integrated_test_predictions.csv"), index=False)
    
    # Export metrics JSON
    with open(os.path.join(reports_dir, "integrated_test_metrics.json"), "w") as f:
        cleaned_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}
        json.dump(cleaned_metrics, f, indent=4)

    # Save metrics CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(reports_dir, "integrated_test_metrics.csv"), index=False)

    # 5. Generate plots (Confusion Matrix, ROC, PR, Calibration)
    logger.info("Generating evaluation figures...")
    plot_evaluation_results(predictions, metrics, figures_dir)
    
    return metrics


def write_summary_report(metrics: Dict[str, Any], path: str):
    """Writes a markdown training summary report file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    report = f"""# ORD-MED Final Research Training Summary

This document summarizes the clinical diagnostic performance of the **ORD-MED** multi-task model trained using the pre-training (APTOS) and fine-tuning (IDRiD) transfer learning pipelines.

## 1. Diabetic Retinopathy (DR) Severity Grading
* **Accuracy**: {metrics.get('dr_accuracy', 0.0)*100:.2f}%
* **Precision (Macro)**: {metrics.get('dr_precision', 0.0)*100:.2f}%
* **Recall (Macro)**: {metrics.get('dr_recall', 0.0)*100:.2f}%
* **F1-Score (Macro)**: {metrics.get('dr_f1_macro', 0.0)*100:.2f}%
* **Quadratic Weighted Kappa (QWK)**: {metrics.get('dr_qwk', 0.0):.4f}
* **Expected Calibration Error (ECE)**: {metrics.get('dr_ece', 0.0):.4f}
* **Brier Score**: {metrics.get('dr_brier', 0.0):.4f}

## 2. Diabetic Macular Edema (DME) Staging
* **Accuracy**: {metrics.get('dme_accuracy', 0.0)*100:.2f}%
* **Precision (Macro)**: {metrics.get('dme_precision', 0.0)*100:.2f}%
* **Recall (Macro)**: {metrics.get('dme_recall', 0.0)*100:.2f}%
* **F1-Score (Macro)**: {metrics.get('dme_f1_macro', 0.0)*100:.2f}%
* **Quadratic Weighted Kappa (QWK)**: {metrics.get('dme_qwk', 0.0):.4f}
* **Expected Calibration Error (ECE)**: {metrics.get('dme_ece', 0.0):.4f}
* **Brier Score**: {metrics.get('dme_brier', 0.0):.4f}

## 3. Clinical Referral Module Analysis
* **AI Referral Rate**: {metrics.get('referral_rate', 0.0)*100:.2f}% (flagged for specialist review)
* **AI Coverage**: {metrics.get('coverage', 0.0)*100:.2f}% (accepted predictions)
* **Accepted Cases Count**: {metrics.get('accepted_count', 0)}
* **DR Accuracy on Accepted Cases**: {metrics.get('accepted_dr_accuracy', 0.0)*100:.2f}%
* **DME Accuracy on Accepted Cases**: {metrics.get('accepted_dme_accuracy', 0.0)*100:.2f}%
* **Integrated Multitask Accuracy on Accepted Cases**: {metrics.get('accepted_multitask_avg_accuracy', 0.0)*100:.2f}%

---
*Report generated automatically on completion of the two-stage training loop.*
"""
    with open(path, "w") as f:
        f.write(report)
    print(f"Research training summary successfully saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="ORD-MED Two-Stage Training Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--epochs_stage1", type=int, default=30, help="Epochs for Stage 1 (APTOS)")
    parser.add_argument("--epochs_stage2", type=int, default=20, help="Epochs for Stage 2 (IDRiD)")
    args = parser.parse_args()

    # Load base configuration
    config = Config.load_from_yaml(args.config)
    os.makedirs(config.trainer.log_dir, exist_ok=True)
    logger = setup_logger(config.trainer.log_dir, "two_stage_orchestrator")
    
    logger.info("Initializing ORD-MED Two-Stage Training Pipeline...")
    set_seed(config.trainer.seed)

    # 1. Split integrated files into stage-specific CSV files
    create_stage_datasets()

    # 2. Stage 1: Pre-train DR on APTOS
    stage1_train_csv = "dataset/aptos_train_stage1.csv"
    stage1_val_csv = "dataset/aptos_val_stage1.csv"
    best_dr_checkpoint = os.path.join(config.trainer.save_dir, "best_dr_model.pt")
    
    stage1_best_pth = run_training_stage(
        stage_num=1,
        config=config,
        train_csv=stage1_train_csv,
        val_csv=stage1_val_csv,
        epochs=args.epochs_stage1,
        checkpoint_load_path=None,
        checkpoint_save_path=best_dr_checkpoint,
        logger=logger
    )

    # 3. Stage 2: Fine-tune multi-task network on IDRiD
    stage2_train_csv = "dataset/idrid_train_stage2.csv"
    stage2_val_csv = "dataset/idrid_val_stage2.csv"
    best_final_checkpoint = os.path.join(config.trainer.save_dir, "best_model.pt")

    # We resume/initialize Stage 2 using the best model from Stage 1
    # Note: we clean the scheduler start_epoch to start fresh
    config.trainer.checkpoint_path = stage1_best_pth
    
    run_training_stage(
        stage_num=2,
        config=config,
        train_csv=stage2_train_csv,
        val_csv=stage2_val_csv,
        epochs=args.epochs_stage2,
        checkpoint_load_path=stage1_best_pth,
        checkpoint_save_path=best_final_checkpoint,
        logger=logger
    )

    # 4. Final Evaluation & Reports
    metrics = run_evaluation_and_reports(config, best_final_checkpoint, logger)

    # 5. Write final markdown research report
    write_summary_report(metrics, "outputs/training_summary.md")
    
    logger.info("\n=======================================================")
    logger.info("ORD-MED TWO-STAGE TRAINING WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("=======================================================")


if __name__ == "__main__":
    main()
