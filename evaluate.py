#!/usr/bin/env python
"""
Evaluation script for assessing an ORD-MED model checkpoint on a dataset split.
Computes multi-task metrics, calibration stats, clinical referral metrics, and exports reports.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch

from config import Config
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint
from datasets.dataset import get_dataloaders
from models import build_model
from evaluators import MultiTaskEvaluator
from visualization.plots import plot_evaluation_results


def main():
    parser = argparse.ArgumentParser(description="ORD-MED Evaluation Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory to save evaluation reports")
    args = parser.parse_args()

    # 1. Load config and setup loggers
    config = Config.load_from_yaml(args.config)
    
    # If the user passed output_dir, override the base_dir of config.outputs
    if args.output_dir != "outputs/":
        config.outputs.base_dir = args.output_dir
        config.outputs.__post_init__()
        config.__post_init__()
        
    os.makedirs(config.outputs.logs, exist_ok=True)
    logger = setup_logger(config.outputs.logs, f"eval_{config.trainer.experiment_name}")
    logger.info(f"Starting evaluation of checkpoint: {args.checkpoint} on split: {args.split}")

    # Set up outputs directories
    predictions_dir = config.outputs.predictions
    figures_dir = config.outputs.plots
    reports_dir = config.outputs.metrics
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 2. Get dataloader
    logger.info("Initializing dataset loaders...")
    # Typically, get_dataloaders gets train/val, we'll retrieve the split requested
    _, eval_loader = get_dataloaders(config, splits=["train", args.split])
    logger.info(f"Total evaluation batches: {len(eval_loader)}")

    # 3. Build Model Architecture
    logger.info("Building model architecture...")
    model = build_model(config)
    
    # 4. Load weights
    logger.info(f"Loading checkpoint weights from {args.checkpoint}...")
    model, _, epoch = load_checkpoint(model, checkpoint_path=args.checkpoint)
    
    device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model placed on {device} (Checkpoint trained up to epoch: {epoch})")

    # 5. Run Evaluation Loop
    logger.info("Running evaluation loop...")
    evaluator = MultiTaskEvaluator(
        model=model,
        dataloader=eval_loader,
        device=device,
        config=config,
        logger=logger
    )

    metrics, predictions = evaluator.evaluate()
    
    logger.info("=== Evaluation Metrics Summary ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 6. Save Predictions DataFrame to CSV
    predictions_path = os.path.join(predictions_dir, f"predictions_{args.split}.csv")
    predictions.to_csv(predictions_path, index=False)
    logger.info(f"Saved evaluation predictions to: {predictions_path}")

    # 7. Save Metrics as JSON and CSV
    metrics_json_path = os.path.join(reports_dir, f"metrics_{args.split}.json")
    with open(metrics_json_path, "w") as f:
        # Convert nan values to null for valid JSON
        cleaned_metrics = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}
        json.dump(cleaned_metrics, f, indent=4)
        
    metrics_csv_path = os.path.join(reports_dir, f"metrics_{args.split}.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved metrics reports to: {metrics_json_path} and {metrics_csv_path}")

    # 8. Generate and save publication-quality plots (Confusion Matrix, ROC, PR, Calibration Curves)
    logger.info(f"Generating publication-quality figures and plots in: {figures_dir}...")
    plot_evaluation_results(predictions, metrics, save_dir=figures_dir)
    logger.info("Figures generated and saved successfully!")


if __name__ == "__main__":
    main()
