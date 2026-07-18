#!/usr/bin/env python
"""
Main training script for the ORD-MED network.
"""

import os
import sys
import torch

from config import parse_args
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from datasets.dataset import get_dataloaders
from models import build_model
from models.losses.multitask_loss import MultiTaskLoss
from trainers import MultiTaskTrainer


def main():
    # 1. Parse configuration and command line args
    config = parse_args()

    # Auto-resume check: automatically detect and load latest interrupted checkpoint if available
    interrupted_path = os.path.join(config.trainer.save_dir, "interrupted.pth")
    if os.path.exists(interrupted_path):
        try:
            checkpoint = torch.load(interrupted_path, map_location="cpu")
            loaded_epoch = checkpoint.get("epoch", 0)
            print(f"Checkpoint found. Resuming from Epoch {loaded_epoch}.")
        except Exception:
            print("Checkpoint found. Resuming from Epoch (unknown).")
        config.trainer.checkpoint_path = interrupted_path
    else:
        print("No checkpoint found. Starting fresh.")

    # 2. Setup logging and output directories
    os.makedirs(config.trainer.save_dir, exist_ok=True)
    os.makedirs(config.trainer.log_dir, exist_ok=True)
    logger = setup_logger(config.trainer.log_dir, config.trainer.experiment_name)
    logger.info("Initializing ORD-MED Training Pipeline...")

    # 3. Enforce reproducibility
    set_seed(config.trainer.seed)
    logger.info(f"Random seed set to {config.trainer.seed}")

    # 4. Initialize DataLoaders
    logger.info("Setting up data loaders...")
    from datasets.dataset import verify_and_report_dataset
    verify_and_report_dataset(config, logger)
    train_loader, val_loader = get_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 5. Build Model Architecture
    logger.info(f"Building model with encoder: {config.encoder.name}...")
    model = build_model(config)
    device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model successfully transferred to {device}")

    # 6. Initialize Multi-task Criterion
    logger.info("Defining multi-task loss criterion...")
    criterion = MultiTaskLoss(config)

    # 7. Setup Optimizer and Learning Rate Scheduler
    logger.info("Setting up optimization packages...")
    params = [p for p in model.parameters() if p.requires_grad]
    
    if config.trainer.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            params,
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay
        )

    # 8. Load checkpoint if configured
    start_epoch = 0
    if config.trainer.checkpoint_path:
        logger.info(f"Resuming training from checkpoint: {config.trainer.checkpoint_path}...")
        model, optimizer, start_epoch = load_checkpoint(model, config.trainer.checkpoint_path, optimizer)

    # Adjust learning rate scheduler
    if config.trainer.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.trainer.epochs,
            last_epoch=start_epoch - 1 if start_epoch > 0 else -1
        )
    else:
        scheduler = None

    # 9. Instantiate Trainer and run training loop
    logger.info("Initializing MultiTaskTrainer...")
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

    logger.info(f"Starting training for {config.trainer.epochs} epochs starting at epoch {start_epoch + 1}...")
    try:
        trainer.fit()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current checkpoint...")
        save_checkpoint(model, optimizer, epoch=-1, path=os.path.join(config.trainer.save_dir, "interrupted.pth"))
    except Exception as e:
        logger.error(f"Training failed due to error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
