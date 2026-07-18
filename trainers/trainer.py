import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict, Optional

from config import Config
from utils.checkpoint import save_checkpoint, load_checkpoint
from evaluators import MultiTaskEvaluator


class MultiTaskTrainer:
    """
    Production-quality Trainer class for the ORD-MED network.
    Coordinates:
      - Training epochs with Automatic Mixed Precision (AMP)
      - Validation epochs evaluating clinical metrics (QWK, F1, ECE)
      - Gradient clipping for training stability
      - Optimizer and learning rate scheduler updates
      - TensorBoard logging
      - Early stopping based on validation loss
      - Automatic checkpoint resumption and saving of best models
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        config: Config,
        logger: Any,
        start_epoch: int = 0
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.start_epoch = start_epoch

        # Early Stopping parameters
        self.patience = getattr(config.trainer, "patience", 10)
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

        # Mixed Precision (AMP) setup
        self.use_amp = config.trainer.use_amp and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Gradient clipping norm
        self.max_grad_norm = getattr(config.trainer, "max_grad_norm", 5.0)

        # TensorBoard Logger
        self.writer = SummaryWriter(log_dir=config.trainer.log_dir)
        self.logger.info(f"TensorBoard logging directory: {config.trainer.log_dir}")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs one full epoch of training.
        
        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: Average training loss values.
        """
        self.model.train()
        running_losses: Dict[str, float] = {}
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.trainer.epochs} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            targets = {
                "dr_label": batch["dr_label"].to(self.device),
                "dme_label": batch["dme_label"].to(self.device)
            }

            self.optimizer.zero_grad()
            
            # Forward pass under autocast context for mixed precision
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets, epoch=epoch)
                loss = loss_dict["loss"]
            
            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            # Step optimizer & scaler update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Record running losses
            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        num_batches = len(self.train_loader)
        epoch_losses = {k: v / num_batches for k, v in running_losses.items()}
        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Runs validation and evaluates both losses and clinical metrics.
        
        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: Validation loss and evaluation metrics.
        """
        self.model.eval()
        running_losses: Dict[str, float] = {}

        # 1. Compute loss values on the validation set
        pbar = tqdm(
            self.val_loader, 
            desc=f"Epoch {epoch+1}/{self.config.trainer.epochs} [Val Losses]",
            leave=False
        )
        for batch in pbar:
            images = batch["image"].to(self.device)
            targets = {
                "dr_label": batch["dr_label"].to(self.device),
                "dme_label": batch["dme_label"].to(self.device)
            }

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets, epoch=epoch)

            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v.item()

        num_batches = len(self.val_loader)
        val_results = {f"val_{k}": v / num_batches for k, v in running_losses.items()}

        # 2. Evaluate clinical metrics (Accuracy, QWK, F1) using MultiTaskEvaluator
        self.logger.info("Computing validation metrics (QWK, F1, Accuracy)...")
        evaluator = MultiTaskEvaluator(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            config=self.config,
            logger=self.logger
        )
        metrics, _ = evaluator.evaluate()
        
        # Merge metrics into validation results
        val_results.update(metrics)
        return val_results

    def fit(self) -> None:
        """Runs the complete multi-epoch training pipeline."""
        self.logger.info("Starting training loop execution...")

        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            # 1. Run training epoch
            train_losses = self.train_one_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log training metrics
            self.logger.info(
                f"Epoch {epoch+1} - Train Loss: {train_losses['loss']:.4f} | "
                f"DR Loss: {train_losses['dr_loss']:.4f} | "
                f"DME Loss: {train_losses['dme_loss']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Log training to TensorBoard
            self.writer.add_scalar("Train/Total_Loss", train_losses["loss"], epoch)
            self.writer.add_scalar("Train/DR_Loss", train_losses["dr_loss"], epoch)
            self.writer.add_scalar("Train/DME_Loss", train_losses["dme_loss"], epoch)
            self.writer.add_scalar("Train/Evidential_Loss", train_losses.get("evidential_loss", 0.0), epoch)
            self.writer.add_scalar("Trainer/Learning_Rate", current_lr, epoch)

            # 2. Run validation epoch
            val_results = self.validate(epoch)
            
            # Log validation loss & metrics
            self.logger.info(
                f"Epoch {epoch+1} - Val Loss: {val_results['val_loss']:.4f} | "
                f"DR Acc: {val_results.get('dr_accuracy', 0.0):.4f} | "
                f"DR QWK: {val_results.get('dr_qwk', 0.0):.4f} | "
                f"DME Acc: {val_results.get('dme_accuracy', 0.0):.4f} | "
                f"DME QWK: {val_results.get('dme_qwk', 0.0):.4f}"
            )

            # Log validation to TensorBoard
            self.writer.add_scalar("Val/Total_Loss", val_results["val_loss"], epoch)
            self.writer.add_scalar("Val/DR_Loss", val_results["val_dr_loss"], epoch)
            self.writer.add_scalar("Val/DME_Loss", val_results["val_dme_loss"], epoch)
            self.writer.add_scalar("Val/DR_Accuracy", val_results.get("dr_accuracy", 0.0), epoch)
            self.writer.add_scalar("Val/DME_Accuracy", val_results.get("dme_accuracy", 0.0), epoch)
            self.writer.add_scalar("Val/DR_QWK", val_results.get("dr_qwk", 0.0), epoch)
            self.writer.add_scalar("Val/DME_QWK", val_results.get("dme_qwk", 0.0), epoch)
            self.writer.add_scalar("Val/DR_F1_Macro", val_results.get("dr_f1_macro", 0.0), epoch)
            self.writer.add_scalar("Val/DME_F1_Macro", val_results.get("dme_f1_macro", 0.0), epoch)
            
            if self.config.heads.use_evidential:
                self.writer.add_scalar("Val/DR_Uncertainty", val_results.get("dr_uncertainty", 0.0), epoch)
                self.writer.add_scalar("Val/DME_Uncertainty", val_results.get("dme_uncertainty", 0.0), epoch)

            # 3. Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results["val_loss"])
                else:
                    self.scheduler.step()

            # 4. Checkpoint saving
            val_loss = val_results["val_loss"]
            is_best = val_loss < self.best_val_loss
            
            # Save latest training progress state
            latest_path = os.path.join(
                self.config.trainer.save_dir, 
                f"{self.config.trainer.experiment_name}_latest.pth"
            )
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                path=latest_path
            )
            
            # Save best performing checkpoint
            if is_best:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.logger.info(f"New best validation loss achieved: {self.best_val_loss:.4f}. Saving best model...")
                best_path = os.path.join(
                    self.config.trainer.save_dir, 
                    f"{self.config.trainer.experiment_name}_best.pth"
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    path=best_path
                )
            else:
                self.early_stopping_counter += 1
                self.logger.info(
                    f"No validation loss improvement. Early stopping counter: "
                    f"{self.early_stopping_counter}/{self.patience}"
                )

            # 5. Early Stopping check
            if self.early_stopping_counter >= self.patience:
                self.logger.info("Early stopping condition triggered. Terminating training pipeline.")
                break

        # Close TensorBoard summary writer
        self.writer.close()
