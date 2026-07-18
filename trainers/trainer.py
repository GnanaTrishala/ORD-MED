import os
import time
import random
import numpy as np
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
        os.makedirs(config.outputs.tensorboard, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.outputs.tensorboard)
        self.logger.info(f"TensorBoard logging directory: {config.outputs.tensorboard}")

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

        # 1. Search for auto-resume checkpoint (last.pt)
        last_checkpoint_path = os.path.join(self.config.trainer.save_dir, "last.pt")
        if os.path.exists(last_checkpoint_path):
            self.logger.info(f"Auto-resume checkpoint found: {last_checkpoint_path}")
            try:
                # Load the checkpoint fail-safely
                checkpoint = torch.load(last_checkpoint_path, map_location=self.device, weights_only=False)
                
                # Restore model weights
                state_dict = checkpoint["state_dict"]
                cleaned_state = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        cleaned_state[k.replace("module.", "")] = v
                    else:
                        cleaned_state[k] = v
                self.model.load_state_dict(cleaned_state, strict=True)
                self.logger.info("Model state dict successfully restored.")
                
                # Restore optimizer state
                if "optimizer" in checkpoint and checkpoint["optimizer"] is not None and self.optimizer is not None:
                    try:
                        self.optimizer.load_state_dict(checkpoint["optimizer"])
                        self.logger.info("Optimizer state successfully restored.")
                    except Exception as e:
                        self.logger.warning(f"Could not restore optimizer state dict: {e}")
                        
                # Restore scheduler state
                if "scheduler" in checkpoint and checkpoint["scheduler"] is not None and self.scheduler is not None:
                    try:
                        self.scheduler.load_state_dict(checkpoint["scheduler"])
                        self.logger.info("Scheduler state successfully restored.")
                    except Exception as e:
                        self.logger.warning(f"Could not restore scheduler state dict: {e}")
                        
                # Restore scaler state
                if "scaler" in checkpoint and checkpoint["scaler"] is not None and hasattr(self, "scaler") and self.scaler is not None:
                    try:
                        self.scaler.load_state_dict(checkpoint["scaler"])
                        self.logger.info("GradScaler state successfully restored.")
                    except Exception as e:
                        self.logger.warning(f"Could not restore GradScaler state dict: {e}")
                
                # Restore training variables
                self.start_epoch = checkpoint.get("epoch", 0)
                self.best_val_loss = checkpoint.get("best_metric", float("inf"))
                self.early_stopping_counter = checkpoint.get("early_stopping_counter", 0)
                
                # Restore random seeds
                if "seed_states" in checkpoint:
                    seed_states = checkpoint["seed_states"]
                    if "torch_rng_state" in seed_states and seed_states["torch_rng_state"] is not None:
                        try:
                            torch.set_rng_state(seed_states["torch_rng_state"].cpu())
                        except Exception:
                            pass
                    if "cuda_rng_state" in seed_states and seed_states["cuda_rng_state"] is not None and torch.cuda.is_available():
                        try:
                            torch.cuda.set_rng_state_all([s.cpu() for s in seed_states["cuda_rng_state"]])
                        except Exception:
                            pass
                    if "numpy_rng_state" in seed_states and seed_states["numpy_rng_state"] is not None:
                        try:
                            np.random.set_state(seed_states["numpy_rng_state"])
                        except Exception:
                            pass
                    if "random_rng_state" in seed_states and seed_states["random_rng_state"] is not None:
                        try:
                            random.setstate(seed_states["random_rng_state"])
                        except Exception:
                            pass
                            
                self.logger.info(f"Resume successful! Resuming from Epoch {self.start_epoch + 1}")
                print(f"\n>>> Resume successful! Resuming from Epoch {self.start_epoch + 1} with Best Val Loss: {self.best_val_loss:.4f}\n")
            except Exception as e:
                self.logger.warning(f"Error loading auto-resume checkpoint: {e}. Starting fresh from epoch {self.start_epoch + 1}")
        else:
            self.logger.info("No auto-resume checkpoint (last.pt) found. Starting from scratch.")

        epoch_times = []

        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            start_time = time.time()
            
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
            if is_best:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Prepare state dict to save
            seed_states = {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state() if hasattr(np.random, "get_state") else None,
                "random_rng_state": random.getstate()
            }
            checkpoint_data = {
                "state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler is not None else None,
                "epoch": epoch + 1,
                "best_metric": self.best_val_loss,
                "early_stopping_counter": self.early_stopping_counter,
                "seed_states": seed_states,
                "config": self.config
            }
            
            # Auto-resume & standard files saving
            saved_paths = []
            
            # last.pt
            last_path = os.path.join(self.config.trainer.save_dir, "last.pt")
            os.makedirs(os.path.dirname(last_path), exist_ok=True)
            torch.save(checkpoint_data, last_path)
            saved_paths.append("last.pt")
            
            # epoch_x.pt
            epoch_path = os.path.join(self.config.trainer.save_dir, f"epoch_{epoch + 1}.pt")
            torch.save(checkpoint_data, epoch_path)
            saved_paths.append(f"epoch_{epoch + 1}.pt")
            
            # best.pt
            if is_best:
                best_path = os.path.join(self.config.trainer.save_dir, "best.pt")
                torch.save(checkpoint_data, best_path)
                saved_paths.append("best.pt")
                self.logger.info(f"New best validation loss achieved: {self.best_val_loss:.4f}. Saving best model...")
                
            # Maintain existing checkpoint functionality as well
            latest_path = os.path.join(self.config.trainer.save_dir, f"{self.config.trainer.experiment_name}_latest.pth")
            save_checkpoint(self.model, self.optimizer, epoch + 1, latest_path)
            
            if is_best:
                best_path_old = os.path.join(self.config.trainer.save_dir, f"{self.config.trainer.experiment_name}_best.pth")
                save_checkpoint(self.model, self.optimizer, epoch + 1, best_path_old)

            # 5. Training display and ETA calculation
            epoch_elapsed = time.time() - start_time
            epoch_times.append(epoch_elapsed)
            remaining_epochs = self.config.trainer.epochs - (epoch + 1)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            
            display_str = (
                f"\n========================================\n"
                f"Epoch: {epoch + 1} / {self.config.trainer.epochs}\n"
                f"Remaining Epochs: {remaining_epochs}\n"
                f"Best Validation Loss: {self.best_val_loss:.4f}\n"
                f"Checkpoint Saved: {', '.join(saved_paths)}\n"
                f"ETA: {eta_str}\n"
                f"========================================\n"
            )
            print(display_str)
            self.logger.info(display_str.strip())

            # 6. Early stopping logic
            if not is_best:
                self.logger.info(
                    f"No validation loss improvement. Early stopping counter: "
                    f"{self.early_stopping_counter}/{self.patience}"
                )
            
            if self.early_stopping_counter >= self.patience:
                self.logger.info("Early stopping condition triggered. Terminating training pipeline.")
                break

        # Close TensorBoard summary writer
        self.writer.close()
