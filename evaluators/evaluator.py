from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from config import Config
from utils.metrics import compute_multitask_metrics


class MultiTaskEvaluator:
    """
    Evaluator class for aggregating predictions and computing multi-task metrics.
    Runs predictions on the given validation/test DataLoader and returns:
      - Dictionary of clinical and calibration metrics.
      - DataFrame of batch predictions, ground truths, and uncertainties.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        config: Config,
        logger: Any
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.logger = logger
        self.use_evidential = config.heads.use_evidential

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs evaluation, collects predictions and annotations, 
        and calculates metrics.

        Returns:
            Tuple[Dict[str, Any], pd.DataFrame]: (metrics_dict, predictions_df)
        """
        self.model.eval()

        all_dr_targets = []
        all_dme_targets = []
        all_dr_probs = []
        all_dme_probs = []
        all_dr_preds = []
        all_dme_preds = []
        
        # Evidential and path tracking placeholders
        all_dr_uncertainties = []
        all_dme_uncertainties = []
        all_image_paths = []

        self.logger.info("Iterating over evaluation loader batches...")
        for batch in self.dataloader:
            images = batch["image"].to(self.device)
            all_image_paths.extend(batch["image_path"])

            # Collect labels if available
            if "dr_label" in batch:
                all_dr_targets.extend(batch["dr_label"].cpu().numpy())
                all_dme_targets.extend(batch["dme_label"].cpu().numpy())

            # Model forward pass
            outputs = self.model(images)

            # Softmax probabilities and predicted classes
            dr_logits = outputs["dr_logits"]
            dme_logits = outputs["dme_logits"]

            # DME uses standard softmax
            dme_probs = torch.softmax(dme_logits, dim=-1)
            dme_preds = torch.argmax(dme_probs, dim=-1)

            # DR uses CORN or EMD
            ordinal_method = getattr(self.config.loss, "ordinal_method", "corn")
            if ordinal_method == "corn":
                num_classes_dr = self.config.heads.dr_num_classes
                corn_logits = dr_logits[:, :num_classes_dr - 1]
                
                sigmoids = torch.sigmoid(corn_logits)
                cum_probs = torch.ones(sigmoids.size(0), num_classes_dr, device=sigmoids.device)
                cum_probs[:, 1:] = torch.cumprod(sigmoids, dim=-1)
                
                dr_probs = torch.zeros_like(cum_probs)
                dr_probs[:, :-1] = cum_probs[:, :-1] - cum_probs[:, 1:]
                dr_probs[:, -1] = cum_probs[:, -1]
                
                dr_preds = (corn_logits > 0).sum(dim=-1)
            else:
                dr_probs = torch.softmax(dr_logits, dim=-1)
                dr_preds = torch.argmax(dr_probs, dim=-1)

            all_dr_probs.extend(dr_probs.cpu().numpy())
            all_dme_probs.extend(dme_probs.cpu().numpy())

            all_dr_preds.extend(dr_preds.cpu().numpy())
            all_dme_preds.extend(dme_preds.cpu().numpy())

            # Epistemic uncertainty u = K / S if evidential is active
            if self.use_evidential and "dr_evidence" in outputs:
                dr_alpha = outputs["dr_evidence"]
                dme_alpha = outputs["dme_evidence"]

                dr_S = torch.sum(dr_alpha, dim=-1)
                dme_S = torch.sum(dme_alpha, dim=-1)
                
                dr_u = dr_alpha.shape[-1] / dr_S
                dme_u = dme_alpha.shape[-1] / dme_S

                all_dr_uncertainties.extend(dr_u.cpu().numpy())
                all_dme_uncertainties.extend(dme_u.cpu().numpy())

        # Construct prediction records dictionary
        pred_data = {
            "image_path": all_image_paths,
            "dr_pred": all_dr_preds,
            "dme_pred": all_dme_preds
        }

        # Add true labels if present
        if len(all_dr_targets) > 0:
            pred_data["dr_target"] = all_dr_targets
            pred_data["dme_target"] = all_dme_targets

        # Add prob variables class wise
        dr_probs_arr = np.array(all_dr_probs)
        dme_probs_arr = np.array(all_dme_probs)
        for i in range(dr_probs_arr.shape[1]):
            pred_data[f"dr_prob_class_{i}"] = dr_probs_arr[:, i]
        for i in range(dme_probs_arr.shape[1]):
            pred_data[f"dme_prob_class_{i}"] = dme_probs_arr[:, i]

        # Add evidential uncertainty
        if self.use_evidential and len(all_dr_uncertainties) > 0:
            pred_data["dr_uncertainty"] = all_dr_uncertainties
            pred_data["dme_uncertainty"] = all_dme_uncertainties

        df_predictions = pd.DataFrame(pred_data)

        # 3. Compute clinical metrics if targets are present
        metrics = {}
        if "dr_target" in df_predictions.columns:
            metrics = compute_multitask_metrics(df_predictions, self.config)

        return metrics, df_predictions
