import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from config import Config
from .ordinal_loss import OrdinalLoss
from .evidential_loss import EvidentialLoss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Ref: "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    
    Supports both binary (BCE-style) and multi-class (CE-style) classification.
    
    Mathematical formulation:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        where p_t is the model's estimated probability for the ground truth class.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        """
        Args:
            gamma (float): Focusing parameter to adjust down-weighting of easy examples.
            alpha (torch.Tensor, optional): Class weights or binary alpha parameter, shape (C,).
            reduction (str): Loss reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss.
        
        Shapes:
            - logits: (B, C) where B is batch size, C is number of classes.
            - targets: (B, C) [one-hot for binary style] or (B,) [indices for multi-class style].
        """
        # If target has the same shape as logits, compute binary/multi-label BCE focal loss
        if logits.dim() == targets.dim():
            p = torch.sigmoid(logits)
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            loss = ce_loss * ((1.0 - p_t) ** self.gamma)
            
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
                loss = alpha_t * loss
        else:
            # Multi-class softmax focal loss
            log_p = F.log_softmax(logits, dim=-1)
            ce_loss = F.nll_loss(log_p, targets, reduction="none")
            
            p = torch.exp(log_p)
            p_t = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            
            loss = ce_loss * ((1.0 - p_t) ** self.gamma)
            
            if self.alpha is not None:
                alpha_t = self.alpha.gather(dim=-1, index=targets)
                loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DMELoss(nn.Module):
    """
    DME Staging Loss.
    Supports standard cross-entropy ('ce'), binary cross-entropy ('bce'), and Focal Loss ('focal').
    Supports class weighting to address severity imbalances.
    """
    def __init__(
        self,
        num_classes: int = 3,
        loss_type: str = "bce",
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0
    ) -> None:
        """
        Args:
            num_classes (int): Number of target classes.
            loss_type (str): Loss formulation ('bce', 'focal', or 'ce').
            weight (torch.Tensor, optional): Class-wise weights tensor of shape (num_classes,).
            gamma (float): Focusing parameter for Focal Loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type.lower()
        self.weight = weight
        self.gamma = gamma
        
        if self.loss_type == "focal":
            self.focal_loss = FocalLoss(gamma=gamma, alpha=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes task-specific DME loss.
        
        Shapes:
            - logits: (B, num_classes) where B is batch size.
            - targets: (B,) containing class indices in the range [0, num_classes-1].
        """
        if self.loss_type == "bce":
            # Convert targets to one-hot encoding for multi-label BCE style
            targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
            
            if self.weight is not None:
                # Multiply element-wise binary losses by class-specific weights
                loss = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
                loss = loss * self.weight.unsqueeze(0)
                return loss.mean()
            return F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="mean")
            
        elif self.loss_type == "focal":
            # Multi-label BCE-style focal loss is applied to the one-hot encoded targets
            targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
            return self.focal_loss(logits, targets_onehot)
            
        elif self.loss_type == "ce":
            # Standard multi-class Cross Entropy
            return F.cross_entropy(logits, targets, weight=self.weight, reduction="mean")
            
        else:
            raise ValueError(f"Unsupported DME loss type: {self.loss_type}")


class MultiTaskLoss(nn.Module):
    """
    Combined Multi-task Loss for ORD-MED.
    This module coordinates:
      - Ordinal classification loss (CORN or EMD) for the DR grading branch.
      - Imbalance-robust loss (weighted BCE/CE or Focal) for the DME staging branch.
      - Evidential deep learning losses for uncertainty estimation (if enabled).
      
    Total Loss Formulation:
        Total Loss = lambda1 * DR Loss + lambda2 * DME Loss + lambda3 * Evidential Loss
        where Evidential Loss is the sum of DR and DME branch evidential losses.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.use_evidential = config.heads.use_evidential

        # 1. Read lambda weights from config
        self.lambda1 = getattr(config.loss, "lambda1", 1.0)
        self.lambda2 = getattr(config.loss, "lambda2", 1.0)
        self.lambda3 = getattr(config.loss, "lambda3", 0.5)

        # 2. Setup DR Ordinal Loss (prefer CORN, fall back to EMD based on config)
        ordinal_method = getattr(config.loss, "ordinal_method", "corn")
        self.dr_ordinal = OrdinalLoss(
            num_classes=config.heads.dr_num_classes,
            method=ordinal_method
        )

        # 3. Setup DME Class Weighting Buffer
        dme_weights_list = getattr(config.loss, "dme_class_weights", None)
        if dme_weights_list is not None:
            self.register_buffer("dme_class_weights", torch.tensor(dme_weights_list, dtype=torch.float))
        else:
            self.dme_class_weights = None

        # 4. Setup DME Staging Loss
        dme_loss_type = getattr(config.loss, "dme_loss_type", "bce")
        focal_gamma = getattr(config.loss, "focal_gamma", 2.0)
        self.dme_loss_fn = DMELoss(
            num_classes=config.heads.dme_num_classes,
            loss_type=dme_loss_type,
            weight=self.dme_class_weights,
            gamma=focal_gamma
        )

        # 5. Setup Evidential Losses if enabled
        if self.use_evidential:
            # We map the standard evidential loss types ('mse' or 'ce')
            evidential_loss_type = getattr(config.loss, "evidential_loss_type", "mse")
            self.dr_evidential = EvidentialLoss(
                num_classes=config.heads.dr_num_classes,
                loss_type=evidential_loss_type
            )
            self.dme_evidential = EvidentialLoss(
                num_classes=config.heads.dme_num_classes,
                loss_type=evidential_loss_type
            )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the multi-task loss.
        
        Args:
            outputs (dict): Predictions containing 'dr_logits', 'dme_logits',
                            and optionally 'dr_evidence', 'dme_evidence'.
            targets (dict): Target labels containing 'dr_label' and 'dme_label'.
            epoch (int): Current epoch number (for KL divergence annealing in evidential loss).

        Returns:
            dict: Containing individual scalar loss values and average epistemic uncertainties.
            
        Shapes:
            - outputs['dr_logits']: (B, K_dr)
            - outputs['dme_logits']: (B, K_dme)
            - targets['dr_label']: (B,)
            - targets['dme_label']: (B,)
            - Outputs in dict: All values are scalar (0D tensors).
        """
        dr_targets = targets["dr_label"]
        dme_targets = targets["dme_label"]

        dr_mask = dr_targets != -100
        dme_mask = dme_targets != -100

        # --- 1. DR Task Loss (Ordinal) ---
        dr_logits = outputs["dr_logits"]
        if dr_mask.any():
            loss_dr = self.dr_ordinal(dr_logits[dr_mask], dr_targets[dr_mask])
        else:
            loss_dr = torch.tensor(0.0, device=dr_logits.device)

        # --- 2. DME Task Loss (BCE/Focal/CE) ---
        dme_logits = outputs["dme_logits"]
        if dme_mask.any():
            loss_dme = self.dme_loss_fn(dme_logits[dme_mask], dme_targets[dme_mask])
        else:
            loss_dme = torch.tensor(0.0, device=dme_logits.device)

        # --- 3. Evidential Losses & Epistemic Uncertainty Extraction ---
        loss_dr_ev = torch.tensor(0.0, device=dr_logits.device)
        loss_dme_ev = torch.tensor(0.0, device=dme_logits.device)
        dr_u_mean = torch.tensor(0.0, device=dr_logits.device)
        dme_u_mean = torch.tensor(0.0, device=dme_logits.device)

        if self.use_evidential and "dr_evidence" in outputs:
            dr_alpha = outputs["dr_evidence"]
            dme_alpha = outputs["dme_evidence"]
            
            # Compute evidential losses and get uncertainties per sample
            if dr_mask.any():
                loss_dr_ev, dr_u = self.dr_evidential(dr_alpha[dr_mask], dr_targets[dr_mask], epoch=epoch)
                dr_u_mean = torch.mean(dr_u)
                
            if dme_mask.any():
                loss_dme_ev, dme_u = self.dme_evidential(dme_alpha[dme_mask], dme_targets[dme_mask], epoch=epoch)
                dme_u_mean = torch.mean(dme_u)

        # --- 4. Final Aggregated Multi-Task Loss ---
        # Evidential loss term combines the uncertainty penalties across both tasks
        evidential_loss = loss_dr_ev + loss_dme_ev
        
        total_loss = (
            self.lambda1 * loss_dr +
            self.lambda2 * loss_dme +
            self.lambda3 * evidential_loss
        )

        return {
            "loss": total_loss,
            "dr_loss": loss_dr,
            "dme_loss": loss_dme,
            "dr_evidential_loss": loss_dr_ev,
            "dme_evidential_loss": loss_dme_ev,
            "evidential_loss": evidential_loss,
            "dr_uncertainty": dr_u_mean,
            "dme_uncertainty": dme_u_mean
        }
