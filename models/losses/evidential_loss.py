import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal


class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning (EDL) Loss for classification.
    Reference: "Evidential Deep Learning on Prior Networks" (Sensoy et al., NeurIPS 2018).
    
    This loss treats predictions as parameters of a Dirichlet distribution (alpha).
    It is comprised of:
      1. Bayes Risk: Expected cross-entropy or mean squared error (MSE) over the Dirichlet distribution.
      2. KL Divergence: Regularizer that penalizes evidence for incorrect classes,
         forcing them toward a uniform prior (alpha_k -> 1).
    
    Dirichlet-based epistemic uncertainty is defined as:
        u_i = K / S_i
        where K is the number of classes, and S_i = sum_{k=1}^K alpha_{i, k} is the Dirichlet strength.
    """
    def __init__(self, num_classes: int, loss_type: Literal["mse", "ce"] = "mse") -> None:
        """
        Args:
            num_classes (int): Number of target classes.
            loss_type (str): Type of predictive risk term to use ('mse' or 'ce').
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type.lower()

    def kl_divergence(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence between the Dirichlet distribution parameterizing
        non-target class probabilities and a uniform Dirichlet distribution.
        
        Args:
            alpha (torch.Tensor): Dirichlet alpha parameters of shape (B, K), alpha >= 1.
            y (torch.Tensor): One-hot targets of shape (B, K).

        Returns:
            torch.Tensor: KL divergence per sample, shape (B,).
        """
        # Exclude the true class from being penalized (set its alpha to 1.0)
        alpha_tilde = y + (1.0 - y) * alpha
        sum_alpha_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        
        first_term = (
            torch.lgamma(sum_alpha_tilde)
            - torch.lgamma(torch.tensor(float(self.num_classes), device=alpha.device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
        )
        
        second_term = torch.sum(
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)),
            dim=-1,
            keepdim=True,
        )
        
        kl = first_term + second_term
        return kl.squeeze(-1)

    def forward(
        self, 
        alpha: torch.Tensor, 
        targets: torch.Tensor, 
        epoch: int = 0, 
        max_epochs: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the evidential loss and returns both the scalar loss and sample uncertainty.
        
        Args:
            alpha (torch.Tensor): Dirichlet concentration parameters of shape (B, K), alpha_k >= 1.
            targets (torch.Tensor): Ground-truth class labels of shape (B,) with values in [0, K-1].
            epoch (int): Current training epoch, used to scale KL regularization.
            max_epochs (int): Number of epochs to run KL annealing.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, uncertainty) where:
                - loss: scalar tensor representing average loss.
                - uncertainty: (B,) tensor representing epistemic uncertainty for each sample.
        
        Shapes:
            - alpha: (B, K)
            - targets: (B,)
            - Output loss: () [scalar]
            - Output uncertainty: (B,)
        """
        # Convert targets to one-hot encoding
        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Dirichlet strength: sum of all alphas per sample
        # Shape: (B, 1)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # Expected predictive class probabilities
        # Shape: (B, K)
        probs = alpha / S

        # 1. Compute Bayes Risk
        if self.loss_type == "mse":
            # Expected Mean Squared Error Risk
            risk_term1 = torch.sum((y_onehot - probs) ** 2, dim=-1)
            risk_term2 = torch.sum(probs * (1.0 - probs) / (S + 1.0), dim=-1)
            risk_loss = risk_term1 + risk_term2  # Shape: (B,)
        elif self.loss_type == "ce":
            # Expected Cross-Entropy Risk (Dirichlet-Multinomial Cross Entropy)
            risk_loss = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)  # Shape: (B,)
        else:
            raise ValueError(f"Unsupported evidential loss type: {self.loss_type}")

        # 2. Compute KL divergence regularization (penalizes misleading evidence)
        kl_loss = self.kl_divergence(alpha, y_onehot)  # Shape: (B,)
        
        # Annealing parameter coefficient: grows linearly from 0 to 1 as epochs progress
        annealing_coef = min(1.0, epoch / max_epochs)
        
        # Compute combined loss per sample, then average across batch
        sample_losses = risk_loss + annealing_coef * kl_loss
        total_loss = torch.mean(sample_losses)
        
        # Epistemic uncertainty: u_i = K / S_i
        # Shape: (B,)
        uncertainty = self.num_classes / S.squeeze(-1)
        
        return total_loss, uncertainty
