from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from config import Config


class ReferralModule(nn.Module):
    """
    Automated Referral Module based on Evidential Uncertainty and severity thresholding.
    In clinical settings, a sample should be referred to a human specialist if:
      1. The predicted severity grade exceeds a safe diagnostic threshold (e.g., Moderate DR or worse).
      2. The model's predictive (epistemic) uncertainty is high, indicating out-of-distribution
         or ambiguous visual features.
    """
    def __init__(
        self,
        config: Optional[Config] = None,
        dr_severity_threshold: Optional[int] = None,
        dme_severity_threshold: Optional[int] = None,
        uncertainty_threshold: Optional[float] = None
    ):
        super().__init__()
        
        # Load from central config or fall back to explicit arguments / defaults
        if config is not None:
            self.dr_severity_threshold = config.referral.dr_severity_threshold
            self.dme_severity_threshold = config.referral.dme_severity_threshold
            self.uncertainty_threshold = config.referral.uncertainty_threshold
        else:
            self.dr_severity_threshold = dr_severity_threshold if dr_severity_threshold is not None else 2
            self.dme_severity_threshold = dme_severity_threshold if dme_severity_threshold is not None else 1
            self.uncertainty_threshold = uncertainty_threshold if uncertainty_threshold is not None else 0.4

    def compute_uncertainty(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes the epistemic uncertainty (u) from Dirichlet alpha parameters.
        u = K / S, where S = sum(alpha_k) and K is the number of classes.
        """
        K = alpha.shape[-1]
        S = torch.sum(alpha, dim=-1)
        u = K / S
        return u

    def forward(
        self,
        dr_preds: torch.Tensor,
        dme_preds: torch.Tensor,
        dr_alpha: torch.Tensor,
        dme_alpha: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Determines referral recommendations for a batch of predictions.

        Args:
            dr_preds (torch.Tensor): Predicted DR grades (B,).
            dme_preds (torch.Tensor): Predicted DME stages (B,).
            dr_alpha (torch.Tensor): Dirichlet alpha parameters for DR (B, K_dr).
            dme_alpha (torch.Tensor): Dirichlet alpha parameters for DME (B, K_dme).

        Returns:
            dict: Containing:
                - 'referral_flag': Binary tensor (B,) indicating if referral is recommended.
                - 'dr_uncertainty': Epistemic uncertainty for DR (B,).
                - 'dme_uncertainty': Epistemic uncertainty for DME (B,).
                - 'reason_code': Code representation of why referral was triggered.
        """
        dr_u = self.compute_uncertainty(dr_alpha)
        dme_u = self.compute_uncertainty(dme_alpha)

        # Condition 1: High severity predictions
        high_severity_dr = dr_preds >= self.dr_severity_threshold
        high_severity_dme = dme_preds >= self.dme_severity_threshold
        high_severity = high_severity_dr | high_severity_dme

        # Condition 2: High predictive uncertainty
        high_uncertainty_dr = dr_u >= self.uncertainty_threshold
        high_uncertainty_dme = dme_u >= self.uncertainty_threshold
        high_uncertainty = high_uncertainty_dr | high_uncertainty_dme

        # Combined referral recommendation
        referral_flag = high_severity | high_uncertainty

        # Construct referral reason codes
        # 1: Severity trigger, 2: Uncertainty trigger, 3: Both, 0: Safe (No referral)
        reason_code = torch.zeros_like(referral_flag, dtype=torch.long)
        reason_code[high_severity] += 1
        reason_code[high_uncertainty] += 2

        return {
            "referral_flag": referral_flag.long(),
            "dr_uncertainty": dr_u,
            "dme_uncertainty": dme_u,
            "reason_code": reason_code
        }
