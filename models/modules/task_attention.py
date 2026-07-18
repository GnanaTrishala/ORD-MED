from typing import Tuple
import torch
import torch.nn as nn


class TaskAttention(nn.Module):
    """
    Task-Gated Attention module for multi-task learning.
    It takes the shared latent representation and maps it into task-specific
    representations for DR grading and DME staging using task-specific gating functions.
    
    This ensures that features important for DR (e.g. microaneurysms, hemorrhages)
    and features important for DME (e.g. macular thickening, hard exudates) are
    dynamically weighted and separated before reaching the respective heads.
    """
    def __init__(self, feature_dim: int = 512):
        """
        Args:
            feature_dim (int): Input shared feature dimension.
        """
        super().__init__()
        
        # DR Gating Network
        self.dr_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

        # DME Gating Network
        self.dme_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

        # Task-specific projection refinements
        self.dr_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.dme_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Shared latent representation of shape (B, feature_dim).

        Returns:
            tuple: (dr_features, dme_features) where each is of shape (B, feature_dim).
        """
        # Compute task-specific gating weights
        dr_weights = self.dr_gate(x)
        dme_weights = self.dme_gate(x)

        # Apply gating (element-wise multiplication)
        dr_gated = x * dr_weights
        dme_gated = x * dme_weights

        # Refine task representations
        dr_features = self.dr_proj(dr_gated)
        dme_features = self.dme_proj(dme_gated)

        return dr_features, dme_features
