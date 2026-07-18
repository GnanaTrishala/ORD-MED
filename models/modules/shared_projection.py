import torch
import torch.nn as nn


class SharedProjection(nn.Module):
    """
    Shared Projection Layer that maps backbone features of arbitrary size
    to a standardized latent space dimension. Swapping backbones with different 
    output dimensions only requires modifying the `in_features` of this projection layer.
    """
    def __init__(self, in_features: int, out_features: int = 512, dropout: float = 0.2):
        """
        Args:
            in_features (int): Number of features from the backbone output.
            out_features (int): Target dimension size for downstream heads.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Backbone representation tensor of shape (B, in_features).

        Returns:
            torch.Tensor: Projected features tensor of shape (B, out_features).
        """
        return self.proj(x)
