import torch
import torch.nn as nn


class DRHead(nn.Module):
    """
    Diabetic Retinopathy (DR) severity grading head.
    Outputs logits for DR grade classification or ordinal boundaries.
    """
    def __init__(self, in_features: int, num_classes: int = 5, dropout: float = 0.3):
        """
        Args:
            in_features (int): Input representation feature dimension.
            num_classes (int): Number of severity grades (usually 5: 0 to 4).
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.num_classes = num_classes

        # Standard representation head with layer normalization and dropout for regularization
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Latent representation tensor of shape (B, in_features).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        return self.head(x)
