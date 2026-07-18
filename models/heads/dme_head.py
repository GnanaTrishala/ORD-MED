import torch
import torch.nn as nn


class DMEHead(nn.Module):
    """
    Diabetic Macular Edema (DME) severity staging head.
    Outputs logits for DME stage classification (typically 3 stages: 0 to 2).
    """
    def __init__(self, in_features: int, num_classes: int = 3, dropout: float = 0.3):
        """
        Args:
            in_features (int): Input representation feature dimension.
            num_classes (int): Number of severity stages.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.num_classes = num_classes

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
