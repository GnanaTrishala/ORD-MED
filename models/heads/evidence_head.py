import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    """
    Evidential Head for subjective uncertainty estimation using Dirichlet distributions.
    Instead of standard softmax probabilities, this head outputs the Dirichlet
    concentration parameters (alpha), representing class-wise "evidence" plus one.
    
    Mathematical relations:
      - Evidence: e_k = f(x)_k >= 0 (using softplus or relu)
      - Dirichlet parameter: alpha_k = e_k + 1
      - Dirichlet strength: S = sum(alpha_k)
      - Epistemic uncertainty: u = K / S  (where K is the number of classes)
    """
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        """
        Args:
            in_features (int): Input representation feature dimension.
            num_classes (int): Number of target classes.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.num_classes = num_classes

        # Evidence MLP mapping
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features // 2, num_classes)
        )
        
        # Softplus activation ensures evidence values are non-negative
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Dirichlet alpha parameters.

        Args:
            x (torch.Tensor): Feature tensor of shape (B, in_features).

        Returns:
            torch.Tensor: Dirichlet alpha values (B, num_classes) where each alpha >= 1.
        """
        raw_evidence = self.fc(x)
        # alpha = softplus(raw_evidence) + 1
        # The +1 represents the uniform prior (no evidence = uniform distribution)
        alpha = self.softplus(raw_evidence) + 1.0
        return alpha
