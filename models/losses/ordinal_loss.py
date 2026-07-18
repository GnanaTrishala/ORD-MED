import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class CORNLoss(nn.Module):
    """
    Conditional Ordinal Regression for Neural Networks (CORN) Loss.
    Ref: "Conditional Ordinal Regression for Neural Networks" (Shi et al., 2021).
    
    CORN converts a K-class ordinal classification problem into K-1 binary
    classification subproblems. Each subproblem models whether the true grade 
    is greater than a threshold j, given that it is greater than or equal to j.
    
    The loss function only penalizes the active binary classifiers (i.e. where targets >= j).
    
    Mathematical formulation:
        For a batch of size B, class grade y_i in [0, K-1], and logits v_{i, j}:
        Loss = - (1 / N_active) * sum_{i=1}^B sum_{j=0}^{K-2} I(y_i >= j) * 
               [ t_{i, j} * log(sigmoid(v_{i, j})) + (1 - t_{i, j}) * log(1 - sigmoid(v_{i, j})) ]
        where t_{i, j} = 1 if y_i > j else 0, and N_active = sum_{i, j} I(y_i >= j).
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the CORN loss.
        
        Shapes:
            - logits: (B, K) or (B, K-1) where B is the batch size, K is num_classes.
                      If shape is (B, K), the last logit is ignored because we model
                      only K-1 binary tasks.
            - targets: (B,) containing class indices in the range [0, K-1].
            - Output: scalar (0D tensor).
        """
        # If the input has K logits, slice it to K-1 logits for the binary tasks
        if logits.size(-1) == self.num_classes:
            logits = logits[..., :-1]
            
        num_tasks = self.num_classes - 1
        device = logits.device
        
        # 1. Create task indices tensor of shape (1, num_tasks)
        task_indices = torch.arange(num_tasks, device=device).unsqueeze(0)
        
        # 2. Reshape targets to (B, 1) for broadcasting
        targets_unsqueezed = targets.unsqueeze(1)
        
        # 3. Create binary targets: t_{i, j} = 1 if y_i > j else 0
        # Shape: (B, num_tasks)
        binary_targets = (targets_unsqueezed > task_indices).float()
        
        # 4. Create active tasks mask: mask_{i, j} = 1 if y_i >= j else 0
        # Shape: (B, num_tasks)
        binary_mask = (targets_unsqueezed >= task_indices).float()
        
        # 5. Compute element-wise BCE loss
        # Shape: (B, num_tasks)
        bce_loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='none')
        
        # 6. Apply mask and calculate the mean loss over active subproblems
        masked_loss = bce_loss * binary_mask
        active_count = binary_mask.sum()
        
        # Guard against division by zero in empty batch contexts
        if active_count == 0:
            return torch.tensor(0.0, device=device)
            
        return masked_loss.sum() / active_count


class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) Loss for Ordinal Regression.
    Measures the distance between the cumulative probability distributions.
    Penalizes distant misclassifications more heavily than adjacent ones.
    
    Mathematical formulation:
        EMD = (1 / (K-1)) * sum_{j=0}^{K-2} (cumsum(P)_j - cumsum(Y)_j)^2
        where P is the predicted probabilities, Y is the one-hot targets.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the EMD ordinal loss.
        
        Shapes:
            - logits: (B, K) where B is the batch size, K is num_classes.
            - targets: (B,) containing class indices in the range [0, K-1].
            - Output: scalar (0D tensor).
        """
        # Convert logits to probability distribution via softmax
        # Shape: (B, K)
        probs = F.softmax(logits, dim=-1)
        
        # One-hot encode targets
        # Shape: (B, K)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute cumulative distributions
        # Shape: (B, K)
        cumsum_probs = torch.cumsum(probs, dim=-1)
        cumsum_targets = torch.cumsum(targets_onehot, dim=-1)
        
        # Compute squared L2 distance between cumulative profiles
        # Omit the last class index as the sum is always 1.0 (difference is 0)
        # Shape: (B, K-1)
        emd_diff = cumsum_probs[:, :-1] - cumsum_targets[:, :-1]
        
        # Mean across the batch and sum across classes
        emd_loss = torch.mean(torch.sum(emd_diff ** 2, dim=-1))
        
        return emd_loss


class OrdinalLoss(nn.Module):
    """
    Modular Ordinal Loss module for Diabetic Retinopathy severity classification.
    Wraps both CORN and EMD implementations.
    """
    def __init__(self, num_classes: int, method: Literal["corn", "emd"] = "corn") -> None:
        """
        Args:
            num_classes (int): Number of ordinal target classes.
            method (str): Ordinal regression method to use ('corn' or 'emd').
        """
        super().__init__()
        self.num_classes = num_classes
        self.method = method.lower()
        
        if self.method == "corn":
            self.loss_fn = CORNLoss(num_classes)
        elif self.method == "emd":
            self.loss_fn = EMDLoss(num_classes)
        else:
            raise ValueError(f"Unsupported ordinal loss method: {method}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - logits: (B, K) or (B, K-1)
            - targets: (B,)
            - Output: scalar (0D tensor).
        """
        return self.loss_fn(logits, targets)
