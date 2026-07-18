from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

from config import Config
from .backbones import get_backbone
from .modules.shared_projection import SharedProjection
from .modules.task_attention import TaskAttention
from .heads.dr_head import DRHead
from .heads.dme_head import DMEHead
from .heads.evidence_head import EvidenceHead


class OrdMedNet(nn.Module):
    """
    Ordinal Multi-task Evidential Diabetic Eye Disease Network (ORD-MED).
    This architecture integrates:
      - Modular backbones (EfficientNet, RETFound, etc.)
      - Shared Projection Layer
      - Cross-task attention mechanism
      - Task-specific regression/classification heads
      - Evidential uncertainty modeling for safe diagnostic referrals
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. Initialize Backbone (Encoder)
        self.backbone, self.feature_dim = get_backbone(
            name=config.encoder.name,
            pretrained=config.encoder.pretrained,
            checkpoint_path=config.encoder.checkpoint_path,
            freeze_features=config.encoder.freeze_features
        )

        # 2. Shared Projection Layer (Dimensionality alignment)
        self.shared_proj = SharedProjection(
            in_features=self.feature_dim,
            out_features=config.heads.projection_dim,
            dropout=config.encoder.dropout
        )

        # 3. Cross-Task Attention Module
        self.task_attention = TaskAttention(
            feature_dim=config.heads.projection_dim
        )

        # 4. Initialize Diagnostic Heads
        self.dr_head = DRHead(
            in_features=config.heads.projection_dim,
            num_classes=config.heads.dr_num_classes,
            dropout=config.encoder.dropout
        )
        self.dme_head = DMEHead(
            in_features=config.heads.projection_dim,
            num_classes=config.heads.dme_num_classes,
            dropout=config.encoder.dropout
        )

        # 5. Optional Evidential Heads (for uncertainty estimation)
        if config.heads.use_evidential:
            self.dr_evidential = EvidenceHead(
                in_features=config.heads.projection_dim,
                num_classes=config.heads.dr_num_classes,
                dropout=config.encoder.dropout
            )
            self.dme_evidential = EvidenceHead(
                in_features=config.heads.projection_dim,
                num_classes=config.heads.dme_num_classes,
                dropout=config.encoder.dropout
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ORD-MED.

        Args:
            x (torch.Tensor): Input batch of retinal fundus images of shape (B, C, H, W).

        Returns:
            dict: Dictionary of outputs containing logits, ordinal predictions,
                  and evidential parameters for DR and DME.
        """
        # Feature extraction
        features = self.backbone(x)  # Shape: (B, feature_dim) or (B, feature_dim, H, W)
        
        # Ensure pooled representation (B, feature_dim)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=[2, 3])

        # Shared projection
        projected = self.shared_proj(features)  # Shape: (B, projection_dim)

        # Task-specific feature conditioning using cross-task attention
        dr_feat, dme_feat = self.task_attention(projected)

        # Head outputs
        dr_logits = self.dr_head(dr_feat)
        dme_logits = self.dme_head(dme_feat)

        outputs = {
            "dr_logits": dr_logits,
            "dme_logits": dme_logits,
            "shared_features": projected
        }

        # Add Evidential outputs if enabled
        if self.config.heads.use_evidential:
            # Evidential heads output subjective evidence alpha parameters (Dirichlet parameters)
            dr_evidence = self.dr_evidential(dr_feat)
            dme_evidence = self.dme_evidential(dme_feat)
            
            outputs.update({
                "dr_evidence": dr_evidence,
                "dme_evidence": dme_evidence
            })

        return outputs


def build_model(config: Config) -> nn.Module:
    """
    Factory function to construct the ORD-MED network based on configuration.
    """
    return OrdMedNet(config)
