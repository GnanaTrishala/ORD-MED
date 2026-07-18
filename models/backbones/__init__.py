from typing import Tuple
import torch.nn as nn

from .efficientnet import get_efficientnet_backbone
from .retfound import get_retfound_backbone


def get_backbone(
    name: str,
    pretrained: bool = True,
    checkpoint_path: str = None,
    freeze_features: bool = False
) -> Tuple[nn.Module, int]:
    """
    Registry function for retrieving backbones.

    Args:
        name (str): Backbone name (e.g., 'efficientnet', 'retfound').
        pretrained (bool): Load standard ImageNet weights if applicable.
        checkpoint_path (str, optional): Path to specialized custom pre-trained checkpoints.
        freeze_features (bool): Freeze backbone weights for feature extraction.

    Returns:
        tuple: (backbone_module, feature_dimension)
    """
    name = name.lower()
    
    if "efficientnet" in name:
        backbone, dim = get_efficientnet_backbone(
            name=name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path
        )
    elif "retfound" in name:
        backbone, dim = get_retfound_backbone(
            pretrained=pretrained,
            checkpoint_path=checkpoint_path
        )
    else:
        raise ValueError(f"Unsupported backbone: {name}. Choose from 'efficientnet', 'retfound'.")

    if freeze_features:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, dim
