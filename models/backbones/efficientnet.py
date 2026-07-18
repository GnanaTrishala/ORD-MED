from typing import Tuple, Optional
import torch
import torch.nn as nn
import timm


def get_efficientnet_backbone(
    name: str = "efficientnet_b4",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> Tuple[nn.Module, int]:
    """
    Constructs an EfficientNet backbone using timm, removing its classification head.

    Args:
        name (str): Specific EfficientNet version (e.g. 'efficientnet_b0', 'efficientnet_b4').
        pretrained (bool): Load ImageNet pre-trained weights.
        checkpoint_path (str, optional): Path to local pre-trained weight checkpoint.

    Returns:
        tuple: (backbone_module, feature_dimension)
    """
    # Create the model using timm
    # features_only=True returns features from intermediate blocks, 
    # but we will load the base model and remove the classification head to extract final feature maps.
    try:
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    except Exception as e:
        print(f"Could not load '{name}' via timm, falling back to efficientnet_b0. Error: {str(e)}")
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        name = "efficientnet_b0"

    # If a custom local checkpoint is provided
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle state_dict mapping if model is wrapped in another module/head in checkpoint
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # Strip prefixes like 'backbone.' if necessary
        cleaned_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                cleaned_state[k.replace("backbone.", "")] = v
            else:
                cleaned_state[k] = v
                
        model.load_state_dict(cleaned_state, strict=False)

    # Get feature dimension of the model
    feature_dim = model.num_features

    return model, feature_dim
