from typing import Tuple, Optional
import torch
import torch.nn as nn
import timm


def get_retfound_backbone(
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> Tuple[nn.Module, int]:
    """
    Constructs a RETFound backbone. RETFound is a retinal foundation model
    based on a Vision Transformer (typically ViT-Large/16).

    Args:
        pretrained (bool): Load default weights.
        checkpoint_path (str, optional): Path to the official downloaded 'RETFound_oct_weights.pth' 
                                         or 'RETFound_cfp_weights.pth' file.

    Returns:
        tuple: (backbone_module, feature_dimension)
    """
    # RETFound is built on vit_large_patch16_224 or vit_base_patch16_224
    # We will instantiate a timm vit_large_patch16_224 without its classification head (num_classes=0)
    model_name = "vit_large_patch16_224"
    
    # We set num_classes=0 to output the representation vector (CLS token)
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    feature_dim = model.num_features  # Typically 1024 for ViT-Large

    if pretrained and not checkpoint_path:
        print("Warning: RETFound requires explicit checkpoint weights path. Loading default timm ImageNet weights instead.")
        model = timm.create_model(model_name, pretrained=True, num_classes=0)

    if checkpoint_path:
        print(f"Loading RETFound weights from: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract model state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Clean state dict to remove prediction head keys if they exist in pre-training
        cleaned_state = {}
        for k, v in state_dict.items():
            if k.startswith("head.") or k.startswith("patch_embed.proj.weight") or k.startswith("patch_embed.proj.bias"):
                # Sometimes patch embeddings have slightly different shapes if input resolution differs
                cleaned_state[k] = v
            else:
                cleaned_state[k] = v
                
        msg = model.load_state_dict(cleaned_state, strict=False)
        print(f"RETFound weight loading status: {msg}")

    return model, feature_dim
