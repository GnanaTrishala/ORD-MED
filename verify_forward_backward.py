#!/usr/bin/env python
"""
Verification script to test one forward and backward pass of the ORD-MED network.
"""

import sys
import torch

from config import Config
from models import build_model
from models.losses.multitask_loss import MultiTaskLoss


def verify_pass():
    print("Initializing ORD-MED configuration...")
    config = Config()
    # Configure lightweight settings for verification
    config.encoder.name = "efficientnet_b0"
    config.encoder.pretrained = False
    config.heads.use_evidential = True
    config.heads.dr_num_classes = 5
    config.heads.dme_num_classes = 3
    config.trainer.use_amp = True

    print("Building model architecture...")
    model = build_model(config)
    model.train()  # Place in training mode to calculate gradients

    # Identify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model placed on device: {device}")

    print("Initializing multi-task criterion...")
    criterion = MultiTaskLoss(config)

    # 1. Generate mock batch (Batch size = 2, Channels = 3, Height = 512, Width = 512)
    print("Generating mock image batch and targets...")
    mock_images = torch.randn(2, 3, 512, 512, device=device)
    mock_targets = {
        "dr_label": torch.tensor([2, 0], dtype=torch.long, device=device),
        "dme_label": torch.tensor([1, 2], dtype=torch.long, device=device)
    }

    # 2. Forward pass with automatic mixed precision (AMP)
    print("Executing forward pass...")
    use_amp = config.trainer.use_amp and (device.type == "cuda")
    
    with torch.amp.autocast("cuda", enabled=use_amp):
        outputs = model(mock_images)
        loss_dict = criterion(outputs, mock_targets, epoch=0)
        total_loss = loss_dict["loss"]

    # Assert outputs are present
    assert "dr_logits" in outputs, "Missing DR logits"
    assert "dme_logits" in outputs, "Missing DME logits"
    assert "dr_evidence" in outputs, "Missing DR evidence"
    assert "dme_evidence" in outputs, "Missing DME evidence"

    print("Forward pass completed successfully!")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  DR Logits Shape: {outputs['dr_logits'].shape}")
    print(f"  DME Logits Shape: {outputs['dme_logits'].shape}")
    print(f"  DR Uncertainty: {loss_dict['dr_uncertainty'].item():.4f}")
    print(f"  DME Uncertainty: {loss_dict['dme_uncertainty'].item():.4f}")

    # 3. Backward pass
    print("Executing backward pass...")
    # Zero gradients first
    model.zero_grad()
    
    # Backward step
    total_loss.backward()

    # 4. Verify gradients are computed
    grad_count = 0
    zero_grad_count = 0
    no_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_count += 1
                if torch.all(param.grad == 0.0):
                    zero_grad_count += 1
            else:
                no_grad_count += 1

    print("Backward pass completed successfully!")
    print(f"  Parameters with gradients computed: {grad_count}")
    print(f"  Parameters with zero gradients: {zero_grad_count}")
    print(f"  Parameters with NO gradients (None): {no_grad_count}")

    if no_grad_count > 0:
        print("Warning: Some parameters that require gradients did not receive gradients.")
    else:
        print("Success: Gradients computed successfully for all parameters!")


if __name__ == "__main__":
    try:
        verify_pass()
        sys.exit(0)
    except ImportError as e:
        print(f"Import Error: Missing dependencies. Please run 'pip install -r requirements.txt'. Error details: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Verification Failed! Error: {str(e)}")
        sys.exit(1)
