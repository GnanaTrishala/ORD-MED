import os
import torch
from typing import Any, Tuple, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str
) -> None:
    """
    Saves model weights, optimizer state, and training progress metadata.

    Args:
        model (nn.Module): Active PyTorch model.
        optimizer (Optimizer): Active training optimizer.
        epoch (int): Completed epoch count.
        path (str): File destination path for the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Handle DataParallel/DistributedDataParallel wrapping
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    checkpoint = {
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint successfully saved to: {path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int]:
    """
    Loads model state and optional optimizer parameters from a checkpoint file.

    Args:
        model (nn.Module): Target PyTorch model.
        checkpoint_path (str): Location of saved checkpoint.
        optimizer (Optimizer, optional): Target training optimizer.
        strict (bool): Whether model keys must match checkpoint keys exactly.

    Returns:
        tuple: (model, optimizer, loaded_epoch)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load model weights
    state_dict = checkpoint["state_dict"]
    
    # Adapt keys if checkpoint was saved with DataParallel 'module.' prefix
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state[k.replace("module.", "")] = v
        else:
            cleaned_state[k] = v

    model.load_state_dict(cleaned_state, strict=strict)
    
    epoch = checkpoint.get("epoch", 0)

    # Load optimizer state if requested
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Checkpoint successfully loaded from {checkpoint_path} (epoch {epoch})")
    
    return model, optimizer, epoch
