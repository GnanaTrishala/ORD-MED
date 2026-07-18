import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets seed values across standard libraries to ensure model training
    reproducibility and deterministic behavior.

    Args:
        seed (int): The seed number to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enforce deterministic convolutions (at the cost of some performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
