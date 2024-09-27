# seed_util.py

import torch
import numpy as np
import random

from misc.logger import get_logger

logger = get_logger()

# Declare the generator as a global variable
g = torch.Generator()

def set_seed(seed):
    """
    Sets the seed for reproducibility across multiple libraries such as 
    torch, numpy, random, and torch.Generator.

    Args:
        seed (int): The seed value to set.
    """
    global g
    # Set seed for torch (PyTorch library)
    torch.manual_seed(seed)

    # Set seed for numpy (NumPy library)
    np.random.seed(seed)

    # Set seed for random (Python's built-in random library)
    random.seed(seed)

    # If CUDA is available, set seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For PyTorch reproducibility (ensure deterministic behavior in cuDNN)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for the global generator
    g.manual_seed(seed)
    logger.info(f"Global generator seed set to {seed}")

def initialize_seed(use_fixed_seed=True, seed_value=42):
    """
    Decides whether to set a fixed seed or leave randomness based on `use_fixed_seed`.

    Args:
        use_fixed_seed (bool): If True, set a fixed seed. If False, use random behavior.
        seed_value (int): The seed value to use if `use_fixed_seed` is True.
    """
    if use_fixed_seed:
        set_seed(seed_value)
    else:
        logger.info("No fixed seed provided. Using random behavior.")
        # No seed set, randomness remains
