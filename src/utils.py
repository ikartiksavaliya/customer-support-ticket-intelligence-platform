"""
Customer Support Ticket Intelligence Platform
Module: utils.py

Foundational utilities shared across training, evaluation, and notebooks.

Design Principles
-----------------
- set_seed() guarantees bitwise-reproducible results across runs by
  seeding Python, NumPy, and PyTorch PRNG sources.  The project standard
  seed is 42 (see docs/PROJECT_MASTER.md § Repository Quality Standards).
- get_device() provides a single point of device selection so notebooks
  and scripts never hardcode 'cpu' or 'cuda'.
- compute_class_weights() implements the class-imbalance mitigation strategy
  documented in DECISION_LOG.md (Decision 4): inverse-frequency weighting
  for CrossEntropyLoss.

Usage
-----
    from src.utils import set_seed, get_device, compute_class_weights

    set_seed(42)
    device = get_device()
    weights = compute_class_weights(label_encoder, train_df)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
"""

import random
from typing import Dict

import numpy as np
import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set all random number generator seeds for full reproducibility.

    Seeds the following sources:
        - Python's built-in `random` module
        - NumPy's global PRNG
        - PyTorch CPU generator
        - PyTorch CUDA generators (all GPUs)

    Additionally enables PyTorch's deterministic mode to force
    reproducible cuDNN convolution algorithms (at a potential
    performance cost — acceptable for a study-oriented project).

    Parameters
    ----------
    seed : int, optional
        The seed value (default 42, per project convention).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms when available.
    # This can slow down training slightly but ensures that
    # re-running a notebook produces identical numbers.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Device Selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Returns
    -------
    torch.device
        'cuda' if a CUDA GPU is available, otherwise 'cpu'.

    Notes
    -----
    The selected device is printed for visibility in notebook outputs,
    so the reader always knows whether training ran on CPU or GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Class Imbalance Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    label_encoder: Dict[str, int],
    train_df: pd.DataFrame,
    column: str = "category",
) -> torch.FloatTensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    The CFPB dataset is highly imbalanced (22.97% Debt Collection vs
    0.004% Virtual Currency).  Inverse-frequency weights ensure that
    the loss function penalises misclassifications of rare classes
    proportionally more, preventing the model from simply predicting
    majority classes.

    Weight formula for class c:
        w_c = N_total / (N_classes × N_c)

    Where:
        N_total   = total number of training samples
        N_classes = number of distinct classes (18)
        N_c       = number of samples in class c

    Parameters
    ----------
    label_encoder : dict[str, int]
        Mapping from category name → integer index
        (e.g. {"Debt collection": 7, "Mortgage": 10, ...}).
    train_df : pd.DataFrame
        Training split DataFrame with a `column` containing category strings.
    column : str, optional
        Column name holding the target labels (default "category").

    Returns
    -------
    torch.FloatTensor, shape (num_classes,)
        Weight vector ordered by class index (i.e. weights[0] is the weight
        for the class whose label_encoder value is 0).

    Raises
    ------
    ValueError
        If `column` is not found in `train_df`.

    Examples
    --------
    >>> weights = compute_class_weights(label_encoder, train_df)
    >>> criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    if column not in train_df.columns:
        raise ValueError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {list(train_df.columns)}"
        )

    num_classes = len(label_encoder)
    n_total = len(train_df)

    # Count samples per category
    class_counts = train_df[column].value_counts()

    # Build weight vector in class-index order
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for category_name, class_idx in label_encoder.items():
        n_c = class_counts.get(category_name, 1)  # fallback to 1 to avoid division by zero
        weights[class_idx] = n_total / (num_classes * n_c)

    return weights
