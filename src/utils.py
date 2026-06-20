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
from typing import Dict, Tuple

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

def get_device(gpu_index: int = 0) -> Tuple[torch.device, str, str]:
    """
    Detect and return the best available compute device, prioritizing TPU > GPU > CPU.

    Parameters
    ----------
    gpu_index : int, optional
        CUDA device index to use (default 0).

    Returns
    -------
    Tuple[torch.device, str, str]
        - device: torch.device or XLA device object.
        - device_type: 'tpu', 'cuda', or 'cpu'.
        - device_name: A descriptive string of the hardware.
    """
    # 1. Check for TPU (PyTorch XLA)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        # This will fail or raise an exception if not running on TPU runtime
        device = xm.xla_device()
        device_type = "tpu"
        device_name = "TPU (Google Colab / Cloud)"
        print(f"Using device: {device} (Type: {device_type}, Name: {device_name})")
        print(f"PyTorch version: {torch.__version__}")
        return device, device_type, device_name
    except (ImportError, RuntimeError, Exception):
        # Gracefully fall back if torch_xla is not installed or XLA device is not present
        pass

    # 2. Check for CUDA GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(gpu_index)
        gpu_mem = torch.cuda.get_device_properties(gpu_index).total_memory
        print(f"Using device: {device} (Type: {device_type}, Name: {device_name}, Memory: {gpu_mem / 1e9:.1f} GB)")
        print(f"PyTorch version: {torch.__version__} | CUDA version: {torch.version.cuda}")
        return device, device_type, device_name

    # 3. Fallback to CPU
    device = torch.device("cpu")
    device_type = "cpu"
    # Try reading CPU model name on Linux
    device_name = "CPU"
    try:
        import subprocess
        cpu_info = subprocess.check_output("lscpu", shell=True).decode()
        for line in cpu_info.splitlines():
            if "Model name" in line:
                device_name = line.split(":")[1].strip()
                break
    except Exception:
        pass
    print(f"Using device: {device} (Type: {device_type}, Name: {device_name})")
    print(f"PyTorch version: {torch.__version__}")
    print("  ⚠️  No GPU/TPU detected — training will be slow on CPU.")
    return device, device_type, device_name



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


# ─────────────────────────────────────────────────────────────────────────────
# Colab Compatibility & Environment Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_colab(mount_drive: bool = False, repo_url: str = None) -> None:
    """
    Configure Google Colab environment: mount drive, clone repository,
    setup path, install packages, and verify imports.

    Parameters
    ----------
    mount_drive : bool, optional
        Whether to prompt to mount Google Drive (default False).
    repo_url : str, optional
        The git URL to clone. Defaults to the main repository URL.
    """
    import sys
    import os
    from pathlib import Path

    if "google.colab" not in sys.modules:
        print("Not running in Google Colab environment. Skipping Colab setup.")
        return

    print("Running Colab compatibility setup...")

    # 1. Mount Google Drive
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("Google Drive successfully mounted.")
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")

    # 2. Clone Repository
    repo_name = "customer-support-ticket-intelligence-platform"
    if repo_url is None:
        repo_url = f"https://github.com/ikartiksavaliya/{repo_name}.git"

    target_dir = Path(f"/content/{repo_name}")
    if not target_dir.exists():
        print(f"Cloning repository from {repo_url}...")
        import subprocess
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
    else:
        print(f"Repository already exists at {target_dir}")

    # 3. Add project root to sys.path and change directory
    os.chdir(str(target_dir))
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))

    # 4. Install requirements automatically
    print("Installing requirements and installing project in editable mode...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    # 5. Verify imports
    try:
        from src.utils import set_seed
        from src.preprocessing import clean_text
        from src.dataset import TicketDataset
        print("✅ Colab setup verified: imports are working perfectly!")
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        raise e
