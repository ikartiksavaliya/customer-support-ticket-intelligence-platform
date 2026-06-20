"""
Customer Support Ticket Intelligence Platform
Module: training.py

Reusable PyTorch training loop with per-epoch metrics and checkpointing.

Design Principles
-----------------
- Functions are model-agnostic: they accept any nn.Module that takes
  input_ids and returns logits.  The same train_model() works for
  BagOfEmbeddings (Phase 4), RNN (Phase 5), LSTM (Phase 6), etc.
- train_one_epoch() and evaluate() return raw predictions and labels
  so that the caller (or evaluation.py) can compute any metric.
- Checkpointing saves the model state_dict (not the full model object)
  for portability — the caller must re-instantiate the model class and
  call load_state_dict() to restore.  This is PyTorch best practice.
- Gradient clipping (max_grad_norm) is supported to prevent exploding
  gradients in recurrent models.  See INTERVIEW_NOTES.md Common Mistake #3.

Usage
-----
    from src.training import train_model

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=10,
        checkpoint_path="outputs/boe_best.pt",
    )
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


# ─────────────────────────────────────────────────────────────────────────────
# Single Epoch: Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
) -> Tuple[float, List[int], List[int]]:
    """
    Train the model for one full pass over the training data.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    dataloader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance (e.g. Adam).
    criterion : nn.Module
        Loss function (e.g. CrossEntropyLoss with class weights).
    device : torch.device
        Target device ('cpu' or 'cuda').
    max_grad_norm : float or None, optional
        If set, clips the total gradient norm to this value after
        each backward pass using torch.nn.utils.clip_grad_norm_().
        Recommended: 1.0 for RNN models. None for BoE (no clipping).

    Returns
    -------
    avg_loss : float
        Mean training loss over all batches.
    all_preds : list[int]
        Predicted class indices for every sample in the epoch.
    all_labels : list[int]
        Ground-truth class indices for every sample in the epoch.
    """
    model.train()

    running_loss = 0.0
    num_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────
        logits = model(input_ids)
        loss = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # ── Gradient clipping (prevents exploding gradients in RNNs) ─────
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # ── Track metrics ─────────────────────────────────────────────────
        running_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Single Epoch: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """
    Evaluate the model on a validation or test set (no gradient computation).

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    dataloader : DataLoader
        Validation or test data loader.
    criterion : nn.Module
        Loss function (same as training for comparable loss values).
    device : torch.device
        Target device.

    Returns
    -------
    avg_loss : float
        Mean loss over all batches.
    all_preds : list[int]
        Predicted class indices for every sample.
    all_labels : list[int]
        Ground-truth class indices for every sample.
    """
    model.eval()

    running_loss = 0.0
    num_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 10,
    checkpoint_path: Optional[str] = None,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    Train a model for multiple epochs with validation and optional checkpointing.

    This is the main entry point for running experiments.  It:
        1. Trains for `epochs` epochs, calling train_one_epoch() each time.
        2. Evaluates on the validation set after each training epoch.
        3. Tracks training loss, validation loss, and validation macro F1.
        4. Saves the best model (by val F1) to `checkpoint_path` if provided.
        5. Prints a formatted summary line per epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train (must already be moved to `device`).
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device.
    epochs : int, optional
        Number of training epochs (default 10).
    checkpoint_path : str or None, optional
        If provided, saves the best model state_dict to this path
        whenever validation F1 improves.
    max_grad_norm : float or None, optional
        If set, clips gradient norms during training.  Recommended
        1.0 for RNN models, None for BoE.

    Returns
    -------
    history : dict
        Training history with keys:
            'train_loss' : list[float]   — per-epoch average training loss
            'val_loss'   : list[float]   — per-epoch average validation loss
            'val_f1'     : list[float]   — per-epoch validation macro F1-score
            'epoch_time' : list[float]   — per-epoch wall-clock time (seconds)
    """
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "epoch_time": [],
    }

    best_val_f1 = 0.0

    print(f"{'Epoch':>7} | {'Train Loss':>11} | {'Val Loss':>9} | {'Val F1':>7} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_grad_norm=max_grad_norm,
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # ── Compute Metrics ───────────────────────────────────────────────
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        epoch_time = time.time() - epoch_start

        # ── Record History ────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["epoch_time"].append(epoch_time)

        # ── Checkpointing ─────────────────────────────────────────────────
        improved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            improved = " ✓"
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": val_f1,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )

        # ── Print Summary ─────────────────────────────────────────────────
        print(
            f"  [{epoch:>2}/{epochs}] | "
            f"{train_loss:>11.4f} | "
            f"{val_loss:>9.4f} | "
            f"{val_f1:>6.4f} | "
            f"{epoch_time:>5.1f}s{improved}"
        )

    print("-" * 55)
    print(f"Best Val F1: {best_val_f1:.4f}")

    return history
