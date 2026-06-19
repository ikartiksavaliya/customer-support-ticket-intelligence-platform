"""
Customer Support Ticket Intelligence Platform
Module: evaluation.py

Evaluation metrics, classification reports, and visualisation utilities.

Design Principles
-----------------
- All metric functions accept raw lists of integer labels (y_true, y_pred)
  so they are framework-agnostic — they work identically whether called
  from a training loop, a notebook, or a Streamlit dashboard.
- Visualisation functions have an optional save_path parameter: when set,
  the figure is saved to disk (for outputs/ artifacts); when None, the
  figure is displayed inline (for notebook use).
- We use sklearn.metrics for numerical computation and matplotlib/seaborn
  for plotting — these are already project dependencies.

Usage
-----
    from src.evaluation import compute_metrics, plot_confusion_matrix

    metrics = compute_metrics(y_true, y_pred)
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")

    plot_confusion_matrix(y_true, y_pred, label_names, save_path="outputs/cm.png")
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Numerical Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Compute core classification metrics.

    We focus on macro-averaged metrics because the CFPB dataset is
    highly imbalanced.  Macro averaging gives equal weight to every
    class regardless of its sample count, which surfaces poor
    performance on minority classes that would be hidden by
    micro/weighted averages.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth class indices.
    y_pred : list[int]
        Predicted class indices.

    Returns
    -------
    dict[str, float]
        Keys: 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> None:
    """
    Pretty-print a full per-class classification report.

    Displays precision, recall, and F1-score for each individual class,
    plus macro/weighted averages.  This is the single most useful output
    for diagnosing which specific CFPB categories the model struggles with.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth class indices.
    y_pred : list[int]
        Predicted class indices.
    label_names : list[str] or None, optional
        Human-readable class names (ordered by class index).
        If None, integer indices are shown.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        digits=4,
    )
    print(report)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 12),
) -> None:
    """
    Plot a heatmap confusion matrix.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth class indices.
    y_pred : list[int]
        Predicted class indices.
    label_names : list[str]
        Human-readable class names (ordered by class index).
    normalize : bool, optional
        If True, normalise rows to show percentages (recall per class).
        Default True because raw counts are hard to compare across
        imbalanced classes.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    figsize : tuple, optional
        Figure size (default (14, 12) for 18-class readability).
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        # Normalise each row (true class) so values represent recall %
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid /0
        cm = cm.astype(float) / row_sums

    # Shorten long label names for readability on the axis
    short_names = [
        name[:30] + "…" if len(name) > 30 else name
        for name in label_names
    ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        cbar_kws={"label": "Recall %" if normalize else "Count"},
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Category", fontsize=12)
    ax.set_ylabel("True Category", fontsize=12)
    ax.set_title("Confusion Matrix (Normalised)" if normalize else "Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training/validation loss and validation F1 over epochs.

    Creates a two-panel figure:
        Left:  Train Loss vs Val Loss  (should converge; gap = overfitting)
        Right: Val F1 over epochs      (should increase and plateau)

    Parameters
    ----------
    history : dict
        Training history from train_model(), must contain keys:
        'train_loss', 'val_loss', 'val_f1'.
    title : str, optional
        Overall figure title.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: Loss curves ───────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2196F3")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#FF5722")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right panel: Validation F1 ────────────────────────────────────────
    ax2.plot(epochs, history["val_f1"], "D-", label="Val F1 (Macro)", color="#4CAF50")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score (Macro)")
    ax2.set_title("Validation F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    plt.show()
