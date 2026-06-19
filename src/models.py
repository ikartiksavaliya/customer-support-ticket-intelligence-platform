"""
Customer Support Ticket Intelligence Platform
Module: models.py

PyTorch neural network architectures for ticket classification.

Design Principles
-----------------
- Each model class is self-contained: it defines its own forward() logic,
  including any pooling or aggregation strategy.
- All models return raw logits (no softmax) because PyTorch's
  CrossEntropyLoss applies LogSoftmax internally.  Applying softmax
  twice would produce incorrect gradients.
- padding_idx=0 is always set in nn.Embedding so that PAD tokens
  (index 0) receive zero embeddings and are never updated by gradients.
- Models are designed to be instantiated with a consistent interface
  so that the training loop in training.py works identically for all.

Architecture Registry (from MODEL_EVOLUTION.md)
-----------------------------------------------
    MODEL-v1 : BagOfEmbeddings   (Phase 4 — this file)
    MODEL-v2 : SimpleRNN         (Phase 5 — future)
    MODEL-v3 : LSTMClassifier    (Phase 6 — future)
    MODEL-v4 : GRUClassifier     (Phase 6 — future)
    MODEL-v5 : BiLSTMClassifier  (Phase 7 — future)
    MODEL-v6 : DeepBiLSTM        (Phase 7 — future)

Usage
-----
    from src.models import BagOfEmbeddings

    model = BagOfEmbeddings(vocab_size=23262, embedding_dim=50, num_classes=18)
    logits = model(input_ids)  # (batch, 18)
"""

import torch
import torch.nn as nn


class BagOfEmbeddings(nn.Module):
    """
    Bag of Embeddings (BoE) classifier — MODEL-v1.

    A sequence-agnostic baseline that averages learned word embeddings
    across the non-padded positions of each input sequence, then projects
    the resulting fixed-size vector through a linear classification head.

    Why "Bag of Embeddings"?
    ------------------------
    This model is the deep-learning analogue of Bag-of-Words: it treats
    each document as an unordered collection of word vectors.  It
    completely ignores word order and syntactic structure.

    Why start here?
    ---------------
    1. **Establishes a lower bound.**  Any sequence model (RNN, LSTM, GRU)
       that scores below BoE on F1 is likely broken (wrong hidden state
       handling, vanishing gradients, etc.).
    2. **Teaches nn.Embedding mechanics** — lookup tables, padding_idx,
       gradient flow through embeddings — before adding recurrence.
    3. **Fast to train** — no sequential bottleneck, so the entire batch
       can be processed in parallel.

    Architecture
    ------------
    ```
    Input IDs (batch, max_len)
         │
         ▼
    nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
         │
         ▼
    Embeddings (batch, max_len, embedding_dim)
         │
         ▼
    Masked Global Average Pooling  ← averages only non-PAD positions
         │
         ▼
    Pooled Vector (batch, embedding_dim)
         │
         ▼
    nn.Linear(embedding_dim, num_classes)
         │
         ▼
    Logits (batch, num_classes)
    ```

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size including special tokens (<PAD>=0, <UNK>=1).
    embedding_dim : int
        Dimensionality of the dense word embedding vectors.
    num_classes : int
        Number of output classes (18 for CFPB categories).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        # ── Embedding Layer ───────────────────────────────────────────────────
        # padding_idx=0 ensures:
        #   1. The embedding vector at index 0 is initialised to all zeros.
        #   2. Gradients are NOT computed for index 0 during backprop.
        # This is critical because PAD tokens are meaningless filler and
        # should contribute zero signal to the pooled representation.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # ── Classification Head ───────────────────────────────────────────────
        # Projects the pooled embedding vector to class logits.
        # No activation here — CrossEntropyLoss handles softmax internally.
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass: embed → masked average pool → classify.

        Parameters
        ----------
        input_ids : torch.LongTensor, shape (batch_size, max_len)
            Integer-encoded, padded input sequences.
            PAD positions contain 0.

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, num_classes)
            Raw (pre-softmax) class scores.
        """
        # ── Step 1: Embedding Lookup ──────────────────────────────────────────
        # (batch, max_len) → (batch, max_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # ── Step 2: Masked Global Average Pooling ─────────────────────────────
        # Why masking matters:
        #   Without masking, PAD positions (which have zero embeddings) would
        #   still be counted in the denominator of the average.  For a short
        #   sequence of 20 real tokens padded to max_len=256, the divisor
        #   would be 256 instead of 20, diluting the real signal by ~13×.
        #
        # Implementation:
        #   1. Build a boolean mask: True where input_ids != 0 (real tokens).
        #   2. Expand mask to embedding_dim for element-wise multiplication.
        #   3. Sum embeddings over sequence length, divide by real token count.

        # (batch, max_len) → True for non-PAD positions
        mask = (input_ids != 0)

        # (batch, max_len) → (batch, max_len, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()

        # Zero out PAD embeddings and sum over sequence dimension
        # (batch, max_len, emb) * (batch, max_len, 1) → (batch, max_len, emb)
        # sum over dim=1 → (batch, emb)
        sum_embeddings = (embedded * mask_expanded).sum(dim=1)

        # Count real (non-PAD) tokens per sample → (batch, 1)
        # clamp(min=1) prevents division by zero for fully-padded sequences
        # (shouldn't happen in practice but is a safety guard)
        token_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)

        # (batch, emb) / (batch, 1) → (batch, emb)
        pooled = sum_embeddings / token_counts

        # ── Step 3: Linear Classification ─────────────────────────────────────
        # (batch, emb) → (batch, num_classes)
        logits = self.fc(pooled)

        return logits
