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
    MODEL-v1 : BagOfEmbeddings        (Phase 4 — this file)
    MODEL-v2 : SimpleRNNClassifier    (Phase 5 — this file)
    MODEL-v3 : LSTMClassifier         (Phase 6 — future)
    MODEL-v4 : GRUClassifier          (Phase 6 — future)
    MODEL-v5 : BiLSTMClassifier       (Phase 7 — future)
    MODEL-v6 : DeepBiLSTM             (Phase 7 — future)

Usage
-----
    from src.models import BagOfEmbeddings, SimpleRNNClassifier

    model = BagOfEmbeddings(vocab_size=23262, embedding_dim=50, num_classes=18)
    logits = model(input_ids)  # (batch, 18)

    rnn = SimpleRNNClassifier(vocab_size=23262, embedding_dim=50, hidden_dim=64, num_classes=18)
    logits = rnn(input_ids, seq_lengths)  # (batch, 18)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


# =============================================================================
# MODEL-v2: Simple RNN Classifier (Phase 5)
# =============================================================================

class SimpleRNNClassifier(nn.Module):
    """
    Vanilla RNN classifier — MODEL-v2.

    The first *sequence-aware* model in our architecture progression.
    Unlike BagOfEmbeddings (which averages all embeddings regardless of
    position), this model processes tokens **one at a time** in order,
    maintaining a hidden state that accumulates context from left to right.

    Why RNNs?
    ---------
    In natural language, word order matters.  Consider these two sentences:
        - "The payment was deducted but the refund was issued."
        - "The refund was issued but the payment was deducted."
    A BoE model would produce identical representations (same words),
    but an RNN can distinguish them because it processes tokens sequentially
    and the hidden state at the end captures the *order* of events.

    Architecture
    ------------
    ```
    Input IDs (batch, max_len) + Lengths (batch,)
         │
         ▼
    nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
         │
         ▼
    Embeddings (batch, max_len, embedding_dim)
         │
         ▼
    pack_padded_sequence  ← skip PAD positions in the RNN computation
         │
         ▼
    nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
         │
         ▼
    Final Hidden State h_T (batch, hidden_dim)
         │      ↑ extracted from the LAST REAL token, not the last PAD
         ▼
    nn.Linear(hidden_dim, num_classes)
         │
         ▼
    Logits (batch, num_classes)
    ```

    Why pack_padded_sequence?
    ------------------------
    Without packing, the RNN processes PAD tokens (zero embeddings) through
    the recurrence.  After 200+ PAD steps, the hidden state is dominated by
    the bias term and tanh saturation, effectively washing out the real
    text signal.  pack_padded_sequence tells PyTorch to skip PAD positions
    entirely, so the final hidden state corresponds to the last *real* token.

    Known Limitation: Vanishing Gradients
    -------------------------------------
    The RNN transition h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b) involves
    repeated multiplication by W_hh during backpropagation.  For long
    sequences, this causes gradients to either vanish (if eigenvalues of
    W_hh < 1) or explode (if > 1).  This is studied in notebook 06.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size including special tokens (<PAD>=0, <UNK>=1).
    embedding_dim : int
        Dimensionality of the dense word embedding vectors.
    hidden_dim : int
        Number of hidden units in the RNN cell.
    num_classes : int
        Number of output classes (18 for CFPB categories).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # ── Embedding Layer ───────────────────────────────────────────────
        # Same embedding configuration as BoE (padding_idx=0).
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # ── Recurrent Layer ──────────────────────────────────────────────
        # batch_first=True: input shape is (batch, seq_len, embedding_dim)
        # This is a single-layer vanilla RNN with tanh activation.
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # ── Classification Head ──────────────────────────────────────────
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.LongTensor,
        seq_lengths: torch.LongTensor = None,
    ) -> torch.Tensor:
        """
        Forward pass: embed → pack → RNN → extract last hidden → classify.

        Parameters
        ----------
        input_ids : torch.LongTensor, shape (batch_size, max_len)
            Integer-encoded, padded input sequences.
        seq_lengths : torch.LongTensor, shape (batch_size,)
            Actual (pre-padding) sequence lengths for each sample.
            Required for pack_padded_sequence.
            If None, assumes all positions are real tokens (no packing).

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, num_classes)
            Raw (pre-softmax) class scores.
        """
        batch_size = input_ids.size(0)

        # ── Step 1: Embedding Lookup ──────────────────────────────────────
        # (batch, max_len) → (batch, max_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # ── Step 2: Pack padded sequences ────────────────────────────────
        # pack_padded_sequence requires lengths on CPU and sorted by
        # descending length (enforce_sorted=False handles this for us).
        if seq_lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                seq_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        else:
            packed = embedded

        # ── Step 3: Initialize hidden state to zeros ────────────────────
        # h_0 shape: (num_layers, batch_size, hidden_dim)
        # Always start from zero — each batch is independent.
        # See INTERVIEW_NOTES.md Common Mistake #2.
        h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=input_ids.device)

        # ── Step 4: Run through RNN ──────────────────────────────────────
        # output: all hidden states   (batch, seq_len, hidden_dim)
        # h_n:    final hidden state   (1, batch, hidden_dim)
        _, h_n = self.rnn(packed, h_0)

        # ── Step 5: Extract final hidden state ───────────────────────────
        # h_n is (1, batch, hidden_dim) → squeeze to (batch, hidden_dim)
        # Because we used pack_padded_sequence, h_n corresponds to the
        # hidden state at the LAST REAL TOKEN of each sequence, not the
        # last PAD position.
        h_final = h_n.squeeze(0)  # (batch, hidden_dim)

        # ── Step 6: Linear Classification ───────────────────────────────
        # (batch, hidden_dim) → (batch, num_classes)
        logits = self.fc(h_final)

        return logits
