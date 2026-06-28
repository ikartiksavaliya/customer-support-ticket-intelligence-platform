"""
Customer Support Ticket Intelligence Platform
Module: dataset.py

PyTorch Dataset and DataLoader utilities for the ticket classification pipeline.

Design Principles
-----------------
- TicketDataset wraps a pandas DataFrame and a Vocabulary; it does NOT load
  data from disk on every __getitem__ call — the DataFrame is kept in memory.
- All encoding, truncation, and padding is done inside __init__.
- Truncation is done PRE-truncation (keep the LAST max_len tokens).
  Rationale (from INTERVIEW_NOTES.md + DATASET_PROFILE.md):
    Consumer complaints in the CFPB corpus often begin with product
    identifiers and then elaborate on the issue. For classification,
    the *concluding details* of the complaint (specific grievances, companies,
    amounts) are frequently more discriminative than the opening boilerplate.
    Pre-truncation preserves this tail context.
  Note: this is a design decision documented in DECISION_LOG and can be
  revisited as an ablation experiment in later phases.
- Padding is POST-padding (zeros appended after the real tokens).
  This is consistent with pack_padded_sequence usage in RNN models.
- return_lengths (Phase 5+): When True, __getitem__ returns a 3-tuple
  (input_ids, label, seq_len) where seq_len is the number of real (non-PAD)
  tokens.  This is required by torch.nn.utils.rnn.pack_padded_sequence
  which needs real sequence lengths to skip PAD positions inside the RNN.

Usage
-----
    import pandas as pd
    from src.preprocessing import clean_text, tokenize
    from src.vocabulary import Vocabulary
    from src.dataset import TicketDataset
    from torch.utils.data import DataLoader

    vocab = Vocabulary.load("outputs/vocab.json")
    label_encoder = {"Debt collection": 0, "Mortgage": 1, ...}

    ds = TicketDataset(df_train, vocab, label_encoder, max_len=256)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    for input_ids, labels in loader:
        # input_ids : (64, 256) long tensor
        # labels    : (64,)     long tensor
        ...
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Tuple

from src.preprocessing import clean_text, tokenize
from src.vocabulary import Vocabulary


class TicketDataset(Dataset):
    """
    PyTorch Dataset for the CFPB Consumer Complaint classification task.

    Each item is a (input_ids, label) pair where:
    - input_ids : LongTensor of shape (max_len,)
        Pre-truncated, post-padded integer sequence.
        PAD index (0) fills positions beyond the actual sequence length.
    - label     : LongTensor scalar (0-dimensional)
        Integer class index for the complaint category.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'issue_description' (str), 'category' (str).
    vocab : Vocabulary
        A fully built Vocabulary instance (must have been built from
        training data only).
    label_encoder : dict[str, int]
        Mapping from category string to integer class index.
        Example: {"Debt collection": 0, "Mortgage": 1, ...}
    max_len : int, optional
        Maximum sequence length in tokens (default 256).
        Sequences longer than max_len are PRE-truncated (tail kept).
        Sequences shorter than max_len are POST-padded with PAD_IDX (0).
    return_lengths : bool, optional
        If True, __getitem__ returns a 3-tuple (input_ids, label, seq_len)
        where seq_len is the actual number of tokens before padding.
        Required for pack_padded_sequence in RNN models (Phase 5+).
        Default False to preserve backward compatibility with BoE (Phase 4).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        label_encoder: Dict[str, int],
        max_len: int = 256,
        return_lengths: bool = False,
    ) -> None:
        # Store a reset-index copy to ensure clean 0-based integer indexing.
        # This prevents subtle bugs when a DataFrame slice is passed in (e.g.
        # the training split from a stratified split, which has non-contiguous
        # indices).
        self._df = df.reset_index(drop=True)
        self._vocab = vocab
        self._label_encoder = label_encoder
        self._max_len = max_len
        self._return_lengths = return_lengths

        # Pre-validate: check required columns exist
        for col in ("issue_description", "category"):
            if col not in self._df.columns:
                raise ValueError(
                    f"DataFrame must contain column '{col}'. "
                    f"Found: {list(self._df.columns)}"
                )

        # Precompute and cache encoded IDs, sequence lengths, and labels in memory
        self._precomputed_ids = []
        self._precomputed_lengths = []
        self._precomputed_labels = []

        descriptions = self._df["issue_description"].tolist()
        categories = self._df["category"].tolist()

        for desc, cat in zip(descriptions, categories):
            # ── Text processing ───────────────────────────────────────────────
            cleaned = clean_text(desc)
            tokens  = tokenize(cleaned)
            ids     = self._vocab.encode(tokens)          # list[int]

            # ── PRE-truncation (keep tail for richer complaint context) ──────
            if len(ids) > self._max_len:
                ids = ids[-self._max_len:]

            # Record actual sequence length BEFORE padding (clamped to max_len)
            # This is needed by pack_padded_sequence in RNN models.
            seq_len = len(ids)

            # ── POST-padding (zeros to the right) ────────────────────────────
            pad_length = self._max_len - len(ids)
            ids = ids + [self._vocab.PAD_IDX] * pad_length

            # ── Label encoding ────────────────────────────────────────────────
            label = self._label_encoder[cat]

            self._precomputed_ids.append(ids)
            self._precomputed_lengths.append(seq_len)
            self._precomputed_labels.append(label)

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset Protocol (required by torch.utils.data.Dataset)
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the total number of samples in this split."""
        return len(self._df)

    def __getitem__(self, idx: int):
        """
        Retrieve the pre-processed and pre-encoded sample at position idx.

        Processing pipeline is done once on initialization, so this is O(1):
            1. Retrieve pre-computed token IDs, label index, and sequence length.
            2. Convert to PyTorch LongTensors.
            3. Return tuple (see below).

        Parameters
        ----------
        idx : int
            Row index (0-based) into the dataset.

        Returns
        -------
        If return_lengths is False (default — BoE / Phase 4):
            (input_ids, label)
        If return_lengths is True (RNN models — Phase 5+):
            (input_ids, label, seq_len)

        Where:
            input_ids : torch.LongTensor, shape (max_len,)
            label     : torch.LongTensor, shape ()
            seq_len   : torch.LongTensor, shape () — actual non-PAD token count
        """
        ids = self._precomputed_ids[idx]
        label = self._precomputed_labels[idx]
        seq_len = self._precomputed_lengths[idx]

        input_ids_t = torch.tensor(ids,   dtype=torch.long)
        label_t     = torch.tensor(label, dtype=torch.long)

        if self._return_lengths:
            return input_ids_t, label_t, torch.tensor(seq_len, dtype=torch.long)
        return input_ids_t, label_t

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection helpers (useful in notebooks)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        """Total number of unique label classes."""
        return len(self._label_encoder)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (including special tokens)."""
        return len(self._vocab)

    def decode_sample(self, idx: int) -> Dict:
        """
        Return a human-readable dict for sample `idx` — useful in notebooks
        for sanity-checking that encoding/decoding is working correctly.

        Parameters
        ----------
        idx : int
            Row index (0-based).

        Returns
        -------
        dict with keys:
            'raw_text'      : original complaint string
            'clean_text'    : output of clean_text()
            'token_count'   : number of tokens before truncation
            'decoded_ids'   : list of str tokens after encoding and decoding
                              (reveals <UNK> substitutions and padding)
            'label_str'     : category string
            'label_int'     : integer class index
        """
        row = self._df.iloc[idx]
        cleaned  = clean_text(row["issue_description"])
        tokens   = tokenize(cleaned)
        ids, lbl = self.__getitem__(idx)

        return {
            "raw_text":    row["issue_description"],
            "clean_text":  cleaned,
            "token_count": len(tokens),
            "decoded_ids": self._vocab.decode(ids.tolist()),
            "label_str":   row["category"],
            "label_int":   lbl.item(),
        }
