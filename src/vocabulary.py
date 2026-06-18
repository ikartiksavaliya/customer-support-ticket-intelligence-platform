"""
Customer Support Ticket Intelligence Platform
Module: vocabulary.py

Stateful word-to-integer vocabulary mapping for the NLP pipeline.

Design Principles
-----------------
- The Vocabulary class is built exclusively from TRAINING data to prevent
  data leakage. Never call build_from_corpus() on validation or test splits.
- Special tokens are always assigned the lowest indices so that:
    <PAD> = 0  →  can be passed directly to nn.Embedding(padding_idx=0)
    <UNK> = 1  →  unknown/rare words during inference map here
- Serialisation uses plain JSON (not pickle) so the vocabulary file is
  human-readable and can be inspected with any text editor.
- min_freq is a critical hyperparameter: see the notebook (02_text_preprocessing)
  for a sensitivity analysis showing its impact on OOV rate and embedding size.

Usage
-----
    from src.preprocessing import clean_text, tokenize
    from src.vocabulary import Vocabulary

    # Build from a list of token lists (training data only)
    token_lists = [tokenize(clean_text(t)) for t in train_df["issue_description"]]
    vocab = Vocabulary()
    vocab.build_from_corpus(token_lists, min_freq=5)

    # Encode a sequence
    ids = vocab.encode(tokenize(clean_text("I was charged XXXX")))

    # Persist
    vocab.save("outputs/vocab.json")

    # Reload
    vocab2 = Vocabulary.load("outputs/vocab.json")
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


class Vocabulary:
    """
    A word-to-integer mapping built from a training corpus.

    Special Tokens
    --------------
    Index 0 : <PAD>  — padding token (must be 0 for nn.Embedding padding_idx)
    Index 1 : <UNK>  — unknown / out-of-vocabulary token

    Attributes
    ----------
    _word2idx : dict[str, int]
        Maps every vocabulary word (including special tokens) to its integer index.
    _idx2word : dict[int, str]
        Inverse mapping for decoding integer sequences back to words.
    _is_built : bool
        Guards against encoding before build_from_corpus() has been called.
    """

    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_IDX:   int = 0
    UNK_IDX:   int = 1

    def __init__(self) -> None:
        self._word2idx: Dict[str, int] = {}
        self._idx2word: Dict[int, str] = {}
        self._is_built: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def build_from_corpus(
        self,
        token_lists: List[List[str]],
        min_freq: int = 5,
    ) -> "Vocabulary":
        """
        Build the vocabulary from a list of token lists (training corpus).

        The procedure is:
            1. Initialise special tokens at fixed indices (0 = PAD, 1 = UNK).
            2. Count all tokens across the corpus with collections.Counter.
            3. Add every token whose count >= min_freq in sorted order
               (sorted for reproducibility — same corpus always yields same vocab).

        ⚠️  Data-Leakage Warning
        ------------------------
        This method must ONLY be called on the TRAINING split.
        Fitting on val/test splits would leak information about the test
        distribution into the vocabulary and artificially reduce the OOV rate.

        Parameters
        ----------
        token_lists : list of list of str
            Each inner list is the token sequence for one training sample.
        min_freq : int, optional
            Minimum corpus frequency for a token to be included (default 5).
            Tokens appearing fewer than `min_freq` times are mapped to <UNK>.

        Returns
        -------
        self : Vocabulary
            Returns self for optional method chaining.

        Raises
        ------
        ValueError
            If `min_freq` < 1.
        """
        if min_freq < 1:
            raise ValueError(f"min_freq must be >= 1, got {min_freq}")

        # Step 1: Reserve special tokens
        self._word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }

        # Step 2: Count all tokens across entire training corpus
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # Step 3: Add qualifying tokens in alphabetical order (reproducibility)
        for word in sorted(counter.keys()):
            if counter[word] >= min_freq:
                self._word2idx[word] = len(self._word2idx)

        # Build inverse mapping
        self._idx2word = {idx: word for word, idx in self._word2idx.items()}
        self._is_built = True
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding / Decoding
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of string tokens to a list of integer indices.

        Any token not in the vocabulary is silently mapped to UNK_IDX (1).
        This is the standard approach for handling out-of-vocabulary words
        at inference time.

        Parameters
        ----------
        tokens : list of str
            Pre-tokenised sequence (output of `tokenize(clean_text(...))`).

        Returns
        -------
        list of int
            Integer-encoded sequence of the same length as `tokens`.

        Raises
        ------
        RuntimeError
            If called before `build_from_corpus`.
        """
        self._check_built("encode")
        return [self._word2idx.get(tok, self.UNK_IDX) for tok in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert a list of integer indices back to string tokens.

        Useful for debugging: inspect what an encoded sequence actually contains
        after truncation, padding, and vocab filtering.

        Parameters
        ----------
        indices : list of int
            Integer-encoded sequence (e.g. from TicketDataset.__getitem__).

        Returns
        -------
        list of str
            String tokens, with <PAD> and <UNK> rendered explicitly.

        Raises
        ------
        RuntimeError
            If called before `build_from_corpus`.
        """
        self._check_built("decode")
        return [self._idx2word.get(idx, self.UNK_TOKEN) for idx in indices]

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the vocabulary to a JSON file.

        JSON is chosen over pickle because:
        - Human-readable: you can open it in any text editor to inspect mappings.
        - Cross-version compatible: no Python version pinning issues.
        - Safely shareable: no arbitrary code execution risk on load.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "outputs/vocab.json").
        """
        self._check_built("save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word2idx": self._word2idx,
            "config": {
                "PAD_TOKEN": self.PAD_TOKEN,
                "UNK_TOKEN": self.UNK_TOKEN,
                "PAD_IDX":   self.PAD_IDX,
                "UNK_IDX":   self.UNK_IDX,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """
        Deserialise a Vocabulary from a JSON file saved by `save()`.

        Parameters
        ----------
        path : str
            Path to the JSON file produced by `save()`.

        Returns
        -------
        Vocabulary
            A fully initialised Vocabulary instance ready for encoding.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = cls()
        vocab._word2idx = {w: int(i) for w, i in payload["word2idx"].items()}
        vocab._idx2word = {int(i): w for w, i in payload["word2idx"].items()}
        vocab._is_built = True
        return vocab

    # ─────────────────────────────────────────────────────────────────────────
    # Properties & Dunder Methods
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return total vocabulary size including special tokens."""
        return len(self._word2idx)

    def __contains__(self, word: str) -> bool:
        """Support `"mortgage" in vocab` membership checks."""
        return word in self._word2idx

    def __repr__(self) -> str:
        status = f"size={len(self)}" if self._is_built else "not built"
        return f"Vocabulary({status})"

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _check_built(self, method_name: str) -> None:
        if not self._is_built:
            raise RuntimeError(
                f"Vocabulary.{method_name}() called before build_from_corpus(). "
                "Build the vocabulary first or load one with Vocabulary.load()."
            )
