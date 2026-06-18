# Model Evolution Log

This log chronicles the structural modifications, parameter tweaks, and learning dynamics across all model versions.

---

## 🧬 Model Version History

### MODEL-v1: Bag of Embeddings (BoE) Classifier
- **Phase**: Phase 4 (Embedding Studies)
- **Parameters**: Embedding layer + Linear output head (e.g. $23,262 \times d + d \times 18$).
- **Structure**:
  - `nn.Embedding(vocab_size=23262, embedding_dim=emb_dim, padding_idx=0)`
  - Global Average Pooling (average embedding vectors across non-padding sequence length)
  - `nn.Linear(emb_dim, 18)`
- **Goal**: Serve as a simple sequence-agnostic deep learning baseline.
- **Key Limitations**: Ignores word order and syntax structures.

---

### MODEL-v2: Vanilla Simple RNN (Word-Level)
- **Phase**: Phase 5
- **Parameters**: Embedding layer + Single hidden recurrent layer + Linear output head.
- **Structure**:
  - `nn.Embedding(vocab_size=23262, embedding_dim=50, padding_idx=0)`
  - `nn.RNN(input_size=50, hidden_size=64, batch_first=True)`
  - `nn.Linear(64, 18)`
- **Key Limitations**: Subject to vanishing gradient problems, making learning difficult.

---

### MODEL-v3: LSTM Classifier
- **Phase**: Phase 6
- **Parameters**: Embedding layer + Single LSTM layer + Linear output head.
- **Structure**:
  - `nn.Embedding(vocab_size=23262, embedding_dim=50, padding_idx=0)`
  - `nn.LSTM(input_size=50, hidden_size=64, batch_first=True)`
  - `nn.Linear(64, 18)`
- **Key Modifications**: Added gating mechanisms (input, forget, output gates) and cell state to preserve long-term context and stabilize gradient flow.

---

### MODEL-v4: GRU Classifier
- **Phase**: Phase 6
- **Parameters**: Embedding layer + Single GRU layer + Linear output head.
- **Structure**:
  - `nn.Embedding(vocab_size=23262, embedding_dim=50, padding_idx=0)`
  - `nn.GRU(input_size=50, hidden_size=64, batch_first=True)`
  - `nn.Linear(64, 18)`
- **Key Modifications**: Employs reset and update gates to merge cell and hidden states for efficiency.

---

### MODEL-v5: Bidirectional LSTM (BiLSTM)
- **Phase**: Phase 7
- **Parameters**: 2x recurrent parameters due to separate forward and backward processing passes.
- **Structure**:
  - `nn.Embedding(vocab_size=23262, embedding_dim=50, padding_idx=0)`
  - `nn.LSTM(input_size=50, hidden_size=64, batch_first=True, bidirectional=True)`
  - Linear classification head takes concatenated state `[h_forward; h_backward]` (dimension 128) and projects to 18 classes.
- **Key Modifications**: Merges left-to-right and right-to-left sentence contexts.

---

### MODEL-v6: Stacked Deep BiLSTM
- **Phase**: Phase 7
- **Parameters**: Multifold parameter scaling due to multiple stacked layers.
- **Structure**:
  - `nn.Embedding(vocab_size=23262, embedding_dim=50, padding_idx=0)`
  - `nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)`
  - Linear classification head projects to 18 classes.
- **Key Modifications**: Hierarchical state stacking for capacity enhancement.
