# Model Evolution Log

This log chronicles the structural modifications, parameter tweaks, and learning dynamics across all model versions.

---

## 🧬 Model Version History

### MODEL-v1: TF-IDF + Logistic Regression (Baseline)
- **Phase**: Phase 4
- **Parameters**: Count depends on vocabulary (TF-IDF features).
- **Structure**: Sparse TF-IDF text representation with character n-grams, fed into a multi-class Logistic Regression model.
- **Goal**: Serve as a simple non-deep learning baseline.
- **Key Limitations**: Ignores word sequence context and sentence semantics.

---

### MODEL-v2: Vanilla Simple RNN (Word-Level)
- **Phase**: Phase 6
- **Parameters**: Embedding layer + Single hidden recurrent layer + Linear output head.
- **Structure**:
  - `nn.Embedding(vocab_size=80, embedding_dim=50)`
  - `nn.RNN(input_size=50, hidden_size=64, batch_first=True)`
  - `nn.Linear(64, 10)`
- **Key Limitations**: Subject to vanishing gradient problems, making learning difficult.

---

### MODEL-v3: Gated RNNs (LSTM & GRU)
- **Phase**: Phase 7
- **Parameters**: Similar size to RNN but with 4x (LSTM) or 3x (GRU) weight parameters in the recurrent cell.
- **Structure**:
  - LSTM: `nn.LSTM(input_size=50, hidden_size=64, batch_first=True)`
  - GRU: `nn.GRU(input_size=50, hidden_size=64, batch_first=True)`
- **Key Modifications**: Added gating mechanisms to preserve long-term hidden states.

---

### MODEL-v4: Bidirectional LSTM (BiLSTM)
- **Phase**: Phase 8
- **Parameters**: 2x recurrent parameters due to separate forward and backward processing passes.
- **Structure**:
  - `nn.LSTM(input_size=50, hidden_size=64, batch_first=True, bidirectional=True)`
  - Linear classification head takes concatenated state `[h_forward; h_backward]` (dimension 128) and projects to 10 classes.
- **Key Modifications**: Merges left-to-right and right-to-left sentence contexts.

---

### MODEL-v5: Stacked Deep BiLSTM
- **Phase**: Phase 8
- **Parameters**: Multifold parameter scaling due to multiple stacked layers.
- **Structure**:
  - `nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)`
- **Key Modifications**: Hierarchical state stacking for capacity enhancement.
