# Decision Log

This log documents key structural, modeling, optimization, and preprocessing decisions made during the project, along with their business and technical rationales.

---

## 🗒️ Log Entries

### Decision 1: Focus on Single-Class Category Classification
- **Status**: Approved
- **Context**: The raw CFPB dataset has multiple target candidates, but we are designing this pipeline as a single-label multi-class classification problem.
- **Decision**: Focus exclusively on `category` (18 classes) mapped from `Product`.
- **Technical Rationale**: Keeps sequence model evaluations clean (RNN vs LSTM vs GRU vs BiLSTM) and avoids confounding optimization issues of multi-task prediction heads early in the learning process.

---

### Decision 2: Implementation of a CPU Benchmarking Pipeline
- **Status**: Approved
- **Context**: The deployment target requires efficient operation on local environments or low-cost clouds.
- **Decision**: Build an explicit inference benchmarks module reporting average inference latency (target < 100ms on CPU), throughput (tickets/sec), model parameter count, and memory footprint.
- **Business Rationale**: Real-world routing needs to be fast and cost-effective; measuring parameter efficiency alongside F1-score provides a holistic engineering view.

---

### Decision 3: Framework and Learning Strategy
- **Status**: Approved
- **Context**: The project has two goals: professional implementation and systematic deep learning study.
- **Decision**: Use **PyTorch** for model architecture definition and custom training loops. Use **Jupyter Notebooks** for interactive studies (gradients, representations, regularization) and modularized `.py` files inside `src/` for reusable pipeline components.
- **Technical Rationale**: PyTorch provides complete, low-level control over hidden states, gradients, and cell structures, which is ideal for studying backpropagation through time.

---

### Decision 4: Class Imbalance Mitigation Strategy
- **Status**: Approved
- **Context**: Real-world CFPB customer complaints exhibit significant class imbalance (e.g., Debt Collection at 23% vs Virtual Currency at 0.004%). Standard training will bias the model towards major classes.
- **Decision**: We will implement:
  1. **Class-Weighted CrossEntropyLoss** to penalize misclassifications of minority classes.
  2. **Evaluation Metrics**: Focus heavily on **Macro-averaged F1-score** and individual class precision/recall curves, rather than raw global accuracy.
- **Pedagogical Rationale**: Handling class imbalance is a vital real-world ML engineering skill. This allows us to study the tradeoffs between minority class sensitivity and overall majority class performance.

---

### Decision 5: Bag of Embeddings (BoE) as Sequence-Agnostic Baseline
- **Status**: Approved
- **Context**: Before building any recurrent model (RNN, LSTM, GRU), we need a baseline that uses learned embeddings but deliberately ignores word order.
- **Decision**: MODEL-v1 is a **Bag of Embeddings** classifier: `nn.Embedding` → Masked Global Average Pooling → `nn.Linear`. It treats each document as an unordered collection of word vectors.
- **Technical Rationale**:
  1. **Lower bound**: Any sequence model (RNN, LSTM) that scores below BoE on F1 is broken — this baseline tests embedding quality in isolation.
  2. **Teaches nn.Embedding mechanics** before adding recurrence: padding_idx, gradient flow through lookup tables, dimensionality tradeoffs.
  3. **Fast to train**: No sequential bottleneck, enabling rapid embedding dimension sweeps (50d, 100d, 200d, 300d).
- **Pedagogical Rationale**: Starting with a "dumb" baseline that ignores sequence order creates a clear motivation for why RNNs are needed — the performance gap between BoE and RNN models will directly quantify the value of sequential processing.

---

### Decision 6: pack_padded_sequence for Correct RNN Sequence Handling
- **Status**: Approved
- **Context**: Post-padded sequences (PAD=0 appended to the right) cause RNNs to process hundreds of meaningless zero-embedding tokens, washing out the hidden state (see INTERVIEW_NOTES.md Common Mistake #1).
- **Decision**: Use `torch.nn.utils.rnn.pack_padded_sequence` in all RNN-based models. The `TicketDataset` returns sequence lengths via `return_lengths=True`, which are passed to the model's `forward()` method.
- **Technical Rationale**: Packing tells the RNN to stop at the last *real* token for each sample. The returned `h_n` then corresponds to the hidden state at the actual sequence end, not at position `max_len`. This is critical for correctness — without it, the classification head receives a hidden state contaminated by 200+ PAD steps.

---

### Decision 7: Gradient Clipping for RNN Training Stability
- **Status**: Approved
- **Context**: Vanilla RNNs involve repeated multiplication by $W_{hh}$ during BPTT. When eigenvalues of $W_{hh} > 1$, gradients explode exponentially (see NLP_CONCEPT_NOTES.md, INTERVIEW_NOTES.md Q3 & Common Mistake #3).
- **Decision**: Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` and before `optimizer.step()` for all recurrent models.
- **Technical Rationale**: Gradient clipping caps the total gradient norm, preventing sudden NaN/divergence without altering the gradient direction. The threshold `max_norm=1.0` is a widely-used default. Note: clipping prevents *exploding* gradients but does NOT solve *vanishing* gradients — that requires architectural changes (LSTM/GRU).

---

### Decision 8: Multi-Environment Support, Package Installation, and Path Portability
- **Status**: Approved
- **Context**: The platform needs to run seamlessly across local development, VS Code with GPU support, and Google Colab (with CPU/GPU/TPU options). The old codebase suffered from `ModuleNotFoundError: No module named 'src'` on Colab due to inconsistent path configurations and absolute imports.
- **Decision**:
  1. Add `pyproject.toml` and `setup.py` to enable editable package installation (`pip install -e .`) so that the project module `src` becomes importable system-wide.
  2. Implement `setup_colab()` and `get_device()` in `src/utils.py` to automatically detect, configure, and output environment settings, supporting CPU, CUDA GPU, and TPU (via `torch_xla`).
  3. Integrate a unified **Notebook Bootstrap Cell** at the top of all notebooks to automate path setups and device detection consistently.
  4. Rename `src/datasets.py` to `src/dataset.py` to match PEP-8 conventions for singular module naming, while preserving backward compatibility with `from src.datasets import *`.
- **Technical Rationale**: Provides path portability, simplifies environment setup, and ensures reproducibility across diverse developer workspaces.

