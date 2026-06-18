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
