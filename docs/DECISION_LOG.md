# Decision Log

This log documents key structural, modeling, optimization, and preprocessing decisions made during the project, along with their business and technical rationales.

---

## 🗒️ Log Entries

### Decision 1: Focus on Single-Class Category Classification First
- **Status**: Approved
- **Context**: The dataset has multiple potential classification targets, including `category` and `priority`.
- **Decision**: Focus exclusively on `category` classification as the primary target. Introduce multi-task learning (category + priority) as an advanced extension (Phase X) after completing the core sequence models.
- **Technical Rationale**: Keeps sequence model evaluations clean (RNN vs LSTM vs GRU vs BiLSTM) and avoids confounding optimization issues of multi-task prediction heads early in the learning process.

---

### Decision 2: Implementation of a CPU Benchmarking Pipeline
- **Status**: Approved
- **Context**: Deployment target requires efficient operation on local environments or low-cost clouds.
- **Decision**: Build an explicit inference benchmarks module reporting average inference latency (target < 100ms on CPU), throughput (tickets/sec), model parameter count, and memory footprint.
- **Business Rationale**: Real-world routing needs to be fast and cost-effective; measuring parameter efficiency alongside F1-score provides a holistic engineering view.

---

### Decision 3: Framework and Learning Strategy
- **Status**: Approved
- **Context**: The project has two goals: professional implementation and systematic deep learning study.
- **Decision**: Use **PyTorch** for model architecture definition and custom training loops. Use **Jupyter Notebooks** for interactive studies (gradients, representations, regularization) and modularized `.py` files inside `src/` for reusable pipeline components.
- **Technical Rationale**: PyTorch provides complete, low-level control over hidden states, gradients, and cell structures, which is ideal for studying backpropagation through time.

---

### Decision 4: Dataset Independence Acknowledgement
- **Status**: Approved
- **Context**: Exploratory analysis revealed that labels are independent of the input text.
- **Decision**: Accept the statistical limitation of the synthetic dataset. Focus evaluation on training convergence speeds, generalization gaps (overfitting study), parameter efficiency, and latency benchmarks, expecting validation accuracy to stabilize around 10% (random guess).
- **Pedagogical Rationale**: Prevents chasing false signals and instead centers the study on structural differences between Simple RNNs, LSTMs, and GRUs.
