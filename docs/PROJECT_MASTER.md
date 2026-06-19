# Project Master Log: Customer Support Ticket Intelligence Platform

This is the central source of truth for tracking the timeline, phase milestones, and repository structure of the Customer Support Ticket Intelligence Platform.

---

## 📅 Timeline & Phases

| Phase | Description | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Project Planning & Documentation Setup | Setup folders, dataset integration, planning notebooks | **Complete** |
| **Phase 2** | Exploratory Data Analysis (EDA) | Ticket length distributions, class distributions, category-length relationships, text noise auditing, `02_eda.ipynb` | **Complete** |
| **Phase 3** | Text Preprocessing & Vocabulary | Cleaning pipelines, tokenization, PyTorch datasets, `03_text_preprocessing.ipynb` | **Complete** |
| **Phase 4** | Embedding Studies | One-hot vs Dense, learned PyTorch embeddings, 50-300 dim sweep, `04_embedding_study.ipynb` | **In Progress** |
| **Phase 5** | Recurrent Neural Network (RNN) Foundations | Simple RNN, vanishing gradients study, `05_simple_rnn.ipynb`, `06_vanishing_gradient_study.ipynb` | Pending |
| **Phase 6** | Gated Sequence Models (LSTM & GRU) | LSTM vs GRU cell structures, gradient flow stability, `07_lstm.ipynb`, `08_gru.ipynb` | Pending |
| **Phase 7** | Advanced Sequence Models | Bidirectional RNNs, Deep multi-layered stacked RNNs/LSTMs, `09_bidirectional_rnn.ipynb`, `10_deep_rnn.ipynb` | Pending |
| **Phase 8** | Regularization & Optimization Study | Dropout, weight decay, optimizer comparison (AdamW/SGD), `11_regularization_study.ipynb` | Pending |
| **Phase 9** | Evaluation, Error Analysis & Final Model | Confusion matrices, error type audits, class imbalance, `12_error_analysis.ipynb`, `13_final_model.ipynb` | Pending |
| **Phase 10** | Production Streamlit Dashboard | Streamlit application, predict view, business impact dashboard, `streamlit_app/app.py` | Pending |
| **Phase 11** | Project Wrap-up & Final Report | Complete final docs, report slides, `14_final_report.ipynb` | Pending |

---

## 🛠️ Repository Mapping

- **`/data`**: Stores the raw CSV `consumer-complaint-database.csv` (ignored by Git) and the generated processed files:
  - `df_clean.csv` (deduplicated full dataset, ignored by Git)
  - `df_sample.csv` (stratified modeling sample of 200,000 rows, ignored by Git)
- **`/docs`**: Central repo for sequence model studies, mathematical theory explanations, and interview prep questions.
- **`/notebooks`**: Sequentially-numbered Jupyter Notebooks for step-by-step training and evaluations.
- **`/src`**: Modularized, clean, linted, PEP-8 compliant Python source modules containing PyTorch datasets, custom neural network modules, and standard preprocessing steps.
- **`/outputs`**: Stores trained model state dict weights (`.pt`), learning curves (`.png`), and error profiles (`.csv`).

---

## 🚦 Repository Quality Standards

1. **Mentor-First Approach**: Code is only written *after* the theoretical foundations, math details, and tradeoffs are outlined in `docs/` and notebooks.
2. **Pedagogical Integrity**: Documentation must explain the "why", not just the "how", detailing common interview questions and bugs.
3. **Reproducibility**: Set explicit manual seeds (`torch.manual_seed(42)`) in all files and notebooks.
4. **No Code Placeholders**: Keep all notebooks and scripts fully executable.
