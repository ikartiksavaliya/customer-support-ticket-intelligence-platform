# Project Master Log: Customer Support Ticket Intelligence Platform

This is the central source of truth for tracking the timeline, phase milestones, and repository structure of the Customer Support Ticket Intelligence Platform.

---

## 📅 Timeline & Phases

| Phase | Description | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Project Planning & Documentation Setup | Setup folders, dataset relocation, write baseline docs, analyze business problem, prepare planning notebook | **In Progress** |
| **Phase 2** | Exploratory Data Analysis (EDA) | Ticket length distributions, class distributions, category/priority relationships, noisy patterns, `02_eda.ipynb` | Pending |
| **Phase 3** | Text Preprocessing & Vocabulary | Cleaning pipelines, lowercasing, tokenization, pad/trunc, Integer encoding, OOV vocabulary dictionary, `03_text_preprocessing.ipynb` | Pending |
| **Phase 4** | Text Representation & Baseline Model | TF-IDF extraction, Logistic Regression/Naive Bayes classification baseline, pipeline check, `04_text_representation.ipynb`, `06_baseline_tfidf.ipynb` | Pending |
| **Phase 5** | Embedding Studies | One-hot vs Dense embeddings, learned PyTorch embeddings, dimensional comparison (50, 100, 200, 300 dim), `05_embedding_study.ipynb` | Pending |
| **Phase 6** | Recurrent Neural Network (RNN) Foundations | Simple RNN implementation, hidden states unrolled in time, vanishing gradients empirical study, `07_simple_rnn.ipynb`, `08_vanishing_gradient_study.ipynb` | Pending |
| **Phase 7** | Gated Sequence Models (LSTM & GRU) | LSTM vs GRU vs Simple RNN cell structures, cell/hidden state gradient flow, learning stability comparison, `09_lstm.ipynb`, `10_gru.ipynb` | Pending |
| **Phase 8** | Advanced Sequence Models | Bidirectional RNNs, Deep multi-layered stacked RNNs/LSTMs, capacity vs parameters study, `11_bidirectional_rnn.ipynb`, `12_deep_rnn.ipynb` | Pending |
| **Phase 9** | Regularization & Optimization Study | Dropout, recurrent dropout, weight decay, early stopping, Adam/AdamW/SGD/RMSprop comparison, `13_regularization_study.ipynb`, `14_hyperparameter_tuning.ipynb` | Pending |
| **Phase 10** | Evaluation, Error Analysis & Final Model | Confusion matrices, error type audits (semantic overlap, noise), deployment candidate selection, `15_model_comparison.ipynb`, `16_error_analysis.ipynb`, `17_final_model.ipynb` | Pending |
| **Phase 11** | Production Streamlit Dashboard | Streamlit application code, inference pipeline, predict view, dashboards for business metrics, deployment docs, `18_streamlit_deployment.ipynb` | Pending |
| **Phase 12** | Project Wrap-up & Final Report | Complete final documentation, portfolio presentation slides/report, `19_final_report.ipynb` | Pending |

---

## 🛠️ Repository Mapping

- **`/data`**: Stores the raw CSV `customer_support_tickets_200k.csv` (and any future generated training/validation splits).
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
