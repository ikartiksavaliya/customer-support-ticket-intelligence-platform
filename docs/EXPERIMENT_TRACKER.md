# Experiment Tracker

This log tracks every empirical run across our representation, architecture, embedding, optimization, and regularization studies.

---

## 📊 Evaluation Metrics Directory

All studies are evaluated under identical conditions:
* **Train / Val / Test Split**: 80% / 10% / 10%
* **Hardware**: CPU (Inference Benchmarks run on single core thread)
* **Target Metric**: Category Classification (10 classes)

---

## 🏃 Experiment Registry

| ID | Study Type | Model / Config | Epochs | Train Loss | Val Loss | F1-Score | Inference Latency (ms) | Throughput (tk/sec) | Param Count | Memory Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline | TF-IDF + Logistic Reg | - | | | | | | | |
| **EXP-02** | Architecture | Simple RNN (50d emb, 64h) | | | | | | | | |
| **EXP-03** | Architecture | LSTM (50d emb, 64h) | | | | | | | | |
| **EXP-04** | Architecture | GRU (50d emb, 64h) | | | | | | | | |
| **EXP-05** | Architecture | BiLSTM (50d emb, 64h) | | | | | | | | |
| **EXP-06** | Architecture | Deep BiLSTM (2-layer, 64h) | | | | | | | | |
| **EXP-07** | Embeddings | GRU (100d emb, 64h) | | | | | | | | |
| **EXP-08** | Embeddings | GRU (200d emb, 64h) | | | | | | | | |
| **EXP-09** | Embeddings | GRU (300d emb, 64h) | | | | | | | | |
| **EXP-10** | Optimizers | GRU (SGD) | | | | | | | | |
| **EXP-11** | Optimizers | GRU (RMSprop) | | | | | | | | |
| **EXP-12** | Optimizers | GRU (AdamW) | | | | | | | | |
| **EXP-13** | Regularization| GRU (Baseline) | | | | | | | | |
| **EXP-14** | Regularization| GRU (Dropout 0.2) | | | | | | | | |
| **EXP-15** | Regularization| GRU (Weight Decay 1e-4) | | | | | | | | |
| **EXP-16** | Regularization| GRU (Combined Regularization)| | | | | | | | |

---

## 📈 Study Summary & Key Observations

*To be updated post-execution of each respective phase.*
