# Model Comparison Report

This report summarizes the final performance, inference profiles, and computational requirements of all sequence models evaluated in this project.

---

## 📊 Performance Comparison Grid

| Model ID | Model Name | Val F1-Score | Parameter Count | Avg Latency (CPU) | Throughput (tk/s) | Peak Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MODEL-v1** | TF-IDF + Logistic Reg | | | | | |
| **MODEL-v2** | Simple RNN | | | | | |
| **MODEL-v3** | LSTM | | | | | |
| **MODEL-v4** | GRU | | | | | |
| **MODEL-v5** | BiLSTM | | | | | |
| **MODEL-v6** | Stacked Deep BiLSTM | | | | | |

---

## ⚖️ Trade-off Analysis

### 1. Accuracy vs Latency
- Vanilla RNNs are fast but suffer from gradient vanishing.
- LSTMs and GRUs introduce gates, increasing parameter counts but solving gradient problems.
- BiLSTMs double recurrent processing steps, increasing CPU latency but capturing bi-directional contexts.

### 2. Parameter Efficiency
- We evaluate how F1-score scales relative to parameter size.
- **GRU vs LSTM**: GRUs combine forget and input gates, resulting in 25% fewer parameters than LSTMs, which can speed up inference on CPU resource-constrained nodes.
