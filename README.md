# Customer Support Ticket Intelligence Platform

A production-grade, portfolio-ready Deep Learning and Natural Language Processing (NLP) system designed to automate customer support ticket classification and priority routing. 

---

## 🎯 Project Objectives

### 1. Business Objective
- Automatically classify customer support tickets to minimize manual triaging and routing delays.
- Reduce average first-response times and improve customer satisfaction scores (CSAT).
- Align technical model performance (Accuracy, F1) to concrete business ROI (hours saved, auto-triaged tickets).

### 2. Learning Objective
- Systematically master sequence modeling in PyTorch:
  - From text preprocessing and vocabulary building,
  - Through dense word embedding learning,
  - To recurrent foundations (Simple RNNs, LSTM, GRU, BiLSTM, and Deep stacked recurrent architectures).
- Understand theoretical optimization challenges (vanishing/exploding gradients) and regularizations (recurrent dropout, weight decay).

### 3. Portfolio Objective
- Demonstrate rigorous deep learning engineering practices: modularized code structure, reproducible experiment logs, model benchmarking (latency, memory, parameters), error analysis profiling, and explainability dashboards.

---

## 🏗️ Project Structure

```text
customer-support-ticket-intelligence-platform/
├── data/                  # Raw and processed support tickets (CSV, mappings)
├── notebooks/             # Step-by-step jupyter notebooks for theory & EDA
├── src/                   # Reusable production-grade python modules
│   ├── preprocessing.py   # Cleaning, tokenization, sequence preparation
│   ├── vocabulary.py      # Vocabulary construction, OOV tracking, integer indexing
│   ├── datasets.py        # PyTorch Dataset & DataLoader utilities
│   ├── models.py          # Sequence models (RNN, LSTM, GRU, BiLSTM, Deep architectures)
│   ├── training.py        # Custom PyTorch training loops with metrics logging
│   ├── evaluation.py      # Performance metrics, inference benchmarking, confusion matrices
│   ├── explainability.py  # Feature attribution, critical words extraction
│   └── utils.py           # Helper logic, path utilities, serialization
├── streamlit_app/         # Streamlit-based model dashboards and web interface
│   ├── app.py             # App entry point
│   └── pages/             # App views (predict, benchmarking, explainability, metrics)
├── docs/                  # Educational writeups, data profiles, and decisions logs
├── outputs/               # Saved model checkpoints, evaluation artifacts, and plots
└── README.md              # Project overview
```

---

## 📚 Sequence Modeling Learning Roadmap

We follow a linear pedagogical sequence to model text:
1. **Representations**: TF-IDF Baseline → Word Embeddings (Learned, Pretrained comparison).
2. **Simple RNNs**: Explaining recurrence, hidden states, cell calculations, and vanishing gradients.
3. **Gated RNNs**: Long Short-Term Memory (LSTM) cells and Gated Recurrent Units (GRU).
4. **Advanced Recurrent Nets**: Bidirectional RNNs, Deep Stacked recurrent networks, and capacity/complexity tradeoffs.
5. **Regularization & Optimization**: Dropout, Weight Decay, and optimizer studies (SGD, RMSprop, Adam, AdamW).
6. **Interpretability & Deploy**: Explainability techniques (e.g., attribution scores) and Streamlit dashboard hosting.

---

## 🔧 Setup & Installation

*Detailed installation guidelines will be populated in `docs/DEPLOYMENT_GUIDE.md`.*

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Streamlit

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-support-ticket-intelligence-platform.git
cd customer-support-ticket-intelligence-platform

# Install dependencies
pip install -r requirements.txt
```
