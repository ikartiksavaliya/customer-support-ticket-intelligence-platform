# Deployment Guide

This guide provides instructions for setting up, configuring, and serving our customer support ticket sequence classifiers using our Streamlit application.

---

## 💻 Environment Setup

### Prerequisites
- Linux OS
- Conda or virtualenv (recommended)
- Python 3.8+

### Installation Steps

You can set up the environment in one of three ways:

#### Option A: Using the Local Setup Script (Recommended for Virtualenv)
```bash
# Run the automated setup script from the repository root
./scripts/setup_local.sh
```

#### Option B: Using Conda (Recommended for Conda users)
```bash
# Create and activate environment from yml file
conda env create -f environment.yml
conda activate customer-support
```

#### Option C: Manual Installation
```bash
# 1. Create and activate a fresh environment
conda create -n ticket_intel python=3.10 -y
conda activate ticket_intel

# 2. Install PyTorch (CPU version is sufficient for low-latency CPU serving)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install other requirements and setup editable package
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Running the Streamlit Application

The Streamlit web application resides in the `streamlit_app/` directory.

To run the application locally:
```bash
streamlit run streamlit_app/app.py
```

### Application Dashboard structure:
- **Page 1: Ticket Prediction**: Input raw ticket text, select model, outputs category, confidence score, and text preprocessing steps.
- **Page 2: Model Comparison Dashboard**: Displays F1-score, accuracy, inference speed, parameters, and memory usage.
- **Page 3: Text Explainability Dashboard**: Attributions highlighting key words that drove the predictions.
- **Page 4: Business Dashboard**: Volume tracker, auto-routing savings calculator, and SLA breach estimations.

---

## ⚙️ Optimization & Serving Configurations

### CPU Thread Constraints
To restrict PyTorch from consuming all CPU cores on shared deployment servers:
```python
import torch
# Restrict to single-threaded CPU execution for deterministic benchmarking
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

### Serialization & Checkpoint Loading
When deploying to production, make sure to load the models in evaluation mode:
```python
model = LSTMClassifier(vocab_size=80, embedding_dim=50, hidden_dim=64, output_dim=10)
checkpoint = torch.load("outputs/lstm_best.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # CRITICAL: Disables dropout layers for stable inference
```
