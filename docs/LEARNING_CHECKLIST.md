# Sequence Modeling Learning Checklist

A topic is only marked as **Complete [x]** when it has been fully **Explained** (in docs/notebooks), **Implemented** (in code/notebooks), **Evaluated** (experimentally), and **Documented** (in final logs).

---

## 🗺️ Topic Checklist

### 1. Sequential Data Modeling
- [ ] **Sequential Data Basics** (What is sequential data, why order matters, temporal/context dependencies)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Bag of Words vs Sequence** (Why BoW/TF-IDF loses temporal structure and context)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 2. Text Representation
- [ ] **Tokenization & Vocabulary** (Text cleaning, splitting tokens, building vocab, integer mapping)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Sequence Length Handling** (Padding, truncation, padding tokens `[PAD]`, unknown tokens `[UNK]`)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 3. Embeddings
- [ ] **One-Hot vs Dense Embeddings** (Sparse high-dimensional vs dense low-dimensional representation)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **PyTorch Embedding Layers** (Embedding parameters, lookup table mechanism, learned representations)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Semantic Similarity & Dimensions** (Cosine similarity, embedding dimension tradeoffs 50-300)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 4. Recurrent Neural Network (RNN) Foundations
- [ ] **RNN Hidden State & Computations** (Recurrent connections, hidden state transition equation, cell unrolling)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Information Propagation** (How context passes from $t=0$ to $t=T$)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 5. Backpropagation Through Time (BPTT)
- [ ] **Computational Graph View** (Unfolding RNN across time, shared weights gradients accumulation)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Gradient Flow & Chain Rule** (BPTT mathematics, matrix multiplications over time steps)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 6. Optimization Problems
- [ ] **Vanishing & Exploding Gradients** (Why deep time unrolling causes product of eigenvalues to vanish/explode)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Long-Term Dependency Limits** (Why vanilla RNNs fail to remember information beyond ~10-20 steps)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 7. Long Short-Term Memory (LSTM)
- [ ] **LSTM Cell State** (Additive gradient highway $C_t$ preventing vanishing gradients)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Gating Mechanisms** (Forget gate $f_t$, Input gate $i_t$, Output gate $o_t$ math and intuition)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 8. Gated Recurrent Unit (GRU)
- [ ] **GRU Architecture** (Reset gate $r_t$, Update gate $z_t$, merge of cell state and hidden state)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **LSTM vs GRU Tradeoffs** (Parameter efficiency, speed, representational capacity)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 9. Advanced Sequence Models
- [ ] **Bidirectional RNNs** (Forward and backward pass hidden states, concatenated representations)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Deep Stacked RNNs** (Hierarchical feature learning, stacking recurrent layers, complexity limits)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 10. Regularization
- [ ] **Dropout in Recurrent Models** (Standard dropout vs Variational/Recurrent Dropout on hidden connections)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Weight Decay & Early Stopping** (L2 penalty on weights, training curves monitoring to prevent overfitting)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 11. Optimization
- [ ] **Optimizer Suites** (SGD, RMSProp, Adam, AdamW mechanisms and parameter updates)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Learning Rate Scheduling** (StepLR, ReduceLROnPlateau, Cosine Annealing)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`

### 12. Deep Learning Theory
- [ ] **Bias-Variance & Model Capacity** (Overfitting vs underfitting in sequence classification)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
- [ ] **Computational Complexity** (Sequence length effects $O(T)$ vs batch size $O(B)$ vs dimensions $O(D^2)$)
  - *Status*: `[ ] Explained | [ ] Implemented | [ ] Evaluated | [ ] Documented`
