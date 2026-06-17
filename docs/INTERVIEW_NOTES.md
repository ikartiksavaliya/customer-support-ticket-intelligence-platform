# Deep Learning & NLP Interview Notes

This document contains standard interview questions, mathematical derivations, intuitions, and common engineering mistakes relating to sequence modeling, recurrent neural networks, and embeddings.

---

## 💻 Technical Interview Questions & Answers

### Q1: Why does Bag of Words (BoW) or TF-IDF lose text information?
- **Intuition**: BoW/TF-IDF represents documents as frequency vectors, completely ignoring word order.
- **Why it matters**: In natural language, changing word order completely changes meaning. For example:
  - *"The payment was deducted but the transaction failed."* (Negative - Payment issues)
  - *"The transaction failed but the payment was refunded."* (Neutral/Informative - Resolved)
  - Under a Bag of Words representation, these two sentences map to identical frequency vectors because they share the exact same word counts.
- **Interview Answer**: "Bag of Words representations assume the bag-of-words assumption, which models a document as a multiset of words. In doing so, it acts as a bag where word positions are shuffled, losing grammatical structures, negations, and temporal dependencies. Sequence models address this by processing tokens step-by-step, maintaining a hidden state that carries historical context."

---

### Q2: Derive the recurrent transition equations and explain the computational complexity of a vanilla RNN.
- **Transition Equations**:
  For a sequence input $x_t \in \mathbb{R}^d$ and hidden state $h_t \in \mathbb{R}^h$:
  
  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
  
  $$y_t = \text{softmax}(W_{hy} h_t + b_y)$$
  
  Where:
  - $W_{xh} \in \mathbb{R}^{h \times d}$ is the input-to-hidden weight matrix.
  - $W_{hh} \in \mathbb{R}^{h \times h}$ is the recurrent hidden-to-hidden weight matrix.
  - $W_{hy} \in \mathbb{R}^{c \times h}$ is the hidden-to-output matrix (for $c$ output classes).
- **Computational Complexity**:
  - For a sequence of length $T$ and batch size $B$, the computation at each step $t$ is dominated by the matrix multiplications:
    - $W_{hh} h_{t-1}$ takes $O(h^2)$ operations.
    - $W_{xh} x_t$ takes $O(h \cdot d)$ operations.
  - Total complexity per step is $O(h^2 + h \cdot d)$.
  - For the entire sequence of length $T$ across batch $B$, the complexity is $O(B \cdot T \cdot (h^2 + h \cdot d))$.
- **Interview Answer**: "A vanilla RNN computes its hidden state recursively by taking the linear combination of the current input and the previous hidden state, followed by a non-linear activation (typically $\tanh$). Its step complexity scales quadratically with the hidden state size $h$ due to the hidden-to-hidden weight matrix transformation."

---

### Q3: Explain why Backpropagation Through Time (BPTT) leads to vanishing or exploding gradients.
- **Mathematical Derivation**:
  Let the loss at step $t$ be $L_t$. The gradient of $L_t$ with respect to the recurrent weights $W_{hh}$ is:
  
  $$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$
  
  The Jacobian of the hidden state at step $t$ with respect to step $k$ is given by the chain rule:
  
  $$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$$
  
  Each term in the product is:
  
  $$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - \tanh^2(\cdot)) W_{hh}^T$$
  
  - If the largest eigenvalue (spectral radius) of $W_{hh}$ is greater than 1, and the activation function doesn't saturate, the product will grow exponentially as $(t - k)$ increases (exploding gradients).
  - If the largest eigenvalue of $W_{hh}$ is less than 1, or the activation function saturates (derivative of $\tanh$ approaches 0), the product will shrink exponentially (vanishing gradients).
- **Interview Answer**: "BPTT unrolls the recurrent network in time, creating a deep computational graph. The gradient is propagated backward through time by multiplying the recurrent Jacobian matrices repeatedly. If the eigenvalues of the weight matrix are far from 1, this repeated multiplication leads to exponential growth or decay of gradients, making it impossible for the model to learn relationships spanning long sequences."

---

### Q4: How do LSTMs solve the vanishing gradient problem?
- **Intuition**: The cell state $C_t$ acts as a 'gradient highway'.
- **Mathematical Formulation**:
  The cell state update equation is additive:
  
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
  
  The derivative of the cell state at step $t$ with respect to step $t-1$ is:
  
  $$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \dots$$
  
  - If the forget gate $f_t$ is close to 1, the gradient can flow backward through time indefinitely without being multiplied by the recurrent weights $W_{hh}$.
  - This additive relationship allows gradients to flow directly without exponential decay.
- **Interview Answer**: "LSTMs introduce a separate cell state $C_t$ that undergoes only linear, element-wise updates governed by an additive forget gate. This creates a path where the derivative $\partial C_t / \partial C_{t-1}$ contains an additive term $f_t$. When $f_t \approx 1$, the gradient flows backward without being subjected to repeated matrix multiplication, effectively mitigating the vanishing gradient problem."

---

## 🚫 Common Mistakes in Sequence Modeling

1. **Incorrect Padding Location**:
   - *Mistake*: Using post-padding in vanilla RNNs without masking or extracting the hidden state from the true (non-padded) sequence end.
   - *Impact*: The model processes multiple trailing zero tokens, saturating the hidden state with zeros and washing out the actual text representation.
   - *Correction*: Use pre-padding for standard forward RNNs, or extract the hidden state using the actual length of each sequence (`torch.nn.utils.rnn.pack_padded_sequence`).

2. **Forgetting to Reset Hidden States between Batches**:
   - *Mistake*: Passing the hidden state of one batch to the next when the batches contain independent sequences.
   - *Impact*: The model attempts to propagate temporal dependencies across unrelated batches, destabilizing convergence.
   - *Correction*: Always initialize the hidden state to zero vectors at the start of each batch unless explicitly training stateful RNNs on continuous streams.

3. **Ignoring Exploding Gradients**:
   - *Mistake*: Training deep sequence models without gradient clipping.
   - *Impact*: Training suddenly diverges or outputs `NaN` values due to gradient explosion.
   - *Correction*: Always apply gradient norm clipping in PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
