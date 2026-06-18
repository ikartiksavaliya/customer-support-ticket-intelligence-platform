# Error Analysis Report

This document outlines our diagnostic framework for auditing model misclassifications, tracking confusion patterns, and understanding error distributions in the CFPB Consumer Complaint classification model.

---

## 🔍 Diagnostic Framework

When the model makes an incorrect prediction, we classify the error into one of the following categories to guide our iterative improvement process:

1. **Semantic Ambiguity & Class Overlap**:
   - The ticket text contains words associated with multiple similar categories.
   - *Example*: A complaint about an unauthorized credit card charge might contain text highly relevant to both `Credit card` and `Debt collection` (if sent to collections).
2. **Extreme Class Imbalance Bias**:
   - The model predicts the majority class (`Debt collection` or `Credit reporting...`) because of training distribution skew, ignoring subtle minority class signals.
3. **Out-of-Vocabulary (OOV) / Sub-word Shift**:
   - The narrative contains key financial terms, specific company names, or typos that were mapped to `<UNK>` during vocabulary construction, causing the sequence model to lose critical semantic context.
4. **Sequence Truncation**:
   - Critical details explaining the complaint were located beyond our max sequence length threshold (e.g., in the middle/end of a 1,000-word narrative) and were truncated.
5. **Noisy Text & Masking Patterns**:
   - An excess of anonymized tokens (`XXXX`, `XX/XX/XXXX`) makes it difficult for recurrent layers to capture coherent syntactic structure and context.

---

## 📊 Error Audit Template

We will extract misclassified complaints and log them in the following tabular format in `outputs/misclassified_tickets.csv` during evaluation phases:

| Ticket ID | Text Input (Truncated) | Ground Truth Product | Predicted Product | Confidence Score | Error Category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | |

---

## 📈 Diagnostic & Improvement Plan

Based on initial data profiling and class distribution skew:
- **Confusion Matrix Auditing**: We will monitor confusion matrices specifically to identify pairwise confusion (e.g., checking if `Credit reporting` is constantly misclassified as the broader `Credit reporting, credit repair services...` class).
- **OOV Analysis**: During inference, we will track the ratio of OOV tokens in misclassified versus correctly classified tickets to determine if we need to raise our vocabulary threshold or switch to subword tokenization (BPE/WordPiece).
- **Dynamic Padding/Truncation Study**: We will experiment with truncation locations (pre-truncation vs post-truncation) to see if consumers write their most descriptive product cues at the very beginning or the end of their complaints.
