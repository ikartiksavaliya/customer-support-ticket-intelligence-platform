# Error Analysis Report

This document outlines our diagnostic framework for auditing model misclassifications, tracking confusion patterns, and understanding error distributions.

---

## 🔍 Diagnostic Framework

When the model makes an incorrect prediction, we classify the error into one of the following categories:

1. **Semantic Ambiguity**: The ticket text contains words associated with multiple categories. (e.g., *"crashes"* could mean `Bug Report` or `Performance Issue`).
2. **Label Noise (Dataset Issue)**: The ground-truth label was assigned randomly, or does not match the actual semantic content of the text.
3. **Out-of-Vocabulary (OOV) Shift**: The text contains key terms that were not present in the model's vocabulary.
4. **Sequence Length Saturation**: Truncation cut off critical words, or padding dominated the sequence state.

---

## 📊 Error Audit Template

We will extract misclassified tickets and log them in the following tabular format in `/outputs/misclassified_audit.csv`:

| Ticket ID | Text Input | Ground Truth | Predicted | Confidence | Error Category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | |

---

## ⚠️ Key Insights from Initial Data Profiling

Because our exploratory analysis proved that **the dataset's target classes are randomly assigned and independent of the text content**:
- **Uniform Error Distribution**: We expect misclassifications to be distributed uniformly across all categories (no single class will have significantly better precision or recall on validation data).
- **Confusion Matrix Baseline**: The off-diagonal entries of our confusion matrix will show uniform numbers (~10% across all cells).
- **Resolution Plan**: This represents a classic "label noise" limit. In a production setting, we would need to relabel the dataset using domain experts or weak supervision. For this pedagogical study, we will use this framework to demonstrate how to audit predictions, calculate confidence scores, and identify OOV words.
