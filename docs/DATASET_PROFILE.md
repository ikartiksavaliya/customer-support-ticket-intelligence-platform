# Dataset Profile

This profile summarizes the statistics, sequence characteristics, and quality observations of the active customer support ticket modeling dataset (`df_sample`).

---

## 📊 Dataset Statistics

* **Total Samples**: 200,000 (after exact deduplication and stratified sampling)
* **Unique Categories (Classes)**: 18
* **Raw Columns Retained**: `issue_description` (mapped from *Consumer complaint narrative*), `category` (mapped from *Product*)

### Class Distributions
Unlike the previous synthetic dataset, the real-world CFPB dataset is **highly imbalanced**, spanning 18 financial service categories:

| Category | Counts | Percentage |
| :--- | :--- | :--- |
| Debt collection | 45,934 | 22.97% |
| Credit reporting, credit repair services, or other personal consumer reports | 43,836 | 21.92% |
| Mortgage | 28,843 | 14.42% |
| Credit reporting | 16,249 | 8.12% |
| Student loan | 11,866 | 5.93% |
| Credit card or prepaid card | 11,599 | 5.80% |
| Credit card | 10,218 | 5.11% |
| Bank account or service | 8,093 | 4.05% |
| Checking or savings account | 7,005 | 3.50% |
| Consumer Loan | 5,144 | 2.57% |
| Vehicle loan or lease | 3,116 | 1.56% |
| Money transfer, virtual currency, or money service | 2,973 | 1.49% |
| Payday loan, title loan, or personal loan | 2,405 | 1.20% |
| Payday loan | 948 | 0.47% |
| Money transfers | 815 | 0.41% |
| Prepaid card | 789 | 0.39% |
| Other financial service | 159 | 0.08% |
| Virtual currency | 8 | 0.00% |

---

## 🔠 Sequence Length Statistics (`issue_description`)

The text complaints show high variance in sequence lengths, requiring careful padding/truncation strategies.

| Metric | Characters | Words (Whitespace split) |
| :--- | :--- | :--- |
| **Average** | 1,115.21 | 202.92 |
| **Median (50%)** | 774.00 | 141.00 |
| **Minimum** | 5.00 | 1.00 |
| **25th Percentile** | 412.00 | 75.00 |
| **75th Percentile** | 1,405.00 | 256.00 |
| **Maximum** | 31,735.00 | 6,314.00 |

---

## 🔤 Vocabulary Size

* **Estimated Unique Vocabulary Size**: **79,605 words** (lowercased, whitespace split, basic punctuation removed).
* This is a massive, real-world vocabulary containing spelling errors, specific terminology, abbreviations, and anonymized placeholders (e.g., `XXXX`, `XX`).

---

## 📝 Text Quality Observations & Representative Examples

The text values in `issue_description` consist of real, unedited consumer complaints submitted to the CFPB. They present several challenges for deep learning model training:

1. **Anonymization / Redaction Masking**: Important names, dates, amounts, and account numbers are replaced with strings like `XXXX` or `XX/XX/XXXX`. These tokens will appear frequently in the vocabulary.
2. **High Imbalance and Semantic Overlap**: Categories like `Credit reporting` and `Credit reporting, credit repair services, or other personal consumer reports` have substantial semantic overlap, making fine-grained distinctions difficult.
3. **Natural Spelling and Grammatical Errors**: Customers write informally, introducing typos and grammatical inconsistencies that increase vocabulary noise.

### Preprocessing Implications
- **Truncation**: With a maximum word length of 6,314 but a median of 141, using the maximum length as our sequence limit is computationally prohibitive. A max sequence length of **128 or 256** is recommended to cover the majority of the text while staying within memory budgets.
- **Out-Of-Vocabulary (OOV) Handling**: Given the large vocabulary (~80k words), many words in the validation/test sets will be unseen. We must use an explicit `<UNK>` token and might benefit from subword tokenizers or capping vocabulary by min-frequency (e.g., ignoring words appearing $< 5$ times).
