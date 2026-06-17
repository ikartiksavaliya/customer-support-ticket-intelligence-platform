# Dataset Profile

This profile summarizes the statistics, sequence characteristics, and quality observations of the customer support ticket dataset.

---

## 📊 Dataset Statistics

* **Total Samples**: 200,000
* **Unique Categories (Classes)**: 10
* **Unique Priorities (Classes)**: 4

### Class Distributions

#### 1. Ticket Category
The target classes are uniformly distributed, meaning there is **no class imbalance**:

| Category | Counts | Percentage |
| :--- | :--- | :--- |
| Feature Request | 20,169 | 10.08% |
| Subscription Cancellation | 20,096 | 10.05% |
| Performance Issue | 20,074 | 10.04% |
| Security Concern | 20,040 | 10.02% |
| Login Issue | 20,002 | 10.00% |
| Payment Problem | 19,997 | 10.00% |
| Bug Report | 19,981 | 9.99% |
| Refund Request | 19,900 | 9.95% |
| Data Sync Issue | 19,877 | 9.94% |
| Account Suspension | 19,864 | 9.93% |

#### 2. Ticket Priority
The priority attribute is also uniformly split:

| Priority | Counts | Percentage |
| :--- | :--- | :--- |
| High | 50,241 | 25.12% |
| Urgent | 50,143 | 25.07% |
| Medium | 49,854 | 24.93% |
| Low | 49,762 | 24.88% |

---

## 🔠 Sequence Length Statistics (`issue_description`)

The dataset text has highly regular, narrow word count ranges.

| Metric | Characters | Words (Whitespace split) |
| :--- | :--- | :--- |
| **Average** | 65.92 | 11.01 |
| **Median** | 68.00 | 11.00 |
| **Minimum** | 55.00 | 9.00 |
| **Maximum** | 79.00 | 13.00 |

---

## 🔤 Vocabulary Size

* **Estimated Unique Vocabulary Size**: **80 words** (lowercased, punctuation removed).
* This is extremely compact and represents a highly simplified, synthetic vocabulary space.

---

## 📝 Text Quality Observations & Representative Examples

The text values in `issue_description` consist of **exactly 10 unique template sentences**. There are no typos, grammatical variations, slang, or emojis, which makes it a very clean, structured, yet synthetic text column.

### The 10 Unique Sentences
1. `"The payment was deducted from my bank account but the transaction shows failed."`
2. `"I found a bug in the latest update affecting report generation."`
3. `"The application crashes whenever I try to upload a file."`
4. `"My subscription was cancelled without my request and I need clarification."`
5. `"The system is not syncing data across devices properly."`
6. `"There seems to be a discrepancy in my billing statement for this month."`
7. `"I would like to request a refund for the recent charge."`
8. `"Two-factor authentication codes are not being delivered to my phone."`
9. `"I am experiencing very slow performance while using the dashboard."`
10. `"I am unable to access my account after entering the correct credentials."`

### Preprocessing Implications
- Since the text has zero variance outside these 10 sentences, tokenization will produce a closed set of sequences.
- We will use post-padding with sequence length `15` to capture every word without truncation.
