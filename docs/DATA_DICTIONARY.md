# Data Dictionary

This document details the schema, data types, business meaning, and preprocessing decisions for the Customer Support Ticket Intelligence Platform.

---

## 📋 Schema Definition

The raw database contains 1,282,355 consumer complaints. Below is the metadata for all 18 columns in `consumer-complaint-database.csv`.

| Column | Data Type | Description / Business Meaning |
| :--- | :--- | :--- |
| `Date received` | Date/String | The date the complaint was received by the CFPB. |
| `Product` | Categorical | **Primary Target (mapped to `category`)**. The financial service or product category (e.g., Mortgage, Credit card). |
| `Sub-product` | Categorical | Specific sub-type of the product (e.g., Conventional home mortgage). |
| `Issue` | Categorical | High-level type of complaint issue (e.g., Troubles during payment process). |
| `Sub-issue` | Categorical | Detailed sub-type of the complaint issue. |
| `Consumer complaint narrative` | Text | **Primary Input (mapped to `issue_description`)**. The free-text complaint description written by the consumer. |
| `Company public response` | Categorical/Text | The company's public-facing response statement. |
| `Company` | Categorical | Name of the financial company target of the complaint. |
| `State` | Categorical | US State where the consumer resides. |
| `ZIP code` | Categorical | ZIP code where the consumer resides. |
| `Tags` | Categorical | Special tags (e.g., Servicemember, Older American). |
| `Consumer consent provided?` | Categorical | Flag indicating if the consumer consented to publishing their narrative. |
| `Submitted via` | Categorical | Submission channel (e.g., Web, Referral, Phone). |
| `Date sent to company` | Date/String | Date when the complaint was forwarded to the company. |
| `Company response to consumer` | Categorical | The outcome/response category from the company to the consumer. |
| `Timely response?` | Categorical | Flag indicating if the company responded within the SLA time (Yes/No). |
| `Consumer disputed?` | Categorical | Flag indicating if the consumer disputed the resolution (Yes/No). |
| `Complaint ID` | Integer | Unique identifier for each complaint record. |

---

## 🎯 Target Definitions

1. **Modeling Input: `issue_description`**
   - Free-text consumer complaint narrative. All other metadata columns are dropped during ingestion to avoid overfitting, geographic/company bias, and data leakage.
2. **Modeling Target: `category`**
   - 18 distinct financial service categories mapped from the raw `Product` column.

---

## ⚠️ Critical Analysis Findings (Class Imbalance)

> [!WARNING]
> Unlike synthetic datasets, the real-world CFPB dataset is **highly imbalanced**.
> - The top two classes (`Debt collection` and `Credit reporting...`) account for **~45%** of the entire dataset.
> - The bottom two classes (`Other financial service` and `Virtual currency`) account for **<0.1%** of the dataset combined.
> - This class imbalance will require mitigation strategies in model building (e.g., class weights in CrossEntropyLoss, focal loss, or stratified batching) to ensure that the model doesn't overfit to major classes.

---

## ⚙️ Text Preprocessing Decisions

1. **Cleaning**: Convert all text to lowercase, remove punctuation (except potentially specific symbols if using tokenizers), and split into word tokens.
2. **Sequence Length**: 
   - Words per narrative range from 1 to 6,314, with a median of 141.
   - We will select a max sequence length of **128 or 256** (with padding/truncation) for recurrent models to balance context retention and compute efficiency.
3. **Padding & Truncation**: Use `post-padding` and `post-truncation` for sequence alignment.
4. **Vocabulary size**: ~79,605 unique words. We will restrict the active vocabulary by excluding rare words (e.g., keeping only words that appear $\ge 5$ times) to reduce model parameter count and prevent learning from noise.
