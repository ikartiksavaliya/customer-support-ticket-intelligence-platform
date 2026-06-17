# Data Dictionary

This document details the schema, data types, business meaning, and preprocessing decisions for the Customer Support Ticket Intelligence Platform.

---

## 📋 Schema Definition

The dataset contains 200,000 customer support tickets. Below is the metadata for all 30 columns.

| Column | Data Type | Missing % | Description / Business Meaning |
| :--- | :--- | :--- | :--- |
| `ticket_id` | Integer | 0.00% | Unique identifier for each support ticket. |
| `customer_name` | String | 0.00% | Full name of the customer. |
| `customer_email` | String | 0.00% | Contact email address of the customer. |
| `product` | Categorical | 0.00% | Product associated with the ticket (10 unique products). |
| `category` | Categorical | 0.00% | **Primary Target**. The classification category of the ticket. |
| `issue_description` | Text | 0.00% | **Primary Input**. Raw text description of the customer's problem. |
| `resolution_notes` | Text | 0.00% | Text notes explaining how the ticket was resolved. |
| `priority` | Categorical | 0.00% | **Secondary Target**. Priority level (Urgent, High, Medium, Low). |
| `status` | Categorical | 0.00% | Lifecycle status of the ticket (Open, Closed, Resolved, etc.). |
| `channel` | Categorical | 0.00% | Customer contact channel (Email, Chat, Social Media, etc.). |
| `region` | Categorical | 0.00% | Geographical region of the customer. |
| `customer_age` | Integer | 0.00% | Customer age in years. |
| `customer_gender` | Categorical | 0.00% | Customer gender (Male, Female, Other). |
| `subscription_type` | Categorical | 0.00% | Subscription tier (Free, Basic, Premium, Enterprise). |
| `customer_tenure_months`| Integer | 0.00% | Number of months the customer has been active. |
| `previous_tickets` | Integer | 0.00% | Count of past support tickets filed by this customer. |
| `customer_satisfaction` | Integer | 0.00% | Customer satisfaction rating (1 to 5 stars). |
| `first_response_time` | Float | 0.00% | Time elapsed (hours) before first agent response. |
| `resolution_time_hours` | Float | 0.00% | Total time elapsed (hours) to resolve the ticket. |
| `ticket_created_date` | Date/String | 0.00% | Date when the ticket was opened. |
| `ticket_resolved_date` | Date/String | 0.00% | Date when the ticket was marked resolved. |
| `escalated` | Categorical | 0.00% | Flag indicating if the ticket was escalated (Yes/No). |
| `sla_breached` | Categorical | 0.00% | Flag indicating if the resolution breached SLA (Yes/No). |
| `operating_system` | Categorical | 0.00% | Customer's operating system. |
| `browser` | Categorical | 20.01% | Customer's web browser (40,023 missing values). |
| `payment_method` | Categorical | 0.00% | Payment method on file. |
| `language` | Categorical | 0.00% | Ticket language preference. |
| `preferred_contact_time`| Categorical | 0.00% | Customer's preferred time of day for contact. |
| `issue_complexity` | Integer | 0.00% | Complexity score rated by agent (1 to 10 scale). |
| `customer_segment` | Categorical | 0.00% | Customer segment (Small Business, Corporate, Individual). |

---

## 🎯 Target Definitions

1. **Primary Target: `category`**
   - 10 distinct ticket classes: `Feature Request`, `Subscription Cancellation`, `Performance Issue`, `Security Concern`, `Login Issue`, `Payment Problem`, `Bug Report`, `Refund Request`, `Data Sync Issue`, `Account Suspension`.
2. **Secondary Target: `priority` (Multi-Task Extension)**
   - 4 distinct classes: `Low`, `Medium`, `High`, `Urgent`.

---

## ⚠️ Critical Analysis Findings (Statistical Independence)

> [!WARNING]
> Rigorous EDA (Chi-Squared Independence and ANOVA tests) reveals that **both target variables (`category` and `priority`) are statistically independent of all features, including the text input (`issue_description`)**. 
> - All Chi-Squared p-values comparing `category` to categorical fields are $> 0.05$.
> - All ANOVA p-values comparing `category` to numerical fields are $> 0.05$.
> - The dataset contains exactly **10 unique issue descriptions** and **10 categories**, distributed uniformly (each description appears ~2,000 times for each category).

### Pedagogical Implications for Deep Learning Study
- **Random Chance Baseline**: The theoretical baseline accuracy for predicting `category` is **10%** (1 in 10 uniform classes).
- **Overfitting Study**: This dataset offers a perfect playground to study **overfitting**. A high-capacity network (e.g., Deep BiLSTM) can easily memorize the mapping from text descriptions to targets on the training set, pushing training accuracy up, but its validation accuracy will remain hard-capped at **10%**.
- **Model Evaluation**: Rather than comparing models purely on validation accuracy, we will evaluate models based on **learning dynamics (training loss convergence, generalization gaps)** and **inference profiles (speed, CPU latency, memory footprint, parameters)**.

---

## ⚙️ Text Preprocessing Decisions

1. **Inputs**: We only use `issue_description` as text input. Metadata fields will not be fed to the sequence model.
2. **Cleaning**: Convert to lowercase, remove punctuation (except where punctuation can be a token like `?`), and split by whitespace.
3. **Sequence Length**: The maximum word count is 13, and the minimum is 9. We will use a sequence length of 15 (with padding) to capture all words.
4. **Padding & Truncation**: Use `post-padding` and `post-truncation` since sequences are short and highly uniform.
5. **Vocabulary size**: Since there are only 10 unique sentences in the dataset, the lowercased vocabulary size is exactly **80 unique words**.
