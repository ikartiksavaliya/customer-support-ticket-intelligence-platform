# Business Impact Analysis

This document translates deep learning evaluation metrics into concrete business metrics and ROI indicators to align our technical system with customer support operational goals.

---

## 📈 Metric Mapping: Technical to Business

| Technical Metric | Target Value | Support Operations Equivalent | Business Impact |
| :--- | :--- | :--- | :--- |
| **Category F1-Score** | Maximize | Routing Accuracy | Fewer misrouted tickets, reducing customer bounce rate between internal support groups. |
| **Inference Latency** | $< 100\text{ ms}$ (CPU) | Auto-Triaging Speed | Immediate sorting of tickets upon submission, eliminating human triaging backlog. |
| **Confidence Score** | Threshold $\tau$ (e.g., $85\%$) | Auto-Routing Rate | High-confidence tickets are routed automatically; low-confidence tickets go to human review. |
| **Priority Precision** | Maximize | SLA Protection | Ensures "Urgent" tickets are correctly routed to escalation teams, avoiding SLA breach penalties. |

---

## 💰 ROI Calculation Framework

We assume the following operational baseline:
* **Weekly Ticket Volume**: 10,000 support tickets.
* **Manual Triaging Time**: 2 minutes (120 seconds) per ticket by a human agent.
* **Agent Fully Burdened Cost**: \$24 / hour (\$0.40 / minute).
* **Weekly Manual Triaging Cost**: $10,000 \times 2 \times \$0.40 = \$8,000 / \text{week}$.

### 1. Automation Rate and Time Savings
By deploying our sequence classifier with a confidence threshold $\tau = 80\%$, we can automate routing for all tickets where prediction confidence exceeds $\tau$.

$$\text{Weekly Time Saved (Hours)} = \frac{\text{Volume} \times \text{Automation Rate} \times 2\text{ minutes}}{60}$$

If the automation rate is **60%**:
- **Time Saved**: $6,000 \text{ tickets} \times 2\text{ min} = 12,000\text{ minutes} = 200\text{ hours/week}$.
- **Cost Savings**: $200\text{ hours} \times \$24 = \$4,800/\text{week}$ (\$249,600 annual savings).

### 2. Time-to-Route Reduction
- **Manual Routing Latency**: Avg. 4 to 24 hours (tickets sit in a queue waiting to be triaged by a lead agent).
- **Automated Routing Latency**: $< 100\text{ milliseconds}$.
- **Result**: Instant routing leads to a **99.9% reduction in triaging queue delays**, directly accelerating first response times.

---

## 🚦 Business Dashboards (Streamlit Implementation)
Our deployed application includes a **Business Impact Dashboard** displaying:
- **Estimated Automation Rate**: Visualizing cumulative confidence distribution.
- **Estimated Hours Saved**: Dynamic slider adjusting weekly ticket volume and showing labor hour reductions.
- **Estimated Financial Savings**: ROI counter updating in real time based on model selection and threshold tuning.
