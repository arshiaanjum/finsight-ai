# Context-Aware Expense Categorization and Spending Forecasting Using Hybrid NLP and Behavioral Features

**Course:** BTech CSE (AI & ML) — Final Year Research Paper Draft  
**System Name:** FinSight AI

---

## Abstract
Automated personal expense categorization is a core component of modern financial management applications. However, existing rule-based and text-only Natural Language Processing (NLP) classifiers struggle significantly on peer-to-peer (P2P) Unified Payments Interface (UPI) transactions (e.g., *"Paid Rs. 12,000 to rahul.s@okaxis"*), where merchant keywords are absent. In this paper, we present **FinSight AI**, a context-aware, full-stack expense intelligence framework that combines TF-IDF text representations with behavioral transaction context—specifically logarithmic spend amount, cyclical time-of-day encoding, early-month billing indicators, weekend flags, and payee recurrence frequency—within a soft-voting ensemble (`Logistic Regression` + `HistGradientBoostingClassifier`). Additionally, FinSight AI incorporates an Active Learning personalization store for user feedback, Isolation Forest-based anomaly detection, and Ridge Regression monthly budget forecasting. Evaluated on a 6,000-transaction multi-category dataset containing 28% ambiguous P2P UPI records (1,200 held-out test transactions, including 353 ambiguous UPI samples), text-only baselines achieve 73.42%–75.42% overall accuracy and only 14.73%–21.53% accuracy on ambiguous UPI transactions. Our proposed hybrid model achieves **96.42% overall accuracy (96.47% Macro F1)** and **92.92% accuracy on ambiguous UPI transactions**, while 4-week active-learning simulation increases personal ambiguous UPI accuracy from **76.53% (Week 1) to 100.00% (Week 4)**.

---

## 1. Introduction
Personal finance management relies heavily on accurate categorization of expenses across categories such as *Food & Dining*, *Groceries*, *Transport & Fuel*, *Rent & Utilities*, *Shopping*, *Entertainment & Subscriptions*, *Health & Medical*, and *Education & Misc*. While merchant-tagged card transactions (e.g., *Swiggy*, *Amazon*, *Apollo Pharmacy*) are straightforward to classify using text patterns, digital P2P ecosystems such as UPI frequently produce opaque strings containing only a personal Virtual Payment Address (VPA) or numeric reference ID.

To address this gap, our contributions are fourfold:
1. **Privacy-Preserving Ingestion & Receipt OCR:** Automatic masking of sensitive account numbers alongside SMS parsing and optical character recognition (OCR) for physical bills.
2. **Hybrid NLP + Behavioral Context Ensemble:** Fusing sparse/SVD text representations with cyclical temporal and monetary context features.
3. **Human-in-the-Loop Active Learning:** Lightweight persistent personalization of ambiguous VPAs from minimal user corrections.
4. **Post-Categorization Intelligence:** Isolation Forest anomaly alerting and Ridge Regression category-level budget forecasting integrated into a full-stack web platform.

---

## 2. System Methodology

### 2.1 Feature Engineering
Each transaction $x_i$ is represented by two complementary views:
- **Text View:** Normalized transaction description and payee string transformed via `TfidfVectorizer` (1–2 n-grams, sublinear TF scaling), paired with `TruncatedSVD` (40 components) for the tree branch.
- **Behavioral Context View (8 features):**
  1. $\log(1 + \text{amount})$
  2. Micro-spend indicator ($\text{amount} \le 150$)
  3. Large-spend indicator ($\text{amount} \ge 5000$)
  4. Cyclical hour encoding: $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$
  5. Early-month billing flag ($\text{day\_of\_month} \le 7$)
  6. Weekend indicator
  7. Log payee recurrence count: $\log(1 + \text{vpa\_repeat\_count})$

### 2.2 Soft-Voting Hybrid Architecture
We combine two complementary learners in a soft-voting ensemble:
- **Branch A:** `LogisticRegression` ($C=2.5$) on sparse TF-IDF + standardized behavioral features.
- **Branch B:** `HistGradientBoostingClassifier` (`max_iter=220`, `max_depth=7`, `learning_rate=0.08`) on SVD-compressed text + raw behavioral context features.

### 2.3 Anomaly Detection & Forecasting
- **Anomaly Detection:** Combines `IsolationForest` (`contamination=0.05`) on log-amount and category encoding with per-category median-multiplier checks and 30-minute duplicate detection.
- **Budget Forecasting:** Aggregates monthly category spend and applies `Ridge` regression ($\alpha=2.0$) blended with a 3-month rolling average (55% trend / 45% rolling mean).

---

## 3. Experimental Setup & Results
We evaluate all models on a stratified 80/20 split of 6,000 transactions (4,800 train / 1,200 test, containing 353 ambiguous UPI test cases).

### Table 1: Model Benchmark Comparison
| Model | Overall Accuracy | Macro F1-Score | Ambiguous UPI Accuracy |
| :--- | :---: | :---: | :---: |
| Naive Bayes (Text Only) | 75.42% | 78.99% | 21.53% |
| Logistic Regression (Text Only) | 74.92% | 77.72% | 19.83% |
| Random Forest (Text Only) | 73.42% | 75.11% | 14.73% |
| **FinSight Hybrid (Text + Context)** | **96.42%** | **96.47%** | **92.92%** |

*Refer to `artifacts/plots/model_comparison.png` and `artifacts/plots/confusion_matrix.png`.*

### Table 2: Per-Category F1-Score (FinSight Hybrid)
| Category | F1-Score |
| :--- | :---: |
| Groceries | 98.59% |
| Transport & Fuel | 98.15% |
| Education & Misc | 96.97% |
| Shopping | 96.50% |
| Entertainment & Subscriptions | 96.00% |
| Rent & Utilities | 95.95% |
| Food & Dining | 95.72% |
| Health & Medical | 93.84% |

### Table 3: 4-Week Active Learning Personalization Simulation
| Evaluation Window | Overall Accuracy | Ambiguous UPI Accuracy | Learned Personal VPAs |
| :--- | :---: | :---: | :---: |
| Week 1 | 90.67% | 76.53% | 0 |
| Week 2 | 92.67% | 84.78% | 3 |
| Week 3 | 97.33% | 93.83% | 6 |
| Week 4 | 99.33% | 100.00% | 8 |

*Refer to `artifacts/plots/personalization_curve.png`.*

---

## 4. Conclusion & Future Work
Our experiments confirm that text-only classifiers are insufficient for modern P2P UPI ecosystems, scoring below 22% on ambiguous UPI descriptions. Integrating behavioral spend context lifts ambiguous UPI classification by over 71 percentage points to **92.92%**, and lightweight Active Learning personalization resolves recurring user-specific VPAs to **100.00%** within four weeks. Future work includes evaluating transformer embeddings (e.g., DeBERTa) and federated on-device model updates across real multi-user banking logs.
