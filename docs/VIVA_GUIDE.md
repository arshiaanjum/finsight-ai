# FinSight AI — Simple Viva & Panel Presentation Guide (For All 4 Members)

Read this before your college presentation or viva. Every member has a clear part to explain in simple English.

---

## 1. Two-Line Project Pitch (Start With This!)
> "Normal expense apps only look at words like 'Swiggy' or 'Uber', so they fail on Indian UPI payments like `'Paid ₹12,000 to rahul@okaxis'`. Our system **FinSight AI** combines **NLP text + spending behavior (amount, time, date, repeat pattern) + user feedback learning**, plus **bill photo OCR, unusual spend alerts, and next-month budget forecasting**."

---

## 2. What Each Team Member Should Say

### Member 1 — Data Pipeline, Privacy & Receipt OCR (`ml/data_and_ocr.py`)
- **My role:** I built the input pipeline for messy real-world data.
- **3 input types:** Manual text, raw Bank/UPI SMS, and Bill Photos (using OCR).
- **Privacy feature:** Before saving any bank SMS, `mask_sensitive_numbers()` hides 10–16 digit account/card numbers (e.g., `123456789012` becomes `XXXXXX9012`).
- **Dataset:** Created 6,000 transactions across 8 categories, including 28% hard P2P UPI transactions (`paid to rahul.s@okaxis`) where text alone gives no clue.

### Member 2 — Core Hybrid AI & Self-Learning (`ml/category_model.py`)
- **My role:** I built the main category prediction AI and ran the research experiments.
- **Why Text-Only AI fails:** On ambiguous UPI messages, Naive Bayes got only **21.53%** and Logistic Regression got **19.83%**.
- **My Hybrid Model:** I combined **TF-IDF text features** with **8 behavioral context features** (`log_amount`, `is_micro_spend`, `is_large_spend`, `hour_sin`, `hour_cos`, `is_early_month`, `is_weekend`, `vpa_repeat_count`) using a Soft-Voting Ensemble (`LogisticRegression` + `HistGradientBoostingClassifier`).
- **Result:** Ambiguous UPI accuracy jumped from **~20% to 92.92%**, and overall accuracy reached **96.42%**.
- **Active Learning:** When the user fixes one wrong UPI prediction, the AI saves that VPA in memory and reaches **100% accuracy on learned personal VPAs by Week 4**.

### Member 3 — Anomaly Detection & Budget Forecasting (`ml/anomaly_and_forecast.py`)
- **My role:** I built the post-categorization intelligence: **alerts** and **future prediction**.
- **Anomaly Detection:** Uses **Isolation Forest** + **per-category median comparison** + **30-minute duplicate check** to catch sudden high bills or accidental double payments.
- **Next-Month Forecasting:** Groups past monthly spending by category and uses **Ridge Regression + 3-month rolling average** to predict next month's budget and warn if a category is going `Over Budget`.

### Member 4 — Full-Stack App & Live System (`backend/main.py`, `backend/db.py`, `frontend/index.html`)
- **My role:** I built the SQLite database, Starlette + Uvicorn REST APIs, and the 4-tab web dashboard.
- **Live Integration:** Connects all 3 AI modules into one web app so users can scan bills, parse SMS, view forecasts, fix wrong categories, and inspect research charts in real time.

---

## 3. Common Panel Questions & Simple Answers

1. **Q: Why didn't you just use ChatGPT / Gemini API for everything?**
   - **A:** Sending private bank SMS to external cloud APIs hurts privacy and costs money per transaction. Our local Hybrid ML model runs offline in milliseconds, keeps user financial data private, and benchmarks cleanly for a research paper.
2. **Q: How is your model better than a simple keyword rule (`if "swiggy" in text`)?**
   - **A:** Keyword rules completely fail on P2P UPI IDs like `suresh.k@okaxis`. Our model uses amount, day of month, time, and repeat frequency—and learns from user corrections.
3. **Q: Why did you use `hour_sin` and `hour_cos` instead of raw `hour`?**
   - **A:** Because 23:00 (11 PM) and 00:00 (midnight) are 1 hour apart, not 23 hours apart! Sine and cosine encode time as a circle so the model understands late-night continuity.
4. **Q: What happens if a user changes their habit or corrects a category?**
   - **A:** The `learned_rules` table updates immediately via Active Learning, and future predictions for that VPA update without retraining the whole model from scratch.
