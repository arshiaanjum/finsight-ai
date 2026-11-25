"""
Module 3 (Member 3): Unusual Spend Anomaly Detection & Next-Month Budget Forecasting.

Includes:
- Anomaly Detection using Isolation Forest + Category Median Multiplier + Duplicate Check
- Next-Month Category Spending Forecaster using Ridge Regression with trend & lag features
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

from ml.data_and_ocr import CATEGORIES


DEFAULT_MONTHLY_BUDGETS: dict[str, float] = {
    "Food & Dining": 6500.0,
    "Groceries": 5500.0,
    "Transport & Fuel": 3500.0,
    "Rent & Utilities": 18000.0,
    "Shopping": 6000.0,
    "Entertainment & Subscriptions": 2500.0,
    "Health & Medical": 3000.0,
    "Education & Misc": 3500.0,
}


def check_single_transaction_anomaly(
    history_df: pd.DataFrame,
    amount: float,
    category: str,
    merchant_or_vpa: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Evaluate whether a single incoming transaction is an unusual spend or duplicate."""
    if history_df.empty:
        return {"is_anomaly": False, "severity": "normal", "reasons": []}

    reasons: list[str] = []
    severity = "normal"

    # 1. Per-category statistical spike check
    cat_rows = history_df[history_df["category"] == category]
    if len(cat_rows) >= 5:
        median_amt = float(cat_rows["amount"].median())
        p95_amt = float(cat_rows["amount"].quantile(0.95))
        if median_amt > 0 and amount > max(p95_amt * 1.25, median_amt * 3.0) and amount >= 500:
            ratio = round(amount / median_amt, 1)
            reasons.append(
                f"Spend spike: ₹{amount:,.0f} is {ratio}x higher than your usual {category} median (₹{median_amt:,.0f})"
            )
            severity = "high"

    # 2. Isolation Forest multivariate check on amount + hour + category
    if len(history_df) >= 25:
        cat_map = {c: idx for idx, c in enumerate(CATEGORIES)}
        feat_df = pd.DataFrame(
            {
                "log_amt": np.log1p(history_df["amount"].astype(float).clip(lower=0)),
                "cat_id": history_df["category"].map(lambda c: cat_map.get(c, 7)),
            }
        )
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(feat_df)
        probe = pd.DataFrame(
            [{"log_amt": np.log1p(max(0.0, amount)), "cat_id": cat_map.get(category, 7)}]
        )
        pred = int(iso.predict(probe)[0])
        if pred == -1 and amount >= 1000 and not reasons:
            reasons.append(
                f"Isolation Forest flagged ₹{amount:,.0f} as an unusual pattern for {category}"
            )
            severity = "medium"

    # 3. Duplicate transaction risk within 30 minutes
    vpa_norm = (merchant_or_vpa or "").strip().lower()
    if vpa_norm and timestamp:
        try:
            cur_dt = datetime.fromisoformat(timestamp)
            hist_ts = pd.to_datetime(history_df["timestamp"], errors="coerce")
            same_payee = history_df[
                history_df["merchant_or_vpa"].fillna("").str.lower() == vpa_norm
            ]
            for idx, r in same_payee.iterrows():
                prev_dt = hist_ts.loc[idx]
                if pd.notna(prev_dt):
                    diff_mins = abs((cur_dt - prev_dt.to_pydatetime()).total_seconds()) / 60.0
                    if diff_mins <= 30 and abs(float(r["amount"]) - amount) <= 1.0:
                        reasons.append(
                            f"Possible duplicate payment: ₹{amount:,.0f} to '{vpa_norm}' within {int(diff_mins) + 1} min"
                        )
                        severity = "high"
                        break
        except Exception:
            pass

    return {
        "is_anomaly": len(reasons) > 0,
        "severity": severity,
        "reasons": reasons,
    }


def detect_historical_anomalies(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    """Scan stored user transactions and return top unusual expenses with clear reasons."""
    if df.empty or len(df) < 10:
        return []

    work = df.copy()
    work["ts_parsed"] = pd.to_datetime(work["timestamp"], errors="coerce")
    alerts: list[dict[str, Any]] = []

    cat_medians = work.groupby("category")["amount"].median().to_dict()
    cat_p95 = work.groupby("category")["amount"].quantile(0.95).to_dict()

    cat_map = {c: idx for idx, c in enumerate(CATEGORIES)}
    iso_feats = pd.DataFrame(
        {
            "log_amt": np.log1p(work["amount"].astype(float).clip(lower=0)),
            "cat_id": work["category"].map(lambda c: cat_map.get(c, 7)),
        }
    )
    iso = IsolationForest(contamination=0.06, random_state=42)
    iso_flags = iso.fit_predict(iso_feats)
    iso_scores = -iso.score_samples(iso_feats)

    for i, (_, row) in enumerate(work.iterrows()):
        amt = float(row["amount"])
        cat = str(row["category"])
        med = float(cat_medians.get(cat, amt))
        p95 = float(cat_p95.get(cat, amt))
        reasons = []
        if med > 0 and amt >= max(p95, med * 2.8) and amt >= 600:
            reasons.append(f"{amt / med:.1f}x higher than normal {cat} spend (median ₹{med:,.0f})")
        elif iso_flags[i] == -1 and amt >= 1500:
            reasons.append(f"Outlier spend detected by Isolation Forest for {cat}")

        if reasons:
            alerts.append(
                {
                    "id": int(row["id"]) if "id" in row else i + 1,
                    "timestamp": str(row["timestamp"]),
                    "description": str(row["description"]),
                    "category": cat,
                    "amount": round(amt, 2),
                    "anomaly_score": round(float(iso_scores[i]), 3),
                    "reason": reasons[0],
                }
            )

    alerts.sort(key=lambda x: (x["anomaly_score"], x["amount"]), reverse=True)
    return alerts[:limit]


def forecast_next_month(
    df: pd.DataFrame,
    budgets: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Forecast next month's spending per category using Ridge Regression on monthly series.
    Returns category forecasts, total forecast, and budget risk warnings.
    """
    budget_map = budgets or DEFAULT_MONTHLY_BUDGETS
    if df.empty:
        return {
            "next_month_label": "Next Month",
            "predicted_total": 0.0,
            "total_budget": round(sum(budget_map.values()), 2),
            "category_forecasts": [],
        }

    work = df.copy()
    work["ts_parsed"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["ts_parsed"])
    work["ym"] = work["ts_parsed"].dt.to_period("M")

    all_months = sorted(work["ym"].unique())
    if not all_months:
        return {
            "next_month_label": "Next Month",
            "predicted_total": 0.0,
            "total_budget": round(sum(budget_map.values()), 2),
            "category_forecasts": [],
        }

    last_period = all_months[-1]
    next_period = last_period + 1
    next_label = next_period.strftime("%b %Y")

    monthly_cat = (
        work.groupby(["ym", "category"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(all_months, fill_value=0.0)
    )

    forecasts: list[dict[str, Any]] = []
    total_pred = 0.0

    for cat in CATEGORIES:
        series = monthly_cat[cat].values.astype(float) if cat in monthly_cat.columns else np.zeros(len(all_months))
        last_actual = float(series[-1]) if len(series) > 0 else 0.0
        avg_recent = float(np.mean(series[-3:])) if len(series) >= 1 else 0.0

        if len(series) >= 3 and np.sum(series) > 0:
            X = np.arange(len(series)).reshape(-1, 1)
            reg = Ridge(alpha=2.0)
            reg.fit(X, series)
            raw_pred = float(reg.predict(np.array([[len(series)]]))[0])
            # Blend trend prediction with 3-month rolling mean for stability
            pred = max(0.0, 0.55 * raw_pred + 0.45 * avg_recent)
            std = float(np.std(series))
        else:
            pred = max(0.0, avg_recent)
            std = pred * 0.15

        pred = round(pred, 2)
        total_pred += pred
        budget = float(budget_map.get(cat, 5000.0))

        if last_actual > 0:
            change_pct = round(((pred - last_actual) / last_actual) * 100.0, 1)
        else:
            change_pct = 0.0

        trend = "up" if change_pct > 5 else ("down" if change_pct < -5 else "stable")
        risk = "over_budget" if pred > budget else ("near_limit" if pred >= budget * 0.85 else "safe")

        forecasts.append(
            {
                "category": cat,
                "last_month_actual": round(last_actual, 2),
                "predicted_next_month": pred,
                "low_estimate": round(max(0.0, pred - 0.6 * std), 2),
                "high_estimate": round(pred + 0.6 * std, 2),
                "budget_limit": budget,
                "change_pct": change_pct,
                "trend": trend,
                "risk": risk,
            }
        )

    forecasts.sort(key=lambda x: x["predicted_next_month"], reverse=True)
    return {
        "next_month_label": next_label,
        "predicted_total": round(total_pred, 2),
        "total_budget": round(sum(budget_map.values()), 2),
        "category_forecasts": forecasts,
    }
