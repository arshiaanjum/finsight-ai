"""
SQLite Database Layer for FinSight AI.

Stores:
- User transactions (with AI confidence, explanation, and ambiguity flag)
- Active Learning personalization memory (`learned_rules`)
- Category monthly budgets
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.anomaly_and_forecast import DEFAULT_MONTHLY_BUDGETS
from ml.data_and_ocr import DATA_DIR, is_ambiguous_upi_text, mask_sensitive_numbers

DB_PATH = DATA_DIR / "finsight.db"


def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                description TEXT NOT NULL,
                raw_sms TEXT DEFAULT '',
                merchant_or_vpa TEXT DEFAULT '',
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_ambiguous_upi INTEGER DEFAULT 0,
                source TEXT NOT NULL,
                explanation TEXT DEFAULT '',
                is_corrected INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_rules (
                vpa_or_key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                category TEXT PRIMARY KEY,
                monthly_limit REAL NOT NULL
            )
            """
        )
        for cat, lim in DEFAULT_MONTHLY_BUDGETS.items():
            conn.execute(
                "INSERT OR IGNORE INTO budgets (category, monthly_limit) VALUES (?, ?)",
                (cat, lim),
            )
        conn.commit()


def seed_demo_transactions(force: bool = False) -> None:
    init_db()
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        if count > 0 and not force:
            return
        if force:
            conn.execute("DELETE FROM transactions")
            conn.execute("DELETE FROM learned_rules")

        rng = np.random.default_rng(7)
        samples = [
            ("Swiggy Dinner Order", "swiggy@icici", "Food & Dining", (180, 520), 20),
            ("Zomato Lunch Bowl", "zomato@hdfc", "Food & Dining", (160, 440), 13),
            ("Paid to raju.stall@paytm", "raju.stall@paytm", "Food & Dining", (35, 95), 9),
            ("Blinkit Milk & Veggies", "blinkit@icici", "Groceries", (220, 780), 10),
            ("DMart Monthly Ration", "dmart@hdfc", "Groceries", (1400, 3400), 17),
            ("Uber Auto Commute", "uber@icici", "Transport & Fuel", (75, 290), 9),
            ("IndianOil Petrol Refill", "iocl@sbi", "Transport & Fuel", (500, 1800), 18),
            ("Paid to suresh.k@okaxis", "suresh.k@okaxis", "Rent & Utilities", (11500, 12500), 10),
            ("Airtel Fiber WiFi Bill", "airtel@icici", "Rent & Utilities", (943, 943), 11),
            ("BESCOM Electricity Bill", "bescom@sbi", "Rent & Utilities", (1100, 2100), 15),
            ("Amazon Electronics Order", "amazon@icici", "Shopping", (650, 3500), 21),
            ("Netflix Monthly Plan", "netflix@icici", "Entertainment & Subscriptions", (499, 499), 22),
            ("PVR Movie Weekend", "pvr@hdfc", "Entertainment & Subscriptions", (450, 980), 19),
            ("Apollo Pharmacy Meds", "apollo@icici", "Health & Medical", (180, 920), 16),
            ("Udemy AI Course", "udemy@icici", "Education & Misc", (499, 899), 19),
        ]

        rows_to_insert = []
        # 5 months: Jul 2026 through Nov 2026
        for month in [7, 8, 9, 10, 11]:
            max_day = 24 if month == 11 else 28
            # Fixed monthly items
            for desc, vpa, cat, amt_rng, hr in samples:
                if cat == "Rent & Utilities" or cat == "Entertainment & Subscriptions" and "Netflix" in desc:
                    day = 2 if "suresh" in vpa else int(rng.integers(3, 12))
                    if day <= max_day:
                        dt = datetime(2026, month, day, hr, int(rng.integers(5, 55)))
                        amt = round(float(rng.uniform(amt_rng[0], amt_rng[1])), 2)
                        ambig = 1 if is_ambiguous_upi_text(desc) else 0
                        rows_to_insert.append(
                            (
                                dt.strftime("%Y-%m-%d %H:%M:%S"),
                                desc,
                                mask_sensitive_numbers(f"Rs.{amt:.2f} debited to VPA {vpa}"),
                                vpa,
                                amt,
                                cat,
                                round(float(rng.uniform(86.0, 96.5)), 1),
                                ambig,
                                "FinSight Hybrid AI (NLP + Context)",
                                "Recurring monthly pattern + amount context matched",
                                0,
                            )
                        )
            # Frequent variable expenses (~22 per month)
            variable_pool = [s for s in samples if s[2] != "Rent & Utilities"]
            for _ in range(22):
                desc, vpa, cat, amt_rng, hr = variable_pool[int(rng.integers(0, len(variable_pool)))]
                day = int(rng.integers(1, max_day + 1))
                dt = datetime(2026, month, day, hr, int(rng.integers(0, 59)))
                amt = round(float(rng.uniform(amt_rng[0], amt_rng[1])), 2)
                ambig = 1 if is_ambiguous_upi_text(desc) else 0
                rows_to_insert.append(
                    (
                        dt.strftime("%Y-%m-%d %H:%M:%S"),
                        desc,
                        mask_sensitive_numbers(f"Rs.{amt:.2f} debited to VPA {vpa}"),
                        vpa,
                        amt,
                        cat,
                        round(float(rng.uniform(79.0, 97.0)), 1),
                        ambig,
                        "FinSight Hybrid AI (NLP + Context)",
                        "Matched merchant keywords and spend time/amount context",
                        0,
                    )
                )

        # Add 2 realistic anomaly events in Nov 2026 for live anomaly dashboard demo
        rows_to_insert.append(
            (
                "2026-11-19 22:40:00",
                "Swiggy Gourmet Party Order",
                "Rs.4850.00 debited to VPA swiggy@icici",
                "swiggy@icici",
                4850.0,
                "Food & Dining",
                94.2,
                0,
                "FinSight Hybrid AI (NLP + Context)",
                "Matched Swiggy dining keywords; unusual high amount flagged",
                0,
            )
        )
        rows_to_insert.append(
            (
                "2026-11-21 15:10:00",
                "Amazon Gadget Purchase",
                "Rs.14999.00 debited to VPA amazon@icici",
                "amazon@icici",
                14999.0,
                "Shopping",
                96.4,
                0,
                "FinSight Hybrid AI (NLP + Context)",
                "Matched Amazon shopping pattern; spike detected",
                0,
            )
        )

        conn.executemany(
            """
            INSERT INTO transactions (
                timestamp, description, raw_sms, merchant_or_vpa, amount,
                category, confidence, is_ambiguous_upi, source, explanation, is_corrected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()


def fetch_transactions_df() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY timestamp DESC, id DESC", conn)
    return df


def get_learned_rules() -> dict[str, str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT vpa_or_key, category FROM learned_rules").fetchall()
    return {r["vpa_or_key"]: r["category"] for r in rows}


def save_learned_rule(vpa_or_key: str, category: str) -> None:
    key = vpa_or_key.strip().lower()
    if not key:
        return
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO learned_rules (vpa_or_key, category, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(vpa_or_key) DO UPDATE SET category=excluded.category, updated_at=excluded.updated_at
            """,
            (key, category, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


def get_budgets() -> dict[str, float]:
    with get_conn() as conn:
        rows = conn.execute("SELECT category, monthly_limit FROM budgets").fetchall()
    return {r["category"]: float(r["monthly_limit"]) for r in rows}


def insert_transaction(item: dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO transactions (
                timestamp, description, raw_sms, merchant_or_vpa, amount,
                category, confidence, is_ambiguous_upi, source, explanation, is_corrected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["timestamp"],
                item["description"],
                mask_sensitive_numbers(item.get("raw_sms", "")),
                item.get("merchant_or_vpa", ""),
                float(item["amount"]),
                item["category"],
                float(item["confidence"]),
                int(item.get("is_ambiguous_upi", 0)),
                item.get("source", "FinSight Hybrid AI"),
                item.get("explanation", ""),
                int(item.get("is_corrected", 0)),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def correct_transaction_category(tx_id: int, new_category: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if not row:
            return None
        vpa = (row["merchant_or_vpa"] or "").strip().lower()
        key = vpa if vpa and vpa != "unknown" else row["description"].strip().lower()
        conn.execute(
            """
            UPDATE transactions
            SET category = ?, confidence = 100.0, is_corrected = 1,
                source = 'User Corrected (Learned by AI)',
                explanation = ?
            WHERE id = ?
            """,
            (new_category, f"Corrected by user; AI learned '{key}' -> {new_category}", tx_id),
        )
        conn.commit()

    save_learned_rule(key, new_category)
    return {"id": tx_id, "learned_key": key, "new_category": new_category}


def count_vpa_repeats(vpa_or_desc: str) -> int:
    key = (vpa_or_desc or "").strip().lower()
    if not key:
        return 1
    with get_conn() as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE LOWER(merchant_or_vpa) = ? OR LOWER(description) LIKE ?",
            (key, f"%{key}%"),
        ).fetchone()[0]
    return max(1, int(cnt))
