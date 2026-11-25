"""
Module 1 (Member 1): Data Pipeline, Privacy Masking, Bank SMS Parser & Receipt OCR.

Includes:
- Privacy masking for account/card numbers before storage
- Bank/UPI SMS parser (extracts amount, VPA/merchant, ambiguous UPI flag)
- Receipt image OCR using macOS native Vision framework binary (`ml/vision_ocr`)
- Synthetic + realistic Indian/Global Expense Dataset generator (6,000 rows)
- Sample receipt image generator for live demo testing
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VISION_OCR_BIN = Path(__file__).resolve().parent / "vision_ocr"

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transport & Fuel",
    "Rent & Utilities",
    "Shopping",
    "Entertainment & Subscriptions",
    "Health & Medical",
    "Education & Misc",
]


def mask_sensitive_numbers(text: str) -> str:
    """Mask long account/card/phone numbers for user financial privacy."""
    if not text:
        return ""
    # Mask 10-16 digit card/account numbers except last 4 digits
    text = re.sub(r"\b(\d{6,12})(\d{4})\b", r"XXXXXX\2", text)
    # Mask patterns like A/c XX1234 or Acct 12345678
    text = re.sub(r"(?i)(a/c|acct|account)\s*(?:no\.?|#)?\s*[x*]*\d{4,}", r"\1 XXXX", text)
    return text.strip()


def clean_description(text: str) -> str:
    """Normalize transaction text for NLP features while preserving merchant/VPA cues."""
    text = mask_sensitive_numbers(text).lower()
    # Replace UPI ref numbers and long numeric IDs with token
    text = re.sub(r"\b\d{6,}\b", " refnum ", text)
    text = re.sub(r"[^a-z0-9@._\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_ambiguous_upi_text(text: str) -> bool:
    """Detect whether a description/VPA is a person-to-person UPI string without clear merchant words."""
    t = text.lower()
    known_merchant_hints = [
        "swiggy", "zomato", "uber", "ola", "rapido", "dmart", "bigbasket", "blinkit",
        "zepto", "amazon", "flipkart", "myntra", "apollo", "medplus", "pharmeasy",
        "netflix", "spotify", "hotstar", "pvr", "inox", "bescom", "airtel", "jio",
        "act", "bpcl", "hpcl", "iocl", "shell", "starbucks", "kfc", "dominos",
        "mcdonald", "cafe", "hotel", "restaurant", "pharmacy", "hospital", "clinic",
        "mart", "supermarket", "petrol", "fuel", "metro", "irctc", "book", "udemy",
        "coursera", "college", "tuition", "electricity", "water", "rent", "wifi",
    ]
    if any(w in t for w in known_merchant_hints):
        return False
    # Typical P2P VPA or generic UPI/IMPS/transfer text
    if "@" in t or any(k in t for k in ["upi/", "paid to", "sent to", "imps", "neft", "vpa"]):
        return True
    return False


def parse_bank_sms(sms_text: str) -> dict[str, Any]:
    """Extract amount, merchant/VPA, and masked summary from raw bank/UPI SMS."""
    masked = mask_sensitive_numbers(sms_text)

    # Extract amount: Rs. 450.00, INR 1,250, ₹350, etc.
    amt_match = re.search(
        r"(?i)(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        sms_text,
    )
    amount = float(amt_match.group(1).replace(",", "")) if amt_match else 0.0

    # Extract VPA if present
    vpa_match = re.search(r"([a-zA-Z0-9._-]+@[a-zA-Z0-9]+)", sms_text)
    vpa = vpa_match.group(1).lower() if vpa_match else ""

    # Extract merchant after 'to' / 'at' if no VPA
    merchant = vpa
    if not merchant:
        to_match = re.search(
            r"(?i)(?:\bto\b|\bat\b|\bmerchant\b)\s+([A-Za-z0-9 &._-]{2,35}?)(?:\s+on\b|\s+via\b|\s+ref\b|\.|,|$)",
            sms_text,
        )
        if to_match:
            merchant = to_match.group(1).strip()

    ambiguous = is_ambiguous_upi_text(masked)
    return {
        "masked_sms": masked,
        "amount": amount,
        "merchant_or_vpa": merchant or "unknown",
        "cleaned_text": clean_description(masked),
        "is_ambiguous_upi": ambiguous,
    }


def run_vision_ocr(image_path: str | Path) -> list[str]:
    """Run compiled macOS Vision OCR binary on an image file and return recognized lines."""
    path = str(Path(image_path).resolve())
    if not VISION_OCR_BIN.exists():
        raise RuntimeError(
            f"OCR binary not found at {VISION_OCR_BIN}. Compile ml/vision_ocr.swift on macOS."
        )
    proc = subprocess.run(
        [str(VISION_OCR_BIN), path],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"OCR failed: {proc.stderr.strip()}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return lines


def parse_receipt_lines(lines: list[str]) -> dict[str, Any]:
    """Extract merchant, total amount, and cleaned text from recognized OCR lines."""
    full_text = "\n".join(lines)

    # First non-trivial line is usually store/restaurant name
    merchant = "Receipt Merchant"
    for ln in lines[:4]:
        if len(ln) >= 3 and not re.match(r"^(tax|invoice|bill|date|gst|tel)", ln, re.I):
            merchant = ln
            break

    # Look for total amount lines first
    total_amount = 0.0
    for ln in reversed(lines):
        if re.search(r"(?i)(grand\s*total|net\s*amount|total\s*payable|\btotal\b|amount\s*paid)", ln):
            nums = re.findall(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", ln)
            if nums:
                val = float(nums[-1].replace(",", ""))
                if 1.0 <= val <= 500000.0:
                    total_amount = val
                    break

    # Fallback: largest decimal currency-like number
    if total_amount == 0.0:
        candidates = []
        for ln in lines:
            for n in re.findall(r"(?:rs\.?|₹|inr)?\s*([0-9][0-9,]*\.[0-9]{2})\b", ln, re.I):
                val = float(n.replace(",", ""))
                if 1.0 <= val <= 500000.0:
                    candidates.append(val)
        if candidates:
            total_amount = max(candidates)

    summary_description = f"{merchant} " + " ".join(lines[:10])
    return {
        "merchant": merchant,
        "amount": round(total_amount, 2),
        "raw_lines": lines,
        "raw_text": full_text,
        "cleaned_text": clean_description(summary_description),
    }


def extract_receipt_info(image_path: str | Path) -> dict[str, Any]:
    """Extract merchant, total amount, items, and clean text from a bill/receipt image."""
    lines = run_vision_ocr(image_path)
    return parse_receipt_lines(lines)


def create_sample_receipts(out_dir: str | Path | None = None) -> list[Path]:
    """Create 3 realistic sample receipt PNG images for demo & OCR testing."""
    target = Path(out_dir) if out_dir else (DATA_DIR / "sample_receipts")
    target.mkdir(parents=True, exist_ok=True)

    receipts = [
        (
            "restaurant_bill.png",
            [
                "SPICE GARDEN RESTAURANT",
                "MG Road, Bengaluru",
                "Bill No: 4821   Date: 2026-11-20 20:15",
                "------------------------------------",
                "2x Paneer Butter Masala      440.00",
                "4x Butter Naan               180.00",
                "2x Masala Soda               120.00",
                "------------------------------------",
                "Subtotal                     740.00",
                "GST 5%                        37.00",
                "GRAND TOTAL                  777.00",
                "Paid via UPI - Thank You!",
            ],
        ),
        (
            "pharmacy_bill.png",
            [
                "APOLLO PHARMACY WELLNESS",
                "Invoice #AP-9931  Date: 2026-11-18 11:20",
                "------------------------------------",
                "Dolo 650mg Strip x2           64.00",
                "Azithromycin 500mg           128.00",
                "Vitamin D3 Capsules          340.00",
                "Cough Syrup 100ml            118.00",
                "------------------------------------",
                "TOTAL AMOUNT                 650.00",
                "Get Well Soon!",
            ],
        ),
        (
            "grocery_bill.png",
            [
                "FRESH MART SUPERMARKET",
                "Receipt #FM-2041  Date: 2026-11-22 10:30",
                "------------------------------------",
                "Aashirvaad Atta 5kg          285.00",
                "Amul Milk 2L                 136.00",
                "Fresh Tomatoes & Onions      145.00",
                "Sunflower Oil 1L             164.00",
                "------------------------------------",
                "NET PAYABLE                  730.00",
            ],
        ),
    ]

    created: list[Path] = []
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    for fname, lines in receipts:
        img = Image.new("RGB", (620, 440), color=(252, 252, 248))
        draw = ImageDraw.Draw(img)
        draw.rectangle([14, 14, 606, 426], outline=(190, 190, 185), width=2)
        y = 32
        for idx, line in enumerate(lines):
            color = (20, 24, 32) if idx == 0 else (45, 50, 60)
            draw.text((34, y), line, fill=color, font=font)
            y += 30
        out_path = target / fname
        img.save(out_path)
        created.append(out_path)
    return created


def generate_expense_dataset(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a 6,000-row realistic dataset combining clear merchants AND
    ~25% ambiguous P2P UPI transactions where behavioral context resolves category.
    """
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    clear_templates = {
        "Food & Dining": [
            ("Swiggy Order #{ref}", "swiggy@icici", (140, 750), (12, 23)),
            ("Zomato Food Delivery", "zomato@hdfcbank", (160, 880), (12, 23)),
            ("Starbucks Coffee", "starbucks@axis", (220, 650), (9, 20)),
            ("Udupi Darshini Breakfast", "udupi@ybl", (45, 180), (7, 11)),
            ("Dominos Pizza Order", "dominos@sbi", (300, 1100), (18, 23)),
            ("College Canteen Tea Snacks", "canteen@paytm", (20, 120), (9, 17)),
        ],
        "Groceries": [
            ("Blinkit Instant Groceries", "blinkit@icici", (180, 1200), (8, 21)),
            ("Zepto Fresh Vegetables Milk", "zepto@axis", (150, 950), (7, 21)),
            ("DMart Supermarket Bill", "dmart@hdfc", (700, 4500), (10, 21)),
            ("BigBasket Monthly Pantry", "bigbasket@sbi", (600, 3800), (9, 20)),
            ("Reliance Fresh Dairy & Fruits", "reliancefresh@ybl", (120, 900), (8, 20)),
        ],
        "Transport & Fuel": [
            ("Uber Auto Ride", "uber@icici", (65, 420), (7, 22)),
            ("Ola Cab Airport Trip", "ola@axis", (180, 1150), (5, 23)),
            ("Rapido Bike Taxi", "rapido@paytm", (35, 160), (8, 21)),
            ("IndianOil Petrol Pump Fuel", "iocl@sbi", (250, 2600), (7, 22)),
            ("HPCL Fuel Station", "hpcl@hdfc", (300, 2400), (7, 22)),
            ("Namma Metro Card Recharge", "metro@ybl", (100, 500), (8, 20)),
        ],
        "Rent & Utilities": [
            ("BESCOM Electricity Bill", "bescom@sbi", (650, 3200), (9, 18)),
            ("Airtel Fiber WiFi Recharge", "airtel@icici", (599, 1499), (9, 21)),
            ("Jio Postpaid Mobile Bill", "jio@axis", (299, 899), (9, 21)),
            ("Apartment Monthly Rent", "owner@hdfc", (7500, 24000), (8, 19)),
            ("Indane LPG Gas Cylinder", "indane@sbi", (850, 1150), (9, 17)),
        ],
        "Shopping": [
            ("Amazon India Order", "amazon@icici", (350, 6500), (10, 23)),
            ("Flipkart Electronics & Home", "flipkart@axis", (450, 8500), (10, 23)),
            ("Myntra Fashion Apparel", "myntra@hdfc", (699, 4200), (11, 23)),
            ("Zudio Clothing Store", "zudio@ybl", (499, 2999), (12, 21)),
            ("Decathlon Sports Gear", "decathlon@icici", (600, 4800), (11, 20)),
        ],
        "Entertainment & Subscriptions": [
            ("Netflix Monthly Subscription", "netflix@icici", (199, 649), (0, 23)),
            ("Spotify Premium Plan", "spotify@axis", (119, 179), (0, 23)),
            ("PVR Cinemas Movie Tickets", "pvr@hdfc", (320, 1400), (13, 22)),
            ("BookMyShow Event Pass", "bms@paytm", (250, 1800), (14, 22)),
            ("YouTube Premium Renewal", "googleplay@sbi", (129, 189), (0, 23)),
        ],
        "Health & Medical": [
            ("Apollo Pharmacy Medicines", "apollo@icici", (120, 1800), (8, 22)),
            ("MedPlus Health Store", "medplus@ybl", (90, 1400), (8, 22)),
            ("City Diagnostic Lab Blood Test", "lab@hdfc", (450, 3200), (7, 14)),
            ("Dental Clinic Consultation", "clinic@axis", (500, 2500), (10, 19)),
            ("PharmEasy Online Order", "pharmeasy@sbi", (220, 2100), (9, 21)),
        ],
        "Education & Misc": [
            ("Udemy Online Course", "udemy@icici", (449, 1299), (10, 23)),
            ("Coursera Specialization Fee", "coursera@hdfc", (1500, 4200), (10, 22)),
            ("Sapna Book House Textbooks", "books@ybl", (250, 1900), (10, 20)),
            ("College Exam & Print Fee", "printshop@paytm", (40, 850), (9, 17)),
            ("AWS / Cloud Lab Credits", "aws@axis", (300, 2200), (10, 22)),
        ],
    }

    # Generic P2P VPAs shared across categories — text alone CANNOT identify the category!
    generic_p2p_vpas = [
        "rahul.s@okaxis", "amit.k@ybl", "priya.m@okicici", "vikas.p@paytm",
        "neha.r@sbi", "rohit.v@hdfc", "deepak.g@ybl", "sanjay.t@okaxis",
        "kiran.b@paytm", "pooja.n@okicici", "arjun.d@ybl", "suresh.m@okaxis",
    ]
    # Behavioral profiles for ambiguous P2P UPI transactions
    ambiguous_behavior = [
        # (category, amt_range, hour_range, day_range, repeat_range, prefer_weekend)
        ("Rent & Utilities", (7500, 22000), (8, 18), (1, 6), (4, 12), False),
        ("Food & Dining", (20, 120), (7, 11), (1, 28), (6, 24), False),
        ("Food & Dining", (160, 580), (19, 23), (1, 28), (2, 10), False),
        ("Groceries", (140, 480), (6, 10), (8, 28), (5, 18), False),
        ("Transport & Fuel", (40, 190), (16, 19), (1, 28), (3, 12), False),
        ("Shopping", (1100, 4800), (13, 21), (8, 28), (1, 2), False),
        ("Entertainment & Subscriptions", (280, 920), (18, 23), (8, 28), (1, 3), True),
        ("Health & Medical", (520, 1800), (11, 15), (8, 28), (1, 2), False),
        ("Education & Misc", (200, 780), (9, 16), (8, 28), (1, 3), False),
    ]

    start_date = datetime(2026, 5, 1)
    rows = []

    for i in range(n_rows):
        is_ambig = rng.random() < 0.28  # 28% realistic ambiguous UPI P2P rows

        if not is_ambig:
            cat = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
            tmpl, vpa, amt_rng, hr_rng = clear_templates[cat][
                int(rng.integers(0, len(clear_templates[cat])))
            ]
            desc_base = tmpl.format(ref=int(rng.integers(1000, 9999)))
            amount = round(float(rng.uniform(amt_rng[0], amt_rng[1])), 2)
            hour = int(rng.integers(hr_rng[0], hr_rng[1] + 1))
            day_offset = int(rng.integers(0, 180))
            if cat == "Rent & Utilities" and "Rent" in desc_base:
                dt = datetime(2026, 5 + (day_offset // 30), int(rng.integers(1, 6)), hour, int(rng.integers(0, 60)))
            else:
                dt = start_date + timedelta(days=day_offset, hours=hour, minutes=int(rng.integers(0, 60)))
            repeat_cnt = int(rng.integers(1, 12))
            if rng.random() < 0.45:
                raw_sms = f"Rs.{amount:.2f} debited from A/c XX4821 to VPA {vpa} ({desc_base}) on {dt.strftime('%d-%m-%y')}. UPI Ref {int(rng.integers(100000, 999999))}."
                desc = desc_base
            else:
                raw_sms = ""
                desc = desc_base
        else:
            vpa = generic_p2p_vpas[int(rng.integers(0, len(generic_p2p_vpas)))]
            prof = ambiguous_behavior[int(rng.integers(0, len(ambiguous_behavior)))]
            cat, amt_rng, hr_rng, day_rng, rep_rng, prefer_wknd = prof
            amount = round(float(rng.uniform(amt_rng[0], amt_rng[1])), 2)
            hour = int(rng.integers(hr_rng[0], hr_rng[1] + 1))
            month_idx = int(rng.integers(5, 11))
            dom = int(rng.integers(day_rng[0], day_rng[1] + 1))
            dt = datetime(2026, month_idx, dom, hour, int(rng.integers(0, 60)))
            repeat_cnt = int(rng.integers(rep_rng[0], rep_rng[1] + 1))
            if prefer_wknd:
                days_to_sat = (5 - dt.weekday()) % 7
                dt = dt + timedelta(days=days_to_sat)
                if dt.month != month_idx:
                    dt = dt - timedelta(days=7)
            sms_style = int(rng.integers(0, 3))
            if sms_style == 0:
                desc = f"Paid to {vpa}"
                raw_sms = f"Rs.{amount:.2f} debited from A/c XX4821 to VPA {vpa} on {dt.strftime('%d-%m-%y')}. UPI Ref {int(rng.integers(100000, 999999))}."
            elif sms_style == 1:
                desc = f"UPI/{int(rng.integers(100000, 999999))}/{vpa}"
                raw_sms = f"INR {amount:.2f} sent via UPI to {vpa} Ref {int(rng.integers(100000, 999999))}."
            else:
                name_part = vpa.split("@")[0].replace(".", " ")
                desc = f"Sent to {name_part} via UPI"
                raw_sms = f"Paid Rs {amount:.2f} to {vpa}."

        # Small realistic label noise (2.5%)
        if rng.random() < 0.025:
            cat = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]

        rows.append(
            {
                "tx_id": f"TXN{i + 1:05d}",
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "description": desc,
                "raw_sms": mask_sensitive_numbers(raw_sms),
                "merchant_or_vpa": vpa,
                "cleaned_text": clean_description(f"{desc} {vpa}"),
                "amount": amount,
                "hour": dt.hour,
                "day_of_month": dt.day,
                "is_weekend": 1 if dt.weekday() >= 5 else 0,
                "vpa_repeat_count": repeat_cnt,
                "is_ambiguous_upi": 1 if is_ambig else 0,
                "category": cat,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "expenses_dataset.csv"
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    df = generate_expense_dataset()
    print(f"Generated dataset: {len(df)} rows -> {DATA_DIR / 'expenses_dataset.csv'}")
    receipts = create_sample_receipts()
    print(f"Created {len(receipts)} sample receipts in {DATA_DIR / 'sample_receipts'}")
    for r in receipts:
        try:
            if VISION_OCR_BIN.exists():
                parsed = extract_receipt_info(r)
                print(f"  OCR Test [{r.name}]: merchant={parsed['merchant']!r}, amount={parsed['amount']}")
        except Exception as exc:
            print(f"  OCR skipped in restricted env [{r.name}]: {exc}")
