"""
Module 4 (Member 4): ASGI Backend (Starlette + Uvicorn + Pydantic)
connecting all 3 AI modules + SQLite + Frontend UI.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from backend.db import (
    correct_transaction_category,
    count_vpa_repeats,
    fetch_transactions_df,
    get_budgets,
    get_learned_rules,
    insert_transaction,
    seed_demo_transactions,
)
from ml.anomaly_and_forecast import (
    check_single_transaction_anomaly,
    detect_historical_anomalies,
    forecast_next_month,
)
from ml.category_model import ARTIFACTS_DIR, METRICS_PATH, PLOTS_DIR, CategoryPredictor, train_and_evaluate
from ml.data_and_ocr import (
    CATEGORIES,
    DATA_DIR,
    create_sample_receipts,
    extract_receipt_info,
    generate_expense_dataset,
    mask_sensitive_numbers,
    parse_bank_sms,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLE_RECEIPTS_DIR = DATA_DIR / "sample_receipts"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

predictor: CategoryPredictor | None = None


@asynccontextmanager
async def lifespan(_: Starlette):
    global predictor
    seed_demo_transactions()
    if not any(SAMPLE_RECEIPTS_DIR.iterdir()):
        create_sample_receipts(SAMPLE_RECEIPTS_DIR)
    if not (ARTIFACTS_DIR / "finsight_hybrid_model.joblib").exists():
        if not (DATA_DIR / "expenses_dataset.csv").exists():
            generate_expense_dataset()
        train_and_evaluate()
    predictor = CategoryPredictor()
    yield


class PredictRequest(BaseModel):
    description: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    timestamp: str | None = None
    merchant_or_vpa: str = ""
    raw_sms: str = ""
    save: bool = False


class CorrectionRequest(BaseModel):
    new_category: str


class SmsRequest(BaseModel):
    sms_text: str = Field(..., min_length=5)
    timestamp: str | None = None
    save: bool = False


def _get_predictor() -> CategoryPredictor:
    global predictor
    if predictor is None:
        predictor = CategoryPredictor()
    return predictor


def _run_predict(req: PredictRequest) -> dict:
    pred_engine = _get_predictor()
    learned = get_learned_rules()
    ts = req.timestamp or datetime.now().isoformat(timespec="seconds")
    repeat_cnt = count_vpa_repeats(req.merchant_or_vpa or req.description)

    result = pred_engine.predict(
        description=req.description,
        amount=req.amount,
        timestamp=ts,
        merchant_or_vpa=req.merchant_or_vpa,
        vpa_repeat_count=repeat_cnt,
        learned_rules=learned,
    )

    hist_df = fetch_transactions_df()
    anomaly = check_single_transaction_anomaly(
        history_df=hist_df,
        amount=req.amount,
        category=result["category"],
        merchant_or_vpa=req.merchant_or_vpa,
        timestamp=ts,
    )

    tx_id = None
    if req.save:
        tx_id = insert_transaction(
            {
                "timestamp": ts.replace("T", " "),
                "description": mask_sensitive_numbers(req.description),
                "raw_sms": req.raw_sms,
                "merchant_or_vpa": req.merchant_or_vpa,
                "amount": req.amount,
                "category": result["category"],
                "confidence": result["confidence"],
                "is_ambiguous_upi": 1 if result["is_ambiguous_upi"] else 0,
                "source": result["source"],
                "explanation": result["explanation"],
                "is_corrected": 0,
            }
        )

    return {
        "prediction": result,
        "anomaly": anomaly,
        "saved_tx_id": tx_id,
    }


async def serve_index(_: Request):
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"detail": "Frontend index.html not found"}, status_code=404)
    return FileResponse(index_path)


async def health(_: Request):
    return JSONResponse({"status": "ok", "categories": CATEGORIES})


async def get_dashboard(_: Request):
    df = fetch_transactions_df()
    learned = get_learned_rules()
    budgets = get_budgets()

    if df.empty:
        return JSONResponse(
            {
                "kpis": {
                    "total_spend": 0,
                    "current_month_spend": 0,
                    "transaction_count": 0,
                    "learned_rules_count": len(learned),
                    "anomaly_count": 0,
                },
                "category_breakdown": [],
                "monthly_trend": [],
                "anomalies": [],
                "forecast": forecast_next_month(df, budgets),
                "learned_rules": learned,
                "categories": CATEGORIES,
            }
        )

    work = df.copy()
    work["ts_parsed"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["ym"] = work["ts_parsed"].dt.strftime("%Y-%m")
    latest_ym = str(work["ym"].dropna().max())

    cur_month_df = work[work["ym"] == latest_ym]
    cat_sum = (
        cur_month_df.groupby("category")["amount"]
        .sum()
        .reindex(CATEGORIES, fill_value=0.0)
        .reset_index()
    )
    cat_breakdown = [
        {
            "category": str(r["category"]),
            "amount": round(float(r["amount"]), 2),
            "budget": float(budgets.get(str(r["category"]), 5000.0)),
        }
        for _, r in cat_sum.iterrows()
    ]

    monthly_series = work.groupby("ym")["amount"].sum().sort_index().reset_index()
    monthly_trend = [
        {"month": str(r["ym"]), "total": round(float(r["amount"]), 2)}
        for _, r in monthly_series.iterrows()
    ]

    anomalies = detect_historical_anomalies(df, limit=6)
    forecast = forecast_next_month(df, budgets)

    return JSONResponse(
        {
            "kpis": {
                "total_spend": round(float(work["amount"].sum()), 2),
                "current_month_spend": round(float(cur_month_df["amount"].sum()), 2),
                "current_month_label": latest_ym,
                "transaction_count": int(len(work)),
                "learned_rules_count": len(learned),
                "anomaly_count": len(anomalies),
            },
            "category_breakdown": cat_breakdown,
            "monthly_trend": monthly_trend,
            "anomalies": anomalies,
            "forecast": forecast,
            "learned_rules": learned,
            "categories": CATEGORIES,
        }
    )


async def list_transactions(request: Request):
    category = request.query_params.get("category", "")
    search = request.query_params.get("search", "")
    limit = int(request.query_params.get("limit", "80"))

    df = fetch_transactions_df()
    if category and category != "All":
        df = df[df["category"] == category]
    if search:
        s = search.lower()
        df = df[
            df["description"].str.lower().str.contains(s, na=False)
            | df["merchant_or_vpa"].fillna("").str.lower().str.contains(s, na=False)
            | df["category"].str.lower().str.contains(s, na=False)
        ]
    df = df.head(limit)
    return JSONResponse({"transactions": df.to_dict(orient="records")})


async def predict_expense(request: Request):
    try:
        payload = await request.json()
        req = PredictRequest(**payload)
    except (ValidationError, Exception) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    return JSONResponse(_run_predict(req))


async def parse_sms_and_predict(request: Request):
    try:
        payload = await request.json()
        req = SmsRequest(**payload)
    except (ValidationError, Exception) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    parsed = parse_bank_sms(req.sms_text)
    if parsed["amount"] <= 0:
        return JSONResponse(
            {"detail": "Could not find an amount (like Rs. 450 or ₹450) in the SMS text."},
            status_code=400,
        )

    ts = req.timestamp or datetime.now().isoformat(timespec="seconds")
    pred_req = PredictRequest(
        description=parsed["masked_sms"],
        amount=parsed["amount"],
        timestamp=ts,
        merchant_or_vpa=parsed["merchant_or_vpa"],
        raw_sms=parsed["masked_sms"],
        save=req.save,
    )
    out = _run_predict(pred_req)
    out["parsed_sms"] = parsed
    return JSONResponse(out)


async def upload_receipt(request: Request):
    save = request.query_params.get("save", "false").lower() == "true"
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"detail": "No file uploaded"}, status_code=400)

    filename = getattr(upload, "filename", "receipt.png") or "receipt.png"
    suffix = Path(filename).suffix or ".png"
    content = await upload.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        ocr_data = extract_receipt_info(tmp_path)
    except Exception as exc:
        return JSONResponse({"detail": f"Receipt OCR error: {exc}"}, status_code=422)
    finally:
        tmp_path.unlink(missing_ok=True)

    amount = ocr_data["amount"] if ocr_data["amount"] > 0 else 100.0
    ts = datetime.now().isoformat(timespec="seconds")
    pred_req = PredictRequest(
        description=f"{ocr_data['merchant']} (Receipt Scan)",
        amount=amount,
        timestamp=ts,
        merchant_or_vpa=ocr_data["merchant"],
        raw_sms=ocr_data["raw_text"][:300],
        save=save,
    )
    out = _run_predict(pred_req)
    out["ocr"] = ocr_data
    return JSONResponse(out)


async def correct_transaction(request: Request):
    tx_id = int(request.path_params["tx_id"])
    try:
        payload = await request.json()
        req = CorrectionRequest(**payload)
    except (ValidationError, Exception) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if req.new_category not in CATEGORIES:
        return JSONResponse({"detail": f"Invalid category. Choose from {CATEGORIES}"}, status_code=400)
    updated = correct_transaction_category(tx_id, req.new_category)
    if not updated:
        return JSONResponse({"detail": "Transaction not found"}, status_code=404)
    return JSONResponse({"status": "learned", **updated})


async def get_research_metrics(_: Request):
    if not METRICS_PATH.exists():
        return JSONResponse({"detail": "Metrics not generated yet"}, status_code=404)
    return JSONResponse(json.loads(METRICS_PATH.read_text()))


async def reset_demo(_: Request):
    seed_demo_transactions(force=True)
    return JSONResponse({"status": "reset_complete"})


routes = [
    Route("/", serve_index, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/dashboard", get_dashboard, methods=["GET"]),
    Route("/api/transactions", list_transactions, methods=["GET"]),
    Route("/api/predict", predict_expense, methods=["POST"]),
    Route("/api/parse-sms", parse_sms_and_predict, methods=["POST"]),
    Route("/api/upload-receipt", upload_receipt, methods=["POST"]),
    Route("/api/transactions/{tx_id:int}/correct", correct_transaction, methods=["POST"]),
    Route("/api/research-metrics", get_research_metrics, methods=["GET"]),
    Route("/api/demo/reset", reset_demo, methods=["POST"]),
    Mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots"),
    Mount("/sample-receipts", StaticFiles(directory=str(SAMPLE_RECEIPTS_DIR)), name="sample_receipts"),
]

app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)
