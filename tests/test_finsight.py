"""End-to-end tests for FinSight AI (ML modules + FastAPI endpoints)."""

from starlette.testclient import TestClient

from backend.main import app
from ml.data_and_ocr import mask_sensitive_numbers, parse_bank_sms, parse_receipt_lines


def test_privacy_masking_and_sms_parser():
    sms = "Rs.485.00 debited from A/c 123456789012 to VPA swiggy@icici on 24-11-26. UPI Ref 882910."
    masked = mask_sensitive_numbers(sms)
    assert "123456789012" not in masked
    assert "XXXX" in masked

    parsed = parse_bank_sms(sms)
    assert parsed["amount"] == 485.0
    assert parsed["merchant_or_vpa"] == "swiggy@icici"
    assert parsed["is_ambiguous_upi"] is False

    ambig = parse_bank_sms("INR 12000.00 paid via UPI to suresh.k@okaxis on 02-11-26.")
    assert ambig["amount"] == 12000.0
    assert ambig["is_ambiguous_upi"] is True


def test_receipt_line_parser():
    sample_lines = [
        "SPICE GARDEN RESTAURANT",
        "2x Paneer Butter Masala 440.00",
        "GRAND TOTAL 777.00",
    ]
    parsed = parse_receipt_lines(sample_lines)
    assert parsed["merchant"] == "SPICE GARDEN RESTAURANT"
    assert parsed["amount"] == 777.0


def test_api_end_to_end():
    with TestClient(app) as client:
        # Reset demo DB
        assert client.post("/api/demo/reset").status_code == 200

        # Dashboard loads KPIs, forecasts, anomalies
        dash = client.get("/api/dashboard")
        assert dash.status_code == 200
        d = dash.json()
        assert d["kpis"]["transaction_count"] > 50
        assert d["forecast"]["predicted_total"] > 0

        # Predict ambiguous UPI Rent (large amount early in month)
        res = client.post(
            "/api/predict",
            json={
                "description": "Paid to rahul.s@okaxis",
                "amount": 13500.0,
                "timestamp": "2026-11-02T10:00:00",
                "merchant_or_vpa": "rahul.s@okaxis",
                "save": True,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["prediction"]["category"] == "Rent & Utilities"
        tx_id = body["saved_tx_id"]
        assert tx_id is not None

        # Active learning correction test
        corr = client.post(
            f"/api/transactions/{tx_id}/correct",
            json={"new_category": "Education & Misc"},
        )
        assert corr.status_code == 200

        # Re-predict same VPA -> should immediately use Personalized AI Memory
        res2 = client.post(
            "/api/predict",
            json={
                "description": "Paid to rahul.s@okaxis",
                "amount": 13500.0,
                "timestamp": "2026-11-05T10:00:00",
                "merchant_or_vpa": "rahul.s@okaxis",
                "save": False,
            },
        )
        assert res2.status_code == 200
        assert res2.json()["prediction"]["category"] == "Education & Misc"
        assert "Personalized AI Memory" in res2.json()["prediction"]["source"]

        # Anomaly spike test
        anom = client.post(
            "/api/predict",
            json={
                "description": "Swiggy Party Order",
                "amount": 6500.0,
                "timestamp": "2026-11-24T22:00:00",
                "merchant_or_vpa": "swiggy@icici",
                "save": False,
            },
        )
        assert anom.status_code == 200
        assert anom.json()["anomaly"]["is_anomaly"] is True

        # Research metrics endpoint
        rm = client.get("/api/research-metrics")
        assert rm.status_code == 200
        assert len(rm.json()["model_comparison"]) == 4

        # Reset cleanly after test
        client.post("/api/demo/reset")
