"""
Suite di Test Automatizzata per:
1. Resampling e Indicatori (FVG, Dealing Range, Killzones)
2. FastAPI Webhook Validation Logic
"""
import pandas as pd
import pytest
from starlette.testclient import TestClient

from config import CONFIG_RISK
from core.data_loader import resample_candles
from core.indicators import (
    calculate_dealing_range,
    detect_smt_divergence,
    find_fair_value_gaps,
    is_in_killzone
)
from webhook_bridge.app import app


def test_killzone_filtering():
    # 09:30 CET -> London Killzone
    ts_london = pd.Timestamp("2026-10-15 09:30:00", tz="Europe/Rome")
    in_kz, sess = is_in_killzone(ts_london)
    assert in_kz is True
    assert sess == "LONDON"

    # 14:30 CET -> New York Killzone
    ts_ny = pd.Timestamp("2026-10-15 14:30:00", tz="Europe/Rome")
    in_kz, sess = is_in_killzone(ts_ny)
    assert in_kz is True
    assert sess == "NEW_YORK"

    # 22:00 CET -> Out of session
    ts_out = pd.Timestamp("2026-10-15 22:00:00", tz="Europe/Rome")
    in_kz, sess = is_in_killzone(ts_out)
    assert in_kz is False
    assert sess == "OUT_OF_SESSION"


def test_smt_divergence_detection():
    # Asset 1 (EURUSD) fa Higher High
    idx = pd.date_range("2026-10-15 09:00", periods=6, freq="5min")
    eu_data = {
        'high': [1.1100, 1.1110, 1.1105, 1.1115, 1.1120, 1.1130],
        'low':  [1.1080, 1.1090, 1.1095, 1.1100, 1.1105, 1.1110],
        'open': [1.1090, 1.1100, 1.1100, 1.1110, 1.1110, 1.1120],
        'close': [1.1100, 1.1105, 1.1110, 1.1115, 1.1115, 1.1125]
    }
    df_eu = pd.DataFrame(eu_data, index=idx)

    # Benchmark (DXY) NON fa Lower Low (fa un Higher Low -> Bearish SMT)
    dxy_data = {
        'high': [103.50, 103.40, 103.45, 103.40, 103.35, 103.45],
        'low':  [103.20, 103.10, 103.15, 103.25, 103.30, 103.35], # Non rompe 103.10!
        'open': [103.40, 103.30, 103.25, 103.30, 103.30, 103.40],
        'close': [103.30, 103.20, 103.20, 103.35, 103.35, 103.40]
    }
    df_dxy = pd.DataFrame(dxy_data, index=idx)

    smts = detect_smt_divergence(df_eu, df_dxy, relation="INVERSE", lookback=4)
    assert len(smts) > 0
    assert any(s.divergence_type == "BEARISH_SMT" for s in smts)


def test_dealing_range_calculation():

    data = {
        'open': [1.1000, 1.1020, 1.1010, 1.1040],
        'high': [1.1010, 1.1030, 1.1020, 1.1050],
        'low': [1.0990, 1.1010, 1.1000, 1.1030],
        'close': [1.1005, 1.1015, 1.1010, 1.1045],
        'volume': [100, 150, 120, 200]
    }
    df = pd.DataFrame(data, index=pd.date_range("2026-10-15 08:00", periods=4, freq="5min"))
    df_dr = calculate_dealing_range(df, lookback=4)
    
    # Range High = 1.1050, Range Low = 1.0990 -> Eq = 1.1020
    assert df_dr.iloc[-1]['equilibrium'] == pytest.approx(1.1020)
    assert bool(df_dr.iloc[-1]['is_premium']) is True
    assert bool(df_dr.iloc[-1]['is_discount']) is False


def test_webhook_bridge_validation():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # 2. Valid Sell alert in London Premium
    valid_payload = {
        "symbol": "EURUSD",
        "timeframe": "5",
        "action": "SELL",
        "price": 1.11610,
        "sl": 1.11700,
        "session": "LONDON",
        "zone": "PREMIUM",
        "reason": "Bearish FVG retest on H4 IMB"
    }
    res_valid = client.post("/webhook", json=valid_payload)
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["accepted"] is True
    assert data["status"] == "APPROVED_FOR_EXECUTION"
    assert data["sl_pips"] == 9.0
    assert data["estimated_rr"] >= CONFIG_RISK.MIN_RISK_REWARD

    # 3. Invalid Sell in Discount zone
    invalid_zone_payload = dict(valid_payload, zone="DISCOUNT")
    res_invalid_zone = client.post("/webhook", json=invalid_zone_payload)
    data_inv = res_invalid_zone.json()
    assert data_inv["accepted"] is False
    assert "Violazione Framework: Operazione SELL consentita esclusivamente in zona PREMIUM." in data_inv["rejection_reasons"]

    # 4. Invalid session
    invalid_sess_payload = dict(valid_payload, session="ASIA")
    res_invalid_sess = client.post("/webhook", json=invalid_sess_payload)
    data_sess = res_invalid_sess.json()
    assert data_sess["accepted"] is False
    assert any("Sessione 'ASIA' non consentita" in r for r in data_sess["rejection_reasons"])
