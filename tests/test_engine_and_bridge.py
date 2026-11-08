"""
Suite di Test Automatizzata per il Framework A-M-D Rifattorizzato:
1. Validazione Finestre Operative e Cambi Periodo (08:00-20:00 CET)
2. Dealing Range HTF ed Equilibrium 50%
3. Validazione Webhook Bridge con i campi contestuali A-M-D
"""
import pandas as pd
import pytest
from starlette.testclient import TestClient

from core.indicators import (
    calculate_htf_range_and_equilibrium,
    is_active_operating_window
)
from webhook_bridge.app import app


def test_extended_operating_window_and_period_changes():
    # 08:05 CET -> In finestra operativa (H4 period open)
    ts_h4 = pd.Timestamp("2026-10-15 08:05:00", tz="Europe/Rome")
    is_active, tag = is_active_operating_window(ts_h4)
    assert is_active is True
    assert tag == "H4_PERIOD_OPEN"

    # 17:05 CET -> In finestra estesa New York PM / London Close (H1 period open)
    ts_h1 = pd.Timestamp("2026-10-15 17:05:00", tz="Europe/Rome")
    is_active, tag = is_active_operating_window(ts_h1)
    assert is_active is True
    assert tag == "H1_PERIOD_OPEN"

    # 22:30 CET -> Fuori finestra operativa
    ts_out = pd.Timestamp("2026-10-15 22:30:00", tz="Europe/Rome")
    is_active, tag = is_active_operating_window(ts_out)
    assert is_active is False
    assert tag == "OUT_OF_WINDOW"


def test_htf_dealing_range_equilibrium():
    data = {
        'open': [1.1000, 1.1050, 1.1100, 1.1150],
        'high': [1.1050, 1.1100, 1.1150, 1.1200],
        'low':  [1.0950, 1.1000, 1.1050, 1.1100],
        'close': [1.1020, 1.1080, 1.1120, 1.1160]
    }
    df = pd.DataFrame(data, index=pd.date_range("2026-10-15", periods=4, freq="4h"))
    df_dr = calculate_htf_range_and_equilibrium(df, lookback=4)

    # Range High = 1.1200, Range Low = 1.0950 -> Eq = 1.1075
    assert df_dr.iloc[-1]['equilibrium'] == pytest.approx(1.1075)
    # Close 1.1160 > 1.1075 -> Premium Zone (solo Short consentiti)
    assert bool(df_dr.iloc[-1]['is_premium']) is True
    assert bool(df_dr.iloc[-1]['is_discount']) is False


def test_webhook_bridge_contextual_validation():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # 2. Alert Valido: Breakout Retest Sell con SL oltre la manipolazione
    valid_payload = {
        "symbol": "EURUSD",
        "timeframe": "M1",
        "entry_type": "BREAKOUT_RETEST",
        "action": "SELL",
        "entry_price": 1.11530,
        "sl_price": 1.11620,         # Estremo manipolazione (9 pips)
        "tp_50_eq": 1.11300,         # 50% Equilibrium
        "tp_structural": 1.11100,    # Target liquidità esterna (43 pips -> 4.77 R)
        "htf_key_level": "H4_PREMIUM_IMB"
    }
    res_valid = client.post("/webhook", json=valid_payload)
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["accepted"] is True
    assert data["status"] == "APPROVED_INSTITUTIONAL_ENTRY"
    assert data["sl_pips"] == 9.0
    assert data["structural_rr"] >= 4.0

    # 3. Alert Invalido: Stop Loss posizionato male (sotto l'entry su un SELL)
    invalid_sl_payload = dict(valid_payload, sl_price=1.11400)
    res_inv = client.post("/webhook", json=invalid_sl_payload)
    data_inv = res_inv.json()
    assert data_inv["accepted"] is False
    assert any("Stop Loss deve essere rigorosamente sopra l'entry" in r for r in data_inv["rejection_reasons"])
