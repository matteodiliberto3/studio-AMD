"""
Suite di Test Automatizzata per il Meccanismo di Entry Istituzionale e Catena Frattale:
1. Validazione Finestre Operative e Cambi Periodo (08:00-20:00 CET)
2. Mappatura del Range MTF ($M$ ed $m$) e Ritracciamento di Fibonacci (0.0, 0.50 Equilibrium, 1.0)
3. Regola Fonte 16: Invalidazione del Key Level con >= 2 chiusure consecutive oltre il KL
4. Rottura di M/m con IMB + almeno 2 chiusure di conferma oltre M/m
5. Selezione del Primo Key Level MTF non mitigato come Target (Ottimizzazione R:R)
6. Validazione Webhook Bridge per AMD_HORIZONTAL_BOX e DYNAMIC_ACCUMULATION_BOS
"""
import pandas as pd
import pytest
from starlette.testclient import TestClient

from core.indicators import (
    HTFKeyLevel,
    MTFFractalTarget,
    detect_breakout_with_imb_and_two_closes,
    is_active_operating_window,
    is_key_level_invalidated_by_closes,
    map_fibonacci_range,
    select_first_mtf_key_level_target
)
from webhook_bridge.app import app


def test_extended_operating_window_and_period_changes():
    ts_h4 = pd.Timestamp("2026-10-15 08:05:00", tz="Europe/Rome")
    is_active, tag = is_active_operating_window(ts_h4)
    assert is_active is True
    assert tag == "H4_PERIOD_OPEN"

    ts_h1 = pd.Timestamp("2026-10-15 17:05:00", tz="Europe/Rome")
    is_active, tag = is_active_operating_window(ts_h1)
    assert is_active is True
    assert tag == "H1_PERIOD_OPEN"

    ts_out = pd.Timestamp("2026-10-15 22:30:00", tz="Europe/Rome")
    is_active, tag = is_active_operating_window(ts_out)
    assert is_active is False
    assert tag == "OUT_OF_WINDOW"


def test_fibonacci_range_mapping_and_discount_premium_filter():
    fib = map_fibonacci_range(high_M=1.16000, low_m=1.15000, timeframe="H1")
    assert fib.fib_05_equilibrium == pytest.approx(1.15500)
    assert fib.range_pips == pytest.approx(100.0)
    assert fib.get_zone(1.15200) == "DISCOUNT"
    assert fib.fib_level_of_price(1.15200) == pytest.approx(0.20)
    assert fib.get_zone(1.15800) == "PREMIUM"
    assert fib.fib_level_of_price(1.15800) == pytest.approx(0.80)


def test_key_level_invalidation_rule_source_16():
    kl = HTFKeyLevel(
        level_type="MTF_IMBALANCE",
        timeframe="H1",
        timestamp=pd.Timestamp("2026-10-15 08:00:00", tz="Europe/Rome"),
        top=1.11600,
        bottom=1.11550,
        midpoint=1.11575,
        direction="BEARISH",
        range_high=1.11700,
        range_low=1.11200,
        equilibrium_50=1.11450,
        fib_position=0.75,
        target_mtf=1.11200
    )

    df_valid = pd.DataFrame(
        {"close": [1.11580, 1.11610, 1.11540, 1.11520]},
        index=pd.date_range("2026-10-15 09:00", periods=4, freq="1min", tz="Europe/Rome")
    )
    assert is_key_level_invalidated_by_closes(df_valid, kl) is False

    df_invalid = pd.DataFrame(
        {"close": [1.11580, 1.11615, 1.11630, 1.11625]},
        index=pd.date_range("2026-10-15 09:00", periods=4, freq="1min", tz="Europe/Rome")
    )
    assert is_key_level_invalidated_by_closes(df_invalid, kl) is True


def test_breakout_with_imb_and_two_closes_beyond_m():
    idx = pd.date_range("2026-10-15 10:00", periods=6, freq="1min", tz="Europe/Rome")
    df = pd.DataFrame({
        "open":  [1.11600, 1.11620, 1.11565, 1.11510, 1.11490, 1.11500],
        "high":  [1.11650, 1.11625, 1.11570, 1.11535, 1.11510, 1.11545],
        "low":   [1.11590, 1.11560, 1.11505, 1.11480, 1.11470, 1.11490],
        "close": [1.11620, 1.11565, 1.11510, 1.11490, 1.11500, 1.11520]
    }, index=idx)

    res = detect_breakout_with_imb_and_two_closes(
        df, manip_idx=0, level_M_or_m=1.11550, direction="SELL", timeframe="M1"
    )
    assert res is not None
    imb, confirm_idx, _ = res
    assert imb.direction == "BEARISH"
    assert imb.top == pytest.approx(1.11560)
    assert imb.bottom == pytest.approx(1.11535)
    assert confirm_idx == 3


def test_select_first_mtf_key_level_target():
    kl = HTFKeyLevel(
        level_type="MTF_IMBALANCE",
        timeframe="H1",
        timestamp=pd.Timestamp("2026-10-15 08:00:00", tz="Europe/Rome"),
        top=1.15200,
        bottom=1.15100,
        midpoint=1.15150,
        direction="BULLISH",
        range_high=1.15800,
        range_low=1.15000,
        equilibrium_50=1.15400,
        fib_position=0.18,
        target_mtf=1.15800
    )
    targets = [
        MTFFractalTarget(
            timeframe="M15",
            target_type="FIRST_MTF_IMB",
            timestamp=pd.Timestamp("2026-10-15 08:30:00", tz="Europe/Rome"),
            direction_for_exit="BUY_TARGET",
            price_level=1.15420,
            zone_top=1.15450,
            zone_bottom=1.15420
        )
    ]
    tp_price, tp_name = select_first_mtf_key_level_target(
        entry_price=1.15220,
        sl_pips=6.0,
        direction="BUY",
        kl=kl,
        active_mtf_targets=targets
    )
    assert tp_price == pytest.approx(1.15420)
    assert tp_name == "M15_FIRST_MTF_IMB"


def test_webhook_bridge_contextual_validation():
    client = TestClient(app)

    valid_payload = {
        "symbol": "EURUSD",
        "timeframe": "M1",
        "entry_type": "AMD_HORIZONTAL_BOX",
        "action": "SELL",
        "entry_price": 1.11535,
        "sl_price": 1.11625,
        "tp_50_eq": 1.11300,
        "tp_structural": 1.11100,
        "htf_key_level": "MTF_PREMIUM_KL"
    }
    res_valid = client.post("/webhook", json=valid_payload)
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["accepted"] is True
    assert data["entry_type"] == "AMD_HORIZONTAL_BOX"
