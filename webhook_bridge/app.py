"""
FastAPI Webhook Server:
Riceve gli alert JSON da TradingView (Pine Script), valida i vincoli istituzionali
(Killzone, Premium/Discount, R:R minimo, Stop Loss) e registra/inoltra il segnale.
"""
from datetime import datetime
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import CONFIG_RISK
from webhook_bridge.models import TradingViewAlertPayload, ValidationResult

app = FastAPI(
    title="FX Institutional Webhook Bridge",
    description="Bridge per la ricezione e validazione automatica di segnali da TradingView ad Antigravity",
    version="1.0.0"
)

LOG_FILE = Path("webhook_bridge/alerts_history.json")


def save_alert_to_history(data: dict):
    history = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    
    history.append({
        "received_at": datetime.now().isoformat(),
        "payload": data
    })
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


@app.get("/health")
def health():
    return {"status": "ok", "server_time": datetime.now().isoformat()}


@app.post("/webhook", response_model=ValidationResult)
async def receive_tradingview_alert(payload: TradingViewAlertPayload):
    """
    Endpoint Webhook invocato da TradingView:
    Valida le regole operative secondo il framework:
    1. Sell valido solo in zona Premium
    2. Buy valido solo in zona Discount
    3. Sessione limitata a London o New York
    4. Calcolo automatico di TP in base al Risk:Reward minimo (1:2.5 / 1:3) se non fornito
    """
    rejections = []

    # 1. Regola Premium / Discount
    if payload.action == "SELL" and payload.zone != "PREMIUM":
        rejections.append("Violazione Framework: Operazione SELL consentita esclusivamente in zona PREMIUM.")
        
    if payload.action == "BUY" and payload.zone != "DISCOUNT":
        rejections.append("Violazione Framework: Operazione BUY consentita esclusivamente in zona DISCOUNT.")

    # 2. Regola Killzone
    if payload.session.upper() not in ["LONDON", "NEW_YORK"]:
        rejections.append(f"Violazione Framework: Sessione '{payload.session}' non consentita (consentite: LONDON, NEW_YORK).")

    # Calcolo SL e TP
    pip_factor = 0.0001 if "JPY" not in payload.symbol.upper() else 0.01
    sl_distance = abs(payload.price - payload.sl)
    sl_pips = round(sl_distance / pip_factor, 1)

    # Calcolo automatico TP minimo se non specificato
    min_tp_distance = sl_distance * CONFIG_RISK.MIN_RISK_REWARD
    if payload.action == "SELL":
        calculated_tp = payload.tp if payload.tp else (payload.price - min_tp_distance)
    else:
        calculated_tp = payload.tp if payload.tp else (payload.price + min_tp_distance)

    tp_distance = abs(calculated_tp - payload.price)
    rr = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0.0

    if rr < CONFIG_RISK.MIN_RISK_REWARD:
        rejections.append(f"Risk:Reward insufficiente ({rr}R < target {CONFIG_RISK.MIN_RISK_REWARD}R).")

    accepted = len(rejections) == 0
    status_label = "APPROVED_FOR_EXECUTION" if accepted else "REJECTED_BY_FRAMEWORK"

    result = ValidationResult(
        accepted=accepted,
        status=status_label,
        symbol=payload.symbol,
        action=payload.action,
        entry_price=payload.price,
        sl=payload.sl,
        tp=round(calculated_tp, 5),
        sl_pips=sl_pips,
        estimated_rr=rr,
        rejection_reasons=rejections
    )

    # Log persistente dell'evento
    save_alert_to_history({
        "alert": payload.model_dump(),
        "validation": result.model_dump()
    })

    print(f"\n[ALERT WEBHOOK] {payload.symbol} {payload.action} @ {payload.price} | Esito: {status_label}")
    if not accepted:
        print(f"  Motivi rifiuto: {rejections}")

    return result
