"""
FastAPI Webhook Server:
Riceve gli alert JSON contestuali da TradingView (Pine Script A-M-D)
e applica il filtraggio istituzionale (SL ancorato alla manipolazione, Target 50% ed esterno).
"""
from datetime import datetime
import json
from pathlib import Path
from fastapi import FastAPI
from webhook_bridge.models import TradingViewAlertPayload, ValidationResult

app = FastAPI(
    title="FX Institutional A-M-D Bridge",
    description="Bridge per la ricezione e validazione di segnali contestuali A-M-D",
    version="2.0.0"
)

LOG_FILE = Path("webhook_bridge/alerts_history.json")


def save_alert(data: dict):
    history = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append({"received_at": datetime.now().isoformat(), "payload": data})
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "server_time": datetime.now().isoformat()}


@app.post("/webhook", response_model=ValidationResult)
async def receive_tradingview_alert(payload: TradingViewAlertPayload):
    rejections = []
    pip_size = 0.0001 if "JPY" not in payload.symbol.upper() else 0.01

    # 1. Validazione Direzione e Posizionamento dello Stop Loss
    if payload.action == "SELL":
        if payload.sl_price <= payload.entry_price:
            rejections.append("Violazione: Per SELL lo Stop Loss deve essere rigorosamente sopra l'entry (estremo manipolazione).")
        if payload.tp_structural >= payload.entry_price or payload.tp_50_eq >= payload.entry_price:
            rejections.append("Violazione: I target SELL devono trovarsi sotto il prezzo di entrata.")
    elif payload.action == "BUY":
        if payload.sl_price >= payload.entry_price:
            rejections.append("Violazione: Per BUY lo Stop Loss deve essere rigorosamente sotto l'entry (estremo manipolazione).")
        if payload.tp_structural <= payload.entry_price or payload.tp_50_eq <= payload.entry_price:
            rejections.append("Violazione: I target BUY devono trovarsi sopra il prezzo di entrata.")

    sl_dist = abs(payload.entry_price - payload.sl_price)
    sl_pips = round(sl_dist / pip_size, 1)

    tp_dist = abs(payload.tp_structural - payload.entry_price)
    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

    # Controllo R:R strutturale minimo (1:2 minimo atteso su target strutturale)
    if rr < 1.8:
        rejections.append(f"Risk:Reward strutturale non sufficiente ({rr}R < target 1.8R minimo).")

    accepted = len(rejections) == 0
    status = "APPROVED_INSTITUTIONAL_ENTRY" if accepted else "REJECTED_BY_FRAMEWORK"

    result = ValidationResult(
        accepted=accepted,
        status=status,
        symbol=payload.symbol,
        entry_type=payload.entry_type,
        action=payload.action,
        entry_price=payload.entry_price,
        sl_price=payload.sl_price,
        tp_50_eq=payload.tp_50_eq,
        tp_structural=payload.tp_structural,
        sl_pips=sl_pips,
        structural_rr=rr,
        htf_key_level=payload.htf_key_level,
        rejection_reasons=rejections
    )

    save_alert({
        "alert": payload.model_dump(),
        "validation": result.model_dump()
    })

    print(f"\n[ALERT A-M-D] {payload.symbol} {payload.action} ({payload.entry_type}) | Esito: {status} | R:R={rr}R | SL={sl_pips}p")
    return result
