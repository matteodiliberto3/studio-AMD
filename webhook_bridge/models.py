"""
Modelli Pydantic per la validazione dei payload Webhook da TradingView.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class TradingViewAlertPayload(BaseModel):
    symbol: str = Field(..., json_schema_extra={"example": "EURUSD"})
    timeframe: str = Field(..., json_schema_extra={"example": "5"})
    action: Literal["BUY", "SELL"]
    price: float = Field(..., gt=0)
    sl: float = Field(..., gt=0)
    tp: Optional[float] = None
    session: str = Field(..., json_schema_extra={"example": "LONDON"})
    zone: Literal["PREMIUM", "DISCOUNT"]
    smt_divergence: Optional[bool] = Field(default=False, json_schema_extra={"example": True})
    reason: str = Field(..., json_schema_extra={"example": "Bearish FVG in Killzone"})
    passphrase: Optional[str] = None


class ValidationResult(BaseModel):
    accepted: bool
    status: str
    symbol: str
    action: str
    entry_price: float
    sl: float
    tp: float
    sl_pips: float
    estimated_rr: float
    rejection_reasons: list[str] = []
