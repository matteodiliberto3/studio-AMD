"""
Modelli Pydantic per il Webhook Bridge Istituzionale A-M-D & Accumulazione Dinamica.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class TradingViewAlertPayload(BaseModel):
    symbol: str = Field(..., json_schema_extra={"example": "EURUSD"})
    timeframe: str = Field(..., json_schema_extra={"example": "M1"})
    entry_type: Literal["AMD_HORIZONTAL_BOX", "DYNAMIC_ACCUMULATION_BOS", "BREAKOUT_RETEST"]
    action: Literal["BUY", "SELL"]
    entry_price: float = Field(..., gt=0)
    sl_price: float = Field(..., gt=0)
    tp_50_eq: float = Field(..., gt=0)
    tp_structural: float = Field(..., gt=0)
    htf_key_level: str = Field(..., json_schema_extra={"example": "MTF_PREMIUM_KL"})
    passphrase: Optional[str] = None


class ValidationResult(BaseModel):
    accepted: bool
    status: str
    symbol: str
    entry_type: str
    action: str
    entry_price: float
    sl_price: float
    tp_50_eq: float
    tp_structural: float
    sl_pips: float
    structural_rr: float
    htf_key_level: str
    rejection_reasons: list[str] = []
