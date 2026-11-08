"""
Configurazione Globale del Framework Operativo FX
Smart Money Concepts / A-M-D / Multi-Timeframe Engine
"""
from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class SessionConfig:
    # Timezone di default: CET / UTC+1
    TIMEZONE: str = "Europe/Rome"

    # London Killzone (CET)
    LONDON_START: time = time(8, 0)
    LONDON_END: time = time(11, 30)

    # New York Killzone (CET)
    NY_START: time = time(13, 0)
    NY_END: time = time(16, 30)

    # Asia Range (per accumulo/liquidità notturna)
    ASIA_START: time = time(0, 0)
    ASIA_END: time = time(7, 0)


@dataclass(frozen=True)
class AssetConfig:
    PRIMARY_PAIRS: tuple = ("EURUSD", "GBPUSD")
    BENCHMARK_INDEX: str = "DXY"
    
    # Pip size standard
    PIP_SIZE_EU: float = 0.0001
    PIP_SIZE_GU: float = 0.0001
    INDEX_POINT_DXY: float = 0.01


@dataclass(frozen=True)
class RiskConfig:
    RISK_PER_TRADE_PCT: float = 1.0     # 1% per operazione
    MIN_RISK_REWARD: float = 2.5        # Minimo 1:2.5
    MAX_SL_PIPS_M1: float = 10.0        # Max Stop Loss consentito per micro-entry M1 (es. 9-10 pips)
    PIP_VALUE_STANDARD_LOT: float = 10.0 # Valore pip standard su EURUSD / GBPUSD ($10 per lotto)


@dataclass(frozen=True)
class StrategyConfig:
    # Parametri Multi-Timeframe
    HTF_TIMEFRAME: str = "4h"           # H4 per Imbalance/POI principale
    DEFAULT_MTF: str = "5min"           # M5 default di sessione
    PRECISION_LTF: str = "1min"         # M1 "radiografia" di precisione
    
    # SMT Divergence
    ENABLE_SMT_FILTER: bool = True
    
    # FVG threshold in pips (minimo gap per essere considerato FVG valido)
    MIN_FVG_PIPS: float = 1.5
    
    # Dealing Range lookback candele per il calcolo di Premium / Discount
    DEALING_RANGE_LOOKBACK: int = 50


CONFIG_SESSION = SessionConfig()
CONFIG_ASSET = AssetConfig()
CONFIG_RISK = RiskConfig()
CONFIG_STRATEGY = StrategyConfig()
