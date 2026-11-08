"""
Configurazione Globale del Framework Operativo FX
Meccanismo Unificato di Entry su Key Level MTF:
1. AMD_HORIZONTAL_BOX (Accumulazione Orizzontale Classica con Box M/m)
2. DYNAMIC_ACCUMULATION_BOS (Accumulazione Dinamica con Rottura dello Swing M/m)

In entrambi i casi:
Rottura di M/m con IMB -> Almeno 2 chiusure oltre M/m -> Rientro (Retest) nell'IMB -> TP su Target MTF.
"""
from dataclasses import dataclass
from datetime import time
from enum import Enum


class AccumulationType(str, Enum):
    AMD_HORIZONTAL_BOX = "AMD_HORIZONTAL_BOX"             # Box laterale orizzontale -> Manipolazione su KL -> Rottura box con IMB
    DYNAMIC_ACCUMULATION_BOS = "DYNAMIC_ACCUMULATION_BOS" # Accumulazione dinamica a swing -> Rottura di M o m con IMB


@dataclass(frozen=True)
class SessionTimingConfig:
    TIMEZONE: str = "Europe/Rome"

    # Finestra Operativa (08:00 - 20:00 CET copre Londra, NY AM e NY PM / London Close)
    ACTIVE_WINDOW_START: time = time(8, 0)
    ACTIVE_WINDOW_END: time = time(20, 0)

    # Monitoraggio Cambi di Periodo OHLC (H1 e H4)
    MONITOR_H1_OPEN_MINUTES: int = 15
    MONITOR_H4_OPEN_MINUTES: int = 30


@dataclass(frozen=True)
class HTFRangeConfig:
    # Criteri di Validazione del Range e Key Level MTF/HTF
    REQUIRE_LIQUIDITY_SWEEP: bool = True
    REQUIRE_STRUCTURAL_BREAK: bool = True
    REQUIRE_IMBALANCE_OR_REJECTION: bool = True

    # Regola Fonte 16: se >= 2 chiusure di corpo consecutive chiudono oltre il Key Level, il livello è invalidato
    MAX_CONSECUTIVE_CLOSES_BEYOND_KL: int = 2

    # Filtro 50% Equilibrium del range MTF
    EQUILIBRIUM_THRESHOLD: float = 0.50


@dataclass(frozen=True)
class ExecutionConfig:
    # Le due forme di accumulazione ammesse
    ALLOWED_ACCUMULATION_TYPES: tuple = (
        AccumulationType.AMD_HORIZONTAL_BOX,
        AccumulationType.DYNAMIC_ACCUMULATION_BOS,
    )

    # Regola di Conferma della Rottura: almeno 2 chiusure oltre M o m
    MIN_CONFIRMATION_CLOSES_BEYOND_M: int = 2

    # Parametri per la Box di Accumulazione Orizzontale
    MIN_BOX_BARS: int = 6
    MAX_BOX_BARS: int = 20
    MAX_BOX_RANGE_PIPS_M1: float = 10.0
    MAX_BOX_RANGE_PIPS_M5: float = 18.0

    # Parametri per l'Imbalance (IMB) sulla rottura di M/m
    MIN_IMB_GAP_PIPS_M1: float = 0.8
    MIN_IMB_GAP_PIPS_M5: float = 1.5
    MIN_DISPLACEMENT_BODY_RATIO: float = 0.55  # Corpo candela >= 55% del range high-low

    # Finestra massima (in barre) per attendere il rientro (retest) nell'IMB dopo le 2 chiusure
    MAX_BARS_WAIT_FOR_IMB_RETEST: int = 25

    # Gestione Stop Loss e Target MTF
    SL_BUFFER_PIPS: float = 1.0
    TARGET_1_PARTIAL_PCT: float = 0.50         # 50% del range verso il Target MTF
    TRAIL_SL_AFTER_T1: bool = True             # Sposta SL dietro il nuovo swing strutturale (MAI a breakeven sull'entry)


@dataclass(frozen=True)
class AssetConfig:
    PRIMARY_PAIRS: tuple = ("EURUSD", "GBPUSD")
    BENCHMARK_INDEX: str = "DXY"
    PIP_SIZE_DEFAULT: float = 0.0001


CONFIG_SESSION = SessionTimingConfig()
CONFIG_HTF = HTFRangeConfig()
CONFIG_EXEC = ExecutionConfig()
CONFIG_ASSET = AssetConfig()
