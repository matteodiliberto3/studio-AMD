"""
Configurazione Globale del Framework Operativo FX
Smart Money Concepts / Sequenziale A-M-D Top-Down
"""
from dataclasses import dataclass
from datetime import time
from enum import Enum


class EntryType(str, Enum):
    MANIPULATION = "ENTRY_MANIPULATION"                 # Entry 1: Aggressiva sul picco di manipolazione del Key Level
    POST_BREAKOUT_RETEST = "ENTRY_POST_BREAKOUT_RETEST" # Entry 2: Conservativa sul retest IMB / bordo rotto


@dataclass(frozen=True)
class SessionTimingConfig:
    TIMEZONE: str = "Europe/Rome"

    # Finestra Operativa Estesa (copre London Open, New York AM, New York PM / London Close 16:00-20:00)
    ACTIVE_WINDOW_START: time = time(8, 0)
    ACTIVE_WINDOW_END: time = time(20, 0)

    # Monitoraggio Cambi di Periodo OHLC
    MONITOR_H1_OPEN_MINUTES: int = 15     # Primi 15 min di ogni nuova candela H1
    MONITOR_H4_OPEN_MINUTES: int = 30     # Primi 30 min di ogni nuova candela H4


@dataclass(frozen=True)
class HTFRangeConfig:
    # Criteri di Validazione HTF Range (H4 / Daily)
    REQUIRE_LIQUIDITY_SWEEP: bool = True       # Presa di liquidità (PDH/PDL o Swing High/Low)
    REQUIRE_STRUCTURAL_BREAK: bool = True      # Rottura di struttura con displacement
    REQUIRE_IMBALANCE_OR_REJECTION: bool = True # Presenza di Imbalance (FVG) o Rejection Block

    # Premium / Discount: soglia 50% Equilibrium del range HTF
    EQUILIBRIUM_THRESHOLD: float = 0.50


@dataclass(frozen=True)
class ExecutionConfig:
    # Tipologie di Ingresso Ammesse
    ALLOWED_ENTRIES: tuple = (EntryType.MANIPULATION, EntryType.POST_BREAKOUT_RETEST)

    # Parametri LTF
    MIN_ACCUMULATION_BARS: int = 8             # Minimo candele per definire accumulo M1/M5
    MAX_ACCUMULATION_PIPS_M1: float = 12.0     # Ampiezza max dell'accumulo su M1
    MIN_BREAKOUT_IMB_PIPS: float = 1.2         # Gap minimo per Imbalance LTF valido

    # Gestione Stop Loss
    SL_BUFFER_PIPS: float = 1.5                # Buffer oltre l'estremo di manipolazione

    # Gestione Target Contestuali (NO R:R FISSO)
    TARGET_1_PARTIAL_PCT: float = 0.50         # Target 1 di sicurezza al 50% del leg/range
    CLOSE_PARTIAL_SIZE_PCT: float = 0.50       # Chiudi il 50% della posizione a Target 1
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
