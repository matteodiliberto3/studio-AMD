"""
Modulo Indicatori Istituzionali:
- Validazione Key Level HTF (Imbalance H4/D1, Rejection Block, Liquidity Sweeps)
- Dealing Range HTF & Equilibrium (Filtro Premium / Discount 50%)
- Finestre Operative Estese & Monitoraggio Cambi Periodo H1 / H4
- Rilevamento Accumulazione LTF (Base di liquidità)
- Rilevamento Manipolazione (Sweep) e Imbalance di Rottura (FVG)
- Divergenza SMT con DXY
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from config import CONFIG_ASSET, CONFIG_HTF, CONFIG_SESSION


@dataclass
class HTFKeyLevel:
    level_type: str         # 'IMBALANCE_H4', 'REJECTION_BLOCK', 'LIQUIDITY_SWEEP'
    timestamp: pd.Timestamp
    top: float
    bottom: float
    midpoint: float
    direction: str          # 'BEARISH' o 'BULLISH'
    mitigated: bool = False


@dataclass
class AccumulationZone:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    high: float
    low: float
    midpoint: float
    range_pips: float
    timeframe: str          # 'M5' o 'M1'


@dataclass
class SMTEvent:
    timestamp: pd.Timestamp
    divergence_type: str    # 'BEARISH_SMT' o 'BULLISH_SMT'
    primary_asset: str
    correlated_asset: str
    description: str


def is_active_operating_window(dt: pd.Timestamp) -> Tuple[bool, str]:
    """
    Verifica se il timestamp si trova nella finestra operativa (08:00 - 20:00 CET)
    e identifica se coincide con un cambio di periodo H1 o H4.
    """
    t = dt.time()
    if not (CONFIG_SESSION.ACTIVE_WINDOW_START <= t <= CONFIG_SESSION.ACTIVE_WINDOW_END):
        return False, "OUT_OF_WINDOW"

    minute = dt.minute
    hour = dt.hour

    is_h4_change = (hour % 4 == 0) and (minute < CONFIG_SESSION.MONITOR_H4_OPEN_MINUTES)
    is_h1_change = minute < CONFIG_SESSION.MONITOR_H1_OPEN_MINUTES

    if is_h4_change:
        return True, "H4_PERIOD_OPEN"
    elif is_h1_change:
        return True, "H1_PERIOD_OPEN"
    else:
        return True, "IN_SESSION"


def calculate_htf_range_and_equilibrium(df_htf: pd.DataFrame, lookback: int = 40) -> pd.DataFrame:
    """
    Calcola il Dealing Range HTF (H4 o D1) e l'Equilibrium al 50%:
    - Short consentiti SOLO se close > equilibrium (Premium)
    - Long consentiti SOLO se close < equilibrium (Discount)
    """
    res = df_htf.copy()
    res['htf_high'] = res['high'].rolling(lookback, min_periods=3).max()
    res['htf_low'] = res['low'].rolling(lookback, min_periods=3).min()
    res['equilibrium'] = (res['htf_high'] + res['htf_low']) / 2.0
    res['is_premium'] = res['close'] > res['equilibrium']
    res['is_discount'] = res['close'] < res['equilibrium']
    return res


def find_htf_key_levels(df_h4: pd.DataFrame, pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT) -> List[HTFKeyLevel]:
    """
    Identifica i Key Levels HTF su H4:
    1. Imbalance / FVG H4 non mitigati
    2. Rejection Blocks (lunghe ombre di reiezione e sweep)
    """
    levels: List[HTFKeyLevel] = []
    if len(df_h4) < 3:
        return levels

    highs = df_h4['high'].values
    lows = df_h4['low'].values
    opens = df_h4['open'].values
    closes = df_h4['close'].values
    times = df_h4.index

    for i in range(2, len(df_h4)):
        # Bearish FVG H4
        if lows[i - 2] - highs[i] >= 2.0 * pip_size:
            top = lows[i - 2]
            bottom = highs[i]
            levels.append(HTFKeyLevel(
                level_type='IMBALANCE_H4',
                timestamp=times[i - 1],
                top=top,
                bottom=bottom,
                midpoint=(top + bottom) / 2.0,
                direction='BEARISH'
            ))

        # Bullish FVG H4
        elif lows[i] - highs[i - 2] >= 2.0 * pip_size:
            top = lows[i]
            bottom = highs[i - 2]
            levels.append(HTFKeyLevel(
                level_type='IMBALANCE_H4',
                timestamp=times[i - 1],
                top=top,
                bottom=bottom,
                midpoint=(top + bottom) / 2.0,
                direction='BULLISH'
            ))

        # Rejection Block / Sweep di liquidità H4
        candle_range = highs[i] - lows[i]
        body = abs(closes[i] - opens[i])
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]

        if candle_range >= 10.0 * pip_size:
            if upper_wick >= body * 2.0:
                levels.append(HTFKeyLevel(
                    level_type='REJECTION_BLOCK',
                    timestamp=times[i],
                    top=highs[i],
                    bottom=highs[i] - upper_wick,
                    midpoint=highs[i] - (upper_wick / 2.0),
                    direction='BEARISH'
                ))
            elif lower_wick >= body * 2.0:
                levels.append(HTFKeyLevel(
                    level_type='REJECTION_BLOCK',
                    timestamp=times[i],
                    top=lows[i] + lower_wick,
                    bottom=lows[i],
                    midpoint=lows[i] + (lower_wick / 2.0),
                    direction='BULLISH'
                ))

    return levels


def detect_accumulation_zone(
    df: pd.DataFrame,
    end_idx: int,
    lookback: int = 15,
    max_pips: float = 12.0,
    timeframe: str = "M1",
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> Optional[AccumulationZone]:
    """
    Rileva se un intervallo di barre costituisce una vera Accumulazione LTF (range compresso).
    Questa zona funge ESCLUSIVAMENTE da base di liquidità per la successiva manipolazione.
    """
    if end_idx < lookback:
        return None

    slice_df = df.iloc[end_idx - lookback:end_idx]
    high = slice_df['high'].max()
    low = slice_df['low'].min()
    range_pips = (high - low) / pip_size

    if range_pips <= max_pips and range_pips >= 2.5:
        return AccumulationZone(
            start_time=slice_df.index[0],
            end_time=slice_df.index[-1],
            high=round(high, 5),
            low=round(low, 5),
            midpoint=round((high + low) / 2.0, 5),
            range_pips=round(range_pips, 1),
            timeframe=timeframe
        )
    return None


def detect_smt_divergence(
    primary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback: int = 5,
    primary_name: str = "EURUSD",
    benchmark_name: str = "DXY"
) -> List[SMTEvent]:
    """
    Rileva Divergenze SMT con il DXY (relazione inversa):
    - Bearish SMT: Primary fa Higher High, DXY fa Higher Low (mancata rottura dei minimi DXY).
    - Bullish SMT: Primary fa Lower Low, DXY fa Lower High (mancata rottura dei massimi DXY).
    """
    events: List[SMTEvent] = []
    common_idx = primary_df.index.intersection(benchmark_df.index)
    if len(common_idx) < lookback + 2:
        return events

    p_df = primary_df.loc[common_idx]
    b_df = benchmark_df.loc[common_idx]

    p_highs = p_df['high'].values
    p_lows = p_df['low'].values
    b_highs = b_df['high'].values
    b_lows = b_df['low'].values
    times = common_idx

    for i in range(lookback, len(common_idx)):
        p_prev_max = np.max(p_highs[i - lookback:i])
        p_prev_min = np.min(p_lows[i - lookback:i])
        b_prev_max = np.max(b_highs[i - lookback:i])
        b_prev_min = np.min(b_lows[i - lookback:i])

        curr_p_h, curr_p_l = p_highs[i], p_lows[i]
        curr_b_h, curr_b_l = b_highs[i], b_lows[i]

        if (curr_p_h > p_prev_max) and (curr_b_l > b_prev_min):
            events.append(SMTEvent(
                timestamp=times[i],
                divergence_type="BEARISH_SMT",
                primary_asset=primary_name,
                correlated_asset=benchmark_name,
                description=f"{primary_name} HH vs {benchmark_name} HL"
            ))
        elif (curr_p_l < p_prev_min) and (curr_b_h < b_prev_max):
            events.append(SMTEvent(
                timestamp=times[i],
                divergence_type="BULLISH_SMT",
                primary_asset=primary_name,
                correlated_asset=benchmark_name,
                description=f"{primary_name} LL vs {benchmark_name} LH"
            ))

    return events
