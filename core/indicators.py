"""
Modulo Indicatori Istituzionali:
- Fair Value Gap (FVG / Imbalance)
- Dealing Range & Premium/Discount (Equilibrium 50%)
- Killzones (London, New York)
- Swings & Market Structure Shift (MSS)
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from config import CONFIG_SESSION, CONFIG_STRATEGY


@dataclass
class FVG:
    index: int
    timestamp: pd.Timestamp
    fvg_type: str        # 'BULLISH' o 'BEARISH'
    top: float
    bottom: float
    midpoint: float
    mitigated: bool = False
    mitigated_at: Optional[pd.Timestamp] = None


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    swing_type: str      # 'HIGH' o 'LOW'


def is_in_killzone(dt: pd.Timestamp) -> Tuple[bool, str]:
    """
    Verifica se un timestamp cade all'interno della London o New York Killzone.
    Ritorna: (in_killzone: bool, session_name: str)
    """
    t = dt.time()
    
    # London Killzone
    if CONFIG_SESSION.LONDON_START <= t <= CONFIG_SESSION.LONDON_END:
        return True, "LONDON"
        
    # New York Killzone
    if CONFIG_SESSION.NY_START <= t <= CONFIG_SESSION.NY_END:
        return True, "NEW_YORK"
        
    return False, "OUT_OF_SESSION"


def is_in_asia_range(dt: pd.Timestamp) -> bool:
    """Verifica se il timestamp cade nel range Asiatico."""
    t = dt.time()
    return CONFIG_SESSION.ASIA_START <= t <= CONFIG_SESSION.ASIA_END


def calculate_dealing_range(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Calcola il Dealing Range corrente su una finestra mobile (lookback):
    - Range High (massimo recente)
    - Range Low (minimo recente)
    - Equilibrium (50%)
    - Premium (>50%) / Discount (<50%)
    """
    res = df.copy()
    res['range_high'] = res['high'].rolling(lookback).max()
    res['range_low'] = res['low'].rolling(lookback).min()
    res['equilibrium'] = (res['range_high'] + res['range_low']) / 2.0
    
    res['is_premium'] = res['close'] > res['equilibrium']
    res['is_discount'] = res['close'] < res['equilibrium']
    return res


def find_fair_value_gaps(df: pd.DataFrame, pip_size: float = 0.0001, min_gap_pips: float = 1.0) -> List[FVG]:
    """
    Rileva i Fair Value Gap (3-candle pattern):
    - Bullish FVG: Low candela 3 > High candela 1 (Gap = Low[i] - High[i-2])
    - Bearish FVG: High candela 3 < Low candela 1 (Gap = Low[i-2] - High[i])
    Tiene traccia dell'eventuale mitigazione successiva.
    """
    fvgs: List[FVG] = []
    min_gap = min_gap_pips * pip_size

    if len(df) < 3:
        return fvgs

    highs = df['high'].values
    lows = df['low'].values
    times = df.index

    for i in range(2, len(df)):
        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]

        # Bullish FVG
        if c3_low - c1_high >= min_gap:
            fvg = FVG(
                index=i - 1,
                timestamp=times[i - 1],
                fvg_type='BULLISH',
                top=c3_low,
                bottom=c1_high,
                midpoint=(c3_low + c1_high) / 2.0,
            )
            # Controllo mitigazione futura
            for j in range(i + 1, len(df)):
                if lows[j] <= fvg.midpoint: # Consequent Encroachment (50%) mitigato
                    fvg.mitigated = True
                    fvg.mitigated_at = times[j]
                    break
            fvgs.append(fvg)

        # Bearish FVG
        elif c1_low - c3_high >= min_gap:
            fvg = FVG(
                index=i - 1,
                timestamp=times[i - 1],
                fvg_type='BEARISH',
                top=c1_low,
                bottom=c3_high,
                midpoint=(c1_low + c3_high) / 2.0,
            )
            for j in range(i + 1, len(df)):
                if highs[j] >= fvg.midpoint:
                    fvg.mitigated = True
                    fvg.mitigated_at = times[j]
                    break
            fvgs.append(fvg)

    return fvgs


def detect_swings(df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> List[SwingPoint]:
    """Rileva swing high e swing low locali."""
    swings: List[SwingPoint] = []
    highs = df['high'].values
    lows = df['low'].values
    times = df.index

    for i in range(left_bars, len(df) - right_bars):
        curr_high = highs[i]
        curr_low = lows[i]

        # Swing High
        if all(curr_high > highs[i - k] for k in range(1, left_bars + 1)) and \
           all(curr_high > highs[i + k] for k in range(1, right_bars + 1)):
            swings.append(SwingPoint(i, times[i], curr_high, 'HIGH'))

        # Swing Low
        if all(curr_low < lows[i - k] for k in range(1, left_bars + 1)) and \
           all(curr_low < lows[i + k] for k in range(1, right_bars + 1)):
            swings.append(SwingPoint(i, times[i], curr_low, 'LOW'))

    return swings


def detect_market_structure_shift(
    df: pd.DataFrame, 
    swings: List[SwingPoint]
) -> List[dict]:
    """
    Identifica i Market Structure Shifts (MSS) con displacement:
    - Bearish MSS: Candela decisa (close) sotto l'ultimo Swing Low rilevante
    - Bullish MSS: Candela decisa (close) sopra l'ultimo Swing High rilevante
    """
    shifts = []
    swing_highs = [s for s in swings if s.swing_type == 'HIGH']
    swing_lows = [s for s in swings if s.swing_type == 'LOW']

    closes = df['close'].values
    times = df.index

    # Cerca rottura strutturale
    for sl in swing_lows:
        # Cerca la prima candela successiva che chiude sotto il minimo dello swing low
        subsequent_closes = np.where((df.index > sl.timestamp) & (closes < sl.price))[0]
        if len(subsequent_closes) > 0:
            first_break_idx = subsequent_closes[0]
            shifts.append({
                'type': 'BEARISH_MSS',
                'broken_swing': sl,
                'break_time': times[first_break_idx],
                'break_index': first_break_idx,
                'break_price': closes[first_break_idx]
            })

    for sh in swing_highs:
        subsequent_closes = np.where((df.index > sh.timestamp) & (closes > sh.price))[0]
        if len(subsequent_closes) > 0:
            first_break_idx = subsequent_closes[0]
            shifts.append({
                'type': 'BULLISH_MSS',
                'broken_swing': sh,
                'break_time': times[first_break_idx],
                'break_index': first_break_idx,
                'break_price': closes[first_break_idx]
            })

@dataclass
class SMTDivergence:
    timestamp: pd.Timestamp
    divergence_type: str    # 'BEARISH_SMT' o 'BULLISH_SMT'
    primary_asset: str
    correlated_asset: str
    relation: str           # 'INVERSE' (DXY) o 'DIRECT' (GU/EU)
    description: str


def detect_smt_divergence(
    primary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    relation: str = "INVERSE",
    lookback: int = 5,
    primary_name: str = "EU",
    benchmark_name: str = "DXY"
) -> List[SMTDivergence]:
    """
    Rileva le divergenze SMT (Smart Money Technique):
    - Relazione INVERSA (EU/GU vs DXY):
      * Bearish SMT: EU/GU fa Higher High ma DXY fa Higher Low (non rompe il minimo).
      * Bullish SMT: EU/GU fa Lower Low ma DXY fa Lower High (non rompe il massimo).
    - Relazione DIRETTA (EU vs GU):
      * Bearish SMT: EU fa Higher High ma GU fallisce e fa Lower High (o viceversa).
      * Bullish SMT: EU fa Lower Low ma GU fallisce e fa Higher Low (o viceversa).
    """
    divergences: List[SMTDivergence] = []
    
    # Allinea gli indici temporali comuni
    common_idx = primary_df.index.intersection(benchmark_df.index)
    if len(common_idx) < lookback + 2:
        return divergences

    p_df = primary_df.loc[common_idx]
    b_df = benchmark_df.loc[common_idx]

    p_highs = p_df['high'].values
    p_lows = p_df['low'].values
    b_highs = b_df['high'].values
    b_lows = b_df['low'].values
    times = common_idx

    for i in range(lookback, len(common_idx)):
        # Finestra precedente escludendo l'attuale
        p_prev_max = np.max(p_highs[i - lookback:i])
        p_prev_min = np.min(p_lows[i - lookback:i])
        b_prev_max = np.max(b_highs[i - lookback:i])
        b_prev_min = np.min(b_lows[i - lookback:i])

        curr_p_high, curr_p_low = p_highs[i], p_lows[i]
        curr_b_high, curr_b_low = b_highs[i], b_lows[i]
        curr_time = times[i]

        if relation == "INVERSE":
            # BEARISH SMT: Primary fa Higher High, DXY NON fa Lower Low (Higher Low o tiene il supporto)
            if (curr_p_high > p_prev_max) and (curr_b_low > b_prev_min):
                divergences.append(SMTDivergence(
                    timestamp=curr_time,
                    divergence_type="BEARISH_SMT",
                    primary_asset=primary_name,
                    correlated_asset=benchmark_name,
                    relation="INVERSE",
                    description=f"{primary_name} Higher High vs {benchmark_name} Higher Low (Mancata rottura dei minimi DXY)"
                ))

            # BULLISH SMT: Primary fa Lower Low, DXY NON fa Higher High (Lower High o fallimento estensione)
            elif (curr_p_low < p_prev_min) and (curr_b_high < b_prev_max):
                divergences.append(SMTDivergence(
                    timestamp=curr_time,
                    divergence_type="BULLISH_SMT",
                    primary_asset=primary_name,
                    correlated_asset=benchmark_name,
                    relation="INVERSE",
                    description=f"{primary_name} Lower Low vs {benchmark_name} Lower High (Mancata estensione massimi DXY)"
                ))

        elif relation == "DIRECT":
            # BEARISH SMT: Primary fa Higher High, Bench fa Lower High
            if (curr_p_high > p_prev_max) and (curr_b_high <= b_prev_max):
                divergences.append(SMTDivergence(
                    timestamp=curr_time,
                    divergence_type="BEARISH_SMT",
                    primary_asset=primary_name,
                    correlated_asset=benchmark_name,
                    relation="DIRECT",
                    description=f"{primary_name} Higher High vs {benchmark_name} Lower High (Debolezza relativa istituzionale)"
                ))
            # BULLISH SMT: Primary fa Lower Low, Bench fa Higher Low
            elif (curr_p_low < p_prev_min) and (curr_b_low >= b_prev_min):
                divergences.append(SMTDivergence(
                    timestamp=curr_time,
                    divergence_type="BULLISH_SMT",
                    primary_asset=primary_name,
                    correlated_asset=benchmark_name,
                    relation="DIRECT",
                    description=f"{primary_name} Lower Low vs {benchmark_name} Higher Low (Forza relativa istituzionale)"
                ))

    return divergences

