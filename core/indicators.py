"""
Modulo Indicatori Istituzionali:
1. Validazione Key Level MTF/HTF (Presa di liquidità + Rottura strutturale + IMB / Rejection Block)
2. Regola di Tenuta vs Accettazione sul Key Level (Fonte 16: >= 2 chiusure oltre il KL invalidano il setup)
3. Rilevamento Accumulazione Orizzontale Classica (Box M/m per A-M-D)
4. Rilevamento Accumulazione Dinamica (Swing strutturale M/m)
5. Rilevamento Rottura di M/m con Imbalance (IMB) e conteggio delle >= 2 chiusure di conferma oltre M/m
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from config import CONFIG_ASSET, CONFIG_EXEC, CONFIG_HTF, CONFIG_SESSION


@dataclass
class HTFKeyLevel:
    level_type: str         # 'MTF_IMBALANCE' o 'MTF_REJECTION_BLOCK'
    timeframe: str          # 'H4' o 'H1'
    timestamp: pd.Timestamp
    top: float
    bottom: float
    midpoint: float
    direction: str          # 'BEARISH' (in Premium) o 'BULLISH' (in Discount)
    range_high: float       # Massimo del range MTF validato
    range_low: float        # Minimo del range MTF validato
    equilibrium_50: float   # 50% del range MTF
    target_mtf: float       # Target MTF opposto che potrebbe generare un'inversione
    invalidated: bool = False


@dataclass
class HorizontalBox:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    high_M: float           # Tetto M della box di accumulazione
    low_m: float            # Pavimento m della box di accumulazione
    range_pips: float
    timeframe: str


@dataclass
class DynamicSwingLevel:
    timestamp: pd.Timestamp
    index: int
    level_price: float      # Prezzo di M (per Long) o m (per Short)
    swing_type: str         # 'M_HIGH' (per Long) o 'm_LOW' (per Short)


@dataclass
class BreakoutImbalance:
    formed_at: pd.Timestamp
    formed_idx: int
    direction: str          # 'BULLISH' o 'BEARISH'
    top: float
    bottom: float
    midpoint: float
    gap_pips: float


def is_active_operating_window(dt: pd.Timestamp) -> Tuple[bool, str]:
    """
    Verifica se il timestamp cade nella finestra operativa (08:00 - 20:00 CET)
    ed evidenzia i cambi di periodo H1 e H4.
    """
    t = dt.time()
    if not (CONFIG_SESSION.ACTIVE_WINDOW_START <= t <= CONFIG_SESSION.ACTIVE_WINDOW_END):
        return False, "OUT_OF_WINDOW"

    minute = dt.minute
    hour = dt.hour

    if (hour % 4 == 0) and (minute < CONFIG_SESSION.MONITOR_H4_OPEN_MINUTES):
        return True, "H4_PERIOD_OPEN"
    if minute < CONFIG_SESSION.MONITOR_H1_OPEN_MINUTES:
        return True, "H1_PERIOD_OPEN"
    return True, "IN_SESSION"


def calculate_htf_range_and_equilibrium(df_htf: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """
    Calcola il Dealing Range MTF/HTF e il livello 50% Equilibrium:
    - Short ammessi SOLO sopra il 50% (Premium)
    - Long ammessi SOLO sotto il 50% (Discount)
    """
    res = df_htf.copy()
    res['htf_high'] = res['high'].rolling(lookback, min_periods=3).max()
    res['htf_low'] = res['low'].rolling(lookback, min_periods=3).min()
    res['equilibrium'] = (res['htf_high'] + res['htf_low']) / 2.0
    res['is_premium'] = res['close'] > res['equilibrium']
    res['is_discount'] = res['close'] < res['equilibrium']
    return res


def find_validated_mtf_key_levels(
    df_mtf: pd.DataFrame,
    tf_label: str = "H1",
    lookback_range: int = 24,
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> List[HTFKeyLevel]:
    """
    Identifica i Range MTF e i relativi Key Levels MTF validati secondo le 3 regole del notebook:
    1. Presa di liquidità precedente (sweep di uno swing MTF precedente)
    2. Rottura strutturale MTF (chiusura oltre l'ultimo Swing Low/High MTF = BoS)
    3. Creazione di Imbalance (IMB / FVG) o Rejection Block nella gamba di rottura
    4. Collocazione in Premium (>50% della gamba MTF) per Short, o Discount (<50% della gamba MTF) per Long
    Il `target_mtf` è l'estremo opposto della gamba MTF validata (il livello MTF che può generare inversione).
    """
    levels: List[HTFKeyLevel] = []
    if len(df_mtf) < 10:
        return levels

    highs = df_mtf['high'].values
    lows = df_mtf['low'].values
    opens = df_mtf['open'].values
    closes = df_mtf['close'].values
    times = df_mtf.index

    min_imb_gap = (2.0 if tf_label == "H4" else 1.2) * pip_size

    for i in range(6, len(df_mtf)):
        win_start = max(1, i - 14)
        swing_highs = []
        swing_lows = []
        for s in range(win_start, i - 1):
            if highs[s] >= highs[s - 1] and highs[s] >= highs[s + 1]:
                swing_highs.append((s, float(highs[s])))
            if lows[s] <= lows[s - 1] and lows[s] <= lows[s + 1]:
                swing_lows.append((s, float(lows[s])))

        if not swing_highs or not swing_lows:
            continue

        last_sh_idx, last_sh_price = swing_highs[-1]
        last_sl_idx, last_sl_price = swing_lows[-1]

        # ----------------------------------------------------------------------
        # CASO A: GAMBA MTF RIBASSISTA VALIDATA (Bearish BoS sotto last_sl_price)
        # ----------------------------------------------------------------------
        leg_slice_start = max(0, min(last_sh_idx, i - 5))
        origin_high_idx = leg_slice_start + int(np.argmax(highs[leg_slice_start:i + 1]))
        origin_high = float(highs[origin_high_idx])

        prev_highs = [p for idx_s, p in swing_highs[:-1]] if len(swing_highs) > 1 else [origin_high]
        took_buy_liquidity = origin_high >= min(prev_highs) - (2.0 * pip_size)

        broke_mtf_low = closes[i] < last_sl_price or any(
            closes[b] < last_sl_price for b in range(origin_high_idx + 1, i + 1)
        )

        if took_buy_liquidity and broke_mtf_low and origin_high > last_sl_price:
            leg_low = float(np.min(lows[origin_high_idx:min(len(df_mtf), i + 3)]))
            leg_range = origin_high - leg_low
            if leg_range >= 8.0 * pip_size:
                eq_50 = (origin_high + leg_low) / 2.0

                gap_bear = lows[i - 2] - highs[i]
                if gap_bear >= min_imb_gap and float(lows[i - 2]) >= eq_50:
                    kl_top = float(lows[i - 2])
                    kl_bot = float(highs[i])
                    levels.append(HTFKeyLevel(
                        level_type="MTF_IMBALANCE",
                        timeframe=tf_label,
                        timestamp=times[i],
                        top=round(kl_top, 5),
                        bottom=round(kl_bot, 5),
                        midpoint=round((kl_top + kl_bot) / 2.0, 5),
                        direction="BEARISH",
                        range_high=round(origin_high, 5),
                        range_low=round(leg_low, 5),
                        equilibrium_50=round(eq_50, 5),
                        target_mtf=round(leg_low, 5)
                    ))

                u_wick = highs[origin_high_idx] - max(opens[origin_high_idx], closes[origin_high_idx])
                if u_wick >= 2.5 * pip_size and origin_high_idx == i - 1:
                    kl_top = origin_high
                    kl_bot = float(max(opens[origin_high_idx], closes[origin_high_idx]))
                    levels.append(HTFKeyLevel(
                        level_type="MTF_REJECTION_BLOCK",
                        timeframe=tf_label,
                        timestamp=times[i],
                        top=round(kl_top, 5),
                        bottom=round(kl_bot, 5),
                        midpoint=round((kl_top + kl_bot) / 2.0, 5),
                        direction="BEARISH",
                        range_high=round(origin_high, 5),
                        range_low=round(leg_low, 5),
                        equilibrium_50=round(eq_50, 5),
                        target_mtf=round(leg_low, 5)
                    ))

        # ----------------------------------------------------------------------
        # CASO B: GAMBA MTF RIALZISTA VALIDATA (Bullish BoS sopra last_sh_price)
        # ----------------------------------------------------------------------
        leg_slice_start_b = max(0, min(last_sl_idx, i - 5))
        origin_low_idx = leg_slice_start_b + int(np.argmin(lows[leg_slice_start_b:i + 1]))
        origin_low = float(lows[origin_low_idx])

        prev_lows = [p for idx_s, p in swing_lows[:-1]] if len(swing_lows) > 1 else [origin_low]
        took_sell_liquidity = origin_low <= max(prev_lows) + (2.0 * pip_size)

        broke_mtf_high = closes[i] > last_sh_price or any(
            closes[b] > last_sh_price for b in range(origin_low_idx + 1, i + 1)
        )

        if took_sell_liquidity and broke_mtf_high and origin_low < last_sh_price:
            leg_high = float(np.max(highs[origin_low_idx:min(len(df_mtf), i + 3)]))
            leg_range = leg_high - origin_low
            if leg_range >= 8.0 * pip_size:
                eq_50 = (leg_high + origin_low) / 2.0

                gap_bull = lows[i] - highs[i - 2]
                if gap_bull >= min_imb_gap and float(highs[i - 2]) <= eq_50:
                    kl_top = float(lows[i])
                    kl_bot = float(highs[i - 2])
                    levels.append(HTFKeyLevel(
                        level_type="MTF_IMBALANCE",
                        timeframe=tf_label,
                        timestamp=times[i],
                        top=round(kl_top, 5),
                        bottom=round(kl_bot, 5),
                        midpoint=round((kl_top + kl_bot) / 2.0, 5),
                        direction="BULLISH",
                        range_high=round(leg_high, 5),
                        range_low=round(origin_low, 5),
                        equilibrium_50=round(eq_50, 5),
                        target_mtf=round(leg_high, 5)
                    ))

                l_wick = min(opens[origin_low_idx], closes[origin_low_idx]) - lows[origin_low_idx]
                if l_wick >= 2.5 * pip_size and origin_low_idx == i - 1:
                    kl_top = float(min(opens[origin_low_idx], closes[origin_low_idx]))
                    kl_bot = origin_low
                    levels.append(HTFKeyLevel(
                        level_type="MTF_REJECTION_BLOCK",
                        timeframe=tf_label,
                        timestamp=times[i],
                        top=round(kl_top, 5),
                        bottom=round(kl_bot, 5),
                        midpoint=round((kl_top + kl_bot) / 2.0, 5),
                        direction="BULLISH",
                        range_high=round(leg_high, 5),
                        range_low=round(origin_low, 5),
                        equilibrium_50=round(eq_50, 5),
                        target_mtf=round(leg_high, 5)
                    ))

    return levels


def is_key_level_invalidated_by_closes(
    df_slice: pd.DataFrame,
    kl: HTFKeyLevel,
    max_consecutive_closes: int = CONFIG_HTF.MAX_CONSECUTIVE_CLOSES_BEYOND_KL
) -> bool:
    """
    Regola Fonte 16:
    Se il prezzo effettua >= 2 chiusure di corpo consecutive oltre il Key Level MTF
    (sopra kl.top per Bearish, sotto kl.bottom per Bullish), il mercato sta accettando
    nuovi prezzi e il Key Level è INVALIDATO.
    """
    closes = df_slice['close'].values
    consec = 0
    for c in closes:
        if kl.direction == "BEARISH":
            if c > kl.top:
                consec += 1
                if consec >= max_consecutive_closes:
                    return True
            else:
                consec = 0
        else:
            if c < kl.bottom:
                consec += 1
                if consec >= max_consecutive_closes:
                    return True
            else:
                consec = 0
    return False


def detect_horizontal_accumulation_box(
    df_ltf: pd.DataFrame,
    manip_idx: int,
    timeframe: str = "M1",
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> Optional[HorizontalBox]:
    """
    Cerca una Box di Accumulazione Orizzontale (`M` ed `m`) immediatamente precedente
    alla candela di manipolazione (`manip_idx`).
    Una vera box orizzontale ha range compresso, almeno 6-14 candele ed escursione laterale
    (con almeno 2 swing/tocchi sia nella metà superiore che nella metà inferiore).
    """
    max_pips = CONFIG_EXEC.MAX_BOX_RANGE_PIPS_M1 if timeframe == "M1" else CONFIG_EXEC.MAX_BOX_RANGE_PIPS_M5
    for length in range(12, CONFIG_EXEC.MIN_BOX_BARS - 1, -1):
        start_i = manip_idx - length
        if start_i < 0:
            continue
        sub = df_ltf.iloc[start_i:manip_idx]
        h_M = float(sub['high'].max())
        l_m = float(sub['low'].min())
        mid = (h_M + l_m) / 2.0
        r_pips = (h_M - l_m) / pip_size

        if 2.0 <= r_pips <= max_pips:
            net_drift = abs(float(sub['close'].iloc[-1]) - float(sub['open'].iloc[0])) / pip_size
            upper_touches = int((sub['high'] >= mid + (h_M - mid) * 0.4).sum())
            lower_touches = int((sub['low'] <= mid - (mid - l_m) * 0.4).sum())
            if net_drift <= r_pips * 0.55 and upper_touches >= 2 and lower_touches >= 2:
                return HorizontalBox(
                    start_time=sub.index[0],
                    end_time=sub.index[-1],
                    high_M=round(h_M, 5),
                    low_m=round(l_m, 5),
                    range_pips=round(r_pips, 1),
                    timeframe=timeframe
                )
    return None


def detect_dynamic_swing_level(
    df_ltf: pd.DataFrame,
    manip_idx: int,
    direction: str,
    lookback: int = 12,
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> Optional[DynamicSwingLevel]:
    """
    Rileva l'ultimo livello strutturale `M` (per Long) o `m` (per Short) di un'Accumulazione Dinamica
    formatosi prima o durante l'arrivo sul Key Level MTF:
    - Per SELL: l'ultimo minimo di swing `m` (Higher Low) che ha preceduto il massimo di manipolazione.
    - Per BUY: l'ultimo massimo di swing `M` (Lower High) che ha preceduto il minimo di manipolazione.
    """
    start_i = max(1, manip_idx - lookback)
    if manip_idx - start_i < 3:
        return None

    highs = df_ltf['high'].values
    lows = df_ltf['low'].values
    times = df_ltf.index

    if direction == "SELL":
        # Cerchiamo l'ultimo minimo relativo `m` prima di manip_idx
        for k in range(manip_idx - 1, start_i, -1):
            if lows[k] <= lows[k - 1] and lows[k] <= lows[min(len(lows) - 1, k + 1)]:
                if (highs[manip_idx] - lows[k]) >= 1.5 * pip_size:
                    return DynamicSwingLevel(
                        timestamp=times[k],
                        index=k,
                        level_price=round(float(lows[k]), 5),
                        swing_type="m_LOW"
                    )
        # Fallback al minimo locale delle ultime 5 barre prima dello spike
        sub_lows = lows[max(0, manip_idx - 6):manip_idx]
        if len(sub_lows) > 0:
            rel_idx = int(np.argmin(sub_lows)) + max(0, manip_idx - 6)
            return DynamicSwingLevel(
                timestamp=times[rel_idx],
                index=rel_idx,
                level_price=round(float(lows[rel_idx]), 5),
                swing_type="m_LOW"
            )

    elif direction == "BUY":
        # Cerchiamo l'ultimo massimo relativo `M` prima di manip_idx
        for k in range(manip_idx - 1, start_i, -1):
            if highs[k] >= highs[k - 1] and highs[k] >= highs[min(len(highs) - 1, k + 1)]:
                if (highs[k] - lows[manip_idx]) >= 1.5 * pip_size:
                    return DynamicSwingLevel(
                        timestamp=times[k],
                        index=k,
                        level_price=round(float(highs[k]), 5),
                        swing_type="M_HIGH"
                    )
        sub_highs = highs[max(0, manip_idx - 6):manip_idx]
        if len(sub_highs) > 0:
            rel_idx = int(np.argmax(sub_highs)) + max(0, manip_idx - 6)
            return DynamicSwingLevel(
                timestamp=times[rel_idx],
                index=rel_idx,
                level_price=round(float(highs[rel_idx]), 5),
                swing_type="M_HIGH"
            )

    return None


def detect_breakout_with_imb_and_two_closes(
    df_ltf: pd.DataFrame,
    manip_idx: int,
    level_M_or_m: float,
    direction: str,
    timeframe: str = "M1",
    max_search_bars: int = 16,
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> Optional[ Tuple[BreakoutImbalance, int, pd.Timestamp] ]:
    """
    Verifica la sequenza obbligatoria dopo la manipolazione:
    1. Rottura del livello `m` (per SELL) o `M` (per BUY) accompagnata da un Imbalance (IMB / FVG).
    2. Almeno 2 chiusure di candela al di sotto di `m` (SELL) o al di sopra di `M` (BUY).

    Ritorna:
      (BreakoutImbalance, confirm_idx, confirm_timestamp)
      dove `confirm_idx` è l'indice della seconda candela chiusa oltre `M`/`m`.
    """
    min_gap = (
        CONFIG_EXEC.MIN_IMB_GAP_PIPS_M1 if timeframe == "M1" else CONFIG_EXEC.MIN_IMB_GAP_PIPS_M5
    ) * pip_size

    highs = df_ltf['high'].values
    lows = df_ltf['low'].values
    opens = df_ltf['open'].values
    closes = df_ltf['close'].values
    times = df_ltf.index

    end_search = min(len(df_ltf), manip_idx + max_search_bars)
    found_imb: Optional[BreakoutImbalance] = None
    closes_beyond_count = 0

    for k in range(manip_idx + 1, end_search):
        # Controlla se nel movimento di rottura si è formato un IMB a 3 candele (tra k-2, k-1, k)
        if k >= manip_idx + 2:
            disp_range = max(1e-9, highs[k - 1] - lows[k - 1])
            disp_body = abs(closes[k - 1] - opens[k - 1])
            is_displacement = (disp_body / disp_range) >= CONFIG_EXEC.MIN_DISPLACEMENT_BODY_RATIO

            if direction == "SELL" and is_displacement:
                gap = lows[k - 2] - highs[k]
                # L'IMB deve accompagnare l'impulso ribassista che rompe o attraversa `m`
                if gap >= min_gap and closes[k - 1] < opens[k - 1] and highs[k] <= level_M_or_m + (1.5 * pip_size):
                    if found_imb is None or gap > (found_imb.top - found_imb.bottom):
                        top_p = float(lows[k - 2])
                        bot_p = float(highs[k])
                        found_imb = BreakoutImbalance(
                            formed_at=times[k],
                            formed_idx=k,
                            direction="BEARISH",
                            top=round(top_p, 5),
                            bottom=round(bot_p, 5),
                            midpoint=round((top_p + bot_p) / 2.0, 5),
                            gap_pips=round(gap / pip_size, 1)
                        )
            elif direction == "BUY" and is_displacement:
                gap = lows[k] - highs[k - 2]
                if gap >= min_gap and closes[k - 1] > opens[k - 1] and lows[k] >= level_M_or_m - (1.5 * pip_size):
                    if found_imb is None or gap > (found_imb.top - found_imb.bottom):
                        top_p = float(lows[k])
                        bot_p = float(highs[k - 2])
                        found_imb = BreakoutImbalance(
                            formed_at=times[k],
                            formed_idx=k,
                            direction="BULLISH",
                            top=round(top_p, 5),
                            bottom=round(bot_p, 5),
                            midpoint=round((top_p + bot_p) / 2.0, 5),
                            gap_pips=round(gap / pip_size, 1)
                        )

        # Conteggio delle chiusure oltre `M` o `m`
        if direction == "SELL":
            if closes[k] < level_M_or_m:
                closes_beyond_count += 1
        elif direction == "BUY":
            if closes[k] > level_M_or_m:
                closes_beyond_count += 1

        # Appena abbiamo SIA l'IMB di rottura SIA almeno 2 chiusure oltre M/m -> CONFERMATO!
        if found_imb is not None and closes_beyond_count >= CONFIG_EXEC.MIN_CONFIRMATION_CLOSES_BEYOND_M:
            return found_imb, k, times[k]

    return None
