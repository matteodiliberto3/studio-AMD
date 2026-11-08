"""
Modulo Indicatori Istituzionali — Catena Frattale dei Timeframe, Fibonacci (M/m) e Target MTF:
1. Mappatura del Range MTF/HTF sugli estremi assoluti (M ed m) e Ritracciamento di Fibonacci (0.0, 0.50 Equilibrium, 1.0)
2. Filtro Obbligatorio di Posizione: Discount (< 0.50 -> SOLO Long) e Premium (> 0.50 -> SOLO Short)
3. Regola di Tenuta vs Accettazione sul Key Level (Fonte 16: >= 2 chiusure oltre il KL invalidano il setup)
4. Rilevamento Accumulazione Orizzontale Classica (Box M/m per A-M-D) e Accumulazione Dinamica (Swing M/m)
5. Rilevamento Rottura di M/m con Imbalance (IMB) e conteggio delle >= 2 chiusure di conferma oltre M/m
6. Catena Frattale dei Target MTF: individuazione del PRIMO Key Level MTF opposto (IMB MTF, Rejection Block MTF o Swing M/m MTF)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from config import CONFIG_ASSET, CONFIG_EXEC, CONFIG_HTF, CONFIG_SESSION


@dataclass
class FibonacciRangeMapping:
    """
    Rappresenta il tracciamento virtuale di Fibonacci (0.0, 0.50 Equilibrium, 1.0)
    sugli estremi assoluti del range MTF attivo: massimo (M) e minimo (m).
    """
    timeframe: str
    high_M: float               # Livello 1.0 del Fibonacci (Massimo assoluto M del range)
    low_m: float                # Livello 0.0 del Fibonacci (Minimo assoluto m del range)
    fib_05_equilibrium: float   # Livello 0.50 Equilibrium ((M + m) / 2)
    range_pips: float

    def fib_level_of_price(self, price: float) -> float:
        """Restituisce la coordinata di Fibonacci (0.0 = m, 0.50 = Equilibrium, 1.0 = M)."""
        span = self.high_M - self.low_m
        if span <= 1e-9:
            return 0.50
        return round((price - self.low_m) / span, 3)

    def get_zone(self, price: float) -> str:
        """
        Determina se il prezzo si trova in:
        - 'DISCOUNT' (< 0.50 -> cercare ESCLUSIVAMENTE setup Long)
        - 'PREMIUM'  (> 0.50 -> cercare ESCLUSIVAMENTE setup Short)
        """
        fib = self.fib_level_of_price(price)
        if fib < CONFIG_HTF.FIB_EQUILIBRIUM:
            return "DISCOUNT"
        return "PREMIUM"


@dataclass
class HTFKeyLevel:
    level_type: str             # 'MTF_IMBALANCE' o 'MTF_REJECTION_BLOCK'
    timeframe: str              # 'H4' o 'H1'
    timestamp: pd.Timestamp
    top: float
    bottom: float
    midpoint: float
    direction: str              # 'BEARISH' (in Premium > 0.50) o 'BULLISH' (in Discount < 0.50)
    range_high: float           # Massimo M del range MTF validato (Fib 1.0)
    range_low: float            # Minimo m del range MTF validato (Fib 0.0)
    equilibrium_50: float       # 50% Equilibrium del range MTF (Fib 0.50)
    fib_position: float         # Coordinata Fibonacci del Key Level dentro il range M/m (es. 0.24 in Discount o 0.78 in Premium)
    target_mtf: float           # Target MTF di default (estremo M o m opposto del range)
    invalidated: bool = False


@dataclass
class MTFFractalTarget:
    """
    Rappresenta un Key Level nei Medium Timeframe (M15 / H1 / H4) che funge da
    bersaglio magnetico ("Dove devono arrivare") per l'uscita ottimizzata del trade.
    """
    timeframe: str              # 'M15', 'H1', 'H4'
    target_type: str            # 'FIRST_MTF_IMB', 'FIRST_MTF_REJECTION_BLOCK', 'FIRST_MTF_SWING_M_m'
    timestamp: pd.Timestamp
    direction_for_exit: str     # 'BUY_TARGET' (resistenza/IMB sopra il prezzo) o 'SELL_TARGET' (supporto/IMB sotto il prezzo)
    price_level: float          # Prezzo esatto di uscita (bordo prossimale del primo Key Level MTF)
    zone_top: float
    zone_bottom: float
    mitigated: bool = False


@dataclass
class HorizontalBox:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    high_M: float               # Tetto M della box di accumulazione
    low_m: float                # Pavimento m della box di accumulazione
    range_pips: float
    timeframe: str


@dataclass
class DynamicSwingLevel:
    timestamp: pd.Timestamp
    index: int
    level_price: float          # Prezzo di M (per Long) o m (per Short)
    swing_type: str             # 'M_HIGH' (per Long) o 'm_LOW' (per Short)


@dataclass
class BreakoutImbalance:
    formed_at: pd.Timestamp
    formed_idx: int
    direction: str              # 'BULLISH' o 'BEARISH'
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


def map_fibonacci_range(
    high_M: float,
    low_m: float,
    timeframe: str = "H1",
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> FibonacciRangeMapping:
    """
    Costruisce la mappatura di Fibonacci (0.0 = m, 0.50 = Equilibrium, 1.0 = M)
    sugli estremi assoluti del range individuato.
    """
    eq_50 = (high_M + low_m) / 2.0
    r_pips = (high_M - low_m) / pip_size
    return FibonacciRangeMapping(
        timeframe=timeframe,
        high_M=round(high_M, 5),
        low_m=round(low_m, 5),
        fib_05_equilibrium=round(eq_50, 5),
        range_pips=round(r_pips, 1)
    )


def calculate_htf_range_and_equilibrium(df_htf: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """
    Calcola il Dealing Range MTF/HTF (estremi assoluti M ed m) e il livello 0.50 Fibonacci Equilibrium:
    - Zona Premium (> 0.50 Fib): ammessi ESCLUSIVAMENTE setup Short
    - Zona Discount (< 0.50 Fib): ammessi ESCLUSIVAMENTE setup Long
    """
    res = df_htf.copy()
    res['htf_high'] = res['high'].rolling(lookback, min_periods=3).max()
    res['htf_low'] = res['low'].rolling(lookback, min_periods=3).min()
    res['equilibrium'] = (res['htf_high'] + res['htf_low']) / 2.0
    span = (res['htf_high'] - res['htf_low']).replace(0, np.nan)
    res['fib_level'] = ((res['close'] - res['htf_low']) / span).fillna(0.50)
    res['is_premium'] = res['fib_level'] > CONFIG_HTF.FIB_EQUILIBRIUM
    res['is_discount'] = res['fib_level'] < CONFIG_HTF.FIB_EQUILIBRIUM
    return res


def find_validated_mtf_key_levels(
    df_mtf: pd.DataFrame,
    tf_label: str = "H1",
    lookback_range: int = 24,
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> List[HTFKeyLevel]:
    """
    Identifica i Range MTF ($M$ ed $m$), traccia il Fibonacci (0, 0.5, 1) e valida i Key Levels MTF:
    1. Presa di liquidità precedente (sweep di uno swing MTF precedente)
    2. Rottura strutturale MTF (chiusura oltre l'ultimo Swing Low/High MTF = BoS)
    3. Creazione di Imbalance (IMB / FVG) o Rejection Block nella gamba di rottura
    4. Filtro Fibonacci Obbligatorio:
       - Key Level in Premium (Fib > 0.50 tra m ed M) -> Setup esclusivamente BEARISH (Short)
       - Key Level in Discount (Fib < 0.50 tra m ed M) -> Setup esclusivamente BULLISH (Long)
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
        # CASO A: GAMBA MTF RIBASSISTA VALIDATA (Range M -> m, POI in PREMIUM > 0.50 Fib)
        # ----------------------------------------------------------------------
        leg_slice_start = max(0, min(last_sh_idx, i - 5))
        origin_high_idx = leg_slice_start + int(np.argmax(highs[leg_slice_start:i + 1]))
        origin_high_M = float(highs[origin_high_idx])

        prev_highs = [p for idx_s, p in swing_highs[:-1]] if len(swing_highs) > 1 else [origin_high_M]
        took_buy_liquidity = origin_high_M >= min(prev_highs) - (2.0 * pip_size)

        broke_mtf_low = closes[i] < last_sl_price or any(
            closes[b] < last_sl_price for b in range(origin_high_idx + 1, i + 1)
        )

        if took_buy_liquidity and broke_mtf_low and origin_high_M > last_sl_price:
            leg_low_m = float(np.min(lows[origin_high_idx:min(len(df_mtf), i + 3)]))
            fib_map = map_fibonacci_range(origin_high_M, leg_low_m, timeframe=tf_label, pip_size=pip_size)

            if fib_map.range_pips >= 8.0:
                gap_bear = lows[i - 2] - highs[i]
                if gap_bear >= min_imb_gap:
                    kl_top = float(lows[i - 2])
                    kl_bot = float(highs[i])
                    kl_mid = (kl_top + kl_bot) / 2.0
                    # Filtro Fibonacci Obbligatorio: per Short il POI deve essere in PREMIUM (> 0.50)
                    if fib_map.get_zone(kl_top) == "PREMIUM":
                        levels.append(HTFKeyLevel(
                            level_type="MTF_IMBALANCE",
                            timeframe=tf_label,
                            timestamp=times[i],
                            top=round(kl_top, 5),
                            bottom=round(kl_bot, 5),
                            midpoint=round(kl_mid, 5),
                            direction="BEARISH",
                            range_high=fib_map.high_M,
                            range_low=fib_map.low_m,
                            equilibrium_50=fib_map.fib_05_equilibrium,
                            fib_position=fib_map.fib_level_of_price(kl_mid),
                            target_mtf=fib_map.low_m
                        ))

                u_wick = highs[origin_high_idx] - max(opens[origin_high_idx], closes[origin_high_idx])
                if u_wick >= 2.5 * pip_size and origin_high_idx == i - 1:
                    kl_top = origin_high_M
                    kl_bot = float(max(opens[origin_high_idx], closes[origin_high_idx]))
                    kl_mid = (kl_top + kl_bot) / 2.0
                    if fib_map.get_zone(kl_mid) == "PREMIUM":
                        levels.append(HTFKeyLevel(
                            level_type="MTF_REJECTION_BLOCK",
                            timeframe=tf_label,
                            timestamp=times[i],
                            top=round(kl_top, 5),
                            bottom=round(kl_bot, 5),
                            midpoint=round(kl_mid, 5),
                            direction="BEARISH",
                            range_high=fib_map.high_M,
                            range_low=fib_map.low_m,
                            equilibrium_50=fib_map.fib_05_equilibrium,
                            fib_position=fib_map.fib_level_of_price(kl_mid),
                            target_mtf=fib_map.low_m
                        ))

        # ----------------------------------------------------------------------
        # CASO B: GAMBA MTF RIALZISTA VALIDATA (Range m -> M, POI in DISCOUNT < 0.50 Fib)
        # ----------------------------------------------------------------------
        leg_slice_start_b = max(0, min(last_sl_idx, i - 5))
        origin_low_idx = leg_slice_start_b + int(np.argmin(lows[leg_slice_start_b:i + 1]))
        origin_low_m = float(lows[origin_low_idx])

        prev_lows = [p for idx_s, p in swing_lows[:-1]] if len(swing_lows) > 1 else [origin_low_m]
        took_sell_liquidity = origin_low_m <= max(prev_lows) + (2.0 * pip_size)

        broke_mtf_high = closes[i] > last_sh_price or any(
            closes[b] > last_sh_price for b in range(origin_low_idx + 1, i + 1)
        )

        if took_sell_liquidity and broke_mtf_high and origin_low_m < last_sh_price:
            leg_high_M = float(np.max(highs[origin_low_idx:min(len(df_mtf), i + 3)]))
            fib_map = map_fibonacci_range(leg_high_M, origin_low_m, timeframe=tf_label, pip_size=pip_size)

            if fib_map.range_pips >= 8.0:
                gap_bull = lows[i] - highs[i - 2]
                if gap_bull >= min_imb_gap:
                    kl_top = float(lows[i])
                    kl_bot = float(highs[i - 2])
                    kl_mid = (kl_top + kl_bot) / 2.0
                    # Filtro Fibonacci Obbligatorio: per Long il POI deve essere in DISCOUNT (< 0.50)
                    if fib_map.get_zone(kl_bot) == "DISCOUNT":
                        levels.append(HTFKeyLevel(
                            level_type="MTF_IMBALANCE",
                            timeframe=tf_label,
                            timestamp=times[i],
                            top=round(kl_top, 5),
                            bottom=round(kl_bot, 5),
                            midpoint=round(kl_mid, 5),
                            direction="BULLISH",
                            range_high=fib_map.high_M,
                            range_low=fib_map.low_m,
                            equilibrium_50=fib_map.fib_05_equilibrium,
                            fib_position=fib_map.fib_level_of_price(kl_mid),
                            target_mtf=fib_map.high_M
                        ))

                l_wick = min(opens[origin_low_idx], closes[origin_low_idx]) - lows[origin_low_idx]
                if l_wick >= 2.5 * pip_size and origin_low_idx == i - 1:
                    kl_top = float(min(opens[origin_low_idx], closes[origin_low_idx]))
                    kl_bot = origin_low_m
                    kl_mid = (kl_top + kl_bot) / 2.0
                    if fib_map.get_zone(kl_mid) == "DISCOUNT":
                        levels.append(HTFKeyLevel(
                            level_type="MTF_REJECTION_BLOCK",
                            timeframe=tf_label,
                            timestamp=times[i],
                            top=round(kl_top, 5),
                            bottom=round(kl_bot, 5),
                            midpoint=round(kl_mid, 5),
                            direction="BULLISH",
                            range_high=fib_map.high_M,
                            range_low=fib_map.low_m,
                            equilibrium_50=fib_map.fib_05_equilibrium,
                            fib_position=fib_map.fib_level_of_price(kl_mid),
                            target_mtf=fib_map.high_M
                        ))

    return levels


def build_mtf_fractal_targets(
    df_mtf: pd.DataFrame,
    tf_label: str = "M15",
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> List[MTFFractalTarget]:
    """
    Costruisce il catalogo dei Key Levels nei Medium Timeframe (M15 / H1 / H4):
    - Imbalance (IMB) MTF opposte che devono essere ribilanciate
    - Massimi ($M$) e Minimi ($m$) strutturali MTF che fungono da primi bersagli magnetici
    """
    targets: List[MTFFractalTarget] = []
    if len(df_mtf) < 5:
        return targets

    highs = df_mtf['high'].values
    lows = df_mtf['low'].values
    times = df_mtf.index
    min_gap = (0.8 if tf_label == "M15" else (1.2 if tf_label == "H1" else 2.0)) * pip_size

    for i in range(2, len(df_mtf) - 1):
        # 1. Imbalance Ribassista MTF -> Bersaglio per posizioni LONG (BUY_TARGET)
        gap_bear = lows[i - 2] - highs[i]
        if gap_bear >= min_gap:
            targets.append(MTFFractalTarget(
                timeframe=tf_label,
                target_type="FIRST_MTF_IMB",
                timestamp=times[i],
                direction_for_exit="BUY_TARGET",
                price_level=round(float(highs[i]), 5),   # Bordo prossimale inferiore dell'IMB ribassista
                zone_top=round(float(lows[i - 2]), 5),
                zone_bottom=round(float(highs[i]), 5)
            ))

        # 2. Imbalance Rialzista MTF -> Bersaglio per posizioni SHORT (SELL_TARGET)
        gap_bull = lows[i] - highs[i - 2]
        if gap_bull >= min_gap:
            targets.append(MTFFractalTarget(
                timeframe=tf_label,
                target_type="FIRST_MTF_IMB",
                timestamp=times[i],
                direction_for_exit="SELL_TARGET",
                price_level=round(float(lows[i]), 5),    # Bordo prossimale superiore dell'IMB rialzista
                zone_top=round(float(lows[i]), 5),
                zone_bottom=round(float(highs[i - 2]), 5)
            ))

        # 3. Swing High MTF ($M$) -> Bersaglio strutturale per posizioni LONG
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and highs[i] >= highs[i + 1]:
            targets.append(MTFFractalTarget(
                timeframe=tf_label,
                target_type="FIRST_MTF_SWING_M",
                timestamp=times[i + 1],
                direction_for_exit="BUY_TARGET",
                price_level=round(float(highs[i]), 5),
                zone_top=round(float(highs[i]), 5),
                zone_bottom=round(float(highs[i]), 5)
            ))

        # 4. Swing Low MTF ($m$) -> Bersaglio strutturale per posizioni SHORT
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1]:
            targets.append(MTFFractalTarget(
                timeframe=tf_label,
                target_type="FIRST_MTF_SWING_m",
                timestamp=times[i + 1],
                direction_for_exit="SELL_TARGET",
                price_level=round(float(lows[i]), 5),
                zone_top=round(float(lows[i]), 5),
                zone_bottom=round(float(lows[i]), 5)
            ))

    return targets


def select_first_mtf_key_level_target(
    entry_price: float,
    sl_pips: float,
    direction: str,
    kl: HTFKeyLevel,
    active_mtf_targets: List[MTFFractalTarget],
    df_m15_slice: Optional[pd.DataFrame] = None,
    pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT
) -> Tuple[float, str]:
    """
    Seleziona il PRIMO Key Level nei MTF (H1 / H4 / M15 NON MITIGATO) nella direzione del trade
    che ottimizza il rapporto R:R (tra 1:2.5 e 1:5.5) senza inseguire utopie oltre il primo
    vero ostacolo MTF capace di generare un'inversione contraria.
    """
    min_dist = sl_pips * 2.5 * pip_size
    max_dist = sl_pips * 6.5 * pip_size

    candidates: List[Tuple[float, str]] = []

    if direction == "BUY":
        # 1. Filtra solo i Key Levels MTF sopra entry_price che NON sono ancora stati mitigati
        for t in active_mtf_targets:
            if t.direction_for_exit == "BUY_TARGET" and t.timestamp >= kl.timestamp - pd.Timedelta(hours=24):
                dist = t.price_level - entry_price
                if min_dist <= dist <= max_dist:
                    is_unmitigated = True
                    if df_m15_slice is not None and len(df_m15_slice) > 1:
                        after_t = df_m15_slice.loc[t.timestamp + pd.Timedelta(minutes=15):]
                        if len(after_t) > 0 and float(after_t['high'].max()) >= t.price_level:
                            is_unmitigated = False
                    if is_unmitigated:
                        candidates.append((t.price_level, f"{t.timeframe}_{t.target_type}"))

        # 2. Livelli strutturali della Catena MTF (Primo POI in Premium 0.786 Fib e Massimo M del Range MTF)
        fib_786_premium = kl.range_low + (kl.range_high - kl.range_low) * 0.786
        for lvl_price, lvl_name in [
            (fib_786_premium, f"{kl.timeframe}_FIRST_PREMIUM_POI"),
            (kl.range_high, f"{kl.timeframe}_RANGE_HIGH_M")
        ]:
            dist = lvl_price - entry_price
            if min_dist <= dist <= max_dist * 1.4:
                candidates.append((round(lvl_price, 5), lvl_name))

        if candidates:
            # Sceglie il PRIMO Key Level MTF non mitigato incontrato nella salita
            candidates.sort(key=lambda x: x[0])
            return round(candidates[0][0], 5), candidates[0][1]

        # Se il Massimo M del range MTF è oltre 1.8R, usa il Massimo M del range MTF (o clampa a 3.5R)
        if kl.range_high > entry_price + (sl_pips * 1.8 * pip_size):
            tp_opt = min(kl.range_high, entry_price + (sl_pips * 4.5 * pip_size))
            return round(tp_opt, 5), f"{kl.timeframe}_RANGE_HIGH_M"
        return round(entry_price + (sl_pips * 3.0 * pip_size), 5), f"{kl.timeframe}_FIRST_MTF_KL"

    else:
        # 1. Da Premium (> 0.50 Fib) cerchiamo il PRIMO Key Level MTF NON MITIGATO sotto entry_price
        for t in active_mtf_targets:
            if t.direction_for_exit == "SELL_TARGET" and t.timestamp >= kl.timestamp - pd.Timedelta(hours=24):
                dist = entry_price - t.price_level
                if min_dist <= dist <= max_dist:
                    is_unmitigated = True
                    if df_m15_slice is not None and len(df_m15_slice) > 1:
                        after_t = df_m15_slice.loc[t.timestamp + pd.Timedelta(minutes=15):]
                        if len(after_t) > 0 and float(after_t['low'].min()) <= t.price_level:
                            is_unmitigated = False
                    if is_unmitigated:
                        candidates.append((t.price_level, f"{t.timeframe}_{t.target_type}"))

        # 2. Livelli strutturali della Catena MTF (Primo POI in Discount 0.214 Fib e Minimo m del Range MTF)
        fib_214_discount = kl.range_low + (kl.range_high - kl.range_low) * 0.214
        for lvl_price, lvl_name in [
            (fib_214_discount, f"{kl.timeframe}_FIRST_DISCOUNT_POI"),
            (kl.range_low, f"{kl.timeframe}_RANGE_LOW_m")
        ]:
            dist = entry_price - lvl_price
            if min_dist <= dist <= max_dist * 1.4:
                candidates.append((round(lvl_price, 5), lvl_name))

        if candidates:
            # Sceglie il PRIMO Key Level MTF non mitigato incontrato nella discesa
            candidates.sort(key=lambda x: x[0], reverse=True)
            return round(candidates[0][0], 5), candidates[0][1]

        if kl.range_low < entry_price - (sl_pips * 1.8 * pip_size):
            tp_opt = max(kl.range_low, entry_price - (sl_pips * 4.5 * pip_size))
            return round(tp_opt, 5), f"{kl.timeframe}_RANGE_LOW_m"
        return round(entry_price - (sl_pips * 3.0 * pip_size), 5), f"{kl.timeframe}_FIRST_MTF_KL"


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
    formatosi prima o durante l'arrivo sul Key Level MTF.
    """
    start_i = max(1, manip_idx - lookback)
    if manip_idx - start_i < 3:
        return None

    highs = df_ltf['high'].values
    lows = df_ltf['low'].values
    times = df_ltf.index

    if direction == "SELL":
        for k in range(manip_idx - 1, start_i, -1):
            if lows[k] <= lows[k - 1] and lows[k] <= lows[min(len(lows) - 1, k + 1)]:
                if (highs[manip_idx] - lows[k]) >= 1.5 * pip_size:
                    return DynamicSwingLevel(
                        timestamp=times[k],
                        index=k,
                        level_price=round(float(lows[k]), 5),
                        swing_type="m_LOW"
                    )
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
) -> Optional[Tuple[BreakoutImbalance, int, pd.Timestamp]]:
    """
    Verifica la sequenza obbligatoria dopo la manipolazione:
    1. Rottura del livello `m` (per SELL) o `M` (per BUY) accompagnata da un Imbalance (IMB / FVG).
    2. Almeno 2 chiusure di candela al di sotto di `m` (SELL) o al di sopra di `M` (BUY).
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
        if k >= manip_idx + 2:
            disp_range = max(1e-9, highs[k - 1] - lows[k - 1])
            disp_body = abs(closes[k - 1] - opens[k - 1])
            is_displacement = (disp_body / disp_range) >= CONFIG_EXEC.MIN_DISPLACEMENT_BODY_RATIO

            if direction == "SELL" and is_displacement:
                gap = lows[k - 2] - highs[k]
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

        if direction == "SELL":
            if closes[k] < level_M_or_m:
                closes_beyond_count += 1
        elif direction == "BUY":
            if closes[k] > level_M_or_m:
                closes_beyond_count += 1

        if found_imb is not None and closes_beyond_count >= CONFIG_EXEC.MIN_CONFIRMATION_CLOSES_BEYOND_M:
            return found_imb, k, times[k]

    return None
