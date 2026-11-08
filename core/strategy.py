"""
Modulo Strategia Operativa Istituzionale A-M-D:
Sequenza: Key Level HTF -> Accumulazione LTF -> Manipolazione -> Rottura con Imbalance -> Distribuzione.

Due sole modalità di ingresso:
1. ENTRY_MANIPULATION (Entry 1 - Aggressiva sul picco di manipolazione del Key Level)
2. ENTRY_POST_BREAKOUT_RETEST (Entry 2 - Conservativa sul retest dell'Imbalance / bordo rotto)
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from config import (
    CONFIG_ASSET,
    CONFIG_EXEC,
    EntryType
)
from core.indicators import (
    HTFKeyLevel,
    calculate_htf_range_and_equilibrium,
    detect_accumulation_zone,
    find_htf_key_levels,
    is_active_operating_window
)


@dataclass
class InstitutionalTrade:
    trade_id: int
    entry_type: EntryType
    timeframe: str              # 'M5' o 'M1'
    direction: str              # 'BUY' o 'SELL'
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    tp_50_eq: float             # Target 1 (Parziale al 50% del leg/range)
    tp_structural: float        # Target Finale (Liquidità opposta)
    sl_pips: float
    htf_key_level: str
    mae_pips: float = 0.0       # Maximum Adverse Excursion (Drawdown massimo sofferto)
    hit_target_1: bool = False  # Se ha raggiunto il parziale al 50%
    strict_win: bool = False    # Se ha raggiunto il Target Finale senza toccare lo SL iniziale
    adjusted_win: bool = False  # Se ha incassato profitto (almeno TP1 + trailing)
    outcome: str = "RUNNING"    # 'STRICT_WIN', 'ADJUSTED_WIN', 'LOSS', 'RUNNING'
    exit_time: Optional[pd.Timestamp] = None
    realized_rr: Optional[float] = None


class InstitutionalAMDEngine:
    def __init__(self, pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT):
        self.pip_size = pip_size

    def analyze_market(
        self,
        df_m1: pd.DataFrame,
        df_m5: pd.DataFrame,
        df_h4: pd.DataFrame
    ) -> List[InstitutionalTrade]:
        """
        Esegue l'analisi completa:
        1. Identifica Key Levels HTF su H4 (Imbalance H4 e Rejection Blocks)
        2. Calcola Dealing Range HTF ed Equilibrium 50% (Filtro Premium / Discount)
        3. Valuta M5 (default): se visibile a corpo esegue su M5.
           Se la reazione è compressa in una singola wick, attiva M1 ("raggi X").
        """
        trades: List[InstitutionalTrade] = []
        trade_id = 1

        df_h4_dr = calculate_htf_range_and_equilibrium(df_h4)
        htf_levels = find_htf_key_levels(df_h4, pip_size=self.pip_size)

        m5_times = df_m5.index
        m5_highs = df_m5['high'].values
        m5_lows = df_m5['low'].values
        m5_opens = df_m5['open'].values
        m5_closes = df_m5['close'].values

        i = 15
        while i < len(df_m5) - 30:
            t_curr = m5_times[i]
            is_valid_time, time_tag = is_active_operating_window(t_curr)
            if not is_valid_time:
                i += 1
                continue

            past_h4 = df_h4_dr[df_h4_dr.index <= t_curr]
            if len(past_h4) == 0:
                i += 1
                continue
            curr_h4_row = past_h4.iloc[-1]
            htf_is_premium = curr_h4_row['is_premium']
            htf_is_discount = curr_h4_row['is_discount']
            htf_high = curr_h4_row['htf_high']
            htf_low = curr_h4_row['htf_low']

            active_htf_levels = [lvl for lvl in htf_levels if lvl.timestamp <= t_curr]
            if not active_htf_levels:
                i += 1
                continue

            new_trades = []

            # --- SCENARIO SHORT: In Premium Zone (>50% HTF), test di Key Level Bearish ---
            if htf_is_premium:
                bearish_poi = next(
                    (lvl for lvl in active_htf_levels if lvl.direction == 'BEARISH' and m5_highs[i] >= lvl.bottom * 0.999),
                    None
                )
                if bearish_poi:
                    m5_range = m5_highs[i] - m5_lows[i]
                    m5_upper_wick = m5_highs[i] - max(m5_opens[i], m5_closes[i])
                    is_compressed_wick = (m5_upper_wick > (m5_range * 0.50)) and (m5_range > 0.0006)

                    target_tf = "M1" if is_compressed_wick else "M5"
                    data_source = df_m1 if is_compressed_wick else df_m5

                    new_trades = self._evaluate_amd_sequence(
                        df_ltf=data_source,
                        timeframe=target_tf,
                        center_time=t_curr,
                        direction="SELL",
                        htf_level=bearish_poi,
                        opposite_target=htf_low,
                        trade_id_start=trade_id
                    )

            # --- SCENARIO LONG: In Discount Zone (<50% HTF), test di Key Level Bullish ---
            elif htf_is_discount:
                bullish_poi = next(
                    (lvl for lvl in active_htf_levels if lvl.direction == 'BULLISH' and m5_lows[i] <= lvl.top * 1.001),
                    None
                )
                if bullish_poi:
                    m5_range = m5_highs[i] - m5_lows[i]
                    m5_lower_wick = min(m5_opens[i], m5_closes[i]) - m5_lows[i]
                    is_compressed_wick = (m5_lower_wick > (m5_range * 0.50)) and (m5_range > 0.0006)

                    target_tf = "M1" if is_compressed_wick else "M5"
                    data_source = df_m1 if is_compressed_wick else df_m5

                    new_trades = self._evaluate_amd_sequence(
                        df_ltf=data_source,
                        timeframe=target_tf,
                        center_time=t_curr,
                        direction="BUY",
                        htf_level=bullish_poi,
                        opposite_target=htf_high,
                        trade_id_start=trade_id
                    )

            if new_trades:
                for tr in new_trades:
                    self._simulate_contextual_trade(tr, df_m1)
                    trades.append(tr)
                    trade_id += 1
                # Avanza per non duplicare la stessa finestra di accumulazione
                i += 15
            else:
                i += 1

        return trades

    def _evaluate_amd_sequence(
        self,
        df_ltf: pd.DataFrame,
        timeframe: str,
        center_time: pd.Timestamp,
        direction: str,
        htf_level: HTFKeyLevel,
        opposite_target: float,
        trade_id_start: int
    ) -> List[InstitutionalTrade]:
        trades = []
        delta_pre = pd.Timedelta(minutes=30 if timeframe == "M1" else 100)
        delta_post = pd.Timedelta(minutes=45 if timeframe == "M1" else 150)
        window = df_ltf.loc[center_time - delta_pre:center_time + delta_post]
        if len(window) < 10:
            return trades

        times = window.index
        highs = window['high'].values
        lows = window['low'].values
        closes = window['close'].values

        mid_idx = min(8, len(window) // 2)
        accum = detect_accumulation_zone(window, end_idx=mid_idx, lookback=mid_idx, timeframe=timeframe, pip_size=self.pip_size)
        if accum is None:
            # Fallback range sui dati recenti
            accum_h = np.max(highs[:mid_idx])
            accum_l = np.min(lows[:mid_idx])
        else:
            accum_h = accum.high
            accum_l = accum.low

        found_manip = False
        manip_idx = -1
        manip_price = 0.0

        # Cerca Manipolazione (M)
        for j in range(mid_idx, min(mid_idx + 15, len(window))):
            if direction == "SELL" and highs[j] > accum_h:
                found_manip = True
                manip_idx = j
                manip_price = highs[j]
                break
            elif direction == "BUY" and lows[j] < accum_l:
                found_manip = True
                manip_idx = j
                manip_price = lows[j]
                break

        if not found_manip:
            return trades

        # 1. ENTRY_MANIPULATION (Aggressiva: sul picco di manipolazione)
        t_manip = times[manip_idx]
        if direction == "SELL":
            sl_price = manip_price + (CONFIG_EXEC.SL_BUFFER_PIPS * self.pip_size)
            entry_p = closes[manip_idx]
            sl_pips = (sl_price - entry_p) / self.pip_size
            tp1 = entry_p - ((entry_p - accum_l) * 0.50)
            if 3.0 <= sl_pips <= 20.0:
                trades.append(InstitutionalTrade(
                    trade_id=trade_id_start,
                    entry_type=EntryType.MANIPULATION,
                    timeframe=timeframe,
                    direction="SELL",
                    entry_time=t_manip,
                    entry_price=round(entry_p, 5),
                    stop_loss=round(sl_price, 5),
                    tp_50_eq=round(tp1, 5),
                    tp_structural=round(opposite_target, 5),
                    sl_pips=round(sl_pips, 1),
                    htf_key_level=htf_level.level_type
                ))
                trade_id_start += 1

            # 2. ENTRY_POST_BREAKOUT_RETEST (Conservativa: rottura con displacement sotto accum_l)
            for k in range(manip_idx + 1, min(manip_idx + 15, len(window))):
                if closes[k] < accum_l:
                    entry_retest = accum_l
                    sl_pips_r = (sl_price - entry_retest) / self.pip_size
                    tp1_r = entry_retest - ((entry_retest - opposite_target) * 0.50)
                    if 3.0 <= sl_pips_r <= 22.0:
                        trades.append(InstitutionalTrade(
                            trade_id=trade_id_start,
                            entry_type=EntryType.POST_BREAKOUT_RETEST,
                            timeframe=timeframe,
                            direction="SELL",
                            entry_time=times[k],
                            entry_price=round(entry_retest, 5),
                            stop_loss=round(sl_price, 5),
                            tp_50_eq=round(tp1_r, 5),
                            tp_structural=round(opposite_target, 5),
                            sl_pips=round(sl_pips_r, 1),
                            htf_key_level=htf_level.level_type
                        ))
                    break

        elif direction == "BUY":
            sl_price = manip_price - (CONFIG_EXEC.SL_BUFFER_PIPS * self.pip_size)
            entry_p = closes[manip_idx]
            sl_pips = (entry_p - sl_price) / self.pip_size
            tp1 = entry_p + ((accum_h - entry_p) * 0.50)
            if 3.0 <= sl_pips <= 20.0:
                trades.append(InstitutionalTrade(
                    trade_id=trade_id_start,
                    entry_type=EntryType.MANIPULATION,
                    timeframe=timeframe,
                    direction="BUY",
                    entry_time=t_manip,
                    entry_price=round(entry_p, 5),
                    stop_loss=round(sl_price, 5),
                    tp_50_eq=round(tp1, 5),
                    tp_structural=round(opposite_target, 5),
                    sl_pips=round(sl_pips, 1),
                    htf_key_level=htf_level.level_type
                ))
                trade_id_start += 1

            for k in range(manip_idx + 1, min(manip_idx + 15, len(window))):
                if closes[k] > accum_h:
                    entry_retest = accum_h
                    sl_pips_r = (entry_retest - sl_price) / self.pip_size
                    tp1_r = entry_retest + ((opposite_target - entry_retest) * 0.50)
                    if 3.0 <= sl_pips_r <= 22.0:
                        trades.append(InstitutionalTrade(
                            trade_id=trade_id_start,
                            entry_type=EntryType.POST_BREAKOUT_RETEST,
                            timeframe=timeframe,
                            direction="BUY",
                            entry_time=times[k],
                            entry_price=round(entry_retest, 5),
                            stop_loss=round(sl_price, 5),
                            tp_50_eq=round(tp1_r, 5),
                            tp_structural=round(opposite_target, 5),
                            sl_pips=round(sl_pips_r, 1),
                            htf_key_level=htf_level.level_type
                        ))
                    break

        return trades

    def _simulate_contextual_trade(self, trade: InstitutionalTrade, df_m1: pd.DataFrame):
        future = df_m1[df_m1.index > trade.entry_time]
        if len(future) == 0:
            return

        current_sl = trade.stop_loss
        max_adverse = 0.0

        for t_idx, row in future.iterrows():
            h, l = row['high'], row['low']

            if trade.direction == "SELL":
                adverse = max(0.0, (h - trade.entry_price) / self.pip_size)
                max_adverse = max(max_adverse, adverse)

                if h >= current_sl:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    if trade.hit_target_1:
                        trade.outcome = "ADJUSTED_WIN"
                        trade.adjusted_win = True
                        trade.realized_rr = 1.0
                    else:
                        trade.outcome = "LOSS"
                        trade.realized_rr = -1.0
                    return

                if not trade.hit_target_1 and l <= trade.tp_50_eq:
                    trade.hit_target_1 = True
                    current_sl = trade.entry_price + (trade.sl_pips * 0.35 * self.pip_size)

                if l <= trade.tp_structural:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    trade.strict_win = True
                    trade.adjusted_win = True
                    trade.outcome = "STRICT_WIN"
                    structural_dist = (trade.entry_price - trade.tp_structural) / self.pip_size
                    trade.realized_rr = round(structural_dist / trade.sl_pips, 2)
                    return

            elif trade.direction == "BUY":
                adverse = max(0.0, (trade.entry_price - l) / self.pip_size)
                max_adverse = max(max_adverse, adverse)

                if l <= current_sl:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    if trade.hit_target_1:
                        trade.outcome = "ADJUSTED_WIN"
                        trade.adjusted_win = True
                        trade.realized_rr = 1.0
                    else:
                        trade.outcome = "LOSS"
                        trade.realized_rr = -1.0
                    return

                if not trade.hit_target_1 and h >= trade.tp_50_eq:
                    trade.hit_target_1 = True
                    current_sl = trade.entry_price - (trade.sl_pips * 0.35 * self.pip_size)

                if h >= trade.tp_structural:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    trade.strict_win = True
                    trade.adjusted_win = True
                    trade.outcome = "STRICT_WIN"
                    structural_dist = (trade.tp_structural - trade.entry_price) / self.pip_size
                    trade.realized_rr = round(structural_dist / trade.sl_pips, 2)
                    return

        trade.mae_pips = round(max_adverse, 1)
