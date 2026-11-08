"""
Modulo Strategia Operativa:
Implementazione del ciclo A-M-D con logica gerarchica M5 (default) vs M1 (radiografia).
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from config import CONFIG_RISK, CONFIG_SESSION, CONFIG_STRATEGY
from core.indicators import (
    calculate_dealing_range,
    detect_swings,
    find_fair_value_gaps,
    is_in_killzone,
    FVG
)


@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: str          # 'SHORT' o 'LONG'
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_pips: float
    target_rr: float
    execution_timeframe: str # 'M5' o 'M1'
    reason: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None    # 'WIN', 'LOSS', 'RUNNING'
    pnl_pips: Optional[float] = None
    realized_rr: Optional[float] = None


class AMDMultiTimeframeEngine:
    def __init__(self, symbol: str = "EURUSD", pip_size: float = 0.0001):
        self.symbol = symbol
        self.pip_size = pip_size

    def analyze_and_backtest(
        self,
        df_m1: pd.DataFrame,
        df_m5: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None
    ) -> List[Trade]:
        """
        Esegue il backtest del modello su dati storici:
        1. Rileva zone di interesse HTF (H4 FVG/IMB)
        2. Analizza le sessioni attive (London / New York Killzones)
        3. Verifica il filtro Premium / Discount
        4. Decide se eseguire su M5 (default) o scendere su M1 (radiografia)
        5. Simula l'esecuzione a mercato con SL e TP
        """
        trades: List[Trade] = []
        trade_counter = 1

        # 1. Rileva Imbalance su H4 se disponibile, altrimenti su M5
        h4_fvgs = find_fair_value_gaps(df_h4 if df_h4 is not None else df_m5, pip_size=self.pip_size, min_gap_pips=4.0)

        # 2. Calcola Dealing Range su M5
        df_m5_dr = calculate_dealing_range(df_m5, lookback=CONFIG_STRATEGY.DEALING_RANGE_LOOKBACK)

        # Raggruppa i dati per giornata/sessione
        m5_index_times = df_m5_dr.index

        # Itera sulle candele M5
        for i in range(10, len(df_m5_dr) - 1):
            curr_time = m5_index_times[i]
            in_kz, session_name = is_in_killzone(curr_time)
            
            if not in_kz:
                continue

            row_m5 = df_m5_dr.iloc[i]
            prev_m5 = df_m5_dr.iloc[i - 1]
            
            # --- SCENARIO SHORT (Reazione ad Imbalance Bearish in Premium Zone) ---
            if row_m5['is_premium']:
                # Controlla se il prezzo tocca un H4 Bearish FVG
                touching_h4_imb = any(
                    fvg.fvg_type == 'BEARISH' and fvg.bottom <= row_m5['high'] <= fvg.top * 1.002
                    for fvg in h4_fvgs if fvg.timestamp < curr_time
                )
                
                # Se non c'è H4 FVG, consideriamo il test del recente Range High
                manipulating_high = (row_m5['high'] > row_m5['range_high'] * 0.9995) or touching_h4_imb

                if manipulating_high:
                    # VALUTAZIONE M5 vs M1:
                    # Se M5 ha una reiezione compressa in una singola wick senza struttura chiara:
                    candle_range = row_m5['high'] - row_m5['low']
                    body = abs(row_m5['close'] - row_m5['open'])
                    upper_wick = row_m5['high'] - max(row_m5['close'], row_m5['open'])
                    is_compressed_wick = (upper_wick > body * 1.5) and (candle_range > 0.0008)

                    if is_compressed_wick:
                        # NECESSARIA RADIOGRAFIA SU M1
                        trade = self._evaluate_m1_radiography(
                            df_m1, 
                            start_time=curr_time - pd.Timedelta(minutes=15),
                            end_time=curr_time + pd.Timedelta(minutes=30),
                            manipulation_extreme=row_m5['high'],
                            direction='SHORT',
                            trade_id=trade_counter
                        )
                        if trade:
                            trades.append(trade)
                            trade_counter += 1
                    else:
                        # M5 HA STRUTTURA CHIARA (candela decisa sotto MSS)
                        if row_m5['close'] < prev_m5['low']: # MSS su M5
                            sl_price = row_m5['high'] + (2.0 * self.pip_size)
                            entry_price = row_m5['close']
                            sl_pips = (sl_price - entry_price) / self.pip_size
                            
                            if sl_pips <= 20.0: # Limite ragionevole per M5
                                tp_price = entry_price - (sl_pips * CONFIG_RISK.MIN_RISK_REWARD * self.pip_size)
                                trade = Trade(
                                    trade_id=trade_counter,
                                    symbol=self.symbol,
                                    direction='SHORT',
                                    entry_time=curr_time,
                                    entry_price=entry_price,
                                    stop_loss=sl_price,
                                    take_profit=tp_price,
                                    sl_pips=round(sl_pips, 1),
                                    target_rr=CONFIG_RISK.MIN_RISK_REWARD,
                                    execution_timeframe='M5',
                                    reason=f"M5 Multi-Candle MSS in {session_name} Killzone (Premium Zone)"
                                )
                                trades.append(trade)
                                trade_counter += 1

        # Simula l'esito dei trade su M1
        self._simulate_outcomes(trades, df_m1)
        return trades

    def _evaluate_m1_radiography(
        self,
        df_m1: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        manipulation_extreme: float,
        direction: str,
        trade_id: int
    ) -> Optional[Trade]:
        """
        Analizza i dati a 1 minuto per trovare il pattern preciso di manipolazione e MSS
        con Stop Loss ridotto al minimo.
        """
        m1_slice = df_m1.loc[start_time:end_time]
        if len(m1_slice) < 5:
            return None

        swings_m1 = detect_swings(m1_slice, left_bars=2, right_bars=2)
        
        if direction == 'SHORT':
            swing_lows = [s for s in swings_m1 if s.swing_type == 'LOW']
            if not swing_lows:
                return None

            last_sl = swing_lows[-1]
            # Cerca rottura decisa di last_sl su M1
            broken_slice = m1_slice[m1_slice.index > last_sl.timestamp]
            breaks = broken_slice[broken_slice['close'] < last_sl.price]
            
            if len(breaks) > 0:
                mss_bar = breaks.iloc[0]
                entry_time = mss_bar.name
                
                # Retest dell'imbalance M1 o del livello rotto
                entry_price = last_sl.price
                sl_price = manipulation_extreme + (1.5 * self.pip_size)
                sl_pips = (sl_price - entry_price) / self.pip_size

                # Solo se rispetta il vincolo di stop stretto
                if 4.0 <= sl_pips <= CONFIG_RISK.MAX_SL_PIPS_M1:
                    tp_price = entry_price - (sl_pips * CONFIG_RISK.MIN_RISK_REWARD * self.pip_size)
                    return Trade(
                        trade_id=trade_id,
                        symbol=self.symbol,
                        direction='SHORT',
                        entry_time=entry_time,
                        entry_price=round(entry_price, 5),
                        stop_loss=round(sl_price, 5),
                        take_profit=round(tp_price, 5),
                        sl_pips=round(sl_pips, 1),
                        target_rr=CONFIG_RISK.MIN_RISK_REWARD,
                        execution_timeframe='M1',
                        reason="M1 Radiografia: Sweep IMB H4 -> Micro-MSS -> Retest con SL Millimetrico"
                    )
        return None

    def _simulate_outcomes(self, trades: List[Trade], df_m1: pd.DataFrame):
        """Simula il raggiungimento di TP o SL candela per candela."""
        highs = df_m1['high']
        lows = df_m1['low']
        
        for trade in trades:
            future_data = df_m1[df_m1.index > trade.entry_time]
            if len(future_data) == 0:
                trade.outcome = 'RUNNING'
                continue

            hit_sl = False
            hit_tp = False

            for t_idx, row in future_data.iterrows():
                h, l = row['high'], row['low']
                
                if trade.direction == 'SHORT':
                    if h >= trade.stop_loss:
                        trade.exit_time = t_idx
                        trade.exit_price = trade.stop_loss
                        trade.outcome = 'LOSS'
                        trade.pnl_pips = -trade.sl_pips
                        trade.realized_rr = -1.0
                        hit_sl = True
                        break
                    elif l <= trade.take_profit:
                        trade.exit_time = t_idx
                        trade.exit_price = trade.take_profit
                        trade.outcome = 'WIN'
                        trade.pnl_pips = trade.sl_pips * trade.target_rr
                        trade.realized_rr = trade.target_rr
                        hit_tp = True
                        break

            if not hit_sl and not hit_tp:
                trade.outcome = 'RUNNING'
