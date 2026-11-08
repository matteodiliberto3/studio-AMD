"""
Modulo Strategia Operativa Istituzionale:
Implementa il meccanismo unificato di Entry per entrambe le morfologie di accumulazione:
1. AMD_HORIZONTAL_BOX:
   Box orizzontale (M/m) vicino al KL MTF -> Manipolazione fino al KL -> Rottura di m/M con IMB ->
   Almeno 2 chiusure oltre m/M -> Rientro (Retest) nell'IMB -> TP su Target MTF.
2. DYNAMIC_ACCUMULATION_BOS:
   Accumulazione dinamica a swing (M/m) sul KL MTF -> Rottura di m/M con IMB ->
   Almeno 2 chiusure oltre m/M -> Rientro (Retest) nell'IMB -> TP su Target MTF.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from config import (
    CONFIG_ASSET,
    CONFIG_EXEC,
    AccumulationType
)
from core.indicators import (
    BreakoutImbalance,
    HTFKeyLevel,
    detect_breakout_with_imb_and_two_closes,
    detect_dynamic_swing_level,
    detect_horizontal_accumulation_box,
    find_validated_mtf_key_levels,
    is_active_operating_window,
    is_key_level_invalidated_by_closes
)


@dataclass
class InstitutionalTrade:
    trade_id: int
    symbol: str
    accumulation_type: AccumulationType
    timeframe: str                  # 'M5' o 'M1'
    direction: str                  # 'BUY' o 'SELL'
    manip_time: pd.Timestamp        # Data/Ora del picco di manipolazione sul Key Level MTF
    confirm_time: pd.Timestamp      # Data/Ora della 2a chiusura di conferma oltre M/m
    entry_time: pd.Timestamp        # Data/Ora del rientro (retest) effettivo nell'IMB
    htf_key_level_type: str         # Tipo e TF del Key Level MTF (es. H1_MTF_IMBALANCE)
    htf_key_level_price: float      # Prezzo di riferimento del Key Level MTF
    level_M_or_m: float             # Prezzo del livello M o m rotto con IMB
    imb_top: float                  # Limite superiore dell'IMB di rottura
    imb_bottom: float               # Limite inferiore dell'IMB di rottura
    entry_price: float              # Prezzo di esecuzione sul rientro nell'IMB
    stop_loss: float                # Stop Loss oltre il picco di manipolazione
    tp_50_eq: float                 # Target intermedio (50% del range verso Target MTF)
    tp_structural: float            # Target Finale MTF (che può generare inversione)
    sl_pips: float
    mae_pips: float = 0.0           # Maximum Adverse Excursion in pips dopo il retest
    hit_target_1: bool = False
    strict_win: bool = False        # Ha raggiunto il Target MTF Finale
    adjusted_win: bool = False      # Ha raggiunto almeno il 50% + trailing su swing
    outcome: str = "RUNNING"        # 'WIN', 'PARTIAL', 'LOSS', 'RUNNING'
    exit_time: Optional[pd.Timestamp] = None
    realized_rr: Optional[float] = None


class InstitutionalAMDEngine:
    def __init__(self, symbol: str = "EURUSD", pip_size: float = CONFIG_ASSET.PIP_SIZE_DEFAULT):
        self.symbol = symbol
        self.pip_size = pip_size

    def analyze_market(
        self,
        df_m1: pd.DataFrame,
        df_m5: pd.DataFrame,
        df_h4: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None
    ) -> List[InstitutionalTrade]:
        """
        Analizza il mercato cercando sui Key Levels MTF validati (H4 e H1)
        la sequenza completa su M5 (default) o su M1 (raggi X quando M5 è compresso).
        """
        trades: List[InstitutionalTrade] = []
        trade_id = 1

        # 1. Costruzione dei Key Levels MTF validati (H4 + H1), ordinati cronologicamente
        mtf_levels: List[HTFKeyLevel] = []
        mtf_levels.extend(find_validated_mtf_key_levels(df_h4, tf_label="H4", lookback_range=15, pip_size=self.pip_size))
        if df_h1 is not None and len(df_h1) >= 6:
            mtf_levels.extend(find_validated_mtf_key_levels(df_h1, tf_label="H1", lookback_range=24, pip_size=self.pip_size))
        mtf_levels.sort(key=lambda x: x.timestamp)

        m5_times = df_m5.index
        m5_highs = df_m5['high'].values
        m5_lows = df_m5['low'].values

        last_entry_End_time = m5_times[0] - pd.Timedelta(days=1)
        kl_left_idx = 0
        n_kls = len(mtf_levels)
        max_kl_age = pd.Timedelta(days=4)

        i = 15
        while i < len(df_m5) - 20:
            t_curr = m5_times[i]

            if t_curr <= last_entry_End_time:
                i += 1
                continue

            is_valid_time, _ = is_active_operating_window(t_curr)
            if not is_valid_time:
                i += 1
                continue

            # Avanza il puntatore sinistro per scartare i Key Levels più vecchi di 4 giorni
            min_kl_time = t_curr - max_kl_age
            while kl_left_idx < n_kls and mtf_levels[kl_left_idx].timestamp < min_kl_time:
                kl_left_idx += 1

            # Filtra i Key Levels MTF attivi nella finestra [t_curr - 4 giorni, t_curr)
            active_kls: List[HTFKeyLevel] = []
            k_ptr = kl_left_idx
            while k_ptr < n_kls and mtf_levels[k_ptr].timestamp < t_curr:
                kl_cand = mtf_levels[k_ptr]
                if not kl_cand.invalidated:
                    active_kls.append(kl_cand)
                k_ptr += 1

            if not active_kls:
                i += 1
                continue

            c_high = float(m5_highs[i])
            c_low = float(m5_lows[i])

            # Cerca se la candela M5 corrente sta toccando/manipolando un Key Level MTF
            touched_kl: Optional[HTFKeyLevel] = None
            for kl in reversed(active_kls):
                if kl.direction == "BEARISH":
                    if c_high >= kl.bottom - (1.0 * self.pip_size) and c_low <= kl.top + (3.0 * self.pip_size):
                        touched_kl = kl
                        break
                elif kl.direction == "BULLISH":
                    if c_low <= kl.top + (1.0 * self.pip_size) and c_high >= kl.bottom - (3.0 * self.pip_size):
                        touched_kl = kl
                        break

            if touched_kl is None:
                i += 1
                continue

            # Regola Fonte 16 & Validità del Range MTF:
            # Verifica sull'intero storico dalla nascita del KL fino ad ora se:
            # 1) Ci sono state >= 2 chiusure consecutive oltre il Key Level MTF
            # 2) L'origine del Range MTF è stata violata (rottura struttura contraria)
            check_slice = df_m5.loc[touched_kl.timestamp:m5_times[min(len(df_m5) - 1, i + 1)]]
            range_broken = False
            if len(check_slice) > 2:
                if touched_kl.direction == "BEARISH" and float(check_slice['high'].iloc[:-1].max()) > touched_kl.range_high + (2.0 * self.pip_size):
                    range_broken = True
                elif touched_kl.direction == "BULLISH" and float(check_slice['low'].iloc[:-1].min()) < touched_kl.range_low - (2.0 * self.pip_size):
                    range_broken = True

            if range_broken or is_key_level_invalidated_by_closes(check_slice, touched_kl):
                touched_kl.invalidated = True
                i += 1
                continue

            # Scelta Dinamica M5 vs M1 ("Raggi X"):
            # 1) Proviamo prima a validare la sequenza completa a corpo su M5
            trade_m5 = self._try_build_entry_on_timeframe(
                df_ltf=df_m5,
                timeframe="M5",
                center_time=t_curr,
                kl=touched_kl,
                trade_id=trade_id
            )

            if trade_m5 is not None:
                self._simulate_contextual_trade(trade_m5, df_m1)
                trades.append(trade_m5)
                touched_kl.invalidated = True
                trade_id += 1
                last_entry_End_time = trade_m5.entry_time + pd.Timedelta(minutes=30)
                i += 6
                continue

            # 2) Se su M5 il movimento è compresso in un'ombra/wick e non mostra la struttura a corpo,
            #    scendiamo a "Raggi X" su M1 per cercare la stessa sequenza chirurgica
            trade_m1 = self._try_build_entry_on_timeframe(
                df_ltf=df_m1,
                timeframe="M1",
                center_time=t_curr,
                kl=touched_kl,
                trade_id=trade_id
            )

            if trade_m1 is not None:
                self._simulate_contextual_trade(trade_m1, df_m1)
                trades.append(trade_m1)
                touched_kl.invalidated = True
                trade_id += 1
                last_entry_End_time = trade_m1.entry_time + pd.Timedelta(minutes=25)
                i += 5
                continue

            i += 1

        return trades

    def _try_build_entry_on_timeframe(
        self,
        df_ltf: pd.DataFrame,
        timeframe: str,
        center_time: pd.Timestamp,
        kl: HTFKeyLevel,
        trade_id: int
    ) -> Optional[InstitutionalTrade]:
        """
        Cerca sul timeframe specificato (M5 o M1) la sequenza completa:
        1. Accumulazione (Orizzontale con Box M/m oppure Dinamica con Swing M/m)
        2. Manipolazione fino al Key Level MTF (senza >= 2 chiusure oltre il KL)
        3. Rottura di M/m con IMB
        4. Almeno 2 chiusure di candela oltre M/m
        5. Rientro (Retest) effettivo dentro quell'IMB
        """
        pre_delta = pd.Timedelta(minutes=25 if timeframe == "M1" else 90)
        post_delta = pd.Timedelta(minutes=60 if timeframe == "M1" else 180)
        window = df_ltf.loc[center_time - pre_delta:center_time + post_delta]
        if len(window) < 14:
            return None

        # Verifica su LTF che non ci siano >= 2 chiusure consecutive oltre l'estremo del KL
        if is_key_level_invalidated_by_closes(window.iloc[:len(window) // 2], kl):
            return None

        highs = window['high'].values
        lows = window['low'].values
        closes = window['close'].values
        times = window.index

        direction = "SELL" if kl.direction == "BEARISH" else "BUY"

        # Individua l'indice del picco di manipolazione sul Key Level intorno a center_time
        center_Zone_end = min(len(window) - 6, int(len(window) * 0.55))
        center_Zone_start = max(4, int(len(window) * 0.15))
        if center_Zone_end <= center_Zone_start:
            return None

        if direction == "SELL":
            manip_idx = center_Zone_start + int(np.argmax(highs[center_Zone_start:center_Zone_end]))
            manip_extreme = float(highs[manip_idx])
            if manip_extreme < kl.bottom - (1.5 * self.pip_size):
                return None
        else:
            manip_idx = center_Zone_start + int(np.argmin(lows[center_Zone_start:center_Zone_end]))
            manip_extreme = float(lows[manip_idx])
            if manip_extreme > kl.top + (1.5 * self.pip_size):
                return None

        manip_time = times[manip_idx]

        # PASSO 1: Verifica se c'è un'Accumulazione Orizzontale Classica (Box M/m)
        horiz_box = detect_horizontal_accumulation_box(
            window, manip_idx=manip_idx, timeframe=timeframe, pip_size=self.pip_size
        )

        accum_type: Optional[AccumulationType] = None
        level_M_or_m: Optional[float] = None

        if horiz_box is not None:
            # Per essere una vera A-M-D sulla box, la manipolazione deve essere uscita dalla box
            if direction == "SELL" and manip_extreme > horiz_box.high_M:
                accum_type = AccumulationType.AMD_HORIZONTAL_BOX
                level_M_or_m = horiz_box.low_m
            elif direction == "BUY" and manip_extreme < horiz_box.low_m:
                accum_type = AccumulationType.AMD_HORIZONTAL_BOX
                level_M_or_m = horiz_box.high_M

        # PASSO 2: Se non è una box orizzontale piatta, verifica l'Accumulazione Dinamica (Swing M/m)
        if accum_type is None:
            dyn_swing = detect_dynamic_swing_level(
                window, manip_idx=manip_idx, direction=direction, lookback=10, pip_size=self.pip_size
            )
            if dyn_swing is None:
                return None
            accum_type = AccumulationType.DYNAMIC_ACCUMULATION_BOS
            level_M_or_m = dyn_swing.level_price

        # PASSO 3: Rottura di M/m con IMB + Almeno 2 Chiusure oltre M/m
        breakout_res = detect_breakout_with_imb_and_two_closes(
            window,
            manip_idx=manip_idx,
            level_M_or_m=level_M_or_m,
            direction=direction,
            timeframe=timeframe,
            max_search_bars=14,
            pip_size=self.pip_size
        )
        if breakout_res is None:
            return None

        imb, confirm_idx, confirm_time = breakout_res

        # PASSO 4: Attesa del RIENTRO (RETEST) dentro quell'IMB dopo le 2 chiusure di conferma
        # Il prezzo di Limit è il bordo prossimale dell'IMB (o il suo midpoint se molto stretto)
        if direction == "SELL":
            limit_entry_price = imb.bottom if imb.gap_pips >= 1.2 else imb.midpoint
            sl_price = manip_extreme + (CONFIG_EXEC.SL_BUFFER_PIPS * self.pip_size)
            sl_pips = (sl_price - limit_entry_price) / self.pip_size
        else:
            limit_entry_price = imb.top if imb.gap_pips >= 1.2 else imb.midpoint
            sl_price = manip_extreme - (CONFIG_EXEC.SL_BUFFER_PIPS * self.pip_size)
            sl_pips = (limit_entry_price - sl_price) / self.pip_size

        max_sl_allowed = 14.0 if timeframe == "M1" else 24.0
        if not (2.5 <= sl_pips <= max_sl_allowed):
            return None

        # Target MTF (struttura opposta che può generare un'inversione) e Target 50% intermedio
        tp_structural = kl.target_mtf
        if direction == "SELL":
            if tp_structural >= limit_entry_price - (sl_pips * 1.5 * self.pip_size):
                tp_structural = round(limit_entry_price - (sl_pips * 3.0 * self.pip_size), 5)
            tp_50 = round(limit_entry_price - ((limit_entry_price - tp_structural) * 0.50), 5)
        else:
            if tp_structural <= limit_entry_price + (sl_pips * 1.5 * self.pip_size):
                tp_structural = round(limit_entry_price + (sl_pips * 3.0 * self.pip_size), 5)
            tp_50 = round(limit_entry_price + ((tp_structural - limit_entry_price) * 0.50), 5)

        # Cerca la candela successiva a `confirm_idx` che RIENTRA nell'IMB per eseguire il Limit
        retest_end = min(len(window), confirm_idx + 1 + CONFIG_EXEC.MAX_BARS_WAIT_FOR_IMB_RETEST)
        entry_time: Optional[pd.Timestamp] = None

        for r_idx in range(confirm_idx + 1, retest_end):
            r_high = float(highs[r_idx])
            r_low = float(lows[r_idx])

            if direction == "SELL":
                # Se il prezzo è già andato a TP1 prima di ritestare l'IMB, l'ordine viene cancellato
                if r_low <= tp_50:
                    return None
                # Il prezzo risale e rientra nell'IMB!
                if r_high >= limit_entry_price:
                    # Verifica che non abbia bucato direttamente lo SL nella stessa candela prima del fill
                    entry_time = times[r_idx]
                    break
            else:
                if r_high >= tp_50:
                    return None
                # Il prezzo scende e rientra nell'IMB!
                if r_low <= limit_entry_price:
                    entry_time = times[r_idx]
                    break

        if entry_time is None:
            return None

        return InstitutionalTrade(
            trade_id=trade_id,
            symbol=self.symbol,
            accumulation_type=accum_type,
            timeframe=timeframe,
            direction=direction,
            manip_time=manip_time,
            confirm_time=confirm_time,
            entry_time=entry_time,
            htf_key_level_type=f"{kl.timeframe}_{kl.level_type}",
            htf_key_level_price=kl.midpoint,
            level_M_or_m=round(level_M_or_m, 5),
            imb_top=imb.top,
            imb_bottom=imb.bottom,
            entry_price=round(limit_entry_price, 5),
            stop_loss=round(sl_price, 5),
            tp_50_eq=round(tp_50, 5),
            tp_structural=round(tp_structural, 5),
            sl_pips=round(sl_pips, 1)
        )

    def _simulate_contextual_trade(self, trade: InstitutionalTrade, df_m1: pd.DataFrame):
        """
        Simula la vita del trade dopo il rientro nell'IMB:
        - Calcola il MAE (drawdown massimo dal prezzo di entry sull'IMB)
        - Al raggiungimento del 50% (tp_50_eq), incassa il parziale e sposta lo SL dietro lo swing
        - Chiude il resto al raggiungimento del Target MTF opposto (tp_structural)
        """
        future = df_m1.loc[
            trade.entry_time + pd.Timedelta(minutes=1) : trade.entry_time + pd.Timedelta(days=5)
        ]
        if len(future) == 0:
            return

        current_sl = trade.stop_loss
        max_adverse = 0.0
        f_highs = future['high'].values
        f_lows = future['low'].values
        f_times = future.index

        for idx in range(len(future)):
            h = float(f_highs[idx])
            l = float(f_lows[idx])
            t_idx = f_times[idx]

            if trade.direction == "SELL":
                adverse = max(0.0, (h - trade.entry_price) / self.pip_size)
                max_adverse = max(max_adverse, adverse)

                if h >= current_sl:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    if trade.hit_target_1:
                        trade.outcome = "PARTIAL"
                        trade.adjusted_win = True
                        dist_t1 = (trade.entry_price - trade.tp_50_eq) / self.pip_size
                        trade.realized_rr = round((dist_t1 / trade.sl_pips) * 0.50, 2)
                    else:
                        trade.outcome = "LOSS"
                        trade.realized_rr = -1.0
                    return

                if not trade.hit_target_1 and l <= trade.tp_50_eq:
                    trade.hit_target_1 = True
                    # Sposta SL dietro il nuovo swing strutturale (MAI a breakeven sull'entrata)
                    current_sl = round(trade.entry_price + (trade.sl_pips * 0.25 * self.pip_size), 5)

                if l <= trade.tp_structural:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    trade.strict_win = True
                    trade.adjusted_win = True
                    trade.outcome = "WIN"
                    dist_fin = (trade.entry_price - trade.tp_structural) / self.pip_size
                    trade.realized_rr = round(dist_fin / trade.sl_pips, 2)
                    return

            elif trade.direction == "BUY":
                adverse = max(0.0, (trade.entry_price - l) / self.pip_size)
                max_adverse = max(max_adverse, adverse)

                if l <= current_sl:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    if trade.hit_target_1:
                        trade.outcome = "PARTIAL"
                        trade.adjusted_win = True
                        dist_t1 = (trade.tp_50_eq - trade.entry_price) / self.pip_size
                        trade.realized_rr = round((dist_t1 / trade.sl_pips) * 0.50, 2)
                    else:
                        trade.outcome = "LOSS"
                        trade.realized_rr = -1.0
                    return

                if not trade.hit_target_1 and h >= trade.tp_50_eq:
                    trade.hit_target_1 = True
                    current_sl = round(trade.entry_price - (trade.sl_pips * 0.25 * self.pip_size), 5)

                if h >= trade.tp_structural:
                    trade.exit_time = t_idx
                    trade.mae_pips = round(max_adverse, 1)
                    trade.strict_win = True
                    trade.adjusted_win = True
                    trade.outcome = "WIN"
                    dist_fin = (trade.tp_structural - trade.entry_price) / self.pip_size
                    trade.realized_rr = round(dist_fin / trade.sl_pips, 2)
                    return

        trade.mae_pips = round(max_adverse, 1)


def export_trades_to_csv(
    trades: List[InstitutionalTrade],
    output_path: Union[str, Path] = "data/trades_executed_log.csv"
) -> pd.DataFrame:
    """
    Esporta l'elenco dettagliato dei trade eseguiti in CSV con tutti gli orari
    e i livelli esatti per la verifica visiva su TradingView.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in trades:
        rows.append({
            "ID": t.trade_id,
            "Symbol": t.symbol,
            "Data_Ora_Manipolazione": t.manip_time.strftime("%Y-%m-%d %H:%M"),
            "Data_Ora_2a_Chiusura_Conferma": t.confirm_time.strftime("%Y-%m-%d %H:%M"),
            "Data_Ora_Entry_Rientro_IMB": t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "Timeframe_Usato": t.timeframe,
            "Tipo_Entry": t.accumulation_type.value,
            "Direzione": t.direction,
            "HTF_Key_Level_Type": t.htf_key_level_type,
            "HTF_Key_Level_Price": t.htf_key_level_price,
            "Livello_M_o_m_Rotto": t.level_M_or_m,
            "IMB_Top": t.imb_top,
            "IMB_Bottom": t.imb_bottom,
            "Entry_Price": t.entry_price,
            "SL_Price": t.stop_loss,
            "TP_50_Price": t.tp_50_eq,
            "TP_Struct_Price": t.tp_structural,
            "SL_Pips": t.sl_pips,
            "MAE_Pips": t.mae_pips,
            "Esito": t.outcome,
            "R_Realizzato": t.realized_rr if t.realized_rr is not None else 0.0
        })

    df_log = pd.DataFrame(rows)
    df_log.to_csv(path, index=False)
    return df_log
