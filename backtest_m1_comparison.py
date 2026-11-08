"""
Backtest Comparativo Diretto su Dati Reali M1 (EURUSD & GBPUSD):
1. Strategia A: Entrate DENTRO l'Accumulazione ad M1 (Anticipazione sul range / bordi)
2. Strategia B: Entrate sulla ROTTURA CON IMBALANCE (Displacement + Retest FVG)
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from config import CONFIG_SESSION


@dataclass
class ComparisonTrade:
    trade_id: int
    strategy: str          # 'ACCUMULATION_EDGE' vs 'BREAKOUT_WITH_IMB'
    symbol: str
    direction: str         # 'BUY' o 'SELL'
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_pips: float
    outcome: Optional[str] = None    # 'WIN', 'LOSS', 'RUNNING'
    pnl_pips: Optional[float] = None
    realized_r: Optional[float] = None
    fakeout_detected: bool = False   # Se è stato spazzato via da un falso breakout


def download_real_m1_data(symbol: str = "EURUSD=X", period: str = "7d") -> pd.DataFrame:
    """Scarica candele M1 reali via Yahoo Finance e normalizza le colonne."""
    print(f"[+] Download dati reali M1 per {symbol} ({period})...")
    raw = yf.download(symbol, period=period, interval="1m", progress=False)
    
    # Gestisci MultiIndex se presente
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [col.lower() for col in raw.columns]

    df = raw[['open', 'high', 'low', 'close']].dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


def run_comparative_m1_backtest(df: pd.DataFrame, symbol_name: str = "EURUSD", pip_size: float = 0.0001):
    """
    Scansiona il mercato cercando fasi di accumulazione M1 e confronta le due modalità di ingresso:
    - Entrata sull'accumulo (anticipazione ai bordi)
    - Entrata sulla rottura con IMB (retest dell'imbalance post-displacement)
    """
    trades_accum: List[ComparisonTrade] = []
    trades_imb: List[ComparisonTrade] = []
    trade_id = 1

    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    times = df.index

    # Parametri operativi
    accum_len = 20           # 20 candele M1 di consolidamento (20 minuti)
    max_accum_pips = 10.0    # Range massimo per essere considerato accumulo (10 pips)
    target_rr = 2.5          # Target 1:2.5

    i = accum_len + 5
    while i < len(df) - 60:
        curr_time = times[i]
        
        # Filtro Killzone (Londra 08:00-11:30 o NY 13:00-16:30 CET / UTC+1)
        # Converti a ora UTC/locale
        t_hour = curr_time.hour
        in_kz = (7 <= t_hour <= 11) or (12 <= t_hour <= 16)
        if not in_kz:
            i += 1
            continue

        # 1. Rileva se le precedenti 20 candele formano un'ACCUMULAZIONE (Range Stretto)
        window_high = np.max(highs[i - accum_len:i])
        window_low = np.min(lows[i - accum_len:i])
        range_pips = (window_high - window_low) / pip_size

        if range_pips > max_accum_pips or range_pips < 3.0:
            i += 1
            continue

        # Range stretto confermato -> Abbiamo una potenziale ACCUMULAZIONE
        # -------------------------------------------------------------
        # SCENARIO SHORT (Rottura ribassista dell'accumulo o test del bordo superiore)
        # -------------------------------------------------------------
        
        # A) STRATEGIA 1: ENTRATA DENTRO L'ACCUMULO (Bordo Superiore)
        # Il trader vende vicino a window_high anticipando l'espansione verso il basso
        entry_accum_short = window_high - (1.0 * pip_size)
        sl_accum_short = window_high + (2.5 * pip_size)  # Stop stretto appena sopra il range
        sl_pips_accum = (sl_accum_short - entry_accum_short) / pip_size
        tp_accum_short = entry_accum_short - (sl_pips_accum * target_rr * pip_size)

        t_accum = ComparisonTrade(
            trade_id=trade_id,
            strategy="ACCUMULATION_EDGE",
            symbol=symbol_name,
            direction="SELL",
            entry_time=curr_time,
            entry_price=round(entry_accum_short, 5),
            stop_loss=round(sl_accum_short, 5),
            take_profit=round(tp_accum_short, 5),
            sl_pips=round(sl_pips_accum, 1)
        )

        # B) STRATEGIA 2: ENTRATA SULLA ROTTURA CON IMB (Displacement + Retest FVG)
        # Cerca nelle successive 1-10 candele se avviene una rottura violenta sotto window_low
        # che crea un Bearish FVG su M1
        imb_found = False
        for k in range(i, min(i + 15, len(df) - 5)):
            # Candela k chiude sotto window_low (Rottura)
            if closes[k] < window_low:
                # Controlla se c'è un Bearish FVG (Low[k-2] > High[k] con gap >= 1.5 pips)
                if k >= 2 and (lows[k - 2] - highs[k]) >= (1.5 * pip_size):
                    fvg_top = lows[k - 2]
                    fvg_bottom = highs[k]
                    entry_imb = (fvg_top + fvg_bottom) / 2.0  # Retest al 50% dell'IMB
                    
                    # Stop loss sopra il massimo di manipolazione recente
                    manipulation_high = np.max(highs[i - 5:k + 1])
                    sl_imb = manipulation_high + (1.5 * pip_size)
                    sl_pips_imb = (sl_imb - entry_imb) / pip_size

                    if 4.0 <= sl_pips_imb <= 12.0:
                        tp_imb = entry_imb - (sl_pips_imb * target_rr * pip_size)
                        t_imb = ComparisonTrade(
                            trade_id=trade_id,
                            strategy="BREAKOUT_WITH_IMB",
                            symbol=symbol_name,
                            direction="SELL",
                            entry_time=times[k],
                            entry_price=round(entry_imb, 5),
                            stop_loss=round(sl_imb, 5),
                            take_profit=round(tp_imb, 5),
                            sl_pips=round(sl_pips_imb, 1)
                        )
                        imb_found = True
                        break

        # Simula l'esito di entrambi nelle candele future
        _simulate_trade(t_accum, df, i)
        trades_accum.append(t_accum)

        if imb_found:
            _simulate_trade(t_imb, df, k)
            trades_imb.append(t_imb)

        trade_id += 1
        # Salta avanti per non sovrapporre la stessa fase di accumulo
        i += 30

    return trades_accum, trades_imb


def _simulate_trade(trade: ComparisonTrade, df: pd.DataFrame, start_idx: int):
    """Simula l'esito del trade candela per candela verificando stop, target o fakeout."""
    highs = df['high'].values
    lows = df['low'].values
    
    for j in range(start_idx, min(start_idx + 120, len(df))): # Finestra max 2 ore
        h, l = highs[j], lows[j]
        
        if trade.direction == "SELL":
            # Hit Stop Loss
            if h >= trade.stop_loss:
                trade.outcome = "LOSS"
                trade.pnl_pips = -trade.sl_pips
                trade.realized_r = -1.0
                # Verifica se era un fakeout (manipolazione che ha toccato lo stop prima di scendere)
                future_min = np.min(lows[j:min(j + 30, len(df))])
                if future_min <= trade.take_profit:
                    trade.fakeout_detected = True
                return
            # Hit Take Profit
            elif l <= trade.take_profit:
                trade.outcome = "WIN"
                trade.pnl_pips = trade.sl_pips * 2.5
                trade.realized_r = 2.5
                return

    trade.outcome = "RUNNING"


def print_comparison_report(trades_accum: List[ComparisonTrade], trades_imb: List[ComparisonTrade], asset_name: str):
    """Stampa il report comparativo dettagliato."""
    def calc_stats(trades: List[ComparisonTrade]):
        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        total = len(wins) + len(losses)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0
        total_r = sum(t.realized_r for t in trades if t.realized_r is not None)
        total_pips = sum(t.pnl_pips for t in trades if t.pnl_pips is not None)
        fakeouts = sum(1 for t in trades if t.fakeout_detected)
        avg_sl = np.mean([t.sl_pips for t in trades]) if trades else 0.0
        return {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_r": total_r,
            "total_pips": total_pips,
            "fakeouts": fakeouts,
            "avg_sl": avg_sl
        }

    s_accum = calc_stats(trades_accum)
    s_imb = calc_stats(trades_imb)

    wl_accum = f"{s_accum['wins']}W / {s_accum['losses']}L"
    wl_imb = f"{s_imb['wins']}W / {s_imb['losses']}L"
    r_accum = f"{s_accum['total_r']:+.1f} R"
    r_imb = f"{s_imb['total_r']:+.1f} R"
    pips_accum = f"{s_accum['total_pips']:+.1f} pips"
    pips_imb = f"{s_imb['total_pips']:+.1f} pips"
    sl_accum = f"{s_accum['avg_sl']:.1f} pips"
    sl_imb = f"{s_imb['avg_sl']:.1f} pips"

    print("\n" + "=" * 76)
    print(f"       REPORT COMPARATIVO M1: {asset_name} (DATI REALI DI MERCATO)")
    print("=" * 76)
    print(f"{'METRICA':<36} | {'ENTRATA ACCUMULO (A)':<17} | {'ROTTURA CON IMB (B)':<16}")
    print("-" * 76)
    print(f"{'Trade Totali Conclusi':<36} | {s_accum['total']:<17} | {s_imb['total']:<16}")
    print(f"{'Win Rate (%)':<36} | {s_accum['win_rate']:>15.1f}% | {s_imb['win_rate']:>14.1f}%")
    print(f"{'Operazioni Vincenti / Perdenti':<36} | {wl_accum:<17} | {wl_imb:<16}")
    print(f"{'Rendimento Netto (R-Multiple)':<36} | {r_accum:>16} | {r_imb:>15}")
    print(f"{'Guadagno Netto in Pips':<36} | {pips_accum:>16} | {pips_imb:>15}")
    print(f"{'Stop Loss Medio (Pips)':<36} | {sl_accum:>16} | {sl_imb:>15}")
    print(f"{'Vittime di Stop Hunt / Fakeout':<36} | {s_accum['fakeouts']:<17} | {s_imb['fakeouts']:<16}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    # Test su EUR/USD
    df_eu = download_real_m1_data("EURUSD=X", period="7d")
    eu_accum, eu_imb = run_comparative_m1_backtest(df_eu, symbol_name="EURUSD")
    print_comparison_report(eu_accum, eu_imb, "EUR/USD")

    # Test su GBP/USD
    df_gu = download_real_m1_data("GBPUSD=X", period="7d")
    gu_accum, gu_imb = run_comparative_m1_backtest(df_gu, symbol_name="GBPUSD")
    print_comparison_report(gu_accum, gu_imb, "GBP/USD")
