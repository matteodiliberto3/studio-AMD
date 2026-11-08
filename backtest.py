"""
Runner di Backtest Istituzionale su Dati Reali:
Confronta le due morfologie del meccanismo di Entry:
1. AMD_HORIZONTAL_BOX (Box orizzontale M/m -> Manipolazione su KL -> Rottura con IMB -> >=2 chiusure -> Rientro in IMB)
2. DYNAMIC_ACCUMULATION_BOS (Accumulo dinamico M/m -> Rottura con IMB -> >=2 chiusure -> Rientro in IMB)

Esporta il registro completo dei trade in `data/trades_executed_log.csv` per verifica su TradingView.
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np

from config import AccumulationType
from core.data_loader import (
    fetch_and_save_real_market_data,
    load_candles_csv,
    resample_candles
)
from core.strategy import (
    InstitutionalAMDEngine,
    InstitutionalTrade,
    export_trades_to_csv
)


def print_comparison_table(
    group1: List[InstitutionalTrade],
    group2: List[InstitutionalTrade],
    name1: str,
    name2: str,
    title: str
):
    def get_stats(subset: List[InstitutionalTrade]):
        total = len(subset)
        if total == 0:
            return {
                "total": 0,
                "strict_wr": 0.0,
                "adj_wr": 0.0,
                "avg_mae": 0.0,
                "avg_sl": 0.0,
                "avg_rr": 0.0,
                "total_rr": 0.0
            }
        strict_w = sum(1 for t in subset if t.strict_win)
        adj_w = sum(1 for t in subset if t.adjusted_win)
        avg_mae = float(np.mean([t.mae_pips for t in subset]))
        avg_sl = float(np.mean([t.sl_pips for t in subset]))
        rrs = [t.realized_rr for t in subset if t.realized_rr is not None]
        avg_rr = float(np.mean(rrs)) if rrs else 0.0
        total_rr = float(np.sum(rrs)) if rrs else 0.0
        return {
            "total": total,
            "strict_wr": (strict_w / total * 100.0),
            "adj_wr": (adj_w / total * 100.0),
            "avg_mae": avg_mae,
            "avg_sl": avg_sl,
            "avg_rr": avg_rr,
            "total_rr": total_rr
        }

    s1 = get_stats(group1)
    s2 = get_stats(group2)

    print("\n" + "=" * 80)
    print(f"   {title}")
    print("=" * 80)
    print(f"{'METRICA':<38} | {name1:<18} | {name2:<18}")
    print("-" * 80)
    print(f"{'Operazioni Totali':<38} | {s1['total']:<18} | {s2['total']:<18}")
    print(f"{'Strict Win Rate (Target MTF Finale)':<38} | {s1['strict_wr']:>16.1f}% | {s2['strict_wr']:>16.1f}%")
    print(f"{'Adjusted Win Rate (Con Parziale 50%)':<38} | {s1['adj_wr']:>16.1f}% | {s2['adj_wr']:>16.1f}%")
    print(f"{'Stop Loss Medio (Pips)':<38} | {s1['avg_sl']:>16.1f} p | {s2['avg_sl']:>16.1f} p")
    print(f"{'MAE Medio / Drawdown da Retest IMB':<38} | {s1['avg_mae']:>16.1f} p | {s2['avg_mae']:>16.1f} p")
    print(f"{'R:R Medio per Operazione':<38} | {s1['avg_rr']:>+16.2f} R | {s2['avg_rr']:>+16.2f} R")
    print(f"{'Rendimento Netto Complessivo':<38} | {s1['total_rr']:>+16.2f} R | {s2['total_rr']:>+16.2f} R")
    print("=" * 80)


def run_institutional_backtest(
    csv_path: str = "data/eurusd_m1_sample.csv",
    symbol: str = "EURUSD",
    refresh_real_data: bool = False
):
    path = Path(csv_path)
    if refresh_real_data or not path.exists():
        ticker = "EUR/USD" if "EUR" in symbol.upper() else "GBP/USD"
        print(f"[+] Download dati REALI ECN di mercato per {symbol} ({ticker})...")
        fetch_and_save_real_market_data(ticker=ticker, output_path=path, days=14)

    print("\n" + "=" * 80)
    print(f"   BACKTEST MECCANISMO DI ENTRY ISTITUZIONALE SU DATI REALI ({symbol})")
    print("   Sequenza: KL MTF -> Accumulo -> Rottura M/m con IMB -> >=2 Chiusure -> Retest IMB")
    print("=" * 80)

    df_m1 = load_candles_csv(path)
    df_m5 = resample_candles(df_m1, target_tf="5min")
    df_m15 = resample_candles(df_m1, target_tf="15min")
    df_h1 = resample_candles(df_m1, target_tf="1h")
    df_h4 = resample_candles(df_m1, target_tf="4h")

    print(f"[+] Candele M1 Reali: {len(df_m1)} ({df_m1.index[0]} -> {df_m1.index[-1]})")
    print(f"[+] Catena Frattale Timeframe: M5={len(df_m5)}, M15={len(df_m15)}, H1={len(df_h1)}, H4={len(df_h4)}")

    engine = InstitutionalAMDEngine(symbol=symbol)
    trades = engine.analyze_market(df_m1=df_m1, df_m5=df_m5, df_h4=df_h4, df_h1=df_h1, df_m15=df_m15)

    if not trades:
        print("[!] Nessun trade trovato nel periodo selezionato.")
        return []

    trades_box = [t for t in trades if t.accumulation_type == AccumulationType.AMD_HORIZONTAL_BOX]
    trades_dyn = [t for t in trades if t.accumulation_type == AccumulationType.DYNAMIC_ACCUMULATION_BOS]

    trades_m5 = [t for t in trades if t.timeframe == "M5"]
    trades_m1 = [t for t in trades if t.timeframe == "M1"]

    print_comparison_table(
        trades_box, trades_dyn,
        "AMD_HORIZ_BOX", "DYNAMIC_BOS_IMB",
        f"CONFRONTO MORFOLOGIA ACCUMULO ({symbol}): BOX ORIZZONTALE A-M-D vs DINAMICA M/m"
    )

    print_comparison_table(
        trades_m5, trades_m1,
        "ESECUZIONE M5", "RAGGI X M1",
        f"CONFRONTO TIMEFRAME ({symbol}): M5 (CORPO VISIBILE) vs M1 (OMBRA COMPRESSA)"
    )

    log_path = Path("data/trades_executed_log.csv")
    export_trades_to_csv(trades, output_path=log_path)

    print(f"\n[+] LOG DETTAGLIATO PER VERIFICA SU TRADINGVIEW (Salvato in {log_path}):")
    print("-" * 124)
    print(
        f"{'ID':<3} | {'MANIP TIME (CET)':<16} | {'ENTRY RETEST':<16} | {'TF':<2} | {'TIPO ACCUMULO':<15} | "
        f"{'DIR':<4} | {'M/m':<7} | {'ENTRY':<7} | {'SL':<7} | {'TP MTF':<7} | {'MAE':<5} | {'ESITO':<7} | {'R:R':<5}"
    )
    print("-" * 124)
    for t in trades:
        m_str = t.manip_time.strftime("%Y-%m-%d %H:%M")
        e_str = t.entry_time.strftime("%Y-%m-%d %H:%M")
        t_label = "AMD_BOX" if t.accumulation_type == AccumulationType.AMD_HORIZONTAL_BOX else "DYNAMIC_M/m"
        r_str = f"{t.realized_rr:+.2f}R" if t.realized_rr is not None else "-"
        print(
            f"{t.trade_id:<3} | {m_str:<16} | {e_str:<16} | {t.timeframe:<2} | {t_label:<15} | "
            f"{t.direction:<4} | {t.level_M_or_m:<7.5f} | {t.entry_price:<7.5f} | {t.stop_loss:<7.5f} | "
            f"{t.tp_structural:<7.5f} | {t.mae_pips:<4.1f}p | {t.outcome:<7} | {r_str:<5}"
        )
    print("=" * 124 + "\n")
    return trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest FX Institutional Framework")
    parser.add_argument("--csv", type=str, default="data/eurusd_m1_sample.csv")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--refresh", action="store_true", help="Scarica nuovi dati reali di mercato")
    args = parser.parse_args()
    run_institutional_backtest(args.csv, symbol=args.symbol, refresh_real_data=args.refresh)
