"""
Runner di Backtest & Analisi Quantitativa del Framework FX
Esegue l'intero workflow istituzionale su dati storici M1 e genera il report delle performance.
"""
import argparse
from pathlib import Path
import pandas as pd

from core.data_loader import load_candles_csv, resample_candles
from core.strategy import AMDMultiTimeframeEngine
from data.sample_data import generate_sample_dataset


def run_backtest(csv_path: str = "data/eurusd_m1_sample.csv"):
    path = Path(csv_path)
    if not path.exists():
        print(f"File {csv_path} non trovato. Generazione dataset di test realistico...")
        generate_sample_dataset(days=5, output_path=csv_path)

    print(f"\n=======================================================")
    print(f"CARICAMENTO DATI E RESAMPLING MULTI-TIMEFRAME: {csv_path}")
    print(f"=======================================================")
    df_m1 = load_candles_csv(csv_path)
    print(f"[+] Candele M1 caricate: {len(df_m1)} (dal {df_m1.index[0]} al {df_m1.index[-1]})")

    df_m5 = resample_candles(df_m1, target_tf="5min")
    print(f"[+] Candele M5 aggregate: {len(df_m5)}")

    df_h4 = resample_candles(df_m1, target_tf="4h")
    print(f"[+] Candele H4 aggregate: {len(df_h4)}")

    print("\n[+] Esecuzione Engine A-M-D con gerarchia M5 (default) vs M1 (radiografia)...")
    engine = AMDMultiTimeframeEngine(symbol="EURUSD")
    trades = engine.analyze_and_backtest(df_m1=df_m1, df_m5=df_m5, df_h4=df_h4)

    print("\n=======================================================")
    print("                    REPORT OPERATIVO                   ")
    print("=======================================================")

    if not trades:
        print("Nessun trade validato dal modello nel periodo analizzato.")
        return

    m5_trades = [t for t in trades if t.execution_timeframe == 'M5']
    m1_trades = [t for t in trades if t.execution_timeframe == 'M1']
    wins = [t for t in trades if t.outcome == 'WIN']
    losses = [t for t in trades if t.outcome == 'LOSS']
    win_rate = (len(wins) / (len(wins) + len(losses)) * 100) if (wins or losses) else 0.0
    total_pips = sum(t.pnl_pips for t in trades if t.pnl_pips is not None)
    total_r = sum(t.realized_rr for t in trades if t.realized_rr is not None)

    print(f"Operazioni totali eseguite: {len(trades)}")
    print(f"  - Eseguite su M5 (Default):      {len(m5_trades)}")
    print(f"  - Eseguite su M1 (Radiografia):  {len(m1_trades)}")
    print(f"Win Rate:                         {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Guadagno Totale (Pips):           {total_pips:+.1f} pips")
    print(f"Rendimento Totale (R-Multiple):   {total_r:+.1f} R")
    print("-------------------------------------------------------")
    print(f"{'ID':<3} | {'TF':<3} | {'DIR':<5} | {'ENTRY TIME':<16} | {'ENTRY':<8} | {'SL':<8} | {'TP':<8} | {'SL PIPS':<7} | {'ESITO':<6} | {'R-MULT':<6}")
    print("-" * 88)
    for t in trades:
        time_str = t.entry_time.strftime("%d/%m %H:%M")
        esito = t.outcome if t.outcome else "RUNNING"
        rmult = f"{t.realized_rr:+.1f}R" if t.realized_rr is not None else "-"
        print(f"{t.trade_id:<3} | {t.execution_timeframe:<3} | {t.direction:<5} | {time_str:<16} | {t.entry_price:<8.5f} | {t.stop_loss:<8.5f} | {t.take_profit:<8.5f} | {t.sl_pips:<7.1f} | {esito:<6} | {rmult:<6}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest FX Institutional Framework")
    parser.add_argument("--csv", type=str, default="data/eurusd_m1_sample.csv", help="Percorso del file CSV M1")
    args = parser.parse_args()

    run_backtest(args.csv)
