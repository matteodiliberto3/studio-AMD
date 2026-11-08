"""
Runner di Backtest Istituzionale A-M-D:
Analizza i dati storici aggregando H4, H1, M5 da dati M1 e calcola:
1. Performance comparativa: ENTRY_MANIPULATION vs ENTRY_POST_BREAKOUT_RETEST
2. Performance e MAE: Setup eseguiti su M5 vs Setup raffinati su M1
3. Metriche: Strict Win Rate, Adjusted Win Rate (parziale 50%), MAE medio, R:R strutturale
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from core.data_loader import load_candles_csv, resample_candles
from core.strategy import InstitutionalAMDEngine, InstitutionalTrade
from data.sample_data import generate_sample_dataset


def run_institutional_backtest(csv_path: str = "data/eurusd_m1_sample.csv"):
    path = Path(csv_path)
    if not path.exists():
        print(f"File {csv_path} non trovato. Generazione dataset...")
        generate_sample_dataset(days=5, output_path=csv_path)

    print(f"\n============================================================================")
    print(f"   BACKTEST STRUTTURALE A-M-D CON TARGET CONTESTUALI (NO R:R FISSO)         ")
    print(f"============================================================================")
    print(f"[+] Caricamento dati da: {csv_path}")
    df_m1 = load_candles_csv(csv_path)
    print(f"[+] Candele M1: {len(df_m1)} (dal {df_m1.index[0]} al {df_m1.index[-1]})")

    df_m5 = resample_candles(df_m1, target_tf="5min")
    df_h1 = resample_candles(df_m1, target_tf="1h")
    df_h4 = resample_candles(df_m1, target_tf="4h")
    print(f"[+] Resampling multi-timeframe completato: M5={len(df_m5)}, H1={len(df_h1)}, H4={len(df_h4)}")

    engine = InstitutionalAMDEngine()
    trades = engine.analyze_market(df_m1=df_m1, df_m5=df_m5, df_h4=df_h4)

    if not trades:
        print("[!] Nessun trade conforme alle rigide condizioni A-M-D nel periodo analizzato.")
        return

    # 1. Suddivisione per Tipologia di Entrata
    trades_manip = [t for t in trades if t.entry_type.value == "ENTRY_MANIPULATION"]
    trades_retest = [t for t in trades if t.entry_type.value == "ENTRY_POST_BREAKOUT_RETEST"]

    # 2. Suddivisione per Timeframe di Esecuzione
    trades_m5 = [t for t in trades if t.timeframe == "M5"]
    trades_m1 = [t for t in trades if t.timeframe == "M1"]

    def print_metrics_table(group1, group2, name1, name2, title):
        def get_stats(subset):
            total = len(subset)
            if total == 0:
                return {"total": 0, "strict_wr": 0.0, "adj_wr": 0.0, "avg_mae": 0.0, "avg_sl": 0.0, "avg_rr": 0.0}
            strict_w = sum(1 for t in subset if t.strict_win)
            adj_w = sum(1 for t in subset if t.adjusted_win)
            avg_mae = np.mean([t.mae_pips for t in subset])
            avg_sl = np.mean([t.sl_pips for t in subset])
            rrs = [t.realized_rr for t in subset if t.realized_rr is not None]
            avg_rr = np.mean(rrs) if rrs else 0.0
            return {
                "total": total,
                "strict_w": strict_w,
                "strict_wr": (strict_w / total * 100),
                "adj_w": adj_w,
                "adj_wr": (adj_w / total * 100),
                "avg_mae": avg_mae,
                "avg_sl": avg_sl,
                "avg_rr": avg_rr
            }

        s1 = get_stats(group1)
        s2 = get_stats(group2)

        print("\n" + "=" * 76)
        print(f"   {title}")
        print("=" * 76)
        print(f"{'METRICA':<36} | {name1:<17} | {name2:<16}")
        print("-" * 76)
        print(f"{'Operazioni Totali':<36} | {s1['total']:<17} | {s2['total']:<16}")
        print(f"{'Strict Win Rate (Target Finale)':<36} | {s1['strict_wr']:>15.1f}% | {s2['strict_wr']:>14.1f}%")
        print(f"{'Adjusted Win Rate (Con Parziale 50%)':<36} | {s1['adj_wr']:>15.1f}% | {s2['adj_wr']:>14.1f}%")
        print(f"{'Stop Loss Medio (Pips)':<36} | {s1['avg_sl']:>15.1f} p | {s2['avg_sl']:>14.1f} p")
        print(f"{'MAE Medio / Drawdown (Pips)':<36} | {s1['avg_mae']:>15.1f} p | {s2['avg_mae']:>14.1f} p")
        print(f"{'R:R Effettivo Realizzato':<36} | {s1['avg_rr']:>15.2f} R | {s2['avg_rr']:>14.2f} R")
        print("=" * 76)

    # Stampa confronto 1: MANIPULATION vs POST_BREAKOUT_RETEST
    print_metrics_table(
        trades_manip, trades_retest,
        "ENTRY_MANIPULATION", "BREAKOUT_RETEST",
        "CONFRONTO MODALITÀ: MANIPOLAZIONE (AGGRESSIVA) vs BREAKOUT RETEST (CONSERVATIVA)"
    )

    # Stampa confronto 2: M5 (Default di Sessione) vs M1 (Raggi X)
    print_metrics_table(
        trades_m5, trades_m1,
        "ESECUZIONE M5", "RAGGI X M1",
        "CONFRONTO TIMEFRAME: M5 (DEFAULT CORPO) vs M1 (RAGGI X WICK COMPRESSA)"
    )

    print("\n[+] Dettaglio Operazioni Valide Eseguite:")
    print("-" * 105)
    print(f"{'ID':<3} | {'TIPO':<16} | {'TF':<3} | {'DIR':<4} | {'ENTRY':<8} | {'SL':<8} | {'TP 50%':<8} | {'TP STRUTT':<9} | {'MAE':<6} | {'ESITO':<12} | {'R:R':<5}")
    print("-" * 105)
    for t in trades[:15]:
        r_str = f"{t.realized_rr:+.1f}R" if t.realized_rr is not None else "-"
        print(f"{t.trade_id:<3} | {t.entry_type.name:<16} | {t.timeframe:<3} | {t.direction:<4} | {t.entry_price:<8.5f} | {t.stop_loss:<8.5f} | {t.tp_50_eq:<8.5f} | {t.tp_structural:<9.5f} | {t.mae_pips:<4.1f} p | {t.outcome:<12} | {r_str:<5}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest FX Institutional Framework")
    parser.add_argument("--csv", type=str, default="data/eurusd_m1_sample.csv")
    args = parser.parse_args()
    run_institutional_backtest(args.csv)
