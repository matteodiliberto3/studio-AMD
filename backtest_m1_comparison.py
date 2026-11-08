"""
Backtest Comparativo su Dati Reali (EURUSD & GBPUSD):
Confronto Scientifico tra le due uniche modalità istituzionali ammesse:
1. ENTRY_MANIPULATION (Entry 1: Aggressiva sul picco della manipolazione del Key Level)
2. ENTRY_POST_BREAKOUT_RETEST (Entry 2: Conservativa sul retest dell'Imbalance post-breakout)

Nessuna entrata arbitraria dentro l'accumulazione.
Target contestuali: Target 1 al 50% Equilibrium e Target Finale su liquidità strutturale.
"""
import pandas as pd
import numpy as np
import yfinance as yf

from core.data_loader import resample_candles
from core.strategy import InstitutionalAMDEngine


def run_comparative_asset_test(ticker: str = "EURUSD=X", name: str = "EUR/USD", period: str = "7d"):
    print(f"\n[+] Scaricamento dati reali {name} (periodo: {period}, intervallo: 1m)...")
    raw = yf.download(ticker, period=period, interval="1m", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [col.lower() for col in raw.columns]

    df_m1 = raw[['open', 'high', 'low', 'close']].dropna().copy()
    df_m1.index = pd.to_datetime(df_m1.index)
    print(f"[+] Candele M1 scaricate: {len(df_m1)}")

    df_m5 = resample_candles(df_m1, target_tf="5min")
    df_h4 = resample_candles(df_m1, target_tf="4h")
    print(f"[+] Resampling: M5={len(df_m5)}, H4={len(df_h4)}")

    engine = InstitutionalAMDEngine()
    trades = engine.analyze_market(df_m1=df_m1, df_m5=df_m5, df_h4=df_h4)

    if not trades:
        print(f"[!] Nessuna sequenza A-M-D valida trovata su {name}.")
        return

    trades_manip = [t for t in trades if t.entry_type.value == "ENTRY_MANIPULATION"]
    trades_retest = [t for t in trades if t.entry_type.value == "ENTRY_POST_BREAKOUT_RETEST"]

    def calc_stats(subset):
        total = len(subset)
        if total == 0:
            return {"total": 0, "strict_wr": 0.0, "adj_wr": 0.0, "avg_sl": 0.0, "avg_mae": 0.0, "avg_rr": 0.0}
        strict_w = sum(1 for t in subset if t.strict_win)
        adj_w = sum(1 for t in subset if t.adjusted_win)
        avg_sl = np.mean([t.sl_pips for t in subset])
        avg_mae = np.mean([t.mae_pips for t in subset])
        rrs = [t.realized_rr for t in subset if t.realized_rr is not None]
        avg_rr = np.mean(rrs) if rrs else 0.0
        return {
            "total": total,
            "strict_wr": (strict_w / total * 100),
            "adj_wr": (adj_w / total * 100),
            "avg_sl": avg_sl,
            "avg_mae": avg_mae,
            "avg_rr": avg_rr
        }

    sm = calc_stats(trades_manip)
    sr = calc_stats(trades_retest)

    print("\n" + "=" * 76)
    print(f"       REPORT COMPARATIVO ISTITUZIONALE: {name} (DATI REALI)")
    print("=" * 76)
    print(f"{'METRICA':<36} | {'MANIPULATION (1)':<17} | {'BREAKOUT RETEST (2)':<16}")
    print("-" * 76)
    print(f"{'Trade Totali Conclusi':<36} | {sm['total']:<17} | {sr['total']:<16}")
    print(f"{'Strict Win Rate (Target Finale)':<36} | {sm['strict_wr']:>15.1f}% | {sr['strict_wr']:>14.1f}%")
    print(f"{'Adjusted Win Rate (Con Parziale 50%)':<36} | {sm['adj_wr']:>15.1f}% | {sr['adj_wr']:>14.1f}%")
    print(f"{'Stop Loss Medio':<36} | {sm['avg_sl']:>15.1f} p | {sr['avg_sl']:>14.1f} p")
    print(f"{'MAE Medio / Drawdown':<36} | {sm['avg_mae']:>15.1f} p | {sr['avg_mae']:>14.1f} p")
    print(f"{'R:R Effettivo Realizzato':<36} | {sm['avg_rr']:>15.2f} R | {sr['avg_rr']:>14.2f} R")
    print("=" * 76)


if __name__ == "__main__":
    run_comparative_asset_test("EURUSD=X", "EUR/USD", "7d")
    run_comparative_asset_test("GBPUSD=X", "GBP/USD", "7d")
