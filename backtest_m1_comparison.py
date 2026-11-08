"""
Backtest Comparativo su Dati Reali di Mercato (EUR/USD & GBP/USD):
Valuta e confronta le due morfologie del meccanismo di Entry:
1. AMD_HORIZONTAL_BOX (Box di Accumulazione Orizzontale A-M-D -> Rottura M/m con IMB -> >=2 Chiusure -> Rientro in IMB)
2. DYNAMIC_ACCUMULATION_BOS (Accumulazione Dinamica M/m -> Rottura M/m con IMB -> >=2 Chiusure -> Rientro in IMB)

Esporta tutti i trade di EUR/USD e GBP/USD in `data/trades_executed_log.csv`.
"""
from pathlib import Path
from typing import List
import pandas as pd

from backtest import print_comparison_table
from config import AccumulationType
from core.data_loader import (
    load_casaos_dukascopy_year,
    resample_candles
)
from core.strategy import (
    InstitutionalAMDEngine,
    InstitutionalTrade,
    export_trades_to_csv
)


def print_monthly_breakdown(trades: List[InstitutionalTrade], symbol: str, year: int):
    print("\n" + "=" * 80)
    print(f"   RENDIMENTO MENSILE {symbol} ({year} - DATI DUKASCOPY CASAOS)")
    print("=" * 80)
    print(f"{'MESE':<12} | {'TRADE':<8} | {'WIN/PART/LOSS':<16} | {'ADJ WR':<10} | {'MAE MEDIO':<10} | {'NET R':<10}")
    print("-" * 80)
    by_month = {}
    for t in trades:
        m_key = t.entry_time.strftime("%Y-%m")
        by_month.setdefault(m_key, []).append(t)

    for m_key in sorted(by_month.keys()):
        m_trades = by_month[m_key]
        tot = len(m_trades)
        w = sum(1 for t in m_trades if t.outcome == "WIN")
        p = sum(1 for t in m_trades if t.outcome == "PARTIAL")
        l = sum(1 for t in m_trades if t.outcome == "LOSS")
        adj_wr = (w + p) / tot * 100.0 if tot else 0.0
        mae = sum(t.mae_pips for t in m_trades) / tot if tot else 0.0
        net_r = sum(t.realized_rr for t in m_trades if t.realized_rr is not None)
        print(f"{m_key:<12} | {tot:<8} | {f'{w}W / {p}P / {l}L':<16} | {adj_wr:>8.1f}% | {mae:>8.1f} p | {net_r:>+8.2f} R")
    print("=" * 80)


def analyze_real_pair_year(symbol: str, year: int = 2025) -> List[InstitutionalTrade]:
    print(f"\n[+] Caricamento 1 ANNO INTERO ({year}) di dati REALI Dukascopy M1 da CasaOS per {symbol}...")
    df_m1 = load_casaos_dukascopy_year(symbol=symbol, year=year)
    df_m5 = resample_candles(df_m1, target_tf="5min")
    df_h1 = resample_candles(df_m1, target_tf="1h")
    df_h4 = resample_candles(df_m1, target_tf="4h")

    print(f"[+] {symbol} ({year}) caricato: {len(df_m1):,} candele M1 ({df_m1.index[0]} -> {df_m1.index[-1]})")
    print(f"[+] Timeframe aggregati: M5={len(df_m5):,}, H1={len(df_h1):,}, H4={len(df_h4):,}")

    engine = InstitutionalAMDEngine(symbol=symbol)
    trades = engine.analyze_market(df_m1=df_m1, df_m5=df_m5, df_h4=df_h4, df_h1=df_h1)

    trades_box = [t for t in trades if t.accumulation_type == AccumulationType.AMD_HORIZONTAL_BOX]
    trades_dyn = [t for t in trades if t.accumulation_type == AccumulationType.DYNAMIC_ACCUMULATION_BOS]

    trades_m5 = [t for t in trades if t.timeframe == "M5"]
    trades_m1 = [t for t in trades if t.timeframe == "M1"]

    print_comparison_table(
        trades_box, trades_dyn,
        "AMD_HORIZ_BOX", "DYNAMIC_BOS_IMB",
        f"REPORT {symbol} ({year} INTERO): BOX ORIZZONTALE A-M-D vs ACCUMULO DINAMICO M/m"
    )

    print_comparison_table(
        trades_m5, trades_m1,
        "ESECUZIONE M5", "RAGGI X M1",
        f"REPORT {symbol} ({year} INTERO): ESECUZIONE M5 vs RAGGI X M1"
    )

    print_monthly_breakdown(trades, symbol, year)

    return trades


if __name__ == "__main__":
    YEAR_TO_TEST = 2025
    all_trades: List[InstitutionalTrade] = []

    eu_trades = analyze_real_pair_year("EURUSD", year=YEAR_TO_TEST)
    all_trades.extend(eu_trades)

    gu_trades = analyze_real_pair_year("GBPUSD", year=YEAR_TO_TEST)
    for idx, tr in enumerate(gu_trades, start=len(all_trades) + 1):
        tr.trade_id = idx
    all_trades.extend(gu_trades)

    log_file = Path("data/trades_executed_log.csv")
    export_trades_to_csv(all_trades, output_path=log_file)
    print(f"\n[OK] Esportati {len(all_trades)} trade reali dell'anno {YEAR_TO_TEST} in {log_file} per verifica su TradingView.\n")
