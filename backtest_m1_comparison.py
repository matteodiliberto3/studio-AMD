"""
Backtest Comparativo su Dati Reali di Mercato (EUR/USD & GBP/USD):
Valuta e confronta le due morfologie del meccanismo di Entry:
1. AMD_HORIZONTAL_BOX (Box di Accumulazione Orizzontale A-M-D -> Rottura M/m con IMB -> >=2 Chiusure -> Rientro in IMB)
2. DYNAMIC_ACCUMULATION_BOS (Accumulazione Dinamica M/m -> Rottura M/m con IMB -> >=2 Chiusure -> Rientro in IMB)

Esporta tutti i trade di EUR/USD e GBP/USD in `data/trades_executed_log.csv`.
"""
from pathlib import Path
from typing import List

from backtest import print_comparison_table
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


def analyze_real_pair(ticker: str, symbol: str, csv_path: str) -> List[InstitutionalTrade]:
    path = Path(csv_path)
    print(f"\n[+] Acquisizione dati REALI ECN M1 per {symbol} ({ticker})...")
    fetch_and_save_real_market_data(ticker=ticker, output_path=path, days=14)

    df_m1 = load_candles_csv(path)
    df_m5 = resample_candles(df_m1, target_tf="5min")
    df_h1 = resample_candles(df_m1, target_tf="1h")
    df_h4 = resample_candles(df_m1, target_tf="4h")

    print(f"[+] {symbol} caricato: {len(df_m1)} candele M1 ({df_m1.index[0]} -> {df_m1.index[-1]})")

    engine = InstitutionalAMDEngine(symbol=symbol)
    trades = engine.analyze_market(df_m1=df_m1, df_m5=df_m5, df_h4=df_h4, df_h1=df_h1)

    trades_box = [t for t in trades if t.accumulation_type == AccumulationType.AMD_HORIZONTAL_BOX]
    trades_dyn = [t for t in trades if t.accumulation_type == AccumulationType.DYNAMIC_ACCUMULATION_BOS]

    trades_m5 = [t for t in trades if t.timeframe == "M5"]
    trades_m1 = [t for t in trades if t.timeframe == "M1"]

    print_comparison_table(
        trades_box, trades_dyn,
        "AMD_HORIZ_BOX", "DYNAMIC_BOS_IMB",
        f"REPORT {symbol} (DATI REALI): BOX ORIZZONTALE A-M-D vs ACCUMULO DINAMICO M/m"
    )

    print_comparison_table(
        trades_m5, trades_m1,
        "ESECUZIONE M5", "RAGGI X M1",
        f"REPORT {symbol} (DATI REALI): ESECUZIONE M5 vs RAGGI X M1"
    )

    return trades


if __name__ == "__main__":
    all_trades: List[InstitutionalTrade] = []

    eu_trades = analyze_real_pair("EURUSD=X", "EURUSD", "data/eurusd_m1_sample.csv")
    all_trades.extend(eu_trades)

    gu_trades = analyze_real_pair("GBPUSD=X", "GBPUSD", "data/gbpusd_m1_real.csv")
    for idx, tr in enumerate(gu_trades, start=len(all_trades) + 1):
        tr.trade_id = idx
    all_trades.extend(gu_trades)

    log_file = Path("data/trades_executed_log.csv")
    export_trades_to_csv(all_trades, output_path=log_file)
    print(f"\n[OK] Esportati {len(all_trades)} trade reali totali in {log_file} per verifica su TradingView.\n")
