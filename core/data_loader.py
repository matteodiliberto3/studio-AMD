"""
Modulo Data Loader:
Caricamento e normalizzazione nativa di dataset OHLCV reali esportati da:
- TradingView (time,open,high,low,close — ISO o Unix epoch)
- MetaTrader 5 (<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>)
- Feed Istituzionale ECN Dukascopy Bank (candele M1 reali con Open, High, Low, Close, Volume)
Con conversione automatica dei timestamp nel fuso orario Europe/Rome (CET/CEST).
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union
import dukascopy_python
import pandas as pd


def load_candles_csv(file_path: Union[str, Path], timezone: str = "Europe/Rome") -> pd.DataFrame:
    """
    Carica un file CSV di candele reali (TradingView, MetaTrader 5 o Dukascopy),
    normalizza le colonne in ['open', 'high', 'low', 'close', 'volume']
    e converte l'indice temporale nel fuso orario `Europe/Rome`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File dati non trovato: {path}")

    df = None
    for sep in [',', '\t', ';']:
        try:
            sample = pd.read_csv(path, sep=sep, nrows=5)
            if len(sample.columns) >= 4:
                df = pd.read_csv(path, sep=sep)
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Impossibile leggere il file CSV {path}.")

    df.columns = [str(col).strip().lower() for col in df.columns]

    # Riconoscimento colonne temporali (TradingView `time` o MT5 `<date>` + `<time>`)
    if '<date>' in df.columns and '<time>' in df.columns:
        dt_raw = df['<date>'].astype(str) + " " + df['<time>'].astype(str)
        dt_series = pd.to_datetime(dt_raw, utc=True)
    elif 'date' in df.columns and 'time' in df.columns:
        dt_raw = df['date'].astype(str) + " " + df['time'].astype(str)
        dt_series = pd.to_datetime(dt_raw, utc=True)
    else:
        time_col = next(
            (c for c in df.columns if c in ['time', 'timestamp', 'datetime', 'date', '<date>']),
            df.columns[0]
        )
        col_data = df[time_col]
        if pd.api.types.is_numeric_dtype(col_data):
            dt_series = pd.to_datetime(col_data, unit='s', utc=True)
        else:
            dt_series = pd.to_datetime(col_data, utc=True)

    df['timestamp'] = dt_series.dt.tz_convert(timezone)
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='last')]

    mapping = {}
    for col in df.columns:
        c_clean = col.replace('<', '').replace('>', '').strip()
        if c_clean in ['o', 'open']:
            mapping[col] = 'open'
        elif c_clean in ['h', 'high']:
            mapping[col] = 'high'
        elif c_clean in ['l', 'low']:
            mapping[col] = 'low'
        elif c_clean in ['c', 'close']:
            mapping[col] = 'close'
        elif c_clean in ['v', 'vol', 'volume', 'tickvol', 'real_volume']:
            mapping[col] = 'volume'

    df = df.rename(columns=mapping)

    for req in ['open', 'high', 'low', 'close']:
        if req not in df.columns:
            raise KeyError(f"Colonna obbligatoria '{req}' mancante nel CSV. Colonne: {list(df.columns)}")

    if 'volume' not in df.columns:
        df['volume'] = 1.0

    clean_df = df[['open', 'high', 'low', 'close', 'volume']].astype(float).dropna()
    return clean_df


def fetch_and_save_real_market_data(
    ticker: str = "EUR/USD",
    output_path: Union[str, Path] = "data/eurusd_m1_sample.csv",
    days: int = 14,
    end_date: datetime = datetime(2025, 11, 8),
    timezone: str = "Europe/Rome"
) -> pd.DataFrame:
    """
    Scarica candele M1 REALI dal feed ECN Dukascopy Bank con vere wicks,
    veri corpi e volumi reali, salvandole in CSV con fuso orario Europe/Rome.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalizza codice strumento per Dukascopy ('EURUSD' o 'EURUSD=X' -> 'EUR/USD')
    clean_sym = ticker.upper().replace("=X", "").strip()
    if "/" not in clean_sym and len(clean_sym) == 6:
        instrument = f"{clean_sym[:3]}/{clean_sym[3:]}"
    else:
        instrument = clean_sym

    start_date = end_date - timedelta(days=days)

    df = dukascopy_python.fetch(
        instrument=instrument,
        interval=dukascopy_python.INTERVAL_MIN_1,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start_date,
        end=end_date,
        limit=30000
    )

    df.index = pd.to_datetime(df.index, utc=True).tz_convert(timezone)
    df.index.name = "timestamp"
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_csv(path)
    return df


def resample_candles(df_m1: pd.DataFrame, target_tf: str = "5min") -> pd.DataFrame:
    """
    Aggrega candele da M1 a timeframe superiori ('5min', '15min', '1h', '4h', '1D')
    preservando l'allineamento OHLC e il fuso orario Europe/Rome.
    """
    resampled = df_m1.resample(target_tf, closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled
