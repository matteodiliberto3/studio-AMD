"""
Modulo Data Loader:
Caricamento e normalizzazione di dataset OHLCV da formati comuni
(TradingView, MT4/MT5, Forexplay, CSV generici) e resampling multi-timeframe.
"""
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np


def load_candles_csv(file_path: Union[str, Path], timezone: str = "Europe/Rome") -> pd.DataFrame:
    """
    Carica un file CSV di candele ed effettua l'auto-rilevamento delle colonne:
    restituisce un DataFrame standardizzato con colonne lowercase:
    ['open', 'high', 'low', 'close', 'volume'] con DatetimeIndex ordinato.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File dati non trovato: {path}")

    # Prova a leggere separatore virgola, punto e virgola o tabulazione
    df = None
    for sep in [',', ';', '\t']:
        try:
            temp_df = pd.read_csv(path, sep=sep, nrows=10)
            if len(temp_df.columns) >= 4:
                df = pd.read_csv(path, sep=sep)
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Impossibile leggere il file CSV {path} con i delimitatori standard.")

    # Normalizza i nomi delle colonne
    df.columns = [col.strip().lower() for col in df.columns]

    # Identificazione della colonna temporale
    date_cols = [c for c in df.columns if any(k in c for k in ['time', 'date', 'gmt', 'timestamp', '<date>'])]
    
    if len(date_cols) == 1:
        dt_series = pd.to_datetime(df[date_cols[0]])
    elif len(date_cols) >= 2:
        # Spesso MT4/MT5 hanno colonne separate per Data e Ora
        dt_series = pd.to_datetime(df[date_cols[0]].astype(str) + " " + df[date_cols[1]].astype(str))
    else:
        # Fallback alla prima colonna
        dt_series = pd.to_datetime(df.iloc[:, 0])

    df['timestamp'] = dt_series
    df = df.set_index('timestamp').sort_index()

    # Mappatura colonne OHLCV
    mapping = {}
    for col in df.columns:
        if any(col.startswith(x) for x in ['o', 'open', '<open>']):
            mapping[col] = 'open'
        elif any(col.startswith(x) for x in ['h', 'high', '<high>']):
            mapping[col] = 'high'
        elif any(col.startswith(x) for x in ['l', 'low', '<low>']):
            mapping[col] = 'low'
        elif any(col.startswith(x) for x in ['c', 'close', '<close>']):
            mapping[col] = 'close'
        elif any(col.startswith(x) for x in ['v', 'vol', 'volume', '<vol>', '<tickvol>']):
            mapping[col] = 'volume'

    df = df.rename(columns=mapping)
    
    # Verifica colonne minime
    for req in ['open', 'high', 'low', 'close']:
        if req not in df.columns:
            raise KeyError(f"Colonna obbligatoria '{req}' non identificata nel CSV. Colonne trovate: {list(df.columns)}")

    if 'volume' not in df.columns:
        df['volume'] = 1.0

    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    clean_df = df[ohlcv_cols].astype(float).dropna()

    return clean_df


def resample_candles(df_m1: pd.DataFrame, target_tf: str = "5min") -> pd.DataFrame:
    """
    Effettua il resampling aggregando da M1 a timeframe superiori:
    '5min' (M5), '15min' (M15), '1h' (H1), '4h' (H4), '1D' (Daily).
    """
    resampled = df_m1.resample(target_tf, closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled
