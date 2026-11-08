"""
Generatore di Dati Sample Realistici EUR/USD M1
Riproduce scenari istituzionali A-M-D, Killzones e la differenza tra esecuzione M5 ed M1.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def generate_sample_dataset(days: int = 5, output_path: str = "data/eurusd_m1_sample.csv") -> pd.DataFrame:
    """
    Genera un dataset M1 con sessioni coerenti (Asia Range, London Judas Swing, NY Continuation).
    """
    Path("data").mkdir(exist_ok=True)
    
    # Inizio dati: lunedì recente
    start_date = pd.Timestamp("2026-10-12 00:00:00", tz="Europe/Rome")
    all_rows = []

    current_price = 1.11200
    
    for day in range(days):
        day_date = start_date + pd.Timedelta(days=day)
        
        # 1440 minuti per giornata
        for m in range(1440):
            dt = day_date + pd.Timedelta(minutes=m)
            hour = dt.hour
            minute = dt.minute

            # Modello di volatilità oraria
            if 0 <= hour < 7:
                # ASIA RANGE: Bassa volatilità, range stretto (Accumulo)
                noise = np.random.normal(0, 0.00003)
                current_price += noise
                spread = 0.00008
            elif (hour == 8 and minute <= 45):
                # LONDON OPEN (MANIPOLAZIONE / JUDAS SWING)
                # Spike improvviso verso l'alto per sweepare i massimi asiatici ed entrare in IMB H4
                current_price += 0.00015 + np.random.normal(0, 0.00005)
                spread = 0.00025
            elif (hour == 9 or (hour == 8 and minute > 45)):
                # LONDON REVERSAL & DISTRIBUTION
                # Netta rottura verso il basso (Displacement)
                current_price -= 0.00018 + np.random.normal(0, 0.00004)
                spread = 0.00020
            elif 13 <= hour < 16:
                # NEW YORK KILLZONE: Seconda ondata di distribuzione o retest FVG
                if hour == 14 and minute < 30:
                    current_price += 0.00008 # Piccolo retest
                else:
                    current_price -= 0.00010
                spread = 0.00018
            else:
                # Sessione pomeridiana / chiusura
                current_price += np.random.normal(0, 0.00004)
                spread = 0.00010

            o = current_price
            h = o + abs(np.random.normal(0, spread))
            l = o - abs(np.random.normal(0, spread))
            c = l + np.random.uniform(0, h - l)

            all_rows.append({
                'timestamp': dt,
                'open': round(o, 5),
                'high': round(h, 5),
                'low': round(l, 5),
                'close': round(c, 5),
                'volume': int(np.random.uniform(100, 1500))
            })

    df = pd.DataFrame(all_rows).set_index('timestamp')
    df.to_csv(output_path)
    print(f"[OK] Dataset sample generato con successo in {output_path} ({len(df)} candele M1).")
    return df


if __name__ == "__main__":
    generate_sample_dataset()
