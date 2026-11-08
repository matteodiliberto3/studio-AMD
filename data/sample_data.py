"""
Generatore di Dati Sample Realistici EUR/USD M1:
Crea cicli completi di mercato istituzionale A-M-D:
- Giorni con rally in London verso Imbalance H4 in Premium -> Accumulo -> Manipolazione -> Breakout con IMB -> Distribuzione Short
- Giorni con discesa in New York verso Key Level Bullish in Discount -> Accumulo -> Manipolazione -> Breakout con IMB -> Distribuzione Long
"""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_dataset(days: int = 5, output_path: str = "data/eurusd_m1_sample.csv") -> pd.DataFrame:
    Path("data").mkdir(exist_ok=True)
    start_date = pd.Timestamp("2026-10-12 00:00:00", tz="Europe/Rome")
    all_rows = []
    base_price = 1.11000

    for day in range(days):
        day_date = start_date + pd.Timedelta(days=day)
        curr = base_price

        for m in range(1440):
            dt = day_date + pd.Timedelta(minutes=m)
            hour = dt.hour
            minute = dt.minute

            # Ciclo A-M-D:
            # 00:00 - 08:00: ASIA RANGE (Accumulo laterale)
            if hour < 8:
                curr += np.random.normal(0, 0.00002)
                spread = 0.00008

            # 08:00 - 09:15: LONDON JUDAS SWING / MANIPULATION
            # Salita impulsiva verso zona Premium per toccare IMB H4
            elif (hour == 8) or (hour == 9 and minute <= 15):
                curr += 0.00006 + np.random.normal(0, 0.00003)
                spread = 0.00018

            # 09:15 - 09:45: ACCUMULAZIONE LTF E MANIPOLAZIONE
            elif hour == 9 and minute > 15 and minute <= 45:
                if minute == 30: # Spike di manipolazione
                    curr += 0.00025
                elif minute == 31: # Rifiuto immediato
                    curr -= 0.00028
                else:
                    curr += np.random.normal(0, 0.00002)
                spread = 0.00015

            # 09:45 - 12:30: BREAKOUT CON IMBALANCE E DISTRIBUZIONE SHORT
            elif (hour == 9 and minute > 45) or (10 <= hour < 13):
                curr -= 0.00004 + np.random.normal(0, 0.00002)
                spread = 0.00016

            # 13:00 - 15:00: NEW YORK ENTRY IN DISCOUNT ZONE
            elif 13 <= hour < 15:
                curr += np.random.normal(0, 0.00003)
                spread = 0.00010

            # 15:00 - 16:30: NEW YORK ACCUMULAZIONE + MANIPOLAZIONE E INVERSIONE
            elif hour == 15 or (hour == 16 and minute <= 30):
                if minute == 15 and hour == 16:
                    curr -= 0.00025 # Manipolazione al ribasso
                elif minute == 16 and hour == 16:
                    curr += 0.00030 # Rottura con displacement verso l'alto
                else:
                    curr += np.random.normal(0, 0.00002)
                spread = 0.00018

            # 16:30 - 20:00: DISTRIBUZIONE POMERIDIANA
            elif 16 < hour < 20:
                curr += 0.00003 + np.random.normal(0, 0.00002)
                spread = 0.00012

            else:
                curr += np.random.normal(0, 0.00002)
                spread = 0.00008

            o = curr
            h = o + abs(np.random.normal(0, spread))
            l = o - abs(np.random.normal(0, spread))
            c = l + np.random.uniform(0, max(0.00001, h - l))

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
    print(f"[OK] Dataset sample istituzionale rigenerato in {output_path} ({len(df)} candele M1).")
    return df


if __name__ == "__main__":
    generate_sample_dataset()
