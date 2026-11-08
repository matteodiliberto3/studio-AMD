"""
Modulo di Acquisizione Dati Reali di Mercato:
I dati sintetici sono tassativamente vietati dal framework.
Questo script scarica e aggiorna i dati reali M1 di EUR/USD e GBP/USD in fuso orario Europe/Rome.
"""
from core.data_loader import fetch_and_save_real_market_data


def download_all_real_datasets():
    df_eu = fetch_and_save_real_market_data(
        ticker="EUR/USD",
        output_path="data/eurusd_m1_sample.csv",
        days=14
    )
    print(f"[OK] Dati REALI ECN EUR/USD M1 salvati in data/eurusd_m1_sample.csv ({len(df_eu)} candele attive).")

    df_gu = fetch_and_save_real_market_data(
        ticker="GBP/USD",
        output_path="data/gbpusd_m1_real.csv",
        days=14
    )
    print(f"[OK] Dati REALI ECN GBP/USD M1 salvati in data/gbpusd_m1_real.csv ({len(df_gu)} candele attive).")


if __name__ == "__main__":
    download_all_real_datasets()
