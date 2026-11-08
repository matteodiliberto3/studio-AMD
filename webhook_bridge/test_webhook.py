"""
Script di Test per il Webhook Bridge:
Invia una serie di payload simulati (sia conformi che non conformi) all'endpoint FastAPI.
"""
import time
import requests

BASE_URL = "http://127.0.0.1:8000"


def run_tests():
    print(f"Verifica connessione con {BASE_URL}...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        print(f"[+] Server attivo: {r.json()}\n")
    except Exception as e:
        print(f"[!] Server non raggiungibile su {BASE_URL}. Assicurati di aver avviato uvicorn:")
        print("    python -m uvicorn webhook_bridge.app:app --reload --port 8000\n")
        return

    # Test 1: Setup Short CONFORME (in London Killzone e Premium Zone)
    valid_short = {
        "symbol": "EURUSD",
        "timeframe": "5",
        "action": "SELL",
        "price": 1.11610,
        "sl": 1.11700,
        "session": "LONDON",
        "zone": "PREMIUM",
        "reason": "Bearish FVG retest on H4 IMB reaction"
    }

    print(">>> Test 1: Invio Alert SELL Conforme (London, Premium)...")
    res1 = requests.post(f"{BASE_URL}/webhook", json=valid_short)
    print("Risposta:", res1.json())
    print("-" * 60)

    # Test 2: Setup NON CONFORME (Sell in zona Discount)
    invalid_sell = {
        "symbol": "EURUSD",
        "timeframe": "5",
        "action": "SELL",
        "price": 1.11200,
        "sl": 1.11300,
        "session": "LONDON",
        "zone": "DISCOUNT",
        "reason": "Falso breakout su supporto"
    }

    print(">>> Test 2: Invio Alert SELL Non Conforme (Zona Discount)...")
    res2 = requests.post(f"{BASE_URL}/webhook", json=invalid_sell)
    print("Risposta:", res2.json())
    print("-" * 60)

    # Test 3: Setup NON CONFORME (Fuori sessione / Asian session)
    invalid_session = {
        "symbol": "EURUSD",
        "timeframe": "5",
        "action": "BUY",
        "price": 1.11100,
        "sl": 1.11000,
        "session": "ASIA",
        "zone": "DISCOUNT",
        "reason": "Rimbalzo notturno"
    }

    print(">>> Test 3: Invio Alert BUY Non Conforme (Fuori Killzone)...")
    res3 = requests.post(f"{BASE_URL}/webhook", json=invalid_session)
    print("Risposta:", res3.json())
    print("-" * 60)


if __name__ == "__main__":
    run_tests()
