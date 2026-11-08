# Analisi Operatività FX — Institutional Framework (A-M-D & Multi-Timeframe)

Questo progetto implementa il framework operativo istituzionale (Smart Money Concepts / ICT / A-M-D) combinando:
1. **Machine Directives (`.cursorrules`)**: Regole ferree multi-timeframe per agenti AI (Google Antigravity e Cursor).
2. **Pine Script v5 per TradingView**: Box Killzones visive (Londra e New York), calcolo dinamico dell'Equilibrium (50% Premium/Discount), identificazione automatica di Imbalance (FVG) e invio di alert Webhook con payload JSON.
3. **Engine di Backtest & Resampling in Python**: Caricamento di dati M1, aggregazione a M5/H1/H4 e logica selettiva di discesa su M1 (*radiografia*) quando M5 è compresso in una singola wick/spike.
4. **FastAPI Webhook Bridge**: Micro-servizio locale/cloud per ricevere gli alert da TradingView, validare i vincoli istituzionali ed escludere automaticamente i trade fuori sessione o contro-zona.

---

## Struttura del Progetto

```
Analisi operatività FX/
├── .cursorrules                     # Direttive per Google Antigravity / Cursor
├── config.py                        # Parametri di sessione, orari Killzone e gestione rischio
├── requirements.txt                 # Dipendenze Python
├── pine_script/
│   └── institutional_amd_fvg.pine   # Indicatore completo Pine Script v5 per TradingView
├── core/
│   ├── data_loader.py               # Lettura CSV multi-formato (MT5, TradingView, Forexplay)
│   ├── indicators.py                # FVG, Dealing Range (50%), Killzones, Swings, MSS
│   └── strategy.py                  # Motore decisionale A-M-D (M5 default vs M1 radiografia)
├── webhook_bridge/
│   ├── app.py                       # Server FastAPI per ricezione e validazione alert
│   ├── models.py                    # Schema Pydantic del payload JSON
│   ├── test_webhook.py              # Script di test client per il Webhook
│   └── alerts_history.json          # Registro persistente degli alert ricevuti
├── tests/
│   └── test_engine_and_bridge.py    # Test automatici pytest
├── data/
│   ├── sample_data.py               # Generatore dataset realistico EUR/USD M1
│   └── eurusd_m1_sample.csv         # Dati M1 di test
└── backtest.py                      # Runner per analisi storica e report metriche
```

---

## 1. Come eseguire il Backtest in Python

Esegui il runner per caricare i dati M1, effettuare il resampling ed eseguire il report operativo:

```bash
python backtest.py
```

Per testare un tuo file CSV esportato da MT5, Forexplay o TradingView:

```bash
python backtest.py --csv "percorso/al/tuo/file.csv"
```

---

## 2. Come usare Pine Script su TradingView

1. Apri TradingView su un grafico (es. EUR/USD su timeframe M5 o M1).
2. Apri il **Pine Editor** in basso.
3. Copia e incolla il codice contenuto in `pine_script/institutional_amd_fvg.pine`.
4. Clicca su **Aggiungi al grafico**.
5. Per attivare il Webhook automatico:
   - Clicca con il tasto destro sull'indicatore -> **Aggiungi avviso**.
   - Spunta la casella **Webhook URL**.
   - Inserisci l'URL del tuo bridge (es. `https://tuo-dominio.ngrok-free.app/webhook`).
   - Nel corpo del messaggio, TradingView userà automaticamente il template JSON definito nello script.

---

## 3. Come avviare il Webhook Bridge locale

Avvia il server FastAPI con Uvicorn:

```bash
python -m uvicorn webhook_bridge.app:app --reload --port 8000
```

Puoi verificare la documentazione interattiva OpenAPI/Swagger all'indirizzo:
`http://127.0.0.1:8000/docs`

Per simulare l'invio di alert di prova:
```bash
python webhook_bridge/test_webhook.py
```
