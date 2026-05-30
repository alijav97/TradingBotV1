# TradingBotV1 — System Architecture & Operations Journal

Last updated: 2026-05-31
Git repo: https://github.com/alijav97/TradingBotV1.git
Active branch: claude/xauusd-strategy-backtest-sO2vM

---

## GOLDEN RULES (never break these)

1. `btc_research/` = BTC work ONLY. Never import from `v2/` inside here.
2. `v2/` = WTI live bot ONLY. Never import from `btc_research/` inside here.
3. BTC Bot 2 is fully standalone — it has its own connectors, journal, settings. Zero v2 dependency.
4. MT5_SERVER_UTC_OFFSET = 3. Pepperstone server is UTC+3. All bar timestamps must subtract 3h to get true UTC. Critical for kill-zone alignment.
5. Never run pip install in a random terminal — always use the correct venv (see VPS layout below).

---

## REPO STRUCTURE

```
TradingBotV1/
│
├── v2/                          ← WTI LIVE BOT (production)
│   ├── main.py                  ← Entry point: python -m v2.main
│   ├── settings.py              ← All config: ACTIVE_SYMBOLS, risk, MT5 creds from .env
│   ├── instrument_config.py     ← Per-instrument constants (XAUUSD, GBPJPY, WTI, NAS100, BTCUSDT, ETHUSDT)
│   ├── scheduler/
│   │   └── scheduler.py        ← APScheduler: scans every 5min (H1), kill-zone 2s scan
│   ├── signals/
│   │   ├── confluence_engine.py ← Master signal scorer
│   │   ├── entry_checklist.py  ← Pre-trade checklist
│   │   ├── signal_ranker.py    ← Ranks signals by score
│   │   └── strategies/         ← Individual strategy files
│   │       ├── ny_momentum_wti.py  ← NYMomentumWTI — active WTI strategy
│   │       ├── ny_momentum.py
│   │       ├── london_breakout.py
│   │       ├── smc_order_block.py
│   │       └── ... (others)
│   ├── connectors/
│   │   ├── mt5_connector.py    ← Connects to Pepperstone MT5
│   │   ├── binance_connector.py
│   │   └── unified_data.py     ← DataFeed class: get_price(), get_ohlcv()
│   ├── journal/
│   │   └── sqlite_journal.py   ← SQLite trade journal (v2_trades.db)
│   ├── trading/
│   │   ├── paper_trader.py     ← Paper trading engine
│   │   └── trade_monitor.py    ← Monitors open trades
│   ├── risk/
│   │   ├── position_sizer.py   ← Lot sizing based on risk %
│   │   ├── atr_sl_engine.py    ← ATR-based stop loss
│   │   └── portfolio_heat.py   ← Max portfolio heat check
│   ├── api/
│   │   ├── telegram_bot.py     ← WTI bot Telegram alerts
│   │   └── api_server.py       ← FastAPI health/trades endpoints
│   ├── analysis/               ← Technical analysis helpers
│   ├── intelligence/           ← News, COT, DXY correlation
│   ├── ml/                     ← ML regime detection
│   └── backtest/               ← Backtesting engine
│
├── btc_research/                ← BTC RESEARCH + BTC LIVE BOTS
│   ├── __init__.py
│   ├── settings.py             ← Shared BTC research settings
│   │
│   ├── btc_bot_1/              ← BTC BOT 1 (SwingLevelBreak v1)
│   │   ├── main.py             ← Entry point: python -m btc_research.btc_bot_1.main
│   │   ├── settings.py         ← Bot 1 settings (KZ_HOURS=[21,22,23], ADX_MIN=20)
│   │   ├── connectors/
│   │   │   ├── mt5_connector.py  ← Standalone MT5 (subtracts UTC+3 offset)
│   │   │   └── unified_data.py   ← Standalone DataFeed for Bot 1
│   │   ├── journal/
│   │   │   └── sqlite_journal.py ← Bot 1 trade DB (btc_trades.db)
│   │   ├── signals/
│   │   │   └── btc_engine.py   ← Signal engine: EMA200 + ADX + confluence
│   │   ├── trading/
│   │   │   └── paper_trader.py ← Paper trader for Bot 1
│   │   ├── scheduler/
│   │   │   └── scheduler.py    ← BTCScheduler: 5min bg scan + 2s KZ scan
│   │   └── api/
│   │       └── telegram_bot.py ← Bot 1 Telegram (uses BTC_TELEGRAM_BOT_TOKEN)
│   │
│   ├── btc_bot_2/              ← BTC BOT 2 (SwingLevelBreak v2 + VB fallback)
│   │   ├── main.py             ← Entry point: python -m btc_research.btc_bot_2.main
│   │   ├── settings.py         ← Bot 2 settings (KZ_HOURS=[1,2,3,8], ADX thresholds)
│   │   ├── connectors/
│   │   │   ├── mt5_connector.py  ← Standalone MT5 (same UTC+3 correction)
│   │   │   └── unified_data.py   ← Standalone DataFeed for Bot 2
│   │   ├── journal/
│   │   │   └── sqlite_journal.py ← Bot 2 trade DB (btc2_trades.db)
│   │   ├── signal_engine.py    ← Signal engine for Bot 2
│   │   ├── paper_trader.py     ← Paper trader for Bot 2
│   │   ├── scheduler.py        ← BTC2Scheduler (APScheduler)
│   │   ├── telegram.py         ← Bot 2 Telegram (uses BTC2_TELEGRAM_BOT_TOKEN)
│   │   ├── strategy/
│   │   │   └── vb_swing_combined.py ← Combined VB + SwingBreak strategy
│   │   └── api.py              ← FastAPI endpoints for Bot 2
│   │
│   ├── strategies/             ← Shared BTC strategy implementations
│   ├── backtest/               ← BTC backtesting engine
│   ├── data/                   ← Data fetchers
│   ├── factors/                ← Signal factors (gold, nasdaq, momentum, time)
│   └── analysis_*.py           ← Research/analysis scripts (not live)
│
├── setup_vps.ps1               ← VPS setup script (Chocolatey + Python + venv)
├── requirements.txt            ← All Python dependencies
├── .env                        ← Credentials (gitignored — never commit this)
├── .gitignore
└── v2/deploy/
    ├── windows_setup.ps1       ← Sets up C:\TradingBotV2\ on VPS
    └── install_service.ps1     ← Registers Windows Task Scheduler service

```

---

## VPS LAYOUT (what's actually on the Windows VPS)

```
C:\TradingBotV2\          ← WTI BOT installation
├── v2\                   ← WTI bot code (copied by windows_setup.ps1)
├── venv\                 ← Python venv with ALL packages
│   └── Scripts\
│       ├── python.exe    ← USE THIS for running ALL bots
│       └── pip.exe       ← USE THIS for installing packages
└── .env                  ← WTI + BTC credentials (see .env section below)

C:\Temp\TradingBotV1\     ← GIT CLONE of the full repo
├── btc_research\         ← BTC bot code (pulled from GitHub)
│   ├── btc_bot_1\
│   └── btc_bot_2\
└── .env                  ← SEPARATE .env for this location (has BTC creds)
```

**CRITICAL**: BTC bots run from `C:\Temp\TradingBotV1` but use `C:\TradingBotV2\venv\Scripts\python.exe` because:
- `btc_research` package only exists at `C:\Temp\TradingBotV1`
- All Python packages (requests, APScheduler, etc.) only installed in `C:\TradingBotV2\venv`

---

## .ENV FILES

### C:\TradingBotV2\.env (WTI bot — full credentials)
```
ACTIVE_SYMBOLS=WTI
MT5_LOGIN=<pepperstone account number>
MT5_PASSWORD=<pepperstone password>
MT5_SERVER=PepperstoneFinancialUAE-MT5-Live01
MT5_TIMEOUT=10000
TELEGRAM_BOT_TOKEN=<WTI bot token>
TELEGRAM_CHAT_ID=7675749781
BTC_TELEGRAM_BOT_TOKEN=<BTC Bot 1 token>
BTC_TELEGRAM_CHAT_ID=7675749781
BTC2_TELEGRAM_BOT_TOKEN=7658687644:AAFkdoEsjU7gPr_QtwSFJeTUAxuWqOMWH6M
BTC2_TELEGRAM_CHAT_ID=7675749781
BTC2_API_KEY=btc2_readonly
BTC2_API_PORT=8002
```

### C:\Temp\TradingBotV1\.env (BTC bots git clone)
Same BTC-related credentials as above. This file is separate and must be kept in sync manually.

### Telegram Bot Tokens
| Bot | Token env var | Fallback | Purpose |
|-----|--------------|---------|---------|
| WTI bot | TELEGRAM_BOT_TOKEN | — | WTI trade alerts |
| BTC Bot 1 | BTC_TELEGRAM_BOT_TOKEN | TELEGRAM_BOT_TOKEN | BTC Bot 1 alerts |
| BTC Bot 2 | BTC2_TELEGRAM_BOT_TOKEN | TELEGRAM_BOT_TOKEN | BTC Bot 2 alerts |

---

## HOW TO RUN (VPS — Windows PowerShell)

### Start WTI Bot (already set up as Windows Task)
```powershell
Start-ScheduledTask -TaskName "TradingBotV1"
# or manually:
cd C:\TradingBotV2; C:\TradingBotV2\venv\Scripts\python.exe -m v2.main
```

### Start BTC Bot 1
```powershell
Start-Process powershell -ArgumentList "-NoExit -Command `"cd C:\Temp\TradingBotV1; C:\TradingBotV2\venv\Scripts\python.exe -m btc_research.btc_bot_1.main`""
```

### Start BTC Bot 2
```powershell
Start-Process powershell -ArgumentList "-NoExit -Command `"cd C:\Temp\TradingBotV1; C:\TradingBotV2\venv\Scripts\python.exe -m btc_research.btc_bot_2.main`""
```

### Kill all BTC bots
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

### Pull latest + kill + restart all BTC bots (all-in-one)
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; cd C:\Temp\TradingBotV1; git pull origin claude/xauusd-strategy-backtest-sO2vM; Start-Process powershell -ArgumentList "-NoExit -Command `"cd C:\Temp\TradingBotV1; C:\TradingBotV2\venv\Scripts\python.exe -m btc_research.btc_bot_1.main`""; Start-Process powershell -ArgumentList "-NoExit -Command `"cd C:\Temp\TradingBotV1; C:\TradingBotV2\venv\Scripts\python.exe -m btc_research.btc_bot_2.main`""
```

### Install/update packages (always use the TradingBotV2 venv pip)
```powershell
C:\TradingBotV2\venv\Scripts\pip.exe install -r C:\Temp\TradingBotV1\requirements.txt
```

---

## BOT STRATEGIES & KILL ZONES

### WTI Bot (v2)
- **Strategy**: NYMomentumWTI — NY session momentum
- **Window**: 13:00–17:00 UTC (NY/NYMEX open)
- **Scan**: Every 5min on H1
- **Telegram**: TELEGRAM_BOT_TOKEN
- **DB**: C:\TradingBotV2\data\v2_trades.db

### BTC Bot 1
- **Strategy**: SwingLevelBreak v1 — EMA200 trend + ADX filter + confluence scoring
- **Kill zones**: 21:00, 22:00, 23:00 UTC (Asia Night → EU session open in UAE time)
- **ADX minimum**: 20 (below this → no trade, DEBUG log only — looks silent but normal)
- **Scan**: 5min background (outside KZ) + 2s inside KZ
- **IMPORTANT**: Log silence during KZ = NORMAL. Bot scans every 2s but only logs at INFO when a signal fires. ADX below 20 = DEBUG level log = invisible.
- **Telegram**: BTC_TELEGRAM_BOT_TOKEN → fallback TELEGRAM_BOT_TOKEN
- **DB**: C:\Temp\TradingBotV1\btc_research\btc_bot_1\data\btc_trades.db

### BTC Bot 2
- **Strategy**: SwingLevelBreak v2 [both 2×ATR] + VB fallback (46.7% WR, $119k 2yr backtest)
- **Kill zones**: 01:00, 02:00, 03:00, 08:00 UTC (Asia Night + EU Open)
- **Risk**: 3% ADX≤25 | 2% ADX 25–40 | 3% ADX≥40
- **Telegram**: BTC2_TELEGRAM_BOT_TOKEN → fallback TELEGRAM_BOT_TOKEN
- **API port**: 8002
- **DB**: C:\Temp\TradingBotV1\btc_research\btc_bot_2\data\btc2_trades.db

---

## MT5 TIMEZONE CORRECTION

Pepperstone MT5 server is UTC+3 (server_tz_offset = 3).
All bar timestamps from MT5 must subtract 3 hours to get true UTC.
This is done in both bot connectors:
```python
# btc_bot_1/connectors/mt5_connector.py
# btc_bot_2/connectors/mt5_connector.py
df["time"] = df["time"] - pd.Timedelta(hours=MT5_SERVER_UTC_OFFSET)  # MT5_SERVER_UTC_OFFSET = 3
```
Without this correction, kill zones would fire at wrong real-world hours.

---

## GIT WORKFLOW

```
Code on local machine (C:\Users\alija\Downloads\TradingBotV1)
    ↓  git push
GitHub (branch: claude/xauusd-strategy-backtest-sO2vM)
    ↓  git pull (on VPS at C:\Temp\TradingBotV1)
VPS runs bots from C:\Temp\TradingBotV1 using C:\TradingBotV2\venv\python.exe
```

**Never commit**: `.env`, `*.db`, `*.log`, `__pycache__`

---

## DEPENDENCY MAP (what imports what)

```
btc_bot_1/main.py
  → btc_bot_1/settings.py          (KZ hours, thresholds, DB path)
  → btc_bot_1/connectors/          (standalone MT5 + DataFeed)
  → btc_bot_1/journal/             (standalone SQLite journal)
  → btc_bot_1/signals/btc_engine   (signal logic)
  → btc_bot_1/trading/paper_trader (trade management)
  → btc_bot_1/scheduler/scheduler  (APScheduler jobs)
  → btc_bot_1/api/telegram_bot     (Telegram alerts)
  ✗ NEVER imports from v2/

btc_bot_2/main.py
  → btc_bot_2/settings.py          (KZ hours, risk params, DB path)
  → btc_bot_2/connectors/          (standalone MT5 + DataFeed)
  → btc_bot_2/journal/             (standalone SQLite journal)
  → btc_bot_2/signal_engine        (signal logic)
  → btc_bot_2/paper_trader         (trade management)
  → btc_bot_2/scheduler            (APScheduler jobs)
  → btc_bot_2/telegram             (Telegram alerts)
  → btc_bot_2/strategy/            (VB + SwingBreak strategies)
  ✗ NEVER imports from v2/

v2/main.py
  → v2/settings.py                 (ACTIVE_SYMBOLS, risk config)
  → v2/instrument_config.py        (per-instrument constants)
  → v2/connectors/                 (MT5 + Binance DataFeed)
  → v2/journal/sqlite_journal      (v2 trade DB)
  → v2/signals/                    (confluence engine + strategies)
  → v2/trading/                    (paper trader + monitor)
  → v2/risk/                       (position sizer, ATR SL, heat)
  → v2/scheduler/scheduler         (scan scheduler)
  → v2/api/telegram_bot            (WTI Telegram alerts)
  ✗ NEVER imports from btc_research/
```

---

## COMMON ISSUES & FIXES

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'btc_research'` | Running python from wrong directory | Run from `C:\Temp\TradingBotV1` |
| `No module named 'requests'` | Using system python instead of venv | Use `C:\TradingBotV2\venv\Scripts\python.exe` |
| `APScheduler not installed` | Not in venv | `C:\TradingBotV2\venv\Scripts\pip.exe install APScheduler` |
| `.\venv\Scripts\Activate.ps1` not found | No venv at C:\Temp\TradingBotV1 | Don't activate venv there — use full path to TradingBotV2 venv python |
| Bot 1 log goes silent at 21:00 UTC | NORMAL — kill zone active, ADX below 20 = DEBUG only | Not a crash. Check log for ERROR lines. |
| `pkill` not found | Linux command — VPS is Windows | Use `Get-WmiObject Win32_Process \| Where-Object ...` |
| `grep` not found | Linux command — VPS is Windows | Use `Select-String` |
