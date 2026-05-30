# TradingBotV1 — System Architecture & Operations Journal

Last updated: 2026-05-31
Git repo: https://github.com/alijav97/TradingBotV1.git
Active branch: claude/xauusd-strategy-backtest-sO2vM

---

## GOLDEN RULES (never break these)

1. `btc_research/` = BTC work ONLY. Never import from `v2/` inside here.
2. `v2/` = WTI live bot ONLY. Never import from `btc_research/` inside here.
3. BTC Bot 2 is fully standalone — own connectors, own journal, own settings. Zero v2 dependency.
4. MT5_SERVER_UTC_OFFSET = 3. Pepperstone server is UTC+3. All bar timestamps subtract 3h to get true UTC. If you skip this, kill zones fire at completely wrong real-world hours.
5. Never run `pip install` in a random terminal — always use `C:\TradingBotV2\venv\Scripts\pip.exe`.
6. Never use Linux commands on the VPS (no `pkill`, `grep`, `tail`, `screen`) — VPS is Windows PowerShell.
7. Never commit `.env`, `*.db`, `*.log`, `__pycache__` to git.

---

## REPO STRUCTURE

```
TradingBotV1/                        ← GitHub repo root
│
├── v2/                              ← WTI LIVE BOT (production)
│   ├── main.py                      ← Entry point: python -m v2.main
│   ├── settings.py                  ← ACTIVE_SYMBOLS, risk params, MT5 creds from .env
│   ├── instrument_config.py         ← Per-instrument constants (WTI, XAUUSD, GBPJPY, NAS100, BTC, ETH)
│   ├── scheduler/scheduler.py       ← APScheduler: 5min H1 scan + 2s KZ scan
│   ├── signals/
│   │   ├── confluence_engine.py     ← Master signal scorer
│   │   ├── entry_checklist.py       ← Pre-trade checklist
│   │   ├── signal_ranker.py         ← Ranks signals by score
│   │   └── strategies/
│   │       └── ny_momentum_wti.py   ← ACTIVE WTI strategy (NY session momentum)
│   ├── connectors/
│   │   ├── mt5_connector.py         ← Pepperstone MT5 connection
│   │   ├── binance_connector.py     ← Binance connection
│   │   └── unified_data.py          ← DataFeed: get_price(), get_ohlcv()
│   ├── journal/sqlite_journal.py    ← SQLite trade journal
│   ├── trading/paper_trader.py      ← Paper trading engine
│   ├── risk/                        ← Position sizing, ATR SL, portfolio heat
│   ├── api/telegram_bot.py          ← WTI Telegram alerts
│   └── backtest/                    ← Backtesting engine
│
├── btc_research/                    ← BTC RESEARCH + BTC LIVE BOTS
│   ├── settings.py                  ← Shared BTC research settings (KZ=21-24 UTC, MR=17-21 UTC)
│   ├── strategy/confluence.py       ← BTC confluence scorer (MR breakout + inter-market factors)
│   ├── factors/                     ← btc_momentum, gold_factor, nasdaq_factor, time_factor
│   ├── strategies/                  ← Strategy implementations (swing_level, vb, etc.)
│   ├── backtest/                    ← BTC backtesting engine
│   ├── analysis_*.py                ← Research scripts — NOT live, not run by bots
│   │
│   ├── btc_bot_1/                   ← BTC BOT 1 — "Version D" live paper trading
│   │   ├── main.py                  ← Entry point: python -m btc_research.btc_bot_1.main
│   │   ├── settings.py              ← KZ=[21,22,23] UTC, ADX_MIN=20, risk 2-3%
│   │   ├── connectors/
│   │   │   ├── mt5_connector.py     ← Standalone MT5 (subtracts UTC+3 offset from timestamps)
│   │   │   └── unified_data.py      ← Standalone DataFeed for Bot 1
│   │   ├── journal/sqlite_journal.py ← Bot 1 trade DB → btc_trades.db
│   │   ├── signals/btc_engine.py    ← Signal engine: EMA200 → ADX → MR breakout → confluence
│   │   ├── trading/paper_trader.py  ← Paper trader: open/monitor/close trades
│   │   ├── scheduler/scheduler.py   ← BTCScheduler: 5min BG scan + 2s KZ scan + heartbeats
│   │   └── api/telegram_bot.py      ← Telegram alerts (BTC_TELEGRAM_BOT_TOKEN)
│   │
│   └── btc_bot_2/                   ← BTC BOT 2 — "SwingLevelBreak v2 + VB" live paper trading
│       ├── main.py                  ← Entry point: python -m btc_research.btc_bot_2.main
│       ├── settings.py              ← KZ=[1,2,3,8] UTC, ADX split risk
│       ├── connectors/
│       │   ├── mt5_connector.py     ← Standalone MT5 (same UTC+3 correction)
│       │   └── unified_data.py      ← Standalone DataFeed for Bot 2
│       ├── journal/sqlite_journal.py ← Bot 2 trade DB → btc2_trades.db
│       ├── signal_engine.py         ← Signal engine: EMA200 → ADX → VBSwing strategy
│       ├── paper_trader.py          ← Paper trader for Bot 2
│       ├── scheduler.py             ← BTC2Scheduler: 5min BG + 2s KZ + 2s post-KZ + heartbeats
│       ├── telegram.py              ← Telegram alerts (BTC2_TELEGRAM_BOT_TOKEN)
│       ├── strategy/
│       │   └── vb_swing_combined.py ← VB + SwingBreak v2 combined strategy
│       └── api.py                   ← FastAPI health/trades endpoints (port 8002)
│
├── SYSTEM_ARCHITECTURE.md           ← THIS FILE — read first if lost
├── setup_vps.ps1                    ← VPS setup script (Chocolatey + Python + venv)
├── requirements.txt                 ← All Python dependencies
├── .env                             ← Credentials (gitignored — NEVER commit)
└── v2/deploy/
    ├── windows_setup.ps1            ← Sets up C:\TradingBotV2\ on VPS
    └── install_service.ps1          ← Registers Windows Task Scheduler service
```

---

## VPS LAYOUT (physical folder structure on the Windows VPS)

```
C:\TradingBotV2\                 ← WTI BOT installation (set up by windows_setup.ps1)
├── v2\                          ← WTI bot code
├── venv\                        ← Python virtual environment — ALL packages live here
│   └── Scripts\
│       ├── python.exe           ← USE THIS to run ALL bots (Bot 1, Bot 2, WTI)
│       └── pip.exe              ← USE THIS to install packages
└── .env                         ← MASTER credentials file (WTI + BTC + BTC2)

C:\Temp\TradingBotV1\            ← Git clone of the full repo (for BTC bots)
├── btc_research\                ← BTC bot source code (pulled from GitHub)
│   ├── btc_bot_1\
│   └── btc_bot_2\
└── .env                         ← Secondary .env (must stay in sync with TradingBotV2\.env)
```

### WHY this split exists:
- `C:\TradingBotV2\` was set up by `windows_setup.ps1` for the WTI bot — it has the venv with all packages but does NOT have `btc_research/`
- `C:\Temp\TradingBotV1\` is the git clone — it has `btc_research/` but has NO venv of its own
- **Solution**: Run BTC bots from `C:\Temp\TradingBotV1` (so Python finds `btc_research`) but use `C:\TradingBotV2\venv\Scripts\python.exe` (which has all packages installed)

---

## .ENV FILE CONTENTS

### C:\TradingBotV2\.env (master file — must have ALL of these)
```
ACTIVE_SYMBOLS=WTI
MT5_LOGIN=51486884
MT5_PASSWORD=<password>
MT5_SERVER=PepperstoneFinancialUAE-MT5-Live01
MT5_TIMEOUT=10000
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=true
TELEGRAM_BOT_TOKEN=<WTI bot token>           ← used by WTI bot + fallback for BTC bots
TELEGRAM_CHAT_ID=7675749781
BTC_TELEGRAM_BOT_TOKEN=<BTC Bot 1 token>    ← dedicated Bot 1 Telegram bot
BTC_TELEGRAM_CHAT_ID=7675749781
BTC2_TELEGRAM_BOT_TOKEN=7658687644:AAFkdoEsjU7gPr_QtwSFJeTUAxuWqOMWH6M
BTC2_TELEGRAM_CHAT_ID=7675749781
BTC2_API_KEY=btc2_readonly
BTC2_API_PORT=8002
ACCOUNT_BALANCE=500
RISK_PER_TRADE_PCT=2.0
```

### C:\Temp\TradingBotV1\.env (must match above for BTC credentials)
Same file — keep in sync manually. BTC bots read from this when running from C:\Temp\TradingBotV1.

### Telegram Token Routing
| Bot | Primary env var | Fallback | Dedicated bot name |
|-----|----------------|----------|--------------------|
| WTI Bot | `TELEGRAM_BOT_TOKEN` | — | WTI alerts |
| BTC Bot 1 | `BTC_TELEGRAM_BOT_TOKEN` | `TELEGRAM_BOT_TOKEN` | BTC Bot 1 alerts |
| BTC Bot 2 | `BTC2_TELEGRAM_BOT_TOKEN` | `TELEGRAM_BOT_TOKEN` | @BTC2TradingBotVBandSwingbot |

---

## HOW TO RUN — VPS PowerShell Commands

### ─── CORRECT START PATTERN (always use this) ───
```powershell
Start-Process "C:\TradingBotV2\venv\Scripts\python.exe" `
    -ArgumentList "-m btc_research.btc_bot_1.main" `
    -WorkingDirectory "C:\Temp\TradingBotV1"
```
- `python.exe` path → from TradingBotV2 venv (has all packages)
- `-WorkingDirectory` → C:\Temp\TradingBotV1 (where btc_research package is)
- This opens a NEW PowerShell window for the bot

---

### Kill Bot 1 only
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot_1" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

### Kill Bot 2 only
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot_2" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

### Kill ALL BTC bots
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

### Start Bot 1
```powershell
Start-Process "C:\TradingBotV2\venv\Scripts\python.exe" -ArgumentList "-m btc_research.btc_bot_1.main" -WorkingDirectory "C:\Temp\TradingBotV1"
```

### Start Bot 2
```powershell
Start-Process "C:\TradingBotV2\venv\Scripts\python.exe" -ArgumentList "-m btc_research.btc_bot_2.main" -WorkingDirectory "C:\Temp\TradingBotV1"
```

### Pull latest code from GitHub + kill + restart BOTH bots (all-in-one)
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; cd C:\Temp\TradingBotV1; git pull origin claude/xauusd-strategy-backtest-sO2vM; Start-Process "C:\TradingBotV2\venv\Scripts\python.exe" -ArgumentList "-m btc_research.btc_bot_1.main" -WorkingDirectory "C:\Temp\TradingBotV1"; Start-Process "C:\TradingBotV2\venv\Scripts\python.exe" -ArgumentList "-m btc_research.btc_bot_2.main" -WorkingDirectory "C:\Temp\TradingBotV1"
```

### Install/update Python packages (always use TradingBotV2 venv pip)
```powershell
C:\TradingBotV2\venv\Scripts\pip.exe install -r C:\Temp\TradingBotV1\requirements.txt
```

### Check if bots are running
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match "btc_bot" } | Select-Object ProcessId, CommandLine
```

### Read live log for Bot 1
```powershell
Get-Content "C:\Temp\TradingBotV1\btc_research\btc_bot_1\data\logs\btc_bot.log" -Tail 50
```

### Read live log for Bot 2
```powershell
Get-Content "C:\Temp\TradingBotV1\btc_research\btc_bot_2\data\logs\btc_bot_2.log" -Tail 50
```

### WTI Bot (runs as Windows Task Scheduler job)
```powershell
Start-ScheduledTask -TaskName "TradingBotV1"   # start
Stop-ScheduledTask  -TaskName "TradingBotV1"   # stop
Get-ScheduledTask   -TaskName "TradingBotV1"   # check status
```

---

## GIT WORKFLOW

```
1. Code on LOCAL machine:  C:\Users\alija\Downloads\TradingBotV1\
        ↓
2. git push origin claude/xauusd-strategy-backtest-sO2vM
        ↓
3. On VPS: cd C:\Temp\TradingBotV1
           git pull origin claude/xauusd-strategy-backtest-sO2vM
        ↓
4. Restart bots with the all-in-one command above
```

**IMPORTANT**: `C:\TradingBotV2\` is NOT a git repo — never try to git pull there. Only `C:\Temp\TradingBotV1\` is the git clone.

---

## BOT STRATEGIES — FULL DETAIL

---

### WTI BOT (v2)
- **Strategy**: NYMomentumWTI
- **Active window**: 13:00–17:00 UTC (NY / NYMEX session open)
- **Instrument**: WTI Crude Oil (Pepperstone MT5: SpotCrude)
- **Scan**: Every 5min on H1
- **Telegram**: `TELEGRAM_BOT_TOKEN`
- **Trade DB**: C:\TradingBotV2\data\v2_trades.db

---

### BTC BOT 1 — "Version D"
- **Instrument**: BTCUSD (Pepperstone MT5)
- **Timeframe**: H1
- **Kill zone**: UTC 21:00, 22:00, 23:00 only
- **UAE time**: 01:00–04:00 AM (runs overnight)
- **Backtest**: 2yr · 223 trades · 43% WR · +$23,733 · MaxDD 16.1%

#### Entry filter chain (ALL must pass in order):
```
1. Kill Zone gate        → must be UTC hour 21, 22, or 23
2. EMA200 filter         → LONG only if price > EMA200(H1)
                           SHORT only if price < EMA200(H1)
3. ADX(14) >= 20         → skip entirely if ADX below 20 (ranging market)
4. Pre-KZ range exists   → need >= 3 H1 bars between 17:00–21:00 UTC (consolidation range)
5. Range breakout        → bar must CLOSE above range high (long)
                           or CLOSE below range low (short)
6. Confluence score >= 3.0:
     BTC momentum factor  × 1.0  (own trend — most important)
     Gold factor          × 0.5  (inter-market confirmation)
     Nasdaq factor        × 0.5  (risk-on/off environment)
     Time factor          × 0.3  (session timing bonus)
7. Live price deviation  → live price must be within 3% of bar close (execution safety)
```

#### Risk sizing (Flipped Risk):
- ADX 20–28 → **3% risk** (early trend — size up)
- ADX > 28  → **2% risk** (extended trend — normal size)

#### Exits:
- **SL**: opposite side of the 17:00–21:00 pre-KZ range
- **TP1**: 2R → partial close + move SL to breakeven
- **TP2**: 5R → full close
- **Trailing SL**: after TP1, trails at 2×ATR below/above price
- **Max hold**: 96 hours (force close)

#### Scheduler jobs:
| Job | Frequency | What it does |
|-----|-----------|-------------|
| KZ scan | Every 2s | Inside KZ only. Logs `KZ scan #N \| UTC HH:MM:SS` then runs signal engine |
| BG scan | Every 5min | Outside KZ only. Logs `BG scan #N \| UTC HH:MM` + market snapshot |
| Post-KZ watch | Every 5s | Outside KZ, only if trade is open — monitors SL/TP |
| Daily briefing | 20:00 UTC | Sends stats + "KZ opens in 1h" Telegram alert |

#### What normal logs look like (inside KZ, no signal):
```
2026-05-31T21:00:02  INFO  btc_bot_1.scheduler  KZ scan #1 | UTC 21:00:02
2026-05-31T21:00:02  INFO  btc_bot_1.scheduler    LONG → SKIP: ADX filter: ADX 14.2 < threshold 20
2026-05-31T21:00:02  INFO  btc_bot_1.scheduler    SHORT → SKIP: EMA200 filter: SHORT blocked — price above EMA200
2026-05-31T21:00:04  INFO  btc_bot_1.scheduler  KZ scan #2 | UTC 21:00:04
...
```

#### What a signal looks like:
```
2026-05-31T22:15:06  INFO  btc_bot_1.signals.btc_engine  BTC SIGNAL LONG @ 107450.00  SL=106800.00  TP1=108750.00  score=4.20  ADX=24.1  risk_pct=3%
```

- **Telegram**: `BTC_TELEGRAM_BOT_TOKEN` → fallback `TELEGRAM_BOT_TOKEN`
- **Trade DB**: C:\Temp\TradingBotV1\btc_research\btc_bot_1\data\btc_trades.db

---

### BTC BOT 2 — "SwingLevelBreak v2 + VB"
- **Instrument**: BTCUSD (Pepperstone MT5)
- **Timeframe**: H1
- **Kill zones**: UTC 01:00, 02:00, 03:00, 08:00
- **UAE time**: 05:00, 06:00, 07:00, 12:00 UAE
- **Backtest**: 46.7% WR · +$119k 2yr

#### Entry filter chain (ALL must pass in order):
```
1. Kill Zone gate   → must be UTC hour 1, 2, 3, or 8
2. Fetch 300 H1 bars (need >= 220 for EMA200 to be valid)
3. EMA200 filter    → bar_close > EMA200 → direction = LONG
                      bar_close < EMA200 → direction = SHORT
4. ADX(14) >= 20    → skip if ADX below 20 (no clear trend)
5. Strategy signal  → VBSwingStrategy (VB breakout OR SwingLevelBreak v2):
                        - VB: volatility breakout of recent swing high/low
                        - SLv2: swing level break with 2×ATR confirmation on both legs
6. One-trade rule   → only 1 open trade at a time (blocks if trade already open)
```

#### Risk sizing (ADX split):
- ADX ≤ 25  → **3% risk** (early trend)
- ADX 25–40 → **2% risk** (transition)
- ADX ≥ 40  → **3% risk** (strong trend — size up again)

#### Exits:
- **TP1**: 2R → partial close + SL to breakeven
- **TP2**: 5R → full close
- **Trailing SL**: 2×ATR after TP1
- **Max hold**: 96 hours

#### Scheduler jobs:
| Job | Frequency | What it does |
|-----|-----------|-------------|
| KZ scan | Every 2s | Inside KZ only. Logs `KZ scan #N \| UTC HH:MM:SS` then runs signal engine |
| BG scan | Every 5min | Outside KZ only. Logs `BG scan #N \| UTC HH:MM (data refresh)` |
| Post-KZ watch | Every 2s | Outside KZ, only if trade open — monitors SL/TP at full speed |
| Trade monitor | Every 60s | Always-on safety net — catches anything 2s job might miss |
| Morning briefing | 02:00 UTC | Daily stats alert |

#### What normal logs look like (outside KZ):
```
2026-05-31T22:55:01  INFO  btc_bot_2.scheduler  BG scan #1 | UTC 22:55 (data refresh — outside KZ)
2026-05-31T23:00:01  INFO  btc_bot_2.scheduler  BG scan #2 | UTC 23:00 (data refresh — outside KZ)
```

#### What normal logs look like (inside KZ, no signal):
```
2026-05-31T01:00:02  INFO  btc_bot_2.scheduler  KZ scan #1 | UTC 01:00:02
2026-05-31T01:00:02  INFO  btc_bot_2.signal_engine    → SKIP: ADX 14.2 < threshold 20 (weak trend) | BTC $107234 | EMA200 $105100
2026-05-31T01:00:04  INFO  btc_bot_2.scheduler  KZ scan #2 | UTC 01:00:04
...
```

- **Telegram**: `BTC2_TELEGRAM_BOT_TOKEN` → fallback `TELEGRAM_BOT_TOKEN`
- **API**: FastAPI on port 8002 (health/trades/performance endpoints)
- **Trade DB**: C:\Temp\TradingBotV1\btc_research\btc_bot_2\data\btc2_trades.db

---

## MT5 TIMEZONE CORRECTION

Pepperstone server returns bars in **UTC+3** (server local time).
Without correction, a bar timestamped 00:00 by MT5 is actually 21:00 UTC — completely wrong for kill-zone matching.

**Fix applied in both bot connectors:**
```python
MT5_SERVER_UTC_OFFSET = 3
df["time"] = df["time"] - pd.Timedelta(hours=MT5_SERVER_UTC_OFFSET)
```

This is in:
- `btc_research/btc_bot_1/connectors/mt5_connector.py`
- `btc_research/btc_bot_2/connectors/mt5_connector.py`

---

## DEPENDENCY MAP — what imports what

```
btc_bot_1/main.py
  → btc_bot_1/settings.py           KZ hours, thresholds, risk, DB path
  → btc_bot_1/connectors/           standalone MT5 + DataFeed (no v2)
  → btc_bot_1/journal/              standalone SQLite journal → btc_trades.db
  → btc_bot_1/signals/btc_engine    EMA200 + ADX + confluence signal logic
  → btc_bot_1/trading/paper_trader  open/monitor/close trades
  → btc_bot_1/scheduler/scheduler   APScheduler jobs (2s KZ, 5min BG)
  → btc_bot_1/api/telegram_bot      Telegram alerts
  → btc_research/strategy/          confluence scorer (shared research module)
  ✗ NEVER imports from v2/

btc_bot_2/main.py
  → btc_bot_2/settings.py           KZ hours, ADX thresholds, risk split, DB path
  → btc_bot_2/connectors/           standalone MT5 + DataFeed (no v2)
  → btc_bot_2/journal/              standalone SQLite journal → btc2_trades.db
  → btc_bot_2/signal_engine         EMA200 + ADX + VBSwing signal logic
  → btc_bot_2/paper_trader          open/monitor/close trades
  → btc_bot_2/scheduler             APScheduler jobs (2s KZ, 5min BG, 2s post-KZ, 60s monitor)
  → btc_bot_2/telegram              Telegram alerts
  → btc_bot_2/strategy/             VB + SwingBreak v2 combined strategy
  → btc_bot_2/api                   FastAPI endpoints
  ✗ NEVER imports from v2/

v2/main.py
  → v2/settings.py                  ACTIVE_SYMBOLS, risk config
  → v2/instrument_config.py         per-instrument constants
  → v2/connectors/                  MT5 + Binance DataFeed
  → v2/journal/sqlite_journal       v2 trade DB
  → v2/signals/                     confluence engine + strategies
  → v2/trading/                     paper trader + monitor
  → v2/risk/                        position sizer, ATR SL, heat checks
  → v2/scheduler/scheduler          scan scheduler
  → v2/api/telegram_bot             WTI Telegram alerts
  ✗ NEVER imports from btc_research/
```

---

## COMMON ERRORS & FIXES

| Error | Root cause | Fix |
|-------|-----------|-----|
| `No module named 'btc_research'` | Running python from wrong directory | Must run from `C:\Temp\TradingBotV1` as WorkingDirectory |
| `No module named 'requests'` | Using system Python not the venv | Use `C:\TradingBotV2\venv\Scripts\python.exe` |
| `APScheduler not installed` | Package missing from venv | `C:\TradingBotV2\venv\Scripts\pip.exe install APScheduler fastapi uvicorn` |
| `.\venv\Scripts\Activate.ps1` not found | No venv at C:\Temp\TradingBotV1 | Never activate venv there — use full python.exe path instead |
| `pkill` not recognized | Linux command — VPS is Windows | Use `Get-WmiObject Win32_Process \| Where-Object { $_.CommandLine -match "btc_bot" } \| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }` |
| `grep` not recognized | Linux command — VPS is Windows | Use `Select-String` |
| No logs in Bot window after startup | Was: normal (DEBUG only). Now: fixed with heartbeats — if still silent, bot crashed on startup | Check for `ERROR` or `Traceback` in the window |
| Bot window shows startup then closes | Python crash — unhandled exception | Scroll up in the window to see the Traceback |
| `git pull` says "not a git repository" | Wrong directory — C:\TradingBotV2 is NOT a git repo | `cd C:\Temp\TradingBotV1` first, then git pull |
| Signals firing at wrong hours | MT5 UTC offset not applied | Check mt5_connector.py has `MT5_SERVER_UTC_OFFSET = 3` subtraction |

---

## HOW TO VERIFY BOTS ARE HEALTHY

After starting both bots, you should see within the first 30 seconds:

**Bot 1 startup (in its window):**
```
INFO  __main__  BTC Bot 1 starting up
INFO  __main__  Kill-zone: 21:00 - 24:00 UTC
INFO  __main__  BTCScheduler started — 4 jobs | kill-zone 21-24 UTC
INFO  __main__  BTC Bot 1 running — press Ctrl+C or send SIGTERM to stop
```
Then every 5min (outside KZ): `BG scan #N | UTC HH:MM`
Then every 2s (inside KZ 21-24 UTC): `KZ scan #N | UTC HH:MM:SS`

**Bot 2 startup (in its window):**
```
INFO  __main__  BTC Bot 2 starting up
INFO  __main__  Kill-zone : 01:00, 02:00, 03:00, 08:00 UTC
INFO  __main__  BTC2Scheduler started — 5 jobs | KZ: [01:00,02:00,03:00,08:00] UTC
INFO  __main__  BTC Bot 2 running — press Ctrl+C or send SIGTERM to stop
```
Then every 5min (outside KZ): `BG scan #N | UTC HH:MM (data refresh — outside KZ)`
Then every 2s (inside KZ hours 1,2,3,8): `KZ scan #N | UTC HH:MM:SS`

**If you see NOTHING after startup — bot has crashed.** Scroll up in the window to find the Traceback.
