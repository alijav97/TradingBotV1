# TradingBotV2 — Master TODO & Setup Guide
**Build started:** 2026-05-25  
**Go-live target:** 2026-06-20  
**Capital:** $500 forex (Pepperstone MT5) + $500 crypto (Binance Futures testnet → live)  
**Risk:** 2% per trade | 6 instruments | fully autonomous paper → live trading

---

## Progress Legend
- `[x]` Done
- `[ ]` Not started
- `[!]` Needs your input / action

---

## WHAT CLAUDE NEEDS FROM YOU (before setup session)

Have these ready when you sit down at your VPS:

### 1. Telegram (10 minutes)
- [ ] Open Telegram → search **@BotFather** → send `/newbot`
- [ ] Choose a name (e.g. `TradingBotV2 Alerts`) and username (e.g. `mytradingv2_bot`)
- [ ] Copy the token BotFather gives you → looks like `1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ`
- [ ] Search **@userinfobot** → send any message → copy your numeric ID (e.g. `987654321`)
- [ ] Search your new bot → click **Start** (required or bot can't message you)

### 2. Binance Testnet (5 minutes)
- [ ] Go to **testnet.binancefuture.com** → log in with Google
- [ ] Profile icon → **API Management** → **Create API Key**
- [ ] Label it `TradingBotV2` → copy **API Key** and **Secret Key** (secret shown once only)

### 3. Pepperstone MT5 (already have this)
- [ ] Your MT5 **login number** (from Pepperstone welcome email)
- [ ] Your MT5 **password**
- [ ] Exact **server name** — open MT5 → File → Login → copy the server name from the dropdown
  - Demo accounts: usually `Pepperstone-Demo`
  - Live accounts: usually `Pepperstone-Edge-Live` or `Pepperstone-Prime-Live`

### 4. VPS Access
- [ ] RDP credentials to your Windows Server 2022 VPS
- [ ] Admin rights on that machine (needed for setup script)
- [ ] MetaTrader 5 terminal already installed and logged in on the VPS

---

## SETUP STEPS (do these in order when home)

### Step 1 — Get the code onto the VPS
```
# Option A: git clone (if git is installed on VPS)
git clone https://github.com/alijav97/tradingbotv1 C:\Temp\TradingBotV1

# Option B: download ZIP from GitHub, extract to C:\Temp\TradingBotV1
```

### Step 2 — Run the one-click setup script
```powershell
# Open PowerShell as Administrator
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd C:\Temp\TradingBotV1\v2\deploy
.\windows_setup.ps1
```
This installs Python 3.11, TA-Lib, all dependencies, and copies
everything to `C:\TradingBotV2\`

### Step 3 — Fill in your credentials
```powershell
notepad C:\TradingBotV2\.env
```
Fill in these 7 values (everything else is already set correctly):
```
MT5_LOGIN=           ← your Pepperstone account number
MT5_PASSWORD=        ← your MT5 password
MT5_SERVER=          ← exact server name from MT5 (e.g. Pepperstone-Demo)
BINANCE_API_KEY=     ← from testnet.binancefuture.com
BINANCE_API_SECRET=  ← from testnet.binancefuture.com
TELEGRAM_BOT_TOKEN=  ← from @BotFather
TELEGRAM_CHAT_ID=    ← from @userinfobot
```
Save and close.

### Step 4 — Test manually (make sure everything connects)
```powershell
cd C:\TradingBotV2
.\start_bot.bat
```
You should see:
- `MT5 connected: Pepperstone-Demo`
- `Binance testnet connected`
- `API server starting on port 8000`
- `BotScheduler started — 5 jobs registered`
- Telegram message: **TradingBotV2 started**

Press Ctrl+C to stop once verified.

### Step 5 — Run the ML warm-start backtest
```powershell
cd C:\TradingBotV2
venv\Scripts\activate
python -m v2.backtest.run_backtest --days 180
```
- Takes 10–20 minutes
- Replays 6 months of historical data through the live signal engine
- Generates ~300–500 labeled trades
- Trains the ML model automatically
- Prints `GOOD` or `FAIR` confidence level at the end
- **This is what gives the bot ML confidence by June 20th instead of month 3**

### Step 6 — Install as Windows Service (auto-starts on reboot)
```powershell
cd C:\TradingBotV2\v2\deploy
.\install_service.ps1
```

### Step 7 — Paper trade for 2 weeks (June 3 → June 17)
- Watch Telegram for trade alerts
- Check logs: `Get-Content C:\TradingBotV2\logs\bot_stdout.log -Tail 50 -Wait`
- API dashboard: `http://localhost:8000/health` and `/trades`
- Don't change anything — let it run

### Step 8 — Go live June 20th
```powershell
# Edit .env: change Binance testnet → live keys
notepad C:\TradingBotV2\.env
# Change:
#   BINANCE_API_KEY=    ← your LIVE Binance key
#   BINANCE_API_SECRET= ← your LIVE Binance secret
#   BINANCE_TESTNET=false

# Restart the service
Restart-Service TradingBotV2
```

---

## BUILD STATUS — ALL CODE COMPLETE

### Module 1 — Data Layer ✅
- [x] `instrument_config.py` — 6 instruments (XAUUSD, GBPJPY, WTI, NAS100, BTCUSDT, ETHUSDT)
- [x] `connectors/mt5_connector.py` — MT5 live + historical, stub mode on non-Windows
- [x] `connectors/binance_connector.py` — Binance Futures, testnet support
- [x] `connectors/unified_data.py` — single OHLCV interface for all 6 instruments

### Module 2 — Analysis Engine ✅
- [x] `analysis/indicators.py` — 14 indicators (VWAP bias bug fixed)
- [x] `analysis/smart_money.py` — order blocks, FVGs, liquidity sweeps
- [x] `analysis/candle_patterns.py` — TA-Lib 61-pattern wrapper with graceful fallback
- [x] `analysis/mtf_analyzer.py` — D1/H4/H1 alignment (majority vote per timeframe)
- [x] `analysis/world_sessions.py` + `session_profiler.py`
- [x] `analysis/sr_mapper.py` — instrument-agnostic (pct-based tolerances)
- [x] `analysis/liquidity_map.py` — adaptive bucket POC

### Module 3 — Signal Engine ✅
- [x] `signals/strategy_registry.py` — 15 strategies with metadata
- [x] `signals/confluence_engine.py` — 12-factor scorer, MIN_SCORE=4.0 (raised from 3.0)
- [x] `signals/entry_checklist.py` — 5-gate validation
- [x] `signals/signal_ranker.py` — composite score (confluence×0.4 + WR×0.3 + ML×0.2 + RR×0.1)

### Module 4 — External Intelligence ✅
- [x] `intelligence/news_filter.py` — Forex Factory calendar parser
- [x] `intelligence/news_monitor.py` — FinBERT sentiment (local, no API cost)
- [x] `intelligence/tweet_monitor.py` — nitter RSS, 3-instance fallback
- [x] `intelligence/dxy_correlation.py` — DXY bias per instrument
- [x] `intelligence/geo_filter.py` — 5-level geopolitical risk scoring
- [x] `intelligence/cot_analyzer.py` — CFTC COT data with 24h cache

### Module 5 — Risk Engine ✅
- [x] `risk/position_sizer.py` — 2% risk per trade, correct lot sizing
- [x] `risk/atr_sl_engine.py` — dynamic ATR stops (session + regime + volatility multipliers)
- [x] `risk/trade_manager.py` — partial TP + trailing SL + risk of ruin
- [x] `risk/portfolio_heat.py` — max 25% heat, correlation groups, max 4 open trades
- [x] `risk/loss_limits.py` — 6% daily / 12% weekly drawdown limits

### Module 6 — Paper Trading Engine ✅
- [x] `trading/paper_trader.py` — full lifecycle: open → TP1 → trail SL → TP2/SL/MAX_HOLD
- [x] `trading/trade_monitor.py` — portfolio heat + summary logging
- [x] `trading/auto_trader.py` — no threading, no global state, no race conditions

### Module 7 — Trade Journal ✅
- [x] `journal/sqlite_journal.py` — 5 tables, WAL mode, schema migrations
  - Columns added: `original_sl`, `factors_json`, `exit_regime`, `exit_atr`, `hold_time_minutes`
  - Transaction fix: close_trade uses single atomic commit (no more split commits)
  - Added: `update_stop_loss()` public method

### Module 8 — ML Layer ✅
- [x] `ml/feature_engineer.py` — 52 features (40 original + 12 factor scores + exit context)
  - Data leakage fix: historical WR only uses trades opened BEFORE current trade
- [x] `ml/lightgbm_trainer.py` — early stopping (patience=20), chronological split
- [x] `ml/hmm_regime.py` — 4-state GaussianHMM + rule-based fallback
- [x] `ml/ml_engine.py` — retrain gate, confidence, regime
- [x] `backtest/backtester.py` — warm-start: 6 months history → 300-500 labeled trades
- [x] `backtest/run_backtest.py` — CLI with --days, --instruments, --clear flags

### Module 9 — API + Alerts ✅
- [x] `api/api_server.py` — FastAPI, 8 endpoints
- [x] `api/api_keys.py` — read/full scope keys
- [x] `api/telegram_bot.py` — alerts: signal / trade opened / trade closed / morning briefing

### Module 10 — Scheduler ✅
- [x] `scheduler/scheduler.py` — 5 jobs: 60s monitor, 1H scan, 4H scan, 06:00 briefing, 23:00 retrain

### Deployment Package ✅
- [x] `deploy/.env.template` — all credentials, small-account defaults (2% risk, $500 balance)
- [x] `deploy/windows_setup.ps1` — one-click: Python, TA-Lib, pip, copy to C:\TradingBotV2
- [x] `deploy/install_service.ps1` — NSSM Windows Service, auto-start, crash recovery
- [x] `deploy/README_DEPLOY.md` — step-by-step guide (Telegram, Binance, Pepperstone)

---

## KEY SETTINGS (already configured in .env.template)

| Setting | Value | Why |
|---|---|---|
| `ACCOUNT_BALANCE` | 500 | Your starting capital per account |
| `RISK_PER_TRADE_PCT` | 2.0 | Sweet spot: 2× growth vs 1%, safe vs 3% |
| `MAX_OPEN_TRADES` | 4 | Fewer simultaneous trades on small account |
| `MAX_PORTFOLIO_HEAT` | 25.0 | Max % of balance at risk across all open trades |
| `DAILY_LOSS_LIMIT` | 6.0 | Stops trading after 3 consecutive losses in a day |
| `WEEKLY_LOSS_LIMIT` | 12.0 | Stops trading if week is down 12% |
| `ML_MIN_TRADES_TO_TRAIN` | 50 | ML activates after 50 trades (backtest provides 300+) |
| `BINANCE_TESTNET` | true | Paper trades on Binance testnet until June 20 |

---

## REALISTIC EXPECTATIONS

### 90-day projection at 2% risk
| Win Rate | After 90 days (forex $500) | After 90 days (crypto $500) |
|---|---|---|
| 45% (cold start) | ~$350 | ~$310 |
| 50% (base case) | ~$680 | ~$620 |
| 55% (good case) | ~$1,100 | ~$980 |

### Timeline
| Phase | When | What's happening |
|---|---|---|
| Setup | Now → June 1 | VPS setup, credentials, backtest |
| Paper trading | June 1 → June 17 | Verify everything works, watch alerts |
| Go live | June 20 | Switch Binance to live keys |
| ML improves | July onwards | Live trades + backtest = 500+ training samples |
| Full confidence | Month 3-4 | ML has seen real market conditions |

---

## INSTRUMENTS BEING TRADED

| # | Instrument | Type | Exchange | Account |
|---|---|---|---|---|
| 1 | XAUUSD | Gold | Pepperstone MT5 | Forex $500 |
| 2 | GBPJPY | Forex | Pepperstone MT5 | Forex $500 |
| 3 | WTI | Oil | Pepperstone MT5 | Forex $500 |
| 4 | NAS100 | Index | Pepperstone MT5 | Forex $500 |
| 5 | BTCUSDT | Crypto | Binance Futures | Crypto $500 |
| 6 | ETHUSDT | Crypto | Binance Futures | Crypto $500 |

---

## BUGS FIXED IN AUDIT SESSION (2026-05-25)

- VWAP bias was inverted (above VWAP labelled "bearish") — fixed
- MIN_SCORE raised from 3.0 → 4.0 (was too easy to trigger false signals)
- ML data leakage: historical win rates now only use past trades, not future ones
- 12 confluence factor scores now stored per trade (was only storing total)
- Exit-time context (regime, ATR, hold time) now captured when trade closes
- Trailing SL after TP1 now actually implemented (was imported but never called)
- Journal close_trade now uses single atomic transaction (two-commit race fixed)
- LightGBM early stopping added (was training all 200 trees, risked overfitting)
- Raw SQL direct DB access in paper_trader replaced with journal public method

---

## NOTES
- MT5 connector: works natively on Windows. Won't connect in cloud/Linux (stub mode)
- TA-Lib: windows_setup.ps1 installs precompiled wheel — no C compiler needed
- FinBERT: ~400MB download on first run (cached to C:\Users\...\\.cache after that)
- No Anthropic API key needed — FinBERT handles all news sentiment locally
- Backtest takes 10-20 minutes for 6 months of data across all 6 instruments
- After backtest, bot starts with trained ML instead of blank model
