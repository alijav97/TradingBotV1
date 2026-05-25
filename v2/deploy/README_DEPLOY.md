# TradingBotV2 — Windows Server 2022 Deployment Guide

**Platform:** Windows Server 2022  
**Broker:** Pepperstone (MT5)  
**Crypto:** Binance Futures Testnet  
**Alerts:** Telegram (personal account)

---

## Overview

This guide covers every credential you need and every step to get the bot running autonomously on your VPS.

**Time required:** ~30 minutes  
**What you'll set up:**
1. Telegram bot + your chat ID
2. Binance testnet API key
3. Pepperstone MT5 credentials
4. Python + dependencies on the VPS
5. Windows Service (auto-starts on reboot)

---

## PART 1 — Telegram Setup

The bot sends you alerts for every trade opened, closed, and a daily morning briefing. You need two things: a **bot token** and your **personal chat ID**.

### Step 1A — Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send: `/newbot`
3. Choose a name, e.g. `TradingBotV2 Alerts`
4. Choose a username (must end in `bot`), e.g. `mytradingv2_bot`
5. BotFather replies with a **token** that looks like:
   ```
   1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
   ```
6. **Save this token** — you'll put it in the `.env` file as `TELEGRAM_BOT_TOKEN`

### Step 1B — Get Your Personal Chat ID

1. Search Telegram for **@userinfobot**
2. Send it any message
3. It replies with your **ID** (a number like `987654321`)
4. **Save this number** — you'll put it in `.env` as `TELEGRAM_CHAT_ID`

> **Why do this?** The bot only sends alerts to the specific chat ID you configure. Nobody else can receive your trade alerts.

### Step 1C — Start a Chat with Your Bot

1. Search Telegram for the bot username you created (e.g. `@mytradingv2_bot`)
2. Click **Start** (or send `/start`)
3. This is required — Telegram won't let the bot message you until you've started the chat

---

## PART 2 — Binance Testnet API Keys

The bot trades BTC/USDT and ETH/USDT on **Binance Futures Testnet** (fake money, real market prices). This is free.

### Step 2A — Get Testnet API Keys

1. Go to **https://testnet.binancefuture.com**
2. Click **Log In** in the top right
3. Log in with your Google account (or create one)
4. Once logged in, click your profile icon → **API Management**
5. Click **Create API Key**
6. Give it a label (e.g. `TradingBotV2`)
7. You'll receive:
   - **API Key** — save as `BINANCE_API_KEY`
   - **Secret Key** — save as `BINANCE_API_SECRET` (shown once — copy it immediately)

> **Note:** Testnet keys only work on testnet.binancefuture.com, not on real Binance. The `.env` template has `BINANCE_TESTNET=true` which routes all requests to testnet automatically.

### Step 2B — Testnet Account Balance

The testnet gives you fake USDT automatically. No deposit needed.

---

## PART 3 — Pepperstone MT5 Credentials

You need MetaTrader5 **installed and running** on the same Windows Server where you run the bot.

### Step 3A — Your MT5 Credentials

From your Pepperstone welcome email, you have:
- **Login number** (e.g. `12345678`) → `MT5_LOGIN`
- **Password** → `MT5_PASSWORD`
- **Server name** → `MT5_SERVER`

Common Pepperstone server names:
| Account type | Server string |
|---|---|
| Demo | `Pepperstone-Demo` |
| Live (Edge) | `Pepperstone-Edge-Live` |
| Live (Prime) | `Pepperstone-Prime-Live` |

> **Tip:** Open MetaTrader5 → File → Login → the server dropdown shows the exact name. Copy it exactly (case-sensitive).

### Step 3B — MT5 Must Be Running

The Python `MetaTrader5` package connects to the MT5 terminal process. The terminal must be running and **logged in** when the bot starts. You don't need to keep the chart window visible — you can minimize it.

**Recommended:** Log in to MT5, enable "Auto trading", then minimize. The bot will connect to it automatically.

---

## PART 4 — VPS Setup (Windows Server 2022)

### Step 4A — Copy Project Files to VPS

1. RDP into your Windows Server 2022 VPS
2. Download or clone this repository to a temporary folder, e.g. `C:\Temp\TradingBotV1`
3. Open **PowerShell as Administrator**

### Step 4B — Run the Setup Script

```powershell
# Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Navigate to the deploy folder
cd C:\Temp\TradingBotV1\v2\deploy

# Run setup (installs Python, TA-Lib, all dependencies, copies files to C:\TradingBotV2)
.\windows_setup.ps1
```

This takes 3–5 minutes. It will:
- Install Python 3.11 if not already present
- Create `C:\TradingBotV2\` with the full project
- Set up a Python virtual environment at `C:\TradingBotV2\venv`
- Install TA-Lib (precompiled wheel — no compiler needed)
- Install all dependencies from `requirements.txt`
- Create `C:\TradingBotV2\.env` from the template

### Step 4C — Fill In Your Credentials

```powershell
notepad C:\TradingBotV2\.env
```

Fill in these values (all others can stay as defaults for now):

```env
# Your Pepperstone MT5 account
MT5_LOGIN=12345678               ← your actual login number
MT5_PASSWORD=YourRealPassword    ← your MT5 password
MT5_SERVER=Pepperstone-Demo      ← exact server name from MT5

# Binance testnet (from testnet.binancefuture.com)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true             ← keep as true until you're ready for live

# Telegram (from steps above)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=987654321

# Account settings (adjust to match your actual balance)
ACCOUNT_BALANCE=10000
RISK_PER_TRADE_PCT=1.0           ← 1% per trade
```

Save and close Notepad.

### Step 4D — Test the Bot Manually (Recommended)

Before installing as a service, do a quick manual test:

```powershell
cd C:\TradingBotV2
.\start_bot.bat
```

You should see log output like:
```
INFO  DataFeed connecting...
INFO  MT5 connected: Pepperstone-Demo (account 12345678)
INFO  Binance testnet connected
INFO  API server starting on port 8000
INFO  BotScheduler started — 5 jobs registered
```

And within seconds, a Telegram message will arrive: **"TradingBotV2 started"**

Press `Ctrl+C` to stop.

---

## PART 5 — Install as Windows Service

Once the manual test succeeds:

```powershell
# Still as Administrator in PowerShell
cd C:\TradingBotV2\v2\deploy
.\install_service.ps1
```

This installs **TradingBotV2** as a Windows Service that:
- Starts automatically when the server reboots
- Restarts automatically if it crashes
- Logs to `C:\TradingBotV2\logs\`

### Verifying the Service

```powershell
# Check status
Get-Service TradingBotV2

# Watch live logs
Get-Content C:\TradingBotV2\logs\bot_stdout.log -Tail 50 -Wait

# Stop the service
Stop-Service TradingBotV2

# Start the service
Start-Service TradingBotV2

# Remove the service
cd C:\TradingBotV2\v2\deploy
.\install_service.ps1 -Uninstall
```

---

## PART 6 — Verify Everything Is Working

### Check 1 — Telegram Alert

You should receive a startup message on Telegram within 30 seconds of the service starting.

### Check 2 — API Health Check

From the VPS (or any browser if the port is open):
```
http://localhost:8000/health
```
Expected response:
```json
{"status": "ok", "uptime_seconds": 42, "open_trades": 0}
```

### Check 3 — First Signal Scan

The H1 signal scan fires at the top of every hour (`:00`). After it runs, check logs:
```powershell
Get-Content C:\TradingBotV2\logs\bot_stdout.log -Tail 100
```
You should see lines like:
```
INFO  Signal scan starting: H1
INFO  SIGNAL XAUUSD long score=4.2 strategy=smc_order_block
INFO  TRADE OPENED [a3f7c2d1]: XAUUSD LONG @ 3312.450 score=4.2
```

And a Telegram message with the trade details.

### Check 4 — After 50 Paper Trades

The ML layer activates automatically when 50 paper trades are in the journal. The nightly retrain job runs at 23:00 GST (19:00 UTC). After retraining, the signal ranker uses ML confidence scores.

---

## Instruments Being Traded

| Instrument | Exchange | Symbol in MT5/Binance |
|---|---|---|
| Gold | Pepperstone MT5 | `XAUUSD` |
| GBP/JPY | Pepperstone MT5 | `GBPJPY` |
| WTI Oil | Pepperstone MT5 | `WTI` or `USOil` |
| NASDAQ 100 | Pepperstone MT5 | `NAS100` |
| Bitcoin | Binance Futures | `BTCUSDT` |
| Ethereum | Binance Futures | `ETHUSDT` |

> **WTI symbol:** Pepperstone uses `WTI` on some accounts and `USOil` on others. Check your MT5 Market Watch for the exact symbol and update `v2/instrument_config.py` if needed.

---

## Switching from Testnet to Live Binance

When you're ready to trade live crypto (real money):

1. Create a **live** Binance API key at https://www.binance.com → Profile → API Management
2. Enable: `Enable Futures`; Disable: `Enable Withdrawals`
3. Whitelist your VPS IP for extra security
4. Update `.env`:
   ```env
   BINANCE_API_KEY=your_live_key
   BINANCE_API_SECRET=your_live_secret
   BINANCE_TESTNET=false
   ```
5. Restart the service: `Restart-Service TradingBotV2`

---

## Risk Settings Reference

All in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ACCOUNT_BALANCE` | `10000` | Your account balance in USD |
| `RISK_PER_TRADE_PCT` | `1.0` | % of balance risked per trade |
| `MAX_OPEN_TRADES` | `6` | Max simultaneous open trades |
| `MAX_PORTFOLIO_HEAT` | `30.0` | Max % of balance at risk across all trades |
| `DAILY_LOSS_LIMIT` | `3.0` | Bot stops trading if daily loss exceeds this % |
| `WEEKLY_LOSS_LIMIT` | `6.0` | Bot stops trading if weekly loss exceeds this % |

---

## Troubleshooting

**MT5 won't connect**
- Make sure MetaTrader5 terminal is running and logged in
- Verify `MT5_SERVER` exactly matches the server name in MT5 (File → Login)
- The Python package requires MT5 terminal version 5.0.37 or later

**Binance connection error**
- For testnet: confirm keys are from testnet.binancefuture.com, not binance.com
- Check `BINANCE_TESTNET=true` is set for testnet keys

**Telegram not receiving messages**
- Confirm you started a chat with your bot (sent `/start`)
- Double-check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
- Chat ID must be your personal chat ID (from @userinfobot), not the bot's ID

**TA-Lib import error**
- The bot runs fine without TA-Lib (candle patterns disabled, graceful fallback)
- To install manually: download the `.whl` from https://github.com/cgohlke/talib-build/releases

**"insufficient data" in logs**
- Normal on first startup — data fills in after a few candles load
- MT5 historical data requires the terminal to have downloaded it; open the chart for each instrument once

**Service won't start / crashes**
```powershell
Get-Content C:\TradingBotV2\logs\bot_stderr.log -Tail 50
```
This shows Python tracebacks and import errors.

---

## File Layout on VPS

```
C:\TradingBotV2\
├── v2\                     ← all Python source code
│   ├── main.py             ← entry point
│   ├── settings.py         ← config (reads from .env)
│   ├── instrument_config.py
│   ├── connectors\
│   ├── analysis\
│   ├── signals\
│   ├── intelligence\
│   ├── risk\
│   ├── trading\
│   ├── journal\
│   ├── ml\
│   ├── api\
│   ├── scheduler\
│   └── deploy\
│       ├── .env.template   ← copy of credential template
│       ├── windows_setup.ps1
│       ├── install_service.ps1
│       └── README_DEPLOY.md  ← this file
├── venv\                   ← Python virtual environment
├── data\
│   └── trades.db           ← SQLite journal (all trades, signals, ML features)
├── logs\
│   ├── bot_stdout.log      ← main log
│   └── bot_stderr.log      ← errors only
├── .env                    ← YOUR CREDENTIALS (never commit this)
└── start_bot.bat           ← manual start shortcut
```
