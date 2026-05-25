# Project 2 — TradingBotV2 Blueprint (TBR)
**Status:** Planning — build starts when back at laptop  
**Date drafted:** 2026-05-25  
**Instruments:** XAUUSD, BTC/USDT, ETH/USDT, GBP/JPY, WTI, NAS100

---

## Overview

Full rebuild of the trading bot with:
- 6 instruments (forex, crypto, commodities, indices)
- Dual data source: MT5 (forex/gold/oil/indices) + Binance Futures (BTC/ETH)
- Fully autonomous paper trading (no human needed to approve trades)
- ML layer that learns from its own paper trade outcomes
- News + tweet sentiment filtering
- FastAPI with API key auth
- Telegram alerts
- VPS-deployable via Docker

---

## Resources Required (All Free)

```
PRICE DATA      MT5 (forex/gold/oil/indices) + Binance API (crypto)
                Historical + live candles — both free

ML MODEL        FinBERT (HuggingFace, download once, runs locally)
                No API cost, no subscription

CANDLE PATTERNS TA-Lib (61 built-in pattern detectors — Hammer, Doji,
                Engulfing, Morning Star, Pinbar, etc.)
                Replaces hand-coding every pattern

STRATEGY PDFs   Drop any trading book/strategy PDF into resources/
                ingest.py extracts rules automatically (already works in V1)

BOT LEARNING    Bot learns from its own paper trade outcomes
                More trades = smarter over time
                No external training dataset needed to start
```

---

## Architecture — 10 Modules

### Module 1: Data Layer
```
MT5Connector     → XAUUSD, GBP/JPY, WTI, NAS100 (live + historical)
BinanceConnector → BTC/USDT, ETH/USDT futures (live + historical)
UnifiedOHLCV     → Same data format regardless of source
HistoricalLoader → Bulk download on setup (first run only)
```

### Module 2: Analysis Engine
```
Indicators       → 14 technical indicators              [FROM V1 — indicators.py]
CandlePatterns   → TA-Lib 61 built-in patterns          [NEW — replaces manual coding]
SmartMoney       → Order blocks, FVGs, liquidity sweeps [FROM V1 — smart_money.py]
MTFAnalyzer      → D1/H4/H1 timeframe alignment         [FROM V1 — mtf_analyzer.py]
SRMapper         → Auto support/resistance zones        [FROM V1 — sr_mapper.py]
LiquidityMap     → Swing clusters, point of control     [FROM V1 — liquidity_map.py]
```

### Module 3: Signal Engine
```
StrategyRegistry → All 47 strategies (new backtest) + 12 playbooks (V1)
ConfluenceScorer → 12-factor scoring, min 3 required    [FROM V1 — needs review]
EntryChecklist   → 5-gate validation before signal fires [FROM V1 — entry_checklist.py]
SignalRanker     → Composite: WR + PF + Sharpe + ML confidence
```

### Module 4: External Intelligence
```
EconomicCalendar → Forex Factory parser                 [FROM V1 — news_filter.py]
NewsSentiment    → RSS headlines + FinBERT classifier   [FROM V1 — needs review]
TweetMonitor     → nitter RSS + Truth Social scraper    [NEW]
DXYCorrelation   → USD basket directional filter        [FROM V1 — dxy_correlation.py]
COTAnalyzer      → CFTC institutional positioning       [FROM V1 — cot_analyzer.py]
GeoFilter        → Conflict/geopolitical risk scoring   [FROM V1 — geo_filter.py]
```

### Module 5: Risk Engine
```
PositionSizer    → Account-based, per instrument pip value [FROM V1 — trade_manager.py]
DynamicSL        → ATR-based context-aware stops           [FROM V1 — atr_sl_engine.py]
PartialTPManager → 50% at 1:2 RR, trail rest to 1:3       [FROM V1 — trade_manager.py]
PortfolioHeat    → Max 30% account at risk across all open trades [NEW]
DailyLossLimit   → Auto-pause scanning if daily drawdown hit [NEW]
CorrelationCheck → Block 3+ correlated longs simultaneously [NEW]
```

### Module 6: Paper Trading Engine
```
AutoTrader       → Signal → paper order, fully automatic   [FROM V1 — needs review]
TradeMonitor     → Check SL/TP/max hold every 60 seconds
TradeLifecycle   → PENDING → OPEN → MONITORING → CLOSED
PnLCalculator    → Per instrument (pip values differ by instrument)
```

### Module 7: Trade Journal (SQLite)
```
Tables:
  trades          → every paper trade, full context snapshot
  signals         → every signal generated (taken AND skipped)
  news_events     → news sentiment score at time of each signal
  performance     → daily/weekly/monthly aggregate summaries
  ml_features     → engineered feature vectors per trade (for ML training)
```

### Module 8: ML Layer
```
FeatureEngineer  → 40+ features per trade extracted from journal
ModelTrainer     → LightGBM (retrain every 50 new paper trades)
StrategyRanker   → ML win probability replaces static composite score
RegimeDetector   → HMM: trending/ranging/volatile/spike classification
WinRatePredictor → Per strategy × per regime × per session slot
```

### Module 9: API + Alerts
```
FastAPI Server   → REST API, API key authentication
Endpoints:
  GET  /signals          → latest signals with scores
  GET  /trades           → open and closed paper trades
  GET  /performance      → P&L by instrument/strategy/period
  GET  /health           → system status
  POST /settings         → adjust risk params (write key required)
TelegramBot      → Instant trade alerts to phone
APIKeyManager    → read-only keys vs full-access keys
```

### Module 10: Scheduler
```
Every 1 min    → Monitor open paper trades (SL/TP/max hold check)
Every 1H       → Full signal scan across all 6 instruments
Every 4H       → 4H timeframe deep scan
Daily 6am      → Morning briefing + load economic calendar for the day
Daily 11pm     → ML retrain if 50+ new paper trades since last run
Weekly Sunday  → Full backtest refresh, update strategy weights
```

---

## Instrument Config

```
Instrument    Data Source   Pip Value      Leverage    Best Session
─────────────────────────────────────────────────────────────────
XAUUSD        MT5           $0.10 / pip    10x         London + NY
GBP/JPY       MT5           ¥0.01 / pip    10x         London open
WTI (Oil)     MT5           $0.01 / pip    10x         NY open
NAS100        MT5           $0.25 / pip    5x          NY open
BTC/USDT      Binance Fut   $1 / unit      3x          24/7
ETH/USDT      Binance Fut   $0.10 / unit   3x          24/7
```

---

## V1 Code Audit — What to Use vs What to Rewrite

> Rule: Don't blindly copy. Read each file first, understand what it does,
> then decide. If it's clean → plug in. If it has issues → fix or rewrite.

### PLUG-AND-PLAY (use directly after reading)
```
indicators.py        611L   14 clean indicators, type hints, NaN handling
smart_money.py      1031L   Order blocks/FVG/liquidity, well-structured
entry_checklist.py  1006L   5-gate checklist, clear constants, robust
paper_trader.py      433L   JSON trade journal, UUID tracking, no heavy deps
instrument_data.py   163L   Lean yfinance wrapper with ticker mapping
data_manager.py       67L   Minimal storage layer, clean API
trade_manager.py     566L   Partial TP, trailing SL, RoR calc — solid math
news_filter.py       227L   Forex Factory parser, 429 handling, rate limiting
sr_mapper.py         ~450L  Support/resistance auto mapper
liquidity_map.py     ~350L  Swing cluster + POC finder
walk_forward.py      ~350L  Walk-forward backtest validation
debug_logger.py      ~600L  Logging utility — check before reusing
world_sessions.py    ~250L  Session time detection (London/NY/Asia)
session_profiler.py  ~380L  Per-session performance stats
cot_analyzer.py      ~280L  CFTC COT data parser
geo_filter.py        ~280L  Geopolitical risk scoring
dxy_correlation.py   ~650L  DXY correlation filter
atr_sl_engine.py     ~290L  ATR-based dynamic stop loss
```

### NEEDS REVIEW (read carefully, test, fix before using)
```
confluence_engine.py  1950L  ISSUE: tight coupling, 8+ optional imports with
                              silent failures; NEEDS: dependency injection,
                              modularization, explicit error logging

ml_engine.py           893L  ISSUE: bare except blocks everywhere, mutable
                              global state (_FF_CACHE) not thread-safe;
                              NEEDS: specific exception types, class-based
                              state, structured logging

auto_trader.py        1122L  ISSUE: complex multithreaded state machine,
                              edge cases untested (signal race conditions,
                              loop escape paths);
                              NEEDS: integration testing before trusting

news_monitor.py        580L  ISSUE: no rate-limit backoff between RSS fetches,
                              one feed timeout blocks others, Claude API dep;
                              NEEDS: async with timeout per feed, circuit
                              breaker, remove Claude API dep for V2
```

### DO NOT USE (check once, then write fresh)
```
bot_chat.py          319kb  Streamlit chat UI — not needed in V2 (API-first)
morning_briefing.py  125kb  Monolithic, Streamlit-specific, too large
mt5_sync.py           44kb  MT5 sync but unclear if it handles Binance;
                             write fresh unified connector for both brokers
backtest.py           57kb  Single-instrument XAUUSD only; V2 needs
                             multi-instrument backtest engine from scratch
auto_trader.py              (also listed above — rewrite after reviewing arch)
```

### NEW FILES NEEDED (don't exist in V1)
```
binance_connector.py   Binance Futures live + historical candles
mt5_connector.py       Clean MT5 wrapper (separate from mt5_sync.py)
unified_data.py        Single OHLCV interface (MT5 or Binance → same format)
candle_patterns.py     TA-Lib wrapper for 61 candlestick patterns
portfolio_heat.py      Cross-instrument risk monitor
tweet_monitor.py       nitter RSS + Truth Social feed parser
scheduler.py           APScheduler setup for all recurring tasks
api_server.py          FastAPI app with auth, all endpoints
api_keys.py            API key management (generate, revoke, scope)
telegram_bot.py        Trade alert bot (signal fired, trade opened/closed)
hmm_regime.py          Hidden Markov Model market regime detector
sqlite_journal.py      SQLite schema + trade logging (replace JSON files)
lightgbm_trainer.py    LightGBM model training + prediction
feature_engineer.py    40+ feature extraction from trade journal
docker/                Dockerfile + docker-compose for VPS deploy
```

---

## Build Sequence — Paper Trading Live by End of Week 1

### Day 1 (Monday)
- Project directory structure
- requirements.txt (all dependencies listed)
- MT5 connector (XAUUSD, GBP/JPY, WTI, NAS100)
- Binance connector (BTC, ETH futures)
- Unified OHLCV interface
- SQLite schema (tables: trades, signals, performance, ml_features)

### Day 2 (Tuesday)
- Port indicators.py (after reading → plug in)
- Port smart_money.py (after reading → plug in)
- Port mtf_analyzer.py + sr_mapper.py
- TA-Lib candle pattern layer
- Strategy registry (start with 10 key strategies, expand later)

### Day 3 (Wednesday)
- Confluence scorer (review confluence_engine.py, fix tight coupling)
- Entry checklist (port entry_checklist.py after reading)
- Risk engine: position sizer per instrument
- Dynamic SL + partial TP (port trade_manager.py after reading)

### Day 4 (Thursday)
- Paper trading engine (full lifecycle: OPEN → MONITOR → CLOSE)
- Portfolio heat monitor
- Daily loss limit
- Trade journal writes to SQLite

### Day 5 (Friday)
- Scheduler (1min trade monitor + 1H signal scan)
- Economic calendar filter (port news_filter.py)
- Telegram alerts ("SIGNAL: BTC LONG @ 77,400 | Score: 8.2/12")
- Full integration test all 6 instruments
- First autonomous paper trades open

### Weekend
- Bot runs unattended
- Collects first batch of paper trades
- Monitor via Telegram

### Week 2
- News sentiment layer (rewrite news_monitor.py, add FinBERT)
- Tweet monitor
- FastAPI server + API keys
- Basic ML layer (first LightGBM model on accumulated trades)

### Week 3+
- HMM regime detector
- ML confidence integrated into signal ranker
- VPS Docker deployment
- Performance dashboard

---

## Key Decisions

### Scan Frequency
```
Week 1-2:  Time-based (every 60 min) — simpler, get it running fast
Week 3+:   Event-driven (candle close triggers scan) — production quality
```

### ML Activation
```
ML starts training automatically once 50 paper trades are in the journal.
Before that threshold: static composite score ranks strategies.
After threshold: ML confidence score replaces/augments static score.
```

### Data Storage
```
V1 used JSON files per instrument.
V2 uses SQLite for everything — queryable, faster, single file.
Backups: nightly copy of .db file to /backups/ folder.
```

### API Auth
```
Read-only key   → GET /signals, /trades, /performance, /health
Full-access key → All of above + POST /settings (risk param changes)
No public endpoints — every route requires Authorization header.
```

---

## Directory Structure (V2)

```
tradingbotv2/
│
├── connectors/
│   ├── mt5_connector.py
│   ├── binance_connector.py
│   └── unified_data.py
│
├── analysis/
│   ├── indicators.py          ← from V1
│   ├── smart_money.py         ← from V1
│   ├── mtf_analyzer.py        ← from V1
│   ├── sr_mapper.py           ← from V1
│   ├── liquidity_map.py       ← from V1
│   └── candle_patterns.py     ← NEW (TA-Lib wrapper)
│
├── signals/
│   ├── confluence_engine.py   ← V1 reviewed + fixed
│   ├── entry_checklist.py     ← from V1
│   ├── strategy_registry.py   ← NEW (unified strategy list)
│   └── signal_ranker.py       ← NEW
│
├── intelligence/
│   ├── news_filter.py         ← from V1
│   ├── news_monitor.py        ← V1 rewritten
│   ├── tweet_monitor.py       ← NEW
│   ├── dxy_correlation.py     ← from V1
│   ├── cot_analyzer.py        ← from V1
│   └── geo_filter.py          ← from V1
│
├── risk/
│   ├── position_sizer.py      ← from V1 trade_manager.py
│   ├── atr_sl_engine.py       ← from V1
│   ├── portfolio_heat.py      ← NEW
│   └── loss_limits.py         ← NEW
│
├── trading/
│   ├── paper_trader.py        ← V1 upgraded (SQLite, not JSON)
│   ├── trade_manager.py       ← from V1
│   └── auto_trader.py         ← V1 reviewed + fixed
│
├── journal/
│   └── sqlite_journal.py      ← NEW
│
├── ml/
│   ├── feature_engineer.py    ← NEW
│   ├── lightgbm_trainer.py    ← NEW
│   ├── hmm_regime.py          ← NEW
│   └── ml_engine.py           ← V1 reviewed + fixed
│
├── api/
│   ├── api_server.py          ← NEW (FastAPI)
│   ├── api_keys.py            ← NEW
│   └── telegram_bot.py        ← NEW
│
├── scheduler.py               ← NEW (APScheduler)
├── settings.py                ← from V1 + extended
├── instrument_config.py       ← NEW (pip values, leverage, sessions)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── resources/                 ← strategy PDFs (from V1)
```

---

## Notes for When You Start

1. **Start with connectors** — everything depends on getting clean OHLCV data first
2. **Test each V1 port in isolation** before wiring it into the main flow
3. **SQLite first** — get the journal schema right before wiring trading engine to it
4. **Telegram early** — set it up on Day 4 so you can see what the bot is doing from phone
5. **Don't activate all 47 strategies on Day 1** — start with top 5 by win rate, expand once stable
6. **confluence_engine.py needs the most attention** — it's the brain of the signal flow, the V1 version has silent failure risks

---

*Saved: 2026-05-25 | Continue in next session*
