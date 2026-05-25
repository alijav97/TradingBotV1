# TradingBotV2 — Build Tracker
**Started:** 2026-05-25  
**Target:** Paper trading live, 6 instruments, fully autonomous

---

## Progress Legend
- `[x]` Done
- `[~]` In progress
- `[ ]` Not started
- `[!]` Blocked / needs attention

---

## MODULE 1 — Data Layer
- [x] `v2/instrument_config.py` — pip values, leverage, session windows, ticker maps
- [x] `v2/connectors/mt5_connector.py` — MT5 live + historical (XAUUSD, GBP/JPY, WTI, NAS100)
- [x] `v2/connectors/binance_connector.py` — Binance Futures live + historical (BTC, ETH)
- [x] `v2/connectors/unified_data.py` — single OHLCV interface regardless of source
- [ ] Integration test: pull 100 candles from each of 6 instruments

## MODULE 2 — Analysis Engine
- [x] `v2/analysis/indicators.py` — 14 indicators (ported from V1, clean)
- [x] `v2/analysis/smart_money.py` — Order blocks, FVGs, liquidity (ported from V1 directly)
- [x] `v2/analysis/candle_patterns.py` — TA-Lib 61 candle pattern wrapper
- [x] `v2/analysis/mtf_analyzer.py` — multi-timeframe D1/H4/H1 alignment
- [x] `v2/analysis/world_sessions.py` — session detection (ported from V1)
- [x] `v2/analysis/session_profiler.py` — per-session stats (ported from V1)
- [ ] `v2/analysis/sr_mapper.py` — auto support/resistance zones (port from V1)
- [ ] `v2/analysis/liquidity_map.py` — swing clusters, POC (port from V1)
- [ ] Integration test: run full analysis on XAUUSD H1 candles

## MODULE 3 — Signal Engine
- [x] `v2/signals/strategy_registry.py` — 15 strategies with full metadata
- [x] `v2/signals/confluence_engine.py` — rebuilt clean (12 factors, no silent failures)
- [x] `v2/signals/entry_checklist.py` — 5-gate checklist (ported + fixed)
- [ ] `v2/signals/signal_ranker.py` — composite score: WR + PF + Sharpe + ML
- [ ] Integration test: run signal scan on 1 instrument end-to-end

## MODULE 4 — External Intelligence
- [x] `v2/intelligence/news_filter.py` — Forex Factory parser (ported from V1)
- [ ] `v2/intelligence/news_monitor.py` — rewritten (async, circuit breaker, no Claude dep)
- [ ] `v2/intelligence/tweet_monitor.py` — nitter RSS + Truth Social (NEW)
- [ ] `v2/intelligence/dxy_correlation.py` — port from V1
- [ ] `v2/intelligence/geo_filter.py` — port from V1
- [ ] `v2/intelligence/cot_analyzer.py` — port from V1

## MODULE 5 — Risk Engine
- [x] `v2/risk/position_sizer.py` — account-based sizing + TP price calculator
- [x] `v2/risk/atr_sl_engine.py` — dynamic ATR stops (ported + cleaned from V1)
- [x] `v2/risk/trade_manager.py` — partial TP + trailing SL + RoR (ported + fixed)
- [x] `v2/risk/portfolio_heat.py` — cross-instrument risk cap + correlation check (NEW)
- [x] `v2/risk/loss_limits.py` — daily/weekly drawdown limits (NEW)
- [ ] Integration test: size a trade on each instrument

## MODULE 6 — Paper Trading Engine
- [x] `v2/trading/paper_trader.py` — autonomous paper trading (SQLite-backed, full lifecycle)
- [ ] `v2/trading/trade_monitor.py` — standalone 60s checker (currently inside paper_trader)
- [ ] `v2/trading/auto_trader.py` — rebuilt from V1 (reviewed, fixed) [Week 2]
- [ ] Integration test: open → monitor → close a paper trade on BTC

## MODULE 7 — Trade Journal
- [x] `v2/journal/sqlite_journal.py` — full SQLite schema + write/read helpers + ML features
- [ ] Integration test: write 5 trades, query by instrument and date

## MODULE 8 — ML Layer
- [ ] `v2/ml/feature_engineer.py` — 40+ features per trade from journal
- [ ] `v2/ml/lightgbm_trainer.py` — LightGBM model (train when 50+ trades)
- [ ] `v2/ml/hmm_regime.py` — HMM regime detection
- [ ] `v2/ml/ml_engine.py` — rebuilt from V1 (fixed bare excepts + global state)
- [ ] Integration test: train on dummy data, get prediction

## MODULE 9 — API + Alerts
- [ ] `v2/api/api_server.py` — FastAPI with auth
- [ ] `v2/api/api_keys.py` — key management
- [ ] `v2/api/telegram_bot.py` — trade alerts
- [ ] Integration test: hit /health and /signals endpoints

## MODULE 10 — Scheduler
- [x] `v2/scheduler/scheduler.py` — APScheduler: 1min + 1H + 4H + daily + nightly jobs
- [ ] Integration test: verify all jobs fire on schedule
- [ ] Wire Telegram alerts into scheduler jobs

## Infrastructure
- [x] `v2/requirements.txt` — all dependencies
- [x] `v2/settings.py` — config (account size, risk %, limits, all env vars)
- [ ] `v2/docker/Dockerfile` — production container
- [ ] `v2/docker/docker-compose.yml` — with volume mounts
- [ ] `v2/tests/` — per-module unit tests
- [ ] End-to-end: all 6 instruments, full signal → paper trade → journal cycle

---

## Build Sessions Log

### Session 1 (2026-05-25) — FOUNDATION COMPLETE
**Built (37 files):**
- Project blueprint `PROJECT_2_TBR.md` + V1 code audit
- Complete V2 directory structure (10 modules)
- `settings.py` — all env vars + config
- `instrument_config.py` — 6 instruments fully configured
- `connectors/mt5_connector.py` — MT5 (stub mode on non-Windows)
- `connectors/binance_connector.py` — Binance Futures
- `connectors/unified_data.py` — single OHLCV interface
- `analysis/indicators.py` — 14 indicators (clean V1 port)
- `analysis/smart_money.py` — SMC (direct V1 port, verified clean)
- `analysis/candle_patterns.py` — TA-Lib 61-pattern wrapper
- `analysis/mtf_analyzer.py` — D1/H4/H1 alignment analyzer
- `analysis/world_sessions.py` + `session_profiler.py` — V1 ports
- `signals/strategy_registry.py` — 15 strategies with metadata
- `signals/confluence_engine.py` — REBUILT clean (not V1 copy)
- `signals/entry_checklist.py` — 5-gate validation (V1 port + fixed)
- `risk/position_sizer.py` — account-based lot calculator
- `risk/atr_sl_engine.py` — dynamic ATR SL (V1 port, V1 deps removed)
- `risk/trade_manager.py` — partial TP + trailing SL (V1 port + fixed)
- `risk/portfolio_heat.py` — cross-instrument risk + correlation check
- `risk/loss_limits.py` — daily/weekly drawdown limits
- `trading/paper_trader.py` — full autonomous lifecycle (SQLite)
- `journal/sqlite_journal.py` — SQLite schema (5 tables)
- `intelligence/news_filter.py` — Forex Factory calendar (V1 port)
- `scheduler/scheduler.py` — APScheduler 5 jobs

**TODO next session:**
- `analysis/sr_mapper.py` — port from V1 sr_mapper.py
- `analysis/liquidity_map.py` — port from V1 liquidity_map.py
- `signals/signal_ranker.py` — composite score ranker
- `intelligence/dxy_correlation.py` — port from V1
- `intelligence/geo_filter.py` — port from V1
- `intelligence/cot_analyzer.py` — port from V1
- `intelligence/news_monitor.py` — REWRITE (async + circuit breaker)
- `intelligence/tweet_monitor.py` — NEW
- Integration tests for all modules

**Week 2 session:**
- `ml/feature_engineer.py`
- `ml/lightgbm_trainer.py`
- `ml/hmm_regime.py`
- `ml/ml_engine.py`
- `api/api_server.py`
- `api/api_keys.py`
- `api/telegram_bot.py`
- Docker setup
- End-to-end test: all 6 instruments live

---

## Known Issues / Notes
- MT5 connector requires MetaTrader5 installed on Windows host — won't connect in cloud
- Binance connector uses python-binance SDK
- TA-Lib requires C library install: `apt-get install libta-lib-dev` or use `TA-Lib-precompiled`
- FinBERT model (~400MB) must be downloaded on first run: `transformers` downloads to ~/.cache
- `confluence_engine.py` was completely rewritten — does NOT copy V1 directly
- `news_monitor.py` V1 had Claude API dep — removed in V2, uses FinBERT locally
- All paper trades go to SQLite, not JSON files

---

## Instruments Status
| Instrument | Connector  | Analysis | Signals | Paper Trade |
|------------|-----------|----------|---------|-------------|
| XAUUSD     | MT5       | [ ]      | [ ]     | [ ]         |
| GBP/JPY    | MT5       | [ ]      | [ ]     | [ ]         |
| WTI        | MT5       | [ ]      | [ ]     | [ ]         |
| NAS100     | MT5       | [ ]      | [ ]     | [ ]         |
| BTC/USDT   | Binance   | [ ]      | [ ]     | [ ]         |
| ETH/USDT   | Binance   | [ ]      | [ ]     | [ ]         |
