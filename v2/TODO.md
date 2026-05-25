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
- [x] `v2/analysis/sr_mapper.py` — instrument-agnostic S/R (V1 port, pct-based tolerances)
- [x] `v2/analysis/liquidity_map.py` — adaptive bucket POC (V1 port, pct-based)
- [ ] Integration test: run full analysis on XAUUSD H1 candles

## MODULE 3 — Signal Engine
- [x] `v2/signals/strategy_registry.py` — 15 strategies with full metadata
- [x] `v2/signals/confluence_engine.py` — rebuilt clean (12 factors, no silent failures)
- [x] `v2/signals/entry_checklist.py` — 5-gate checklist (ported + fixed)
- [x] `v2/signals/signal_ranker.py` — composite scorer (confluence×0.4 + WR×0.3 + ML×0.2 + RR×0.1)
- [ ] Integration test: run signal scan on 1 instrument end-to-end

## MODULE 4 — External Intelligence
- [x] `v2/intelligence/news_filter.py` — Forex Factory parser (ported from V1)
- [x] `v2/intelligence/news_monitor.py` — rewritten (ThreadPoolExecutor, circuit breaker, FinBERT)
- [x] `v2/intelligence/tweet_monitor.py` — nitter RSS, 3-instance fallback, keyword scoring
- [x] `v2/intelligence/dxy_correlation.py` — 3-stage DXY fetch + per-instrument correlation
- [x] `v2/intelligence/geo_filter.py` — 5-level risk scoring + SL multiplier
- [x] `v2/intelligence/cot_analyzer.py` — CFTC COT with 24h disk cache + fallback

## MODULE 5 — Risk Engine
- [x] `v2/risk/position_sizer.py` — account-based sizing + TP price calculator
- [x] `v2/risk/atr_sl_engine.py` — dynamic ATR stops (ported + cleaned from V1)
- [x] `v2/risk/trade_manager.py` — partial TP + trailing SL + RoR (ported + fixed)
- [x] `v2/risk/portfolio_heat.py` — cross-instrument risk cap + correlation check (NEW)
- [x] `v2/risk/loss_limits.py` — daily/weekly drawdown limits (NEW)
- [ ] Integration test: size a trade on each instrument

## MODULE 6 — Paper Trading Engine
- [x] `v2/trading/paper_trader.py` — autonomous paper trading (SQLite-backed, full lifecycle)
- [x] `v2/trading/trade_monitor.py` — standalone checker with heat + summary logging
- [x] `v2/trading/auto_trader.py` — rebuilt clean (no threading, no global state, no race conditions)
- [ ] Integration test: open → monitor → close a paper trade on BTC

## MODULE 7 — Trade Journal
- [x] `v2/journal/sqlite_journal.py` — full SQLite schema + write/read helpers + ML features
- [ ] Integration test: write 5 trades, query by instrument and date

## MODULE 8 — ML Layer
- [x] `v2/ml/feature_engineer.py` — 40 features (time, trade, price action, context, historical)
- [x] `v2/ml/lightgbm_trainer.py` — LightGBM, chronological split, balanced classes
- [x] `v2/ml/hmm_regime.py` — 4-state GaussianHMM + rule-based fallback
- [x] `v2/ml/ml_engine.py` — orchestrator (retrain gate, confidence, regime) rebuilt clean
- [ ] Integration test: train on dummy data, get prediction

## MODULE 9 — API + Alerts
- [x] `v2/api/api_server.py` — FastAPI, 8 endpoints, lifespan wiring
- [x] `v2/api/api_keys.py` — read/full scope keys, SQLite-backed
- [x] `v2/api/telegram_bot.py` — outbound alerts (signal/trade opened/closed/briefing)
- [ ] Integration test: hit /health and /signals endpoints

## MODULE 10 — Scheduler
- [x] `v2/scheduler/scheduler.py` — APScheduler: 1min + 1H + 4H + daily + nightly jobs + Telegram
- [ ] Integration test: verify all jobs fire on schedule

## Infrastructure
- [x] `v2/requirements.txt` — all dependencies
- [x] `v2/settings.py` — config (account size, risk %, limits, all env vars)
- [x] `v2/docker/Dockerfile` — multi-stage build, TA-Lib included, MT5 excluded
- [x] `v2/docker/docker-compose.yml` — with persistent data volume + log rotation
- [x] `v2/docker/.env.example` — all required env vars documented
- [x] `v2/main.py` — full entry point, all 10 modules wired, SIGINT/SIGTERM clean shutdown
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

**Session 2 (2026-05-25) — ALL MODULES COMPLETE**
Built remaining 20+ files across all 10 modules using 6 parallel agents.
53 Python files total. Bot is feature-complete.

**Remaining (integration + deployment):**
- [ ] Set up `.env` file with real credentials
- [ ] Install TA-Lib C library on VPS: `apt-get install libta-lib-dev`
- [ ] `pip install -r v2/requirements.txt`
- [ ] Configure MT5 broker credentials in `.env`
- [ ] Configure Binance API keys (testnet first) in `.env`
- [ ] Set Telegram bot token + chat ID in `.env`
- [ ] Run `python v2/main.py` — bot starts, paper trades begin
- [ ] Monitor first paper trades via Telegram
- [ ] After 50 paper trades: ML layer activates automatically

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
