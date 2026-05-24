HOW TO USE YOUR TRADING BOT
════════════════════════════════════════════════════════

FIRST TIME ONLY (run once, takes 10-20 minutes):
─────────────────────────────────────────────────
  python setup.py

  This will:
    1. Ingest all your PDF/DOCX/image/URL trading rules into data/rules.json
    2. Download 2 years of XAUUSD (Gold) hourly history
    3. Backtest all rules and save results to data/backtest_results.json
    4. Analyse historical patterns and save to data/historical_patterns.json

  Only needs to run ONCE. After that, daily briefings are fast.


EVERY TRADING SESSION (runs in under 60 seconds):
─────────────────────────────────────────────────
  python morning_briefing.py

  Shows you:
    • Today's gold sentiment and news summary
    • Live price, RSI, ATR, trend direction
    • Historical pattern bias (how gold moved after similar setups)
    • Top 3 trade setups with entry / stop-loss / take-profit
    • Your weekly stats (win rate, pips, best pattern)
    • Whether it is safe to trade (events check)


WEEKLY REFRESH (run every weekend — takes 3-5 minutes):
────────────────────────────────────────────────────────
  python setup.py --refresh

  Re-downloads recent price history, re-runs backtest, updates patterns.
  Does NOT re-ingest your resources (skips step 1).


ADD NEW TRADING RESOURCES:
───────────────────────────
  1. Drop your PDF / DOCX / image / URL files into the  resources/  folder.
  2. Run:
       python setup.py --ingest-only

  This reads the new files, extracts rules, and updates data/rules.json.


LOGGING A TRADE (optional):
────────────────────────────
  python learning.py

  Records your trade entry so the bot can track live performance and
  improve pattern confidence over time.


FILES OVERVIEW:
───────────────
  setup.py           — First-time setup and weekly refresh
  morning_briefing.py — Fast daily briefing (< 60 s)
  learning.py        — Log and close trades, performance report
  ingest.py          — PDF/DOCX/URL rule ingestion engine
  backtest.py        — Historical backtesting engine
  pattern_matcher.py — Historical similarity scanner
  news_monitor.py    — Parallel news fetcher + sentiment
  news_filter.py     — Today's economic calendar events
  _progress.py       — Shared progress bar utilities

  data/rules.json              — Extracted trading rules
  data/backtest_results.json   — Backtest results per rule
  data/historical_patterns.json — Historical pattern analysis
  data/historical_xauusd.csv   — Downloaded price history
  data/trade_log.json          — Your personal trade journal


EXPECTED RUN TIMES:
────────────────────
  python setup.py             → ~10-20 minutes (first time)
  python setup.py --refresh   → ~3-5 minutes
  python morning_briefing.py  → under 60 seconds


QUICK REFERENCE:
─────────────────
  Daily  →  python morning_briefing.py
  Setup  →  python setup.py
  Refresh→  python setup.py --refresh
  Ingest →  python setup.py --ingest-only
  Trades →  python learning.py


════════════════════════════════════════════════════════
Good luck with your trading. Stay disciplined.
════════════════════════════════════════════════════════
