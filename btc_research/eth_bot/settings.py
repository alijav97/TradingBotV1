"""
btc_research/eth_bot/settings.py — ETH Bot runtime configuration.

Fully standalone — reads from .env, no dependency on any other bot's settings.

== KILL-ZONE (S4 optimised set) ==
  KZ_HOURS = [2, 5, 6, 7, 10, 14, 15] — S4 config, validated against real trades.

  BASELINE (original 3 slots):
    02 UTC → RSI 50-Cross            (Asia Night,    WR=50.0%, AvgR=+0.786)
    06 UTC → MACD+ADX                (EU Pre-Open,   WR=45.0%, AvgR=+0.660)
    10 UTC → RSI+EMA Stack           (EU Mid-Session,WR=48.1%, AvgR=+0.706)

  EXPANSION + S4 OPTIMISATION:
    05 UTC → EMA Cross 9/21          (Asia Morning,  WR=52.4%, AvgR=+0.787) ★
    07 UTC → RSI 50-Cross            (S4 ADD — biggest CAGR lever, 135.9%→170.5%)
    14 UTC → Keltner → EMA fallback  (NY Pre-Open,   WR=48.3%/44.7%)
    15 UTC → Keltner ONLY            (NY Open, S4 DROPPED macd_adx fallback)

  S4 also applies two risk overlays (see OCT_RISK_FACTOR / CB_MONTHLY_DD_LIMIT):
    - October risk × 0.5  (only month with negative avg return)
    - Monthly circuit breaker: halt new entries once a month is down -10%

  S4 backtest result: CAGR≈+170.5% | 5yr $500→$72,429 | MaxDD=-30.0% | ~5.1 trades/mo
  Override via ETH_KZ_HOURS in .env (comma-separated, e.g. "2,5,6,7,10,14,15").

== STRATEGY ==
  See eth_combined.py for all 5 paths and fallback routing (Paths A–E).

== .env KEYS ==
  ETH_TELEGRAM_BOT_TOKEN  — Telegram bot token (dedicated ETH bot from BotFather)
  ETH_TELEGRAM_CHAT_ID    — Telegram chat/channel ID for ETH alerts
  ETH_KZ_HOURS            — comma-separated UTC hours e.g. "1,2,3,8"
  ETH_API_PORT            — FastAPI port (default 8003)

  Shared with other bots (same Pepperstone MT5 account):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
"""
from __future__ import annotations

import os
from pathlib import Path

# ── .env loader ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[2] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

# ── Telegram — dedicated ETH bot ───────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("ETH_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("ETH_TELEGRAM_CHAT_ID",   "")

# ── MT5 credentials (shared Pepperstone account) ───────────────────────────────
MT5_LOGIN             = int(os.environ.get("MT5_LOGIN",    "0") or "0")
MT5_PASSWORD          = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER            = os.environ.get("MT5_SERVER",   "")
MT5_SERVER_UTC_OFFSET = 3   # Pepperstone server is UTC+3

# ── Symbols ────────────────────────────────────────────────────────────────────
SYMBOL = "ETHUSD"   # Pepperstone MT5 symbol for Ethereum

# ── Kill-zone hours (UTC) ──────────────────────────────────────────────────────
# Set from 6-year ETH backtest phase-2 results (25 strategies, BTC-aligned):
#   02 UTC  → RSI 50-Cross:   WR=50.0%  AvgR=+0.786  PF=2.57  N=46  (Asia Night)
#   06 UTC  → MACD+ADX:       WR=45.0%  AvgR=+0.660  PF=2.65  N=20  (EU Pre-Open)
#   10 UTC  → RSI+EMA:        WR=48.1%  AvgR=+0.706  PF=2.36  N=27  (EU Mid-Session)
# All three passed the OK threshold (WR≥45%, AvgR≥0.40R, PF≥1.20) on BTC-aligned trades.
# Hours 14-16 tested as combined swing_keltner but did not clear the OK threshold
# after the TP2 reduction (5R→4R) in phase-2.
_kz_env = os.environ.get("ETH_KZ_HOURS", "")
if _kz_env:
    KZ_HOURS: list[int] = [int(h.strip()) for h in _kz_env.split(",") if h.strip()]
else:
    KZ_HOURS = [2, 5, 6, 7, 10, 14, 15]   # S4 set (added H07 rsi_50)

# ── Risk & position sizing ─────────────────────────────────────────────────────
# Same ADX-split logic as BTC Bot 2 — validated on crypto in general.
# Revisit after ETH-specific backtest.
STARTING_BALANCE      = 500.0   # USD paper trading account
# ADX-split risk — Config D, confirmed optimal by 6yr ETH ADX sweep:
#   ADX 20-25: weak zone (WR=45%, AvgR=+0.34) → risk LESS
#   ADX 25-40: sweet spot  (WR=47-60%, AvgR=+0.87-1.64) → normal risk
#   ADX ≥ 40:  strong trend (WR=60%, AvgR=+0.62) → risk MORE
# Config D outperformed all alternatives: CAGR +37.9% vs BTC-style +28.7%
RISK_PCT_EARLY_TREND  = 0.02    # 2% — ADX ≤ 25 (early trend, weakest quality zone)
RISK_PCT_TRANSITION   = 0.03    # 3% — ADX 25-40 (sweet spot — best WR/AvgR bucket)
RISK_PCT_STRONG       = 0.05    # 5% — ADX ≥ 40 (strong trend, high conviction)
                                 # 5% on $500 = $25 at risk, scales up with balance

ADX_SPLIT_EARLY_MAX   = 25      # ADX ≤ 25  → early trend  → 2% risk (weakest bucket)
ADX_SPLIT_STRONG_MIN  = 40      # ADX ≥ 40  → strong trend → 5% risk (high conviction)
                                 # ADX 25-40 → sweet spot   → 3% risk (best WR/AvgR)

# ── S4 risk overlays (validated by backtest_optimised.py) ──────────────────────
# October seasonality: October is the only month with a negative average return
# (-3.7%) across the 6-year sample -> halve risk for that month.
OCT_RISK_FACTOR       = 0.5     # multiply risk_pct by this in October (month == 10)
# Monthly circuit breaker: once a calendar month's REALISED return draws down to
# this level, halt all new entries for the rest of that month. Kills the 2026
# 8-loss cluster; MaxDD -33.5% -> -30.0%, MaxCL 8 -> 7.
CB_MONTHLY_DD_LIMIT   = -0.10   # -10% realised month drawdown -> stop new trades

# Per-strategy ADX minimum (overrides global ADX_THRESHOLD for specific strategies):
# rsi_ema at H10 collapses at ADX 20-25 (WR=41.7%, AvgR=+0.070, PF=1.12 — near random).
# Applying ADX≥25 for that path removes 12 junk trades and lifts its quality to WR≥57%.
RSI_EMA_ADX_MIN       = 25      # used by Path C (rsi_ema, hour 10) in eth_combined.py

# ── TP / SL ratios ─────────────────────────────────────────────────────────────
TP1_RR          = 2.0    # TP1 at 2R — partial close (50%), SL to breakeven
TP2_RR          = 4.0    # TP2 at 4R — full close (remaining 50%)
                          # Reduced from 5R: phase-1 backtest showed avg_r ~0.5R
                          # across all strategies → very few trades reached 5R on ETH.
                          # 4R materially increases hit rate while preserving 2:1 TP2/TP1 ratio.
TRAIL_ATR_MULT  = 2.0    # Trailing SL after TP1: peak/trough ± 2×ATR
MAX_HOLD_BARS   = 96     # 96 H1 bars = 4 days — force-close if still open

# ── Signal filters ─────────────────────────────────────────────────────────────
ADX_THRESHOLD   = 20     # Skip trade if ADX < 20 (no clear trend)
ADX_PERIOD      = 14
EMA200_PERIOD   = 200    # EMA200 — only longs above, only shorts below
MIN_CONFLUENCE_SCORE = 0.0   # Placeholder — set after backtest

# ── Strategy parameters ────────────────────────────────────────────────────────
# SwingLevel v2 mode — "both" checks retest first then break (same as BTC Bot 2)
SWING_ENTRY_MODE = "both"
SWING_MAX_SL_ATR = 2.0    # SL cap for break entries (× ATR)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent
DATA_DIR = _BOT_DIR / "data"
DB_PATH  = DATA_DIR / "eth_trades.db"
LOG_DIR  = DATA_DIR / "logs"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("ETH_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("ETH_API_PORT", "8003"))
