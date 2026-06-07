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
    # VBSwing config (eth_vbswing.py best config). The ported BTC VBSwing strategy
    # is the first robust winner on ETH (TRAIN +0.81R / TEST +0.70R, not curve-fit),
    # and its best hour set is the BTC Asia-Night + EU-open kill-zone.
    KZ_HOURS = [1, 2, 3, 8]

# ── Risk & position sizing ─────────────────────────────────────────────────────
# Same ADX-split logic as BTC Bot 2 — validated on crypto in general.
# Revisit after ETH-specific backtest.
STARTING_BALANCE      = 500.0   # USD paper trading account
# ADX-split risk — Config D, confirmed optimal by 6yr ETH ADX sweep:
#   ADX 20-25: weak zone (WR=45%, AvgR=+0.34) → risk LESS
#   ADX 25-40: sweet spot  (WR=47-60%, AvgR=+0.87-1.64) → normal risk
#   ADX ≥ 40:  strong trend (WR=60%, AvgR=+0.62) → risk MORE
# Config D outperformed all alternatives: CAGR +37.9% vs BTC-style +28.7%
# Tier B sizing (realistic_risk_sweep, TP1-fixed one-position sim): the 3/4/6
# profile + deep -25% throttle realised ~$71k from $500 at MaxDD -31.5% / MaxCL 8,
# comfortably inside the hard limits (CL < 16, DD > -42%). Sized up from the old
# 2/3/5 after the TP1 50/50 fix showed spare risk budget (corrected DD only -21.6%).
# VBSwing FINAL DECISION: FLAT 8% base risk (no ADX split). The Monte Carlo over the
# real VBSwing-on-ETH R-distribution (eth_vbswing_final.py) made the 6/8/10 choice on
# the full upside/pain tradeoff: 8% gives ~35% P($10k)/6mo and ~87%/12mo at 0% ruin,
# with a p95 worst-drawdown of ~58% — the chosen survivable middle path. All three
# ADX tiers are set equal so get_risk_pct() returns a flat 8% regardless of ADX.
RISK_PCT_EARLY_TREND  = 0.08    # 8% flat (ADX ≤ 25)
RISK_PCT_TRANSITION   = 0.08    # 8% flat (ADX 25-40)
RISK_PCT_STRONG       = 0.08    # 8% flat (ADX ≥ 40)

ADX_SPLIT_EARLY_MAX   = 25      # ADX ≤ 25  → early trend  → 3% risk (weakest bucket)
ADX_SPLIT_STRONG_MIN  = 40      # ADX ≥ 40  → strong trend → 6% risk (high conviction)
                                 # ADX 25-40 → sweet spot   → 4% risk (best WR/AvgR)

# ── Risk overlays ──────────────────────────────────────────────────────────────
# DISABLED for the VBSwing config. The eth_vbswing_cb.py circuit-breaker test showed
# that "stop after losses" overlays HURT this strategy: its edge is the rare 5R
# winners that arrive right after a losing streak, and any hard pause / monthly halt
# skips exactly those recovery winners. The validated damage-control rule is the
# consecutive-loss THROTTLE-2 below (halve risk after 2 losses, keep every trade).

# October seasonality cut — was an S4 artifact; VBSwing's edge does not have it.
OCT_RISK_FACTOR       = 1.0     # 1.0 = disabled (no October cut)

# Monthly circuit breaker — DISABLED (set False). A month-level hard halt is the
# "hard pause" the CB test rejected; it forfeits the post-streak recovery winners.
MONTHLY_CB_ENABLED    = False
CB_MONTHLY_DD_LIMIT   = -0.10   # unused while MONTHLY_CB_ENABLED is False

# Equity HWM throttle — DISABLED (THROTTLE_FACTOR = 1.0 short-circuits it in
# signal_engine._equity_throttled). Superseded by THROTTLE-2 (consecutive-loss based).
THROTTLE_DD_TRIGGER   = -0.25   # unused while THROTTLE_FACTOR >= 1.0
THROTTLE_FACTOR       = 1.0     # 1.0 = HWM throttle disabled

# ── THROTTLE-2 (consecutive-loss brake — the chosen damage control) ─────────────
# After THROTTLE2_LOSSES losing trades in a row, multiply risk-per-trade by
# THROTTLE2_FACTOR until the next win. Validated by eth_vbswing_final.py: at 8% base
# this softens the worst drawdowns while KEEPING every trade (the recovery winners
# a hard pause would skip). Set THROTTLE2_FACTOR = 1.0 to disable.
THROTTLE2_LOSSES      = 2       # halve risk after 2 consecutive losses
THROTTLE2_FACTOR      = 0.50    # cut risk-per-trade to 50% until the next win

# Per-strategy ADX minimum (overrides global ADX_THRESHOLD for specific strategies):
# rsi_ema at H10 collapses at ADX 20-25 (WR=41.7%, AvgR=+0.070, PF=1.12 — near random).
# Applying ADX≥25 for that path removes 12 junk trades and lifts its quality to WR≥57%.
RSI_EMA_ADX_MIN       = 25      # used by Path C (rsi_ema, hour 10) in eth_combined.py

# ── TP / SL ratios ─────────────────────────────────────────────────────────────
TP1_RR          = 2.0    # TP1 at 2R — partial close (50%), SL to breakeven
TP2_RR          = 5.0    # TP2 at 5R — full close (remaining 50%)
                          # VBSwing standardises TP1=2R / TP2=5R (vb_swing_combined.py).
                          # The +0.79R AvgR / PF 2.11 backtest edge is built on the rare
                          # 5R runners — do NOT cut this to 4R for VBSwing.
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
