"""
btc_research/btc_bot_2/settings.py — BTC Bot 2 FINAL configuration.

Session  : Asia Night + EU Open  01:00, 02:00, 03:00, 08:00 UTC
           (05:00, 06:00, 07:00, 12:00 UAE)
Strategy : Volatility Breakout + Swing Level Break v2 (entry_mode="both", max_sl_atr=2.0)
           (no Morning Range — pre-session range is Bot 1's active KZ, not a consolidation)

== BACKTEST RESULTS (2yr, 2024-05-30 → 2026-05-30, $500 start) ==
  --- v1 baseline (SL = prior swing structure, avg 4.42×ATR) ---
  Trades    : 176
  Win Rate  : 50.6%
  Avg R     : +0.89R
  PnL       : +$20,354
  Max DD    : 13.1%

  --- v2 FINAL — Mode "both" 2×ATR + SWING-FIRST priority (CHOSEN CONFIG) ---
  Trades    : 270  (break + retest + VB fallback)
  Win Rate  : 46.7%
  PnL       : +$119,329  (6× improvement over v1)
  Max DD    : 17.5%
  Prof halves : 5/5 (100%) | Quarters : 9/9 (100%) | Months : 24/24 (100%)
  Max consec losing months : 0  |  PF : 2.50
  Priority  : SwingLevelBreakV2 FIRST, VB fallback (vs VB-first: +33.5% more PnL)

== WHY THESE HOURS ==
  Single-hour scan (2yr, EMA200+ADX20, VB+SL):
    01:00 UTC → 50.0% WR  +0.92R  MaxDD 7.3%   ← BEST single hour (hidden gem)
    03:00 UTC → 45.3% WR  +0.72R  MaxDD 12.4%  ← GOOD
    02:00 UTC → 42.7% WR  +0.67R  MaxDD 15.4%  ← MARGINAL but needed for trade count
    08:00 UTC → 45.6% WR  +0.62R  MaxDD 9.6%   ← EU open overlap, clean trends

  01:00 UTC is Asia Night momentum (BTC institutional accumulation/distribution).
  08:00 UTC is early EU session open — fresh trend with clean structure.
  Hours 04, 05, 06, 07 are marginal or negative — excluded.

== WHY NO MORNING RANGE ==
  Morning Range fires on pre-KZ consolidation breaks. At 01-03 UTC the
  "pre-KZ range" is the Bot 1 session (21-24 UTC) — an active trending
  session, not a quiet consolidation. Morning Range logic breaks here.
  VB and Swing Level both show their BEST sessions in Asia Night.

== ADX-SPLIT RISK SIZING ==
  ADX 20-25 (early trend):  3% — market just starting to move, entries align
  ADX 25-40 (transition):   2% — dead zone in Asia Night, size conservatively
  ADX 40+   (strong trend): 3% — powerful move, ride it with more size
  This outperforms both flipped and normal risk for this session.

== FILTERS ==
  EMA200: Yes — directional alignment is essential
  H4 EMA20: NO — hurts, removes the best 01:00 UTC entries
  D1 EMA96: Optional — marginal +$493 improvement
  ADX >= 20: Yes
"""
from __future__ import annotations
import os
from pathlib import Path

# ── .env loader ───────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[2] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

# ── MT5 credentials (shared Pepperstone account — same as Bot 1) ──────────────
MT5_LOGIN    = int(os.environ.get("MT5_ACCOUNT", os.environ.get("MT5_LOGIN", "0")) or "0")
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "")

# Pepperstone server time is UTC+3.
# This offset is subtracted from MT5 bar timestamps to get TRUE UTC.
# Critical for kill-zone hour alignment — do not change without re-testing.
MT5_SERVER_UTC_OFFSET = 3

# ── Kill-zone hours (NOT a contiguous range — use a list) ─────────────────────
# Hours 1, 2, 3 = Asia Night  |  Hour 8 = EU session open
KZ_HOURS     = [1, 2, 3, 8]    # UTC hours to scan  (05,06,07,12 UAE)
# Legacy range fields kept for compatibility with any range-checking code
KZ_START_UTC = 1
KZ_END_UTC   = 4    # covers 1,2,3 — hour 8 handled separately via KZ_HOURS

# ── Strategy ──────────────────────────────────────────────────────────────────
VB_ATR_MULTIPLIER = 1.2
VB_CLOSE_ZONE     = 0.45

# Swing Level Break v2 — Mode 6 "both" 2×ATR  (FINAL CHOSEN CONFIG)
# "both" = retest entry (tight SL ~0.6×ATR) preferred; falls back to ATR-capped break entry
# max_sl_atr = 2.0 caps the break-mode SL so it never exceeds 2×ATR from entry
SWING_ENTRY_MODE = "both"   # "break"|"break_capped"|"retest"|"retest_preferred"|"both"
SWING_MAX_SL_ATR = 2.0      # SL cap (× ATR) for break entries

# ── Risk & sizing (VALIDATED by backtest) ─────────────────────────────────────
STARTING_BALANCE     = 500.0
RISK_PCT             = 0.02    # 2% base (used as flat fallback)
RISK_PCT_EARLY_TREND = 0.03    # 3% — ADX 20-25 early trend zone
RISK_PCT_STRONG      = 0.03    # 3% — ADX 40+ strong trend zone
RISK_PCT_TRANSITION  = 0.02    # 2% — ADX 25-40 dead zone (conservative)

# ADX-split thresholds
ADX_SPLIT_EARLY_MAX  = 25      # ADX <= this → RISK_PCT_EARLY_TREND (3%)
ADX_SPLIT_STRONG_MIN = 40      # ADX >= this → RISK_PCT_STRONG (3%)
# Between 25-40 → RISK_PCT_TRANSITION (2%)

TP1_RR           = 2.0
TP2_RR           = 5.0
TRAIL_ATR_MULT   = 2.0
MAX_HOLD_BARS    = 96

# ── Filters ───────────────────────────────────────────────────────────────────
EMA200_PERIOD    = 200
ADX_PERIOD       = 14
ADX_THRESHOLD    = 20          # skip if ADX < 20 (no clear trend)
ADX_EARLY_TREND_MAX = 28       # kept for reference, overridden by ADX-split logic

USE_EMA200       = True        # directional alignment filter
USE_H4_EMA       = False       # EMA20 H4 proxy — hurts, leave OFF
USE_D1_EMA96     = False       # EMA96 D1 proxy — marginal (+$493), optional

# ── Confluence scoring (for live bot — same as Bot 1) ─────────────────────────
MIN_CONFLUENCE_SCORE = 3.0

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOL      = "BTCUSD"
GOLD_SYMBOL = "XAUUSD"
NAS_SYMBOL  = "NAS100"

# ── Paths ─────────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent
DATA_DIR  = _BOT_DIR / "data"
DB_PATH   = DATA_DIR / "btc2_trades.db"
LOG_DIR   = DATA_DIR / "logs"

# ── API ───────────────────────────────────────────────────────────────────────
API_PORT = int(os.environ.get("BTC2_API_PORT", "8002"))
