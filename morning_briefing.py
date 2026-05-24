"""
morning_briefing.py - FAST Daily Briefing for TradingBotV1

Target: under 60 seconds total.

Loads pre-built data (no ingestion, no backtesting, no full downloads):
  data/rules.json               ← built by setup.py
  data/backtest_results.json    ← built by setup.py
  data/historical_patterns.json ← built by setup.py

Only fetches live data each session:
  • Last 100 candles via yfinance  (~2s)
  • Today's news headlines          (~10s)
  • Current price                   (from candles)

If any step exceeds 10 s it is skipped and marked [SKIPPED - timeout].
"""

import os
import json
import time
import signal
import threading
import concurrent.futures
from collections import Counter
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from _progress import Spinner, ProgressBar, _bar, _fmt_time, OK, FAIL, SKIP, WARN

# ── Debug logger ──────────────────────────────────────────────────────────────
try:
    from debug_logger import (
        log_info, log_signal, log_rejected, log_error,
        log_session_start, log_session_end,
        log_playbook_check, log_confluence, log_checklist,
        log_signal_rejection, save_signal_detail,
    )
    _LOG_OK = True
except ImportError:
    _LOG_OK = False
    def log_info(m): pass
    def log_signal(m): pass
    def log_rejected(m): pass
    def log_error(module="", function="", error="", fallback="", impact=""): pass
    def log_session_start(**kw): pass
    def log_session_end(**kw): pass
    def log_playbook_check(**kw): pass
    def log_confluence(**kw): pass
    def log_checklist(*a, **kw): pass
    def log_signal_rejection(**kw): pass
    def save_signal_detail(*a, **kw): pass

# ── Engine modules (imported lazily to avoid hard failures) ───────────────────
try:
    from confluence_engine  import score_confluences
    _CE_OK = True
except ImportError:
    _CE_OK = False

try:
    from strategy_playbooks import get_active_playbooks, format_playbook_signal, _enrich as _pb_enrich
    _PB_OK = True
except ImportError:
    _PB_OK = False

try:
    from entry_checklist    import validate_entry
    _EC_OK = True
except ImportError:
    _EC_OK = False

try:
    from mtf_analyzer import MultiTimeframeAnalyzer as _MTFAnalyzer, get_htf_context, print_htf_report
    _MTF_OK = True
except ImportError:
    _MTF_OK = False

try:
    from spread_monitor import check_spread as _check_spread
    _SM_OK = True
except ImportError:
    _SM_OK = False
    def _check_spread(symbol="XAUUSD"):  # type: ignore[misc]
        return {"spread_usd": None, "status": "unavailable", "blocked": False,
                "reason": "spread_monitor not available", "recommendation": "",
                "bid": None, "ask": None}

_VA_WARNED = False  # track one-time volume_analyzer warning

try:
    from reversal_hunter import hunt_reversals as _hunt_reversals
    _RH_MB_OK = True
except ImportError:
    _RH_MB_OK = False
    def _hunt_reversals(*a, **kw): return []  # type: ignore[misc]

try:
    from fundamental_bias import get_fundamental_bias as _get_fund_bias
    _FB_MB_OK = True
except ImportError:
    _FB_MB_OK = False
    def _get_fund_bias(*a, **kw): return {"fundamental_bias": "NEUTRAL", "total_score": 0, "display_line": "📊 Fundamental: Unavailable", "available": False, "confidence": 5.0, "factors": {}}  # type: ignore[misc]

from settings import load_settings as _load_settings

try:
    from signal_tracker import register_signal as _st_register
    _ST_OK = True
except ImportError:
    _ST_OK = False
    def _st_register(*a, **kw): return ""  # type: ignore[misc]

try:
    from dxy_correlation import get_dxy_context, print_dxy_report, get_macro_context, get_yields_context
    _DXY_OK = True
except ImportError:
    _DXY_OK = False
    def get_macro_context(*a, **kw): return {"available": False}  # type: ignore[misc]
    def get_yields_context(*a, **kw): return {"available": False}  # type: ignore[misc]

try:
    from geo_filter import get_geopolitical_score as _get_geo_score
    _GEO_OK = True
except ImportError:
    _GEO_OK = False
    def _get_geo_score(*a, **kw): return {"available": False, "geo_risk_level": "normal", "sl_atr_multiplier": 0.0, "confidence_adjustment": 0.0}  # type: ignore[misc]

try:
    from cot_analyzer import fetch_cot_data as _fetch_cot_mb
    _COT_MB_OK = True
except ImportError:
    _COT_MB_OK = False
    def _fetch_cot_mb(): return {"available": False}  # type: ignore[misc]

try:
    from liquidity_map import build_liquidity_map as _build_liq_mb, format_liquidity_map as _fmt_liq_mb
    _LIQ_MB_OK = True
except ImportError:
    _LIQ_MB_OK = False
    def _build_liq_mb(df, p): return {"available": False}  # type: ignore[misc]
    def _fmt_liq_mb(l, p): return ""  # type: ignore[misc]

try:
    from walk_forward import (
        run_walk_forward_optimization as _run_wfo_mb,
        check_if_sunday_run_needed    as _check_wfo_mb,
        get_wfo_summary               as _get_wfo_summary_mb,
    )
    _WFO_MB_OK = True
except ImportError:
    _WFO_MB_OK = False
    def _run_wfo_mb():         return {"optimized": False, "reason": "walk_forward not available"}  # type: ignore[misc]
    def _check_wfo_mb():       return False   # type: ignore[misc]
    def _get_wfo_summary_mb(): return ""      # type: ignore[misc]

try:
    from session_handoff import (
        get_ny_session_bias    as _get_ny_bias_mb,
        format_session_handoff as _format_handoff_mb,
    )
    _SH_MB_OK = True
except ImportError:
    _SH_MB_OK = False
    def _get_ny_bias_mb(df):    return {"ny_bias": "NEUTRAL", "confidence": "LOW", "fake_break_alert": False, "summary": "", "recommendation": "unavailable"}  # type: ignore[misc]
    def _format_handoff_mb(h):  return ""  # type: ignore[misc]

try:
    _ATR_SL_OK = True
except ImportError:
    _ATR_SL_OK = False
    def _calc_dyn_sl(*a, **kw): return {}  # type: ignore[misc]

try:
    from volume_analyzer import check_volume_confluence as _check_vol_confluence
    _VA_OK = True
except ImportError:
    _VA_OK = False
    def _check_vol_confluence(*a, **kw): return {"climax": False, "strategy_optimal": True, "score": 0}  # type: ignore[misc]

try:
    from market_context import (detect_gold_regime, score_news_sentiment,
                                  get_pattern_win_rate, get_regime_strategy_config,
                                  save_regime_snapshot, get_regime_history)
    _MC_OK = True
except ImportError:
    _MC_OK = False

try:
    from mt5_sync import (
        get_account_info   as _mt5_account,
        get_open_positions as _mt5_positions,
        get_today_pnl      as _mt5_today_pnl,
        sync_to_journal    as _mt5_sync_journal,
        auto_match_and_update as _mt5_auto_match,
        get_live_price     as _get_live_price,
    )
    _MT5_OK = True
except ImportError:
    _MT5_OK = False
    def _get_live_price(symbol="XAUUSD"):  # type: ignore[misc]
        return {"price": None, "source": "unavailable", "is_live": False,
                "stale_warning": "mt5_sync not available", "timestamp_uae": "—",
                "bid": None, "ask": None, "spread": None, "age_seconds": 9999}

try:
    from world_sessions import (
        get_active_sessions as _ws_active,
        get_session_summary_line as _ws_summary_line,
        get_full_session_board as _ws_board,
    )
    _WS_OK = True
except ImportError:
    _WS_OK = False
    def _ws_active(*a, **kw): return []          # type: ignore[misc]
    def _ws_summary_line(*a, **kw): return "—"   # type: ignore[misc]
    def _ws_board(*a, **kw): return ""            # type: ignore[misc]

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
RULES_FILE       = os.path.join(BASE_DIR, "data", "rules.json")
BACKTEST_FILE    = os.path.join(BASE_DIR, "data", "backtest_results.json")
PATTERNS_FILE    = os.path.join(BASE_DIR, "data", "historical_patterns.json")
TRADE_LOG_FILE   = os.path.join(BASE_DIR, "data", "trade_log.json")
SETTINGS_FILE    = os.path.join(BASE_DIR, "data", "user_settings.json")


# ── Thresholds ─────────────────────────────────────────────────────────────────
STEP_TIMEOUT     = 10       # seconds per step before skip
TOTAL_TIMEOUT    = 60       # hard ceiling
MIN_CONFIDENCE   = 5

# Tiered signal thresholds (must match backtest.py)
TIER_A_WR = 55.0;  TIER_A_PF = 1.2    # Strong  — use with full confidence
TIER_B_WR = 45.0;  TIER_B_PF = 0.9    # Moderate — use with caution
TIER_C_WR = 35.0;  TIER_C_PF = 0.7    # Weak    — paper trade only

TIER_LABELS = {
    "A": "STRONG SIGNAL",
    "B": "MODERATE SIGNAL",
    "C": "WEAK SIGNAL - paper trade only",
    "D": "UNVERIFIED - skip for now",
}


def _get_tier(wr: float, pf: float) -> str:
    """Return tier letter A/B/C/D based on win rate and profit factor."""
    if wr >= TIER_A_WR and pf >= TIER_A_PF:
        return "A"
    if wr >= TIER_B_WR and pf >= TIER_B_PF:
        return "B"
    if wr >= TIER_C_WR and pf >= TIER_C_PF:
        return "C"
    return "D"

# ── Session ───────────────────────────────────────────────────────────────────
GST              = timezone(timedelta(hours=4))

WARNING_KEYWORDS = ["fed", "fomc", "nfp", "non-farm", "powell", "payroll", "cpi"]

PIP_SIZE         = 0.1
PRICE_CACHE_FILE = os.path.join(BASE_DIR, "data", "price_cache.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Timeout helper
# ═══════════════════════════════════════════════════════════════════════════════

def _run_with_timeout(fn, timeout: int, label: str):
    """
    Run fn() in a thread. Returns (result, elapsed, timed_out).
    """
    result_box = [None]
    exc_box    = [None]

    def _worker():
        try:
            result_box[0] = fn()
        except Exception as e:
            exc_box[0] = e

    t0     = time.time()
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    elapsed = time.time() - t0

    if thread.is_alive():
        return None, elapsed, True      # timed out
    if exc_box[0]:
        return None, elapsed, False     # exception
    return result_box[0], elapsed, False


# ═══════════════════════════════════════════════════════════════════════════════
# Step helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gst_now() -> str:
    return datetime.now(GST).strftime("%Y-%m-%d %H:%M GST")


def _day_str() -> str:
    return datetime.now(GST).strftime("%A %d %B %Y")


def _current_session_str() -> str:
    """Return the current trading session name (UAE-accurate via world_sessions)."""
    if _WS_OK:
        return _ws_summary_line()
    # Fallback: UTC-hour approximation
    hour = datetime.now(timezone.utc).hour
    if 12 <= hour < 15:
        return "Overlap"
    if 7 <= hour < 12:
        return "London"
    if 13 <= hour < 17:
        return "NewYork"
    if 0 <= hour < 7:
        return "Asian"
    return "Off-Hours"


def get_mtf_confluence_score(symbol: str = "GC=F") -> dict:
    """
    Score MTF confluence across D1, H4, H1, M15 using EMA50 vs EMA200.

    Returns dict with keys:
        d1_bias, h4_bias, h1_bias, m15_bias,
        overall_bias, confluence_score (0-4),
        aligned (bool, True if score >= 3),
        summary (str)
    """
    import numpy as np

    def _bias_from_df(df) -> str:
        """Return 'bullish' | 'bearish' | 'ranging' from enriched DataFrame."""
        if df is None or df.empty:
            return "ranging"
        try:
            row    = df.iloc[-1]
            price  = float(row["close"])
            ema200 = float(row.get("ema200", float("nan")))
            ema50  = float(row.get("ema50",  float("nan")))
            if any(np.isnan(v) for v in [price, ema200, ema50]):
                return "ranging"
            gap_pct = abs(price - ema200) / ema200
            if gap_pct < 0.005:
                return "ranging"
            if price > ema200 and ema50 > ema200:
                return "bullish"
            if price < ema200 and ema50 < ema200:
                return "bearish"
        except Exception:
            pass
        return "ranging"

    def _fetch_m15(sym: str):
        """Fetch M15 via yfinance and enrich with EMA50/EMA200."""
        try:
            import yfinance as yf
            import pandas as pd
            df = yf.Ticker(sym).history(interval="15m", period="5d", auto_adjust=True)
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df = df.copy()
            close = df["close"]
            df["ema50"]  = close.ewm(span=50,  adjust=False).mean()
            df["ema200"] = close.ewm(span=200, adjust=False).mean()
            return df.dropna(subset=["ema200"])
        except Exception:
            return None

    # Fetch all four timeframes
    _mtf_fetch = None
    if _MTF_OK:
        try:
            from mtf_analyzer import _fetch_df as _mtf_fetch_df
            _mtf_fetch = _mtf_fetch_df
        except Exception:
            pass
    d1_df  = _mtf_fetch(symbol, "D1") if _mtf_fetch else None
    h4_df  = _mtf_fetch(symbol, "H4") if _mtf_fetch else None
    h1_df  = _mtf_fetch(symbol, "H1") if _mtf_fetch else None
    m15_df = _fetch_m15(symbol)

    d1_bias  = _bias_from_df(d1_df)
    h4_bias  = _bias_from_df(h4_df)
    h1_bias  = _bias_from_df(h1_df)
    m15_bias = _bias_from_df(m15_df)

    # Overall bias = D1 + H4 agreement; conflict → ranging
    if d1_bias == h4_bias and d1_bias != "ranging":
        overall_bias = d1_bias
    else:
        overall_bias = "ranging"

    # Score: each TF aligned with overall_bias = +1
    if overall_bias == "ranging":
        score = 0
    else:
        score = sum(
            1 for b in [d1_bias, h4_bias, h1_bias, m15_bias]
            if b == overall_bias
        )

    aligned = score >= 3

    if overall_bias == "ranging":
        summary = "D1/H4 conflict — no directional bias"
    elif aligned:
        summary = f"{score}/4 timeframes {overall_bias} — proceed"
    else:
        summary = f"{score}/4 timeframes {overall_bias} — caution, mixed signals"

    return {
        "d1_bias":          d1_bias,
        "h4_bias":          h4_bias,
        "h1_bias":          h1_bias,
        "m15_bias":         m15_bias,
        "overall_bias":     overall_bias,
        "confluence_score": score,
        "aligned":          aligned,
        "summary":          summary,
    }


# ── Global-news keyword lists ─────────────────────────────────────────────────
_BULLISH_GOLD_KW = [
    "war", "attack", "missile", "invasion", "conflict", "crisis",
    "sanctions", "escalation", "explosion", "terror", "troops",
    "collapse", "recession", "default", "bank run", "contagion",
]
_BEARISH_GOLD_KW = [
    "ceasefire", "peace deal", "resolution", "recovery", "growth",
    "surplus", "strong economy", "rate hike", "dollar rally",
]
_VOLATILE_KW = [
    "tweet", "trump", "president", "executive order", "breaking",
    "unexpected", "surprise", "shock", "emergency", "opec cut",
    "oil embargo", "nuclear", "assassination",
]

# Gold-impact classification for scheduled economic events
_SCHED_BULLISH_EVENTS = [
    "unemployment", "jobless", "non-farm", "nfp", "gdp miss",
    "retail sales miss", "cpi higher", "inflation", "fed pause",
]
_SCHED_BEARISH_EVENTS = [
    "rate hike", "fed hike", "strong gdp", "jobs beat",
    "dollar strength", "rate decision",
]
_SCHED_VOLATILE_EVENTS = [
    "fomc", "fed", "powell", "payroll", "cpi", "pce", "ppi",
    "ecb", "boe", "rba", "nfp", "non-farm",
]


def get_global_news_context() -> dict:
    """
    Combine scheduled economic events + live global headlines into one
    gold-impact verdict.

    Returns dict with:
        gold_bias_from_news  : "bullish" | "bearish" | "volatile" | "conflicted" | "neutral"
        volatility_warning   : bool
        volatility_triggers  : list[str]
        bullish_triggers     : list[str]
        bearish_triggers     : list[str]
        scheduled_bias       : str
        headline_bias        : str
        key_event            : str
        all_headlines        : list[str]  (top 10 relevant)
        recommendation       : str
        trade_with_caution   : bool
        confidence           : float  (0-10)
    """
    bullish_triggers:  list[str] = []
    bearish_triggers:  list[str] = []
    volatile_triggers: list[str] = []
    all_headlines:     list[str] = []
    key_event = "None identified"
    headline_bias = "neutral"
    scheduled_bias = "neutral"

    # ── PART A — Scheduled events ─────────────────────────────────────────────
    try:
        from news_filter import get_todays_events as _gte
        events = _gte({"High", "Medium"})
        sched_bull = 0
        sched_bear = 0
        for ev in events:
            title_l = ev.get("title", "").lower()
            if any(kw in title_l for kw in _SCHED_VOLATILE_EVENTS):
                volatile_triggers.append(ev["title"])
            if any(kw in title_l for kw in _SCHED_BULLISH_EVENTS):
                sched_bull += 1
            elif any(kw in title_l for kw in _SCHED_BEARISH_EVENTS):
                sched_bear += 1
        if sched_bull > sched_bear:
            scheduled_bias = "bullish"
        elif sched_bear > sched_bull:
            scheduled_bias = "bearish"
        elif volatile_triggers:
            scheduled_bias = "volatile"
        else:
            scheduled_bias = "neutral"
        if events:
            key_event = events[0]["title"] + " @ " + events[0]["time_gst"]
    except Exception:
        pass

    # ── PART B — Live headlines ───────────────────────────────────────────────
    try:
        from news_monitor import fetch_news as _fn, get_market_sentiment as _gms
        items = _fn()
        sentiment = _gms(items) or {}
        gold_sent = sentiment.get("gold", {}) or {}
        headline_bias_raw = str(gold_sent.get("bias", "wait")).lower()
        headline_bias = {
            "buy": "bullish", "sell": "bearish",
            "bullish": "bullish", "bearish": "bearish",
        }.get(headline_bias_raw, "neutral")
        if not key_event or key_event == "None identified":
            key_event = sentiment.get("key_event_today", "None identified")

        # Scan headline titles for keyword triggers
        for item in items:
            title_l = item.get("title", "").lower()
            for kw in _BULLISH_GOLD_KW:
                if kw in title_l:
                    bullish_triggers.append(item["title"])
                    break
            for kw in _BEARISH_GOLD_KW:
                if kw in title_l:
                    bearish_triggers.append(item["title"])
                    break
            for kw in _VOLATILE_KW:
                if kw in title_l:
                    volatile_triggers.append(item["title"])
                    break

        # Top 10 relevant headlines (GOLD + MACRO categories first)
        ranked = sorted(
            items,
            key=lambda i: 0 if i.get("category") in ("GOLD", "MACRO", "RISK") else 1,
        )
        all_headlines = [i["title"] for i in ranked[:10]]
    except Exception:
        pass

    # Deduplicate trigger lists
    bullish_triggers  = list(dict.fromkeys(bullish_triggers))[:5]
    bearish_triggers  = list(dict.fromkeys(bearish_triggers))[:5]
    volatile_triggers = list(dict.fromkeys(volatile_triggers))[:5]

    # ── PART C — Combine ──────────────────────────────────────────────────────
    volatility_warning = bool(volatile_triggers)

    # Build keyword-derived headline direction
    kw_bull = len(bullish_triggers)
    kw_bear = len(bearish_triggers)
    if kw_bull > kw_bear + 1:
        kw_bias = "bullish"
    elif kw_bear > kw_bull + 1:
        kw_bias = "bearish"
    else:
        kw_bias = headline_bias   # fall back to sentiment analysis

    # Merge scheduled + live
    if volatility_warning:
        gold_bias_from_news = "volatile"
        trade_with_caution  = True
        confidence          = 4.0
        recommendation      = "Volatility trigger detected — wait for candle close before entry."
    elif scheduled_bias == kw_bias and kw_bias != "neutral":
        gold_bias_from_news = kw_bias
        trade_with_caution  = False
        confidence          = 7.5
        recommendation      = f"Scheduled events + live headlines both {kw_bias} — high confidence."
    elif scheduled_bias != "neutral" and kw_bias != "neutral" and scheduled_bias != kw_bias:
        gold_bias_from_news = "conflicted"
        trade_with_caution  = True
        confidence          = 3.5
        recommendation      = "Scheduled events and live news conflict — trade smaller size or skip."
    elif kw_bias != "neutral":
        gold_bias_from_news = kw_bias
        trade_with_caution  = False
        confidence          = 6.0
        recommendation      = f"Live headlines lean {kw_bias} — watch for confirmation."
    elif scheduled_bias != "neutral":
        gold_bias_from_news = scheduled_bias
        trade_with_caution  = False
        confidence          = 5.5
        recommendation      = f"Scheduled events lean {scheduled_bias} — monitor for follow-through."
    else:
        gold_bias_from_news = "neutral"
        trade_with_caution  = False
        confidence          = 5.0
        recommendation      = "No strong news bias — rely on technicals."

    return {
        "gold_bias_from_news": gold_bias_from_news,
        "volatility_warning":  volatility_warning,
        "volatility_triggers": volatile_triggers,
        "bullish_triggers":    bullish_triggers,
        "bearish_triggers":    bearish_triggers,
        "scheduled_bias":      scheduled_bias,
        "headline_bias":       headline_bias,
        "key_event":           key_event or "None identified",
        "all_headlines":       all_headlines,
        "recommendation":      recommendation,
        "trade_with_caution":  trade_with_caution,
        "confidence":          confidence,
    }


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _load_trade_log() -> list[dict]:
    return _load_json(TRADE_LOG_FILE, [])


# ── Step 1: Load rules database ───────────────────────────────────────────────

def _step1_load_rules():
    rules = _load_json(RULES_FILE, [])

    # Count by tier
    tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in rules:
        tier = r.get("tier") or _get_tier(_bt_wr(r), _pf(r))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return {
        "rules":           rules,
        "above_threshold": tier_counts.get("A", 0),
        "tier_counts":     tier_counts,
    }


# ── Step 2: Fetch news headlines ─────────────────────────────────────────────

def _step2_fetch_news():
    try:
        from news_monitor import fetch_news, get_market_sentiment
        items     = fetch_news()
        sentiment = get_market_sentiment(items) or {}
        gold_bias = str(sentiment.get("gold", {}).get("bias", "wait")).upper()

        gold_items = [
            i for i in items
            if i.get("category") in ("GOLD", "MACRO", "RISK")
        ]
        return {
            "items":      items,
            "sentiment":  sentiment,
            "gold_count": len(gold_items),
            "gold_bias":  gold_bias,
            "conf":       sentiment.get("gold", {}).get("confidence", "—"),
            "key_event":  sentiment.get("key_event_today", "None"),
        }
    except ImportError:
        return {"error": "news_monitor not available"}


# ── Step 3: Get live market data (last 100 candles) ───────────────────────────

def _step3_live_data():
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd

        ticker = yf.Ticker("GC=F")
        df     = ticker.history(period="5d", interval="1h", auto_adjust=True)

        if df.empty:
            return {"error": "No data from yfinance"}

        df.columns = [c.lower() for c in df.columns]
        df = df.tail(100).copy()

        # Basic indicators (pure pandas)
        close   = df["close"]
        ema50   = close.ewm(span=50,  adjust=False).mean()
        ema200  = close.ewm(span=200, adjust=False).mean()
        delta   = close.diff()
        gain    = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss    = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        rs      = gain / loss.replace(0, float("nan"))
        rsi     = (100 - 100 / (1 + rs)).iloc[-1]

        prev_c  = close.shift(1)
        tr      = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr     = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

        price    = float(close.iloc[-1])
        e50      = float(ema50.iloc[-1])
        e200     = float(ema200.iloc[-1])
        rsi_val  = float(rsi)
        # For XAUUSD/Gold: ATR is already in dollars — no pip conversion needed
        atr_dollar = round(float(atr), 2)

        trend = "BULLISH" if price > e200 else "BEARISH"
        rsi_zone = "oversold" if rsi_val < 35 else ("overbought" if rsi_val > 65 else "neutral")

        # Write price cache so strategy_playbooks can read latest bid/ask approx
        try:
            import json as _json
            import os as _os
            _os.makedirs(os.path.dirname(PRICE_CACHE_FILE) if os.path.dirname(PRICE_CACHE_FILE) else ".", exist_ok=True)
            spread = atr_dollar * 0.001   # approximate 0.1% spread for gold
            with open(PRICE_CACHE_FILE, "w") as _f:
                _json.dump({
                    "symbol": "XAUUSD",
                    "ask":    round(price + spread, 2),
                    "bid":    round(price - spread, 2),
                    "ts":     datetime.now(GST).isoformat(),
                }, _f)
        except Exception:
            pass

        # Attach enriched df so Step 5 can use it without re-fetching
        df["ema50"]       = ema50
        df["ema200"]      = ema200
        df["rsi"]         = rsi
        df["atr"]         = tr.ewm(alpha=1/14, adjust=False).mean()
        e12               = close.ewm(span=12, adjust=False).mean()
        e26               = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        if "open" not in df.columns:
            df["open"]    = df["close"].shift(1).fillna(df["close"])

        # ── Upgrade price from live source ────────────────────────────────────
        try:
            lp = _get_live_price()
            if lp.get("price") and lp["price"] > 0 and lp.get("is_live"):
                price = lp["price"]
                # Overwrite last row close so all downstream calcs use live price
                df.iloc[-1, df.columns.get_loc("close")] = price
                # Re-compute trend with live price
                trend = "BULLISH" if price > float(ema200.iloc[-1]) else "BEARISH"
                # Update price cache with real spread if available
                if lp.get("bid") and lp.get("ask"):
                    _spread = round(lp["ask"] - lp["bid"], 2)
                    try:
                        import json as _json2
                        with open(PRICE_CACHE_FILE, "w") as _f2:
                            _json2.dump({
                                "symbol": "XAUUSD",
                                "ask": lp["ask"],
                                "bid": lp["bid"],
                                "ts": datetime.now(GST).isoformat(),
                            }, _f2)
                    except Exception:
                        pass
        except Exception:
            pass

        return {
            "price":    round(price, 2),
            "ema50":    round(e50, 2),
            "ema200":   round(e200, 2),
            "rsi":      round(rsi_val, 1),
            "rsi_zone": rsi_zone,
            "atr":      atr_dollar,
            "atr_pips": atr_dollar,   # kept for backward-compat; now dollar value
            "trend":    trend,
            "df":       df,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Step 4: Load historical patterns ─────────────────────────────────────────

def _step4_patterns():
    data = _load_json(PATTERNS_FILE, {})
    if not data:
        return {"error": "historical_patterns.json not found — run setup.py"}
    return data


# ── Step MTF: Multi-Timeframe Bias ────────────────────────────────────────────

def _step_mtf_bias(symbol: str = "GC=F", h1_price: float | None = None) -> dict:
    """
    Fetch D1 and H4 data and compute top-down HTF bias.
    Returns the full context dict from mtf_analyzer.get_htf_context().
    On any failure returns {'available': False, 'bias_line': 'MTF unavailable'}.
    """
    if not _MTF_OK:
        return {"available": False, "bias_line": "MTF module unavailable",
                "htf_bias": {}, "htf_levels": {}}
    try:
        ctx = get_htf_context(symbol=symbol, h1_price=h1_price)
        return ctx
    except Exception as exc:
        return {"available": False, "bias_line": f"MTF error: {exc}",
                "htf_bias": {}, "htf_levels": {}}


# ── Step DXY / Macro: US Dollar Index + US10Y Yields correlation ────────────

def _step_dxy(gold_dir: str = "") -> dict:
    """
    Fetch DXY + US10Y yields and return the combined macro context dict.
    Falls back to plain DXY context if get_macro_context fails.
    On any failure returns a safe stub dict.
    """
    _stub = {"available": False, "display_line": "DXY/Macro unavailable",
             "dxy_trend": "sideways", "dxy_rsi": 50.0, "momentum_strength": "weak",
             "macro_score": 0.0, "macro_bias": "neutral",
             "macro_confirmed": False, "macro_opposed": False,
             "confidence_adjustment": 0.0, "summary": "",
             "dxy": {}, "yields": {}}
    if not _DXY_OK:
        return {**_stub, "display_line": "DXY module unavailable"}
    try:
        direction = gold_dir.strip().lower() or "long"
        return get_macro_context(direction)
    except Exception as exc:
        try:
            return get_dxy_context()  # plain fallback
        except Exception:
            return {**_stub, "display_line": f"DXY error: {exc}"}


# ── Step 5: Multi-stage signal scan ─────────────────────────────────────────

def _bt_wr(rule: dict) -> float:
    bt = rule.get("backtest", {})
    if isinstance(bt, dict):
        v = bt.get("win_rate")
        if v is not None:
            val = float(v)
            return val * 100 if val <= 1 else val
    for k in ("win_rate", "success_rate"):
        v = rule.get(k)
        if v is not None:
            val = float(v)
            return val * 100 if val <= 1 else val
    return 0.0


def _pf(rule: dict) -> float:
    bt = rule.get("backtest", {})
    if isinstance(bt, dict):
        v = bt.get("profit_factor")
        if v is not None:
            return float(v)
    return float(rule.get("profit_factor", 0.0))


def _conf(rule: dict) -> float:
    for k in ("confidence_score", "confidence"):
        v = rule.get(k)
        if v is not None:
            return float(v)
    return 5.0


def _asset(rule: dict) -> str:
    text = " ".join([rule.get(k, "") or "" for k in ("name", "pattern_name", "description", "asset", "condition", "entry_condition")]).lower()
    if any(k in text for k in ["xau", "gold", "bullion"]): return "XAUUSD"
    if any(k in text for k in ["silver", "xag"]):           return "XAGUSD"
    if any(k in text for k in ["oil", "crude", "wti"]):     return "OIL"
    return "FOREX"


def _step5_scan_signals(
    rules:     list[dict],
    sentiment: dict,
    df=None,               # enriched DataFrame from _step3_live_data
    htf_ctx:   dict | None = None,  # result from _step_mtf_bias()
    dxy_ctx:   dict | None = None,  # result from _step_dxy() / _step_macro()
    mtf_confluence: dict | None = None,  # result from get_mtf_confluence_score()
    global_news_ctx: dict | None = None,  # result from get_global_news_context()
    macro_ctx: dict | None = None,  # alias for dxy_ctx when using get_macro_context()
    geo_ctx:   dict | None = None,  # result from get_geopolitical_score()
) -> tuple[list[dict], dict]:
    """
    4-stage signal scan:
      STAGE 1 — Playbook signals (strategy_playbooks.py)
      STAGE 2 — Rules database signals (rules.json, Tier B+)
      STAGE 3 — Combine + deduplicate + rank
      STAGE 4 — Return top 3

    Returns (signals, meta) where meta holds pipeline counts.
    """
    import pandas as pd
    import numpy as np

    gold_bias    = str(sentiment.get("gold", {}).get("bias", "wait")).lower()
    gold_dir     = {"buy": "long", "sell": "short"}.get(gold_bias, "")
    risk_penalty = 2 if str(sentiment.get("overall_risk", "")).lower() == "high" else 0

    # Macro context — prefer explicit macro_ctx, fall back to dxy_ctx, default 0 adj
    _macro = macro_ctx or dxy_ctx or {}
    _macro_conf_adj = float(_macro.get("confidence_adjustment", 0.0))
    _macro_bias     = str(_macro.get("macro_bias", "neutral"))

    # Geo context — safe-haven confidence boost
    _geo             = geo_ctx or {}
    _geo_conf_adj    = float(_geo.get("confidence_adjustment", 0.0))
    _geo_sl_mult     = float(_geo.get("sl_atr_multiplier", 0.0))
    _geo_risk_level  = str(_geo.get("geo_risk_level", "normal"))

    # Pull htf_bias dict for confluence engine and filter logic
    htf_bias_dict = (htf_ctx or {}).get("htf_bias") or None
    htf_overall   = str((htf_bias_dict or {}).get("overall_bias", "neutral")).lower()
    htf_strength  = str((htf_bias_dict or {}).get("bias_strength", "weak")).lower()

    # If HTF bias is available, tighten direction filter
    # STRONG bias = only allow signals in that direction
    # CONFLICTED = allow both but log warning
    htf_dir_filter = ""
    if htf_bias_dict and htf_strength == "strong":
        if htf_overall == "bearish":
            htf_dir_filter = "short"
        elif htf_overall == "bullish":
            htf_dir_filter = "long"

    # MTF-4 confluence gate — skip all signals if score < 2
    _mtf4        = mtf_confluence or {}
    _mtf4_score  = int(_mtf4.get("confluence_score", 4))  # default 4 = no filter if not provided
    _mtf4_bias   = str(_mtf4.get("overall_bias", "ranging"))
    _mtf4_skip   = _MTF_OK and mtf_confluence is not None and _mtf4_score < 2

    # ── Brain 1: load auto-filters ────────────────────────────────────────────
    _auto_filters: list[dict] = []
    try:
        import json as _json
        _af_path = os.path.join(BASE_DIR, "data", "auto_filters.json")
        if os.path.exists(_af_path):
            with open(_af_path, encoding="utf-8") as _af_f:
                _af_data = _json.load(_af_f)
            _auto_filters = _af_data if isinstance(_af_data, list) else []
    except Exception:
        _auto_filters = []

    meta = {
        "playbook_found":      0,
        "playbook_passed":     0,
        "rules_found":         0,
        "rules_passed":        0,
        "sl_rejected":         0,
        "conf_rejected":       0,
        "mtf4_rejected":       0,
        "fatigue_rejected":    0,
        "active_playbooks":    [],
        "strongest":           None,
        "htf_overall":         htf_overall,
        "htf_strength":        htf_strength,
        "htf_dir_filter":      htf_dir_filter,
        "mtf4_score":          _mtf4_score,
        "mtf4_bias":           _mtf4_bias,
        "mtf4_skipped":        _mtf4_skip,
    }

    if _mtf4_skip:
        log_rejected(
            f"[MTF-4] All signals skipped — confluence score {_mtf4_score}/4 < 2 "
            f"({_mtf4.get('summary', '')})"
        )
        return [], meta

    # ── Spread check ─────────────────────────────────────────────────────────
    _spread_result: dict = {"status": "unavailable", "blocked": False}
    try:
        _spread_result = _check_spread("XAUUSD")
        meta["spread_check"] = _spread_result
        if _spread_result.get("blocked"):
            log_rejected(f"[SPREAD] All signals flagged — {_spread_result['reason']}")
    except Exception as _se:
        log_error("morning_briefing", "spread_check", str(_se))

    # ── Volume analyzer availability warning (one-time) ───────────────────────
    global _VA_WARNED
    if not _VA_OK and not _VA_WARNED:
        log_info("volume_analyzer not available — volume gate disabled")
        _VA_WARNED = True

    # ── Market Context (regime + news) ────────────────────────────────────────
    regime_data = None
    if _MC_OK and df is not None and not df.empty:
        try:
            regime_data = detect_gold_regime(df)
            meta["regime"]       = regime_data["regime"]
            meta["regime_label"] = regime_data["regime_label"]
            meta["regime_note"]  = regime_data["regime_note"]
            meta["size_mult"]    = regime_data["position_size_multiplier"]
            meta["best_playbooks"]  = regime_data["best_playbooks"]
            meta["avoid_playbooks"] = regime_data["avoid_playbooks"]
        except Exception:
            regime_data = None

    # ── STAGE 1 — Playbook signals ────────────────────────────────────────────
    playbook_signals: list[dict] = []

    if _PB_OK and df is not None and not df.empty:
        try:
            pb_hits = get_active_playbooks(df, sentiment, top_n=6)
            meta["playbook_found"]   = len(pb_hits)
            meta["active_playbooks"] = [h["playbook"]["name"] for h in pb_hits]
            if pb_hits:
                meta["strongest"] = (
                    f"{pb_hits[0]['playbook']['name']} "
                    f"({pb_hits[0]['score']}/10)"
                )

            for hit in pb_hits:
                pb        = hit["playbook"]
                direction = hit["direction"]
                entry_p   = hit["entry"]
                sl_p      = hit["stop_loss"]
                tp_p      = hit["take_profit"]
                score     = hit["score"]

                # HTF direction filter — skip if HTF strongly disagrees
                if htf_dir_filter and direction != htf_dir_filter:
                    log_rejected(
                        f"Playbook {pb.get('name','?')} skipped — HTF filter requires {htf_dir_filter}"
                    )
                    continue

                sig = {
                    "source":        "playbook",
                    "asset":         "XAUUSD",
                    "direction":     direction,
                    "confidence":    score,
                    "confidence_score": score,
                    "pattern_name":  pb["name"],
                    "playbook_id":   pb["id"],
                    "description":   pb["notes"],
                    "entry":         entry_p,
                    "stop_loss":     sl_p,
                    "take_profit":   tp_p,
                    "bt_win_rate":   pb.get("win_rate_expected", 0),
                    "profit_factor": 0.0,
                    "tier":          "B",
                    "tier_label":    TIER_LABELS["B"],
                    "note":          f"Playbook: {pb['timeframe']}",
                    "confluence_met":    hit["met_list"],
                    "confluence_missed": [],
                    "checklist_results": None,
                    "is_breakout":   "breakout" in pb["id"].lower(),
                    "smc_context":   None,
                    "entry_quality": None,
                }

                # SMC context enrichment (playbook path)
                if df is not None and not df.empty:
                    try:
                        from smart_money import SmartMoneyAnalyzer as _SMABrief2
                        _smc_ctx_pb = _SMABrief2().get_smc_context(df, direction)
                        sig["smc_context"]  = _smc_ctx_pb
                        sig["entry_quality"] = _smc_ctx_pb.get("entry_quality", None)
                        _smc_adj_pb = float(_smc_ctx_pb.get("confidence_adjustment", 0.0))
                        sig["confidence"] = round(min(10.0, max(0.0, sig["confidence"] + _smc_adj_pb)), 1)
                        if _smc_ctx_pb.get("entry_quality") == "D":
                            sig["note"] = sig.get("note", "") + " ⚠ No SMC confluence"
                    except Exception as _smce_pb:
                        log_error("morning_briefing", "smc_context_playbook", str(_smce_pb), "SMC context skipped")

                # Log playbook trigger
                log_playbook_check(
                    name=pb["name"],
                    number=meta["playbook_found"],
                    conditions_met=len(hit.get("met_list", [])),
                    conditions_total=5,
                    conditions=[{"name": c, "passed": True, "detail": ""} for c in hit.get("met_list", [])],
                    triggered=True,
                    direction=direction,
                )

                # Entry checklist gate (need 4/5 or 5/5)
                if _EC_OK and df is not None:
                    try:
                        ck = validate_entry(sig, df)
                        sig["checklist_results"] = ck
                        sig["confidence"]        = ck["final_confidence"]
                        log_checklist(pb["name"], direction, ck)
                        if ck["checks_passed"] >= 4:
                            # ── Volume check gate ──────────────────────────
                            _vol_gate_ok = True
                            _vol = _check_vol_confluence(
                                df, direction, pb.get("name", "unknown")
                            )
                            sig["volume"] = _vol
                            if _vol.get("climax"):
                                log_rejected(f"{pb['name']} — volume_climax_detected")
                                meta["vol_rejected"] = meta.get("vol_rejected", 0) + 1
                                _vol_gate_ok = False
                            elif not _vol.get("strategy_optimal", True):
                                log_rejected(f"{pb['name']} — volume_suboptimal")
                                meta["vol_rejected"] = meta.get("vol_rejected", 0) + 1
                                _vol_gate_ok = False
                            if not _vol_gate_ok:
                                continue
                            # ── /Volume check gate ─────────────────────────
                            # ── Fatigue gate ───────────────────────────────
                            try:
                                from pattern_fatigue import check_strategy_fatigue
                                _fatigue = check_strategy_fatigue(
                                    sig["pattern_name"],
                                    df if df is not None else pd.DataFrame(),
                                    direction
                                )
                                fatigue_level = _fatigue.get("fatigue_level", "none")
                                sig["fatigue_level"] = fatigue_level
                                sig["fatigue_recommendation"] = _fatigue.get("recommendation", "")
                                if fatigue_level == "critical":
                                    log_rejected(f"{sig['pattern_name']} — fatigue_critical")
                                    meta["fatigue_rejected"] = meta.get("fatigue_rejected", 0) + 1
                                    save_signal_detail(sig, "REJECTED", rejection_reason="Pattern fatigue critical",
                                                      rejection_stage="fatigue_gate",
                                                      spread_usd=meta.get("spread_check", {}).get("spread_usd"))
                                    continue
                                elif fatigue_level in ("high", "moderate"):
                                    sig["confidence"] = max(0, sig["confidence"] - 0.5)
                                    sig["note"] = sig.get("note", "") + f" ⚠ Fatigue {fatigue_level}"
                            except Exception as _fe:
                                log_error("morning_briefing", "fatigue_gate", str(_fe),
                                          "fatigue check skipped")
                            # ── /Fatigue gate ──────────────────────────────
                            meta["playbook_passed"] += 1
                            playbook_signals.append(sig)
                            save_signal_detail(
                                sig, "SHOWN_TO_USER",
                                session=_current_session_str(),
                                gold_price=float(df["close"].iloc[-1]) if df is not None else 0,
                                d1_bias=meta.get("htf_overall","—"),
                                h4_bias=meta.get("htf_overall","—"),
                                dxy_status=str((dxy_ctx or {}).get("dxy_trend","—")),
                                regime=meta.get("regime","—"),
                                checklist_result=ck,
                                spread_usd=meta.get("spread_check", {}).get("spread_usd"),
                            )
                        else:
                            reason = ck.get("rejection_reason", "checklist failed")
                            log_signal_rejection(
                                signal_name=pb["name"], direction=direction,
                                reason=reason, stage="entry_checklist"
                            )
                            save_signal_detail(
                                sig, "REJECTED",
                                session=_current_session_str(),
                                gold_price=float(df["close"].iloc[-1]) if df is not None else 0,
                                d1_bias=meta.get("htf_overall","—"),
                                checklist_result=ck,
                                rejection_reason=reason,
                                rejection_stage="entry_checklist",
                                spread_usd=meta.get("spread_check", {}).get("spread_usd"),
                            )
                    except Exception as _e:
                        log_error("morning_briefing", "playbook_checklist",
                                  str(_e), "signal included without checklist")
                        playbook_signals.append(sig)   # fallback: include anyway
                else:
                    playbook_signals.append(sig)
        except Exception:
            pass   # playbook engine unavailable — continue to Stage 2

    # ── STAGE 2 — Rules database signals ─────────────────────────────────────
    rules_signals: list[dict] = []

    if rules:
        TIER_MIN_CONF = {"A": MIN_CONFIDENCE, "B": 4, "C": 3}

        for rule in rules:
            wr   = _bt_wr(rule)
            pf   = _pf(rule)
            tier = rule.get("tier") or _get_tier(wr, pf)

            if tier in ("C", "D"):
                continue    # Rules: Tier B and above only

            asset     = _asset(rule)
            direction = str(rule.get("direction", "both")).lower()

            if asset == "XAUUSD":
                if gold_bias == "wait":
                    continue
                if direction == "both":
                    direction = gold_dir or direction
                elif gold_dir and direction != gold_dir:
                    continue

            live_wr = rule.get("live_win_rate")
            eff_c   = _conf(rule)
            if live_wr is not None:
                if float(live_wr) > wr + 5:
                    eff_c = min(10, eff_c + 1)
                elif float(live_wr) < wr - 10:
                    eff_c = max(1, eff_c - 1)
            eff_c -= risk_penalty
            # Apply macro confidence adjustment (+1 if confirmed, -1 if opposed, 0 neutral)
            eff_c += _macro_conf_adj
            # Apply geo-risk confidence adjustment (safe-haven bid on extreme/high events)
            eff_c += _geo_conf_adj

            tier_min = TIER_MIN_CONF.get(tier, MIN_CONFIDENCE)
            if eff_c < tier_min:
                continue

            # Confluence gate (score >= 3)
            c_met, c_missed, detail_lines_sig = [], [], []
            _cr_raw: dict = {}
            if _CE_OK and df is not None and not df.empty:
                try:
                    _cr_raw      = score_confluences(df, direction, htf_bias=htf_bias_dict, dxy_ctx=dxy_ctx)
                    met_d        = _cr_raw.get("confluences_met",    [])
                    missed_d     = _cr_raw.get("confluences_failed", [])
                    c_met        = [m.get("check", "?") for m in met_d   if m.get("result") != "neutral"]
                    c_missed     = [m.get("check", "?") for m in missed_d]
                    detail_lines_sig = _cr_raw.get("detail_lines", [])
                    n_met        = len(c_met)
                    log_confluence(asset, direction, raw_result=_cr_raw)
                    if n_met < 3:
                        log_signal_rejection(
                            signal_name=rule.get("name","?"), direction=direction,
                            reason=f"Confluence too low ({n_met}/6 met, need 3)",
                            stage="confluence_gate",
                        )
                        continue   # Fails confluence gate
                except Exception as _ce:
                    log_error("morning_briefing", "confluence_gate", str(_ce),
                              "confluence check skipped", "signal may be low quality")
                    pass   # Engine error — don't block the signal

            # SMC context enrichment
            if df is not None and not df.empty:
                try:
                    from smart_money import SmartMoneyAnalyzer as _SMABrief
                    _smc_ctx = _SMABrief().get_smc_context(df, direction)
                    sig_smc_store = _smc_ctx
                    _smc_adj = float(_smc_ctx.get("confidence_adjustment", 0.0))
                    eff_c = round(min(10.0, max(0.0, eff_c + _smc_adj)), 1)
                    if _smc_ctx.get("entry_quality") == "D":
                        _geo_note_smc = "⚠ No SMC confluence"
                    else:
                        _geo_note_smc = ""
                except Exception as _smce:
                    log_error("morning_briefing", "smc_context_rules", str(_smce), "SMC context skipped")
                    sig_smc_store = None
                    _geo_note_smc = ""
            else:
                sig_smc_store = None
                _geo_note_smc = ""

            # Geo risk note (warn SHORT under extreme/high, confirm LONG)
            _geo_note = ""
            if _geo_risk_level in ("extreme", "high"):
                if direction == "short":
                    _geo_note = f"⚠ Geo risk ({_geo_risk_level}) — SHORT into safe-haven bid; reduce size"
                else:
                    _geo_note = f"🌍 Geo risk ({_geo_risk_level}) — LONG confirmed by safe-haven demand"
            elif _geo_risk_level == "elevated":
                _geo_note = "⚡ Elevated geo risk — monitor for escalation"

            _base_note = f"[−{risk_penalty} risk]" if risk_penalty else ""
            _combined_note = " | ".join(n for n in [_base_note, _geo_note, _geo_note_smc] if n)

            sig = {
                "source":        "rules",
                "asset":         asset,
                "direction":     direction,
                "confidence":    eff_c,
                "confidence_score": eff_c,
                "tier":          tier,
                "tier_label":    TIER_LABELS[tier],
                "pattern_name":  rule.get("name") or rule.get("pattern_name", "Unnamed"),
                "description":   rule.get("description", ""),
                "entry":         rule.get("entry_price", 0.0),
                "stop_loss":     rule.get("stop_loss",   0.0),
                "take_profit":   rule.get("take_profit",  0.0),
                "bt_win_rate":   round(wr, 1),
                "profit_factor": round(pf, 2),
                "live_win_rate": live_wr,
                "note":          _combined_note,
                "sl_atr_multiplier": _geo_sl_mult,
                "geo_risk_level":    _geo_risk_level,
                "confluence_met":    c_met,
                "confluence_missed": c_missed,
                "detail_lines":      detail_lines_sig,
                "checklist_results": None,
                "_confluence_raw":   _cr_raw,
                "smc_context":       sig_smc_store,
                "entry_quality":     (sig_smc_store or {}).get("entry_quality", None),
            }

            # Pattern history boost
            if _MC_OK and regime_data:
                try:
                    pname   = rule.get("name") or rule.get("pattern_name", "")
                    sess    = "London"  # default; could detect from time
                    hist    = get_pattern_win_rate(pname, regime_data["regime"], sess)
                    sig["hist_win_rate"]      = hist["win_rate"]
                    sig["hist_sample_size"]   = hist["sample_size"]
                    sig["confidence_boost"]   = hist["confidence_boost"]
                    sig["confidence"]         = min(10.0, sig["confidence"] + hist["confidence_boost"])
                    sig["regime"]             = regime_data["regime"]
                    sig["regime_label"]       = regime_data["regime_label"]
                    sig["size_multiplier"]    = regime_data["position_size_multiplier"]
                except Exception:
                    pass

            # ── SL distance pre-filter (min ATR×0.5) ─────────────────────────
            if df is not None and not df.empty:
                try:
                    _s = _load_settings()
                    _entry_p = float(sig.get("entry",     0) or 0)
                    _sl_p    = float(sig.get("stop_loss", 0) or 0)
                    if _entry_p and _sl_p:
                        _atr_val    = float(df["atr"].iloc[-1]) if "atr" in df.columns else 5.0
                        _sl_dist    = abs(_entry_p - _sl_p)
                        _min_sl     = _atr_val * 0.5
                        _conf_min   = _s.get("min_confidence", 7.5)
                        if _sl_dist < _min_sl:
                            meta["sl_rejected"] += 1
                            log_signal_rejection(
                                signal_name=sig["pattern_name"], direction=direction,
                                reason=f"SL distance ${_sl_dist:.2f} below ATR noise floor ${_min_sl:.2f}",
                                stage="sl_quality_check",
                                detail=f"ATR={_atr_val:.2f}, min_sl=ATR×0.5",
                            )
                            save_signal_detail(
                                sig, "REJECTED",
                                session=_current_session_str(),
                                gold_price=_entry_p,
                                d1_bias=meta.get("htf_overall","—"),
                                regime=meta.get("regime","—"),
                                rejection_reason=f"SL too tight (${_sl_dist:.2f} < ${_min_sl:.2f})",
                                rejection_stage="sl_quality_check",
                                spread_usd=meta.get("spread_check", {}).get("spread_usd"),
                            )
                            continue
                        if sig["confidence"] < _conf_min:
                            meta["conf_rejected"] += 1
                            log_signal_rejection(
                                signal_name=sig["pattern_name"], direction=direction,
                                reason=f"Confidence {sig['confidence']:.1f} below minimum {_conf_min}",
                                stage="confidence_filter",
                            )
                            save_signal_detail(
                                sig, "REJECTED",
                                session=_current_session_str(),
                                gold_price=_entry_p,
                                rejection_reason=f"Confidence too low ({sig['confidence']:.1f} < {_conf_min})",
                                rejection_stage="confidence_filter",
                                spread_usd=meta.get("spread_check", {}).get("spread_usd"),
                            )
                            continue
                except Exception as _fe:
                    log_error("morning_briefing", "sl_filter", str(_fe),
                              "SL filter skipped", "signal may have tight SL")

            # Entry checklist gate (need 4/5 or 5/5)
            if _EC_OK and df is not None:
                try:
                    ck = validate_entry(sig, df)
                    sig["checklist_results"] = ck
                    sig["confidence"]        = ck["final_confidence"]
                    log_checklist(sig["pattern_name"], direction, ck)
                    if ck["checks_passed"] >= 4:
                        # ── Fatigue gate ───────────────────────────────────
                        try:
                            from pattern_fatigue import check_strategy_fatigue
                            _fatigue_r = check_strategy_fatigue(
                                sig["pattern_name"],
                                df if df is not None else pd.DataFrame(),
                                direction
                            )
                            fatigue_level_r = _fatigue_r.get("fatigue_level", "none")
                            sig["fatigue_level"] = fatigue_level_r
                            sig["fatigue_recommendation"] = _fatigue_r.get("recommendation", "")
                            if fatigue_level_r == "critical":
                                log_rejected(f"{sig['pattern_name']} — fatigue_critical")
                                meta["fatigue_rejected"] = meta.get("fatigue_rejected", 0) + 1
                                save_signal_detail(sig, "REJECTED", rejection_reason="Pattern fatigue critical",
                                                  rejection_stage="fatigue_gate",
                                                  spread_usd=meta.get("spread_check", {}).get("spread_usd"))
                                continue
                            elif fatigue_level_r in ("high", "moderate"):
                                sig["confidence"] = max(0, sig["confidence"] - 0.5)
                                sig["note"] = sig.get("note", "") + f" ⚠ Fatigue {fatigue_level_r}"
                        except Exception as _fe_r:
                            log_error("morning_briefing", "fatigue_gate", str(_fe_r),
                                      "fatigue check skipped")
                        # ── /Fatigue gate ──────────────────────────────────
                        meta["rules_passed"] += 1
                        rules_signals.append(sig)
                        save_signal_detail(
                            sig, "SHOWN_TO_USER",
                            session=_current_session_str(),
                            gold_price=float(sig.get("entry", 0) or 0),
                            d1_bias=meta.get("htf_overall","—"),
                            h4_bias=meta.get("htf_overall","—"),
                            dxy_status=str((dxy_ctx or {}).get("dxy_trend","—")),
                            regime=meta.get("regime","—"),
                            checklist_result=ck,
                            confluence_result=sig.get("_confluence_raw", {}),
                            settings=_load_settings(),
                            spread_usd=meta.get("spread_check", {}).get("spread_usd"),
                        )
                    else:
                        reason = ck.get("rejection_reason", "checklist failed")
                        log_signal_rejection(
                            signal_name=sig["pattern_name"], direction=direction,
                            reason=reason, stage="entry_checklist",
                        )
                        save_signal_detail(
                            sig, "REJECTED",
                            session=_current_session_str(),
                            gold_price=float(sig.get("entry", 0) or 0),
                            checklist_result=ck,
                            rejection_reason=reason,
                            rejection_stage="entry_checklist",
                            spread_usd=meta.get("spread_check", {}).get("spread_usd"),
                        )
                        continue
                except Exception as _cke:
                    log_error("morning_briefing", "rules_checklist", str(_cke),
                              "signal included without checklist")
                    rules_signals.append(sig)
            else:
                rules_signals.append(sig)

        meta["rules_found"] = len(rules_signals) + (len(rules) - len(rules_signals))

    # ── STAGE 3 — Combine + deduplicate + rank ────────────────────────────────
    all_signals = playbook_signals + rules_signals

    # ── Apply dynamic ATR SL to every signal ─────────────────────────────────
    if _ATR_SL_OK and df is not None and not df.empty:
        _dyn_sess = _current_session_str()
        _dyn_reg  = (regime_data or {}).get("regime", "RANGING") if regime_data else "RANGING"
        _dyn_geo  = float((geo_ctx or {}).get("sl_atr_multiplier", 0.0)) if geo_ctx else 0.0
        for _s in all_signals:
            try:
                _entry_v = float(_s.get("entry") or 0)
                if _entry_v <= 0:
                    continue
                _dyn = _calc_dyn_sl(
                    df,
                    _s.get("direction", "long"),
                    _entry_v,
                    session        = _dyn_sess,
                    regime         = _dyn_reg,
                    geo_multiplier = _dyn_geo,
                    strategy_name  = _s.get("pattern_name", ""),
                )
                if _dyn:
                    _s["stop_loss"]        = _dyn["sl_price"]
                    _s["take_profit"]      = _dyn["tp2_price"]
                    _s["sl_breakdown"]     = _dyn["sl_breakdown"]
                    _s["volatility_state"] = _dyn["volatility_state"]
                    _s["atr_percentile"]   = _dyn["atr_percentile"]
                    _s["atr"]              = _dyn["atr_value"]
            except Exception:
                pass

    # Deduplicate: same direction on same asset → keep highest confidence
    seen: dict = {}
    for s in all_signals:
        k = (s["asset"], s["direction"])
        if k not in seen or s["confidence"] > seen[k]["confidence"]:
            seen[k] = s

    # ── Spread scoring ──────────────────────────────────────────────────────
    _sp_status = _spread_result.get("status", "unavailable")
    _sp_usd    = _spread_result.get("spread_usd")
    _sp_reason = _spread_result.get("reason", "")
    for s in seen.values():
        old_note = s.get("note", "")
        if _sp_status == "blocked":
            s["confidence"] = max(1.0, s["confidence"] - 1.0)
            s["note"] = (old_note + " | " if old_note else "") + \
                f"⛔ Spread too wide: {_sp_reason}"
        elif _sp_status == "warning" and _sp_usd is not None:
            s["note"] = (old_note + " | " if old_note else "") + \
                f"⚠ Wide spread: ${_sp_usd:.2f}"

    # ── Global news scoring ───────────────────────────────────────────────────
    _gnc = global_news_ctx or {}
    _gnc_bias = str(_gnc.get("gold_bias_from_news", "neutral")).lower()
    for s in seen.values():
        sig_dir = str(s.get("direction", "")).lower()   # "long" or "short"
        sig_gold_dir = "bullish" if sig_dir == "long" else "bearish"
        old_note = s.get("note", "")
        if _gnc_bias in ("volatile", "conflicted") or _gnc.get("volatility_warning"):
            s["confidence"] = max(1.0, s["confidence"] - 0.5)
            s["note"] = (old_note + " | " if old_note else "") + \
                "⚠ Global news volatile — wait for candle close"
        elif _gnc_bias == sig_gold_dir:
            s["confidence"] = min(10.0, s["confidence"] + 0.5)
            s["note"] = (old_note + " | " if old_note else "") + \
                "✓ Global news confirms direction"
        elif _gnc_bias in ("bullish", "bearish") and _gnc_bias != sig_gold_dir:
            s["confidence"] = max(1.0, s["confidence"] - 0.5)
            s["note"] = (old_note + " | " if old_note else "") + \
                "⚠ Global news opposes direction"

    ranked = sorted(seen.values(), key=lambda x: -x["confidence"])

    # ── Reversal Hunter — prepend STRONG reversals ────────────────────────────
    if _RH_MB_OK and df is not None:
        try:
            _rev_sigs = _hunt_reversals(df)
            _strong_r = [r for r in _rev_sigs if r["reversal_strength"] == "STRONG"]
            ranked    = _strong_r + ranked   # prepend strong reversals
        except Exception:
            pass

    # ── Regime strategy filter ────────────────────────────────────────────────
    if regime_data and _MC_OK:
        try:
            regime_config = get_regime_strategy_config(regime_data["regime"])
            meta["regime_config"] = regime_config
            for _rs in ranked:
                _ef   = regime_config["entry_filter"]
                _pn   = str(_rs.get("pattern_name", "")).lower()
                _src  = str(_rs.get("source", ""))
                _rnote = _rs.get("note", "")

                if _ef == "sr_bounce_only":
                    _sr_patterns = ["rsi oversold bounce", "double top", "double bottom",
                                    "fibonacci", "s/r bounce", "support", "resistance"]
                    if not any(p in _pn for p in _sr_patterns):
                        _rs["confidence"] = max(1.0, float(_rs["confidence"]) - 1.0)
                        _rs["note"] = (_rnote + " | " if _rnote else "") + \
                            "\u26a0 Not ideal for ranging market"

                elif _ef == "breakout_only":
                    if "breakout" not in _pn:
                        _rs["confidence"] = max(1.0, float(_rs["confidence"]) - 1.5)
                        _rs["note"] = (_rnote + " | " if _rnote else "") + \
                            "\u26a0 Wait for breakout in squeeze"

                elif _ef == "news_fade_only":
                    if "news" not in _pn:
                        _rs["confidence"] = max(1.0, float(_rs["confidence"]) - 1.0)
                        _rs["note"] = (_rnote + " | " if _rnote else "") + \
                            "\u26a0 High vol \u2014 news fade preferred"

                if _src == "reversal_hunter" and regime_data["regime"] == "TRENDING_STRONG":
                    _rs["confidence"] = max(1.0, float(_rs["confidence"]) - 0.5)
                    _rs["note"] = (_rnote + " | " if _rnote else "") + \
                        "\u26a0 COUNTER-TREND \u2014 dominant trend is TRENDING_STRONG \u2014 smaller size recommended"

                _rs["size_multiplier"] = regime_data["position_size_multiplier"]
                _rs["regime_config"]   = regime_config
            # Re-sort after confidence adjustments
            ranked.sort(key=lambda x: -float(x.get("confidence", 0)))
        except Exception:
            pass

    # ── STAGE 4 — Top 3 ──────────────────────────────────────────────────────
    top3 = ranked[:3]

    # ── Brain 1: apply auto-filters to top 3 ─────────────────────────────────
    if _auto_filters:
        for s in top3:
            pname = str(s.get("pattern_name", "")).lower()
            for af in _auto_filters:
                if pname == str(af.get("pattern", "")).lower():
                    n_times = int(af.get("times_triggered", 0))
                    s["confidence"] = max(1.0, s["confidence"] - 1.0)
                    flag = " 🚫 Pattern flagged — review" if n_times >= 5 else ""
                    old_note = s.get("note", "")
                    s["note"] = (
                        (old_note + " | " if old_note else "") +
                        f"⚠ Auto-filter: failed {n_times}x for {af.get('filter_reason','?')}"
                        + flag
                    )

    # ── Brain 2: register each top signal with signal_tracker ─────────────────
    if _ST_OK and top3:
        try:
            _spread_usd = meta.get("spread_check", {}).get("spread_usd") or 0.0
            _mtf4_sc    = meta.get("mtf4_score", 0)
            _gnews_bias = str((global_news_ctx or {}).get("gold_bias_from_news", ""))
            for s in top3:
                s["spread_at_signal"] = _spread_usd
                s["mtf_score"]        = _mtf4_sc
                s["global_news_bias"] = _gnews_bias
                s["signal_id"]        = _st_register(s)
        except Exception:
            pass  # never break the briefing

    return top3, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Session summary
# ═══════════════════════════════════════════════════════════════════════════════

CHECK = "\u2713"
CROSS = "\u2717"


def _print_signal_card(
    idx:      int,
    sig:      dict,
    patterns_data: dict,
) -> None:
    """Print the detailed card for one signal."""
    asset     = sig["asset"]
    direction = sig["direction"].upper()
    pattern   = sig["pattern_name"]
    source    = sig.get("source", "rules")
    src_tag   = f"Playbook" if source == "playbook" else f"Rules DB · Tier {sig.get('tier','?')}"
    conf      = sig["confidence"]
    entry_p   = sig["entry"]
    sl_p      = sig["stop_loss"]
    tp_p      = sig["take_profit"]
    c_met     = sig.get("confluence_met",    [])
    c_missed  = sig.get("confluence_missed", [])
    ck        = sig.get("checklist_results") or {}
    ck_passed = ck.get("checks_passed", "?") if ck else "?"
    ck_ok     = CHECK if (isinstance(ck_passed, int) and ck_passed >= 4) else (CROSS if ck else "?")
    bt_wr     = sig.get("bt_win_rate", 0)
    desc      = sig.get("description", "") or ""

    W = 51
    title   = f"SETUP {idx} \u2014 {asset} {direction}"
    sub     = f"Strategy: {pattern} ({src_tag})"

    # ── Live price header (always fresh from system clock) ──────────────────
    _now_uae   = datetime.now(GST)
    _date_s    = _now_uae.strftime("%A %d %B %Y")
    _time_s    = _now_uae.strftime("%I:%M %p UAE")
    _sess_line = _ws_summary_line() if _WS_OK else _current_session_str()
    try:
        _lp = _get_live_price()
        _lp_val  = _lp["price"] if _lp.get("price") and _lp["price"] > 0 else entry_p
        _lp_src  = _lp.get("source", "—")
        _lp_warn = f"  ⚠ {_lp['stale_warning']}" if _lp.get("stale_warning") else ""
    except Exception:
        _lp_val, _lp_src, _lp_warn = entry_p, "—", ""

    print(f"\n  {'═' * W}")
    print(f"  📅 {_date_s}   🕐 {_time_s}")
    print(f"  💰 Live: ${_lp_val:,.2f}  [{_lp_src}]")
    print(f"  🌐 {_sess_line}")
    if _lp_warn:
        print(f"{_lp_warn}")
    print(f"  {'═' * W}")
    print(f"  {title}")
    print(f"  {sub}")
    print(f"  {'═' * W}")

    # Entry / SL / TP
    if isinstance(entry_p, (int, float)) and entry_p:
        rr_num  = sig.get("profit_factor") or 0
        # Use playbook risk_reward if available
        if source == "playbook":
            from strategy_playbooks import PLAYBOOKS
            pb = PLAYBOOKS.get(sig.get("playbook_id", ""), {})
            rr_num = pb.get("risk_reward", 0) or 0
        risk_dist = abs(entry_p - sl_p) if sl_p else 0
        rew_dist  = abs(entry_p - tp_p) if tp_p else 0
        rr_calc   = round(rew_dist / risk_dist, 1) if risk_dist else rr_num
        sl_sign   = "-" if direction == "LONG" else "+"
        tp_sign   = "+" if direction == "LONG" else "-"
        # XAUUSD: show dollar distances; FX: show pips
        is_xau_sig = "XAU" in (sig.get("asset", "") or "") or "XAU" in (sig.get("symbol", "") or "")
        print(f"  Entry:       ${entry_p:>10,.2f}")
        if is_xau_sig:
            sl_dist = round(risk_dist, 2)
            tp_dist = round(rew_dist, 2)
            print(f"  Stop Loss:   ${sl_p:>10,.2f}  ({sl_sign}${sl_dist:,.2f})")
            print(f"  Take Profit: ${tp_p:>10,.2f}  ({tp_sign}${tp_dist:,.2f})")
        else:
            sl_pips = round(risk_dist / PIP_SIZE, 0) if sl_p else 0
            tp_pips = round(rew_dist  / PIP_SIZE, 0) if tp_p else 0
            print(f"  Stop Loss:   ${sl_p:>10,.2f}  ({sl_sign}{sl_pips:.0f} pips)")
            print(f"  Take Profit: ${tp_p:>10,.2f}  ({tp_sign}{tp_pips:.0f} pips)")
        print(f"  Risk/Reward: 1:{rr_calc}")
    else:
        print(f"  Entry:       Market price")
        print(f"  Stop Loss:   As per pattern")
        print(f"  Take Profit: As per pattern")
    print(f"  Confidence:  {conf}/10")

    # Confluence section
    print()
    print(f"  CONFLUENCE:")
    detail_lines = sig.get("detail_lines", [])
    if detail_lines:
        for ln in detail_lines:
            print(f"  {ln}")
    else:
        all_checks = c_met + c_missed
        if not all_checks:
            all_checks = ["Trend", "Momentum", "Structure", "Candle", "Session", "Volatility"]
            c_met_set  = set()
        else:
            c_met_set = set(c_met)
        for chk in all_checks:
            icon = CHECK if chk in c_met_set else CROSS
            print(f"  {icon} {chk}")
        if not all_checks:
            print(f"  (Confluence data unavailable)")

    # Checklist result
    print()
    if ck:
        print(f"  CHECKLIST: {ck_passed}/5 PASSED {ck_ok}")
        if not ck.get("passed"):
            print(f"  Reason: {ck.get('rejection_reason', '—')}")
    else:
        print(f"  CHECKLIST: not run")

    # Why this trade
    print()
    print(f"  WHY THIS TRADE:")
    if desc:
        words = desc.split()
        line  = ""
        for w in words:
            if len(line) + len(w) + 1 <= 48:
                line = (line + " " + w).strip()
            else:
                print(f"  {line}")
                line = w
        if line:
            print(f"  {line}")
    if bt_wr:
        print(f"  Historical win rate: {bt_wr}%")

    # Historical match (from patterns data if available)
    outcomes  = (patterns_data or {}).get("outcomes", {})
    n_match   = outcomes.get("total_matches", 0)
    verdict   = outcomes.get("verdict", "")
    exp_pips  = outcomes.get("expected_pips", 0)
    if n_match > 0:
        print()
        print(f"  HISTORICAL MATCH:")
        sign  = "+" if exp_pips >= 0 else ""
        print(f"  {n_match} similar setups  →  {sign}{exp_pips} pips expected")
        print(f"  Bias: {verdict.split()[0] if verdict else '—'}")

    # Regime + pattern memory
    regime_lbl = sig.get("regime_label", "")
    if regime_lbl:
        print()
        print(f"  REGIME:  {regime_lbl}")
        size_mult = sig.get("size_multiplier", 1.0)
        if size_mult != 1.0:
            print(f"  Lot size multiplier: ×{size_mult:.1f}")
    hist_wr = sig.get("hist_win_rate")
    hist_n  = sig.get("hist_sample_size", 0)
    boost   = sig.get("confidence_boost", 0.0)
    if hist_n and hist_n > 0:
        print()
        print(f"  PATTERN HISTORY:  {hist_n} similar setups → {hist_wr:.0%} win rate"
              + (f"  (+{boost:.1f} boost)" if boost else ""))

    # Volume section
    vol = sig.get("volume", {})
    if vol:
        print()
        print(f"  VOLUME:")
        print(f"  Ratio: {vol.get('volume_ratio', '?')}x avg  |  Class: {vol.get('volume_class', '?')}")
        for vln in vol.get("details", []):
            print(f"  {vln}")
        if vol.get("climax"):
            print(f"  ⚠  Volume climax detected — exhaustion signal present")

    print(f"  {'═' * W}")


def _print_session_summary(
    rules_data:      dict,
    news_data:       dict,
    live_data:       dict,
    patterns_data:   dict,
    signals:         list[dict],
    scan_meta:       dict | None = None,
    htf_ctx:         dict | None = None,
    dxy_ctx:         dict | None = None,
    global_news_ctx: dict | None = None,
) -> None:
    sentiment    = news_data.get("sentiment", {}) if isinstance(news_data, dict) else {}
    gold_bias    = str((sentiment.get("gold") or {}).get("bias", "wait")).upper()
    overall_risk = str(sentiment.get("overall_risk", "unknown")).upper()
    key_event    = news_data.get("key_event") if isinstance(news_data, dict) else "—"
    if not key_event:
        key_event = sentiment.get("key_event_today", "None identified")
    if scan_meta is None:
        scan_meta = {}

    mood = {"BUY": "Bullish", "SELL": "Bearish", "BULLISH": "Bullish", "BEARISH": "Bearish"}.get(gold_bias, "Neutral")

    event_warning = any(kw in str(key_event).lower() for kw in WARNING_KEYWORDS)
    safe_to_trade = "YES" if overall_risk != "HIGH" and not event_warning else "NO"

    # Historical summary (used in header + signal cards)
    outcomes      = patterns_data.get("outcomes", {}) if isinstance(patterns_data, dict) else {}
    hist_matches  = outcomes.get("total_matches", 0)
    hist_verdict  = outcomes.get("verdict", "")
    hist_up_pct   = outcomes.get("up_pct",  0)
    hist_dn_pct   = outcomes.get("down_pct", 0)

    if hist_matches > 0:
        if "BULLISH" in str(hist_verdict):
            hist_bias_str = f"Bullish {hist_up_pct}% based on {hist_matches} similar past setups"
        elif "BEARISH" in str(hist_verdict):
            hist_bias_str = f"Bearish {hist_dn_pct}% based on {hist_matches} similar past setups"
        else:
            hist_bias_str = f"Mixed — {hist_matches} similar setups"
    else:
        hist_bias_str = "N/A (run setup.py first)"

    # Weekly stats
    trades      = _load_trade_log()
    week_start  = (datetime.now(timezone.utc)
                   .replace(hour=0, minute=0, second=0, microsecond=0)
                   - timedelta(days=datetime.now(timezone.utc).weekday()))
    week_trades = []
    for t in trades:
        if t.get("status") != "closed":
            continue
        try:
            dt = datetime.fromisoformat(str(t.get("closed_datetime") or t.get("datetime", "")).replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= week_start.replace(tzinfo=timezone.utc):
                week_trades.append(t)
        except Exception:
            pass
    week_total = len(week_trades)
    week_wins  = sum(1 for t in week_trades if t.get("result") == "WIN")
    week_pips  = sum(t.get("pips", 0) or 0 for t in week_trades)
    week_wr    = round(week_wins / week_total * 100, 1) if week_total else 0
    pat_wins   = {}
    for t in week_trades:
        if t.get("result") == "WIN":
            pn = t.get("pattern_name", "?")
            pat_wins[pn] = pat_wins.get(pn, 0) + 1
    best_pat = max(pat_wins, key=lambda k: pat_wins[k]) if pat_wins else "N/A"

    print("\n" + "═" * 58)
    print("=== YOUR TRADING BRIEFING ===")
    print(f"=== {_day_str()} | {datetime.now(GST).strftime('%H:%M')} GST ===")
    print("═" * 58)

    # ── World session board ───────────────────────────────────────────────────
    if _WS_OK:
        print()
        print(_ws_board())
        print()

    # ── Live price line ───────────────────────────────────────────────────────
    try:
        _lp2 = _get_live_price()
        if _lp2.get("price") and _lp2["price"] > 0:
            _lp_src2 = _lp2.get("source", "—")
            _lp_ts2  = _lp2.get("timestamp_uae", "—")
            print(f"  💰 LIVE PRICE: ${_lp2['price']:,.2f}  [{_lp_src2}]  @ {_lp_ts2}")
            if _lp2.get("stale_warning"):
                print(f"  ⚠️  {_lp2['stale_warning']}")
    except Exception:
        pass

    print(f"\n  MARKET MOOD       : {mood}")
    print(f"  HISTORICAL BIAS   : {hist_bias_str}")
    print(f"  SAFE TO TRADE     : {safe_to_trade}")
    print(f"  RISK LEVEL        : {overall_risk}")

    # ── Session profile block ─────────────────────────────────────────────────
    try:
        from session_profiler import get_current_session_profile as _gsp_brief
        _spr  = _gsp_brief()
        _sn   = _spr.get("current_session", "London")
        _sg   = _spr.get("session_grade",   "B")
        _slm  = _spr.get("lot_multiplier",  1.0)
        _ssm  = _spr.get("sl_multiplier",   1.0)
        _stm  = _spr.get("tp_multiplier",   1.0)
        _srec = _spr.get("trading_recommended", True)
        _spips= _spr.get("avg_move_pips",   0.0)
        _swrl = _spr.get("win_rate_long",   0.0)
        _swrs = _spr.get("win_rate_short",  0.0)
        _trade_str = "RECOMMENDED" if _srec else ("CAUTION" if _sn == "Asian" else "AVOID")
        _lot_lbl  = "normal" if _slm == 1.0 else ("reduced" if _slm < 1.0 else "increased")
        _sl_lbl   = "normal" if _ssm == 1.0 else ("wider"   if _ssm > 1.0 else "tighter")
        _tp_lbl   = "normal" if _stm == 1.0 else ("extended" if _stm > 1.0 else "reduced")
        print(f"\n  SESSION PROFILE   : [{_sn}] Grade [{_sg}]")
        print(f"  Lot multiplier    : x{_slm:.1f} ({_lot_lbl})")
        print(f"  SL multiplier     : x{_ssm:.1f} ({_sl_lbl})")
        print(f"  TP multiplier     : x{_stm:.1f} ({_tp_lbl})")
        if _spips > 0:
            print(f"  Avg move          : {_spips:.1f} pips | Win rate long: {_swrl:.0f}% short: {_swrs:.0f}%")
        print(f"  Trading           : {_trade_str}")
    except Exception:
        pass   # session_profiler not available — skip block

    # ── Market regime block ───────────────────────────────────────────────────
    _regime_key   = scan_meta.get("regime", "")
    _regime_lbl   = scan_meta.get("regime_label", "")
    _regime_note  = scan_meta.get("regime_note", "")
    _regime_mult  = scan_meta.get("size_mult", 1.0)
    _regime_best  = scan_meta.get("best_playbooks", [])
    _regime_avoid = scan_meta.get("avoid_playbooks", [])
    if _regime_key:
        print(f"\n  MARKET REGIME     : {_regime_lbl}")
        print(f"  Size multiplier   : x{_regime_mult:.1f}")
        if _regime_best:
            print(f"  Best strategies   : {', '.join(_regime_best[:3])}")
        if _regime_avoid:
            print(f"  Avoid             : {', '.join(_regime_avoid[:3])}")
        print(f"  Note              : {_regime_note}")

    # Session handoff summary
    _sh_meta = scan_meta.get("session_handoff", {})
    if _sh_meta and _sh_meta.get("summary"):
        print(f"\n  SESSION HANDOFF   : {_sh_meta['summary']}")
        print(f"  \u2192 {_sh_meta.get('recommendation', '')}")

    # ── KEY LEVELS (S/R map) ──────────────────────────────────────────────────
    try:
        from sr_mapper import get_sr_levels as _get_sr_lvls
        _live_p = live_data.get("price", 0) if isinstance(live_data, dict) else 0
        _live_df = live_data.get("df") if isinstance(live_data, dict) else None
        if _live_df is not None and _live_p and float(_live_p) > 0:
            _sr = _get_sr_lvls(_live_df, float(_live_p))
            _SL = "─" * 38
            print(f"\n  KEY LEVELS TODAY:")
            print(f"  {_SL}")
            _res = _sr.get("resistance_levels", [])[:3]
            _sup = _sr.get("support_levels",    [])[:3]
            for _rl in _res:
                _dist = f"+${_rl['distance_usd']:,.1f}"
                _str  = _rl["strength"][:3].upper()
                print(f"  Resistance:  ${_rl['price']:>10,.2f}  ({_rl['label'][:30]})  [{_str}] {_dist}")
            print(f"\n  Current:  →  ${float(_live_p):,.2f}\n")
            for _sl in _sup:
                _dist = f"-${_sl['distance_usd']:,.1f}"
                _str  = _sl["strength"][:3].upper()
                print(f"  Support:     ${_sl['price']:>10,.2f}  ({_sl['label'][:30]})  [{_str}] {_dist}")
            print(f"  {_SL}")
            _pw_h = _sr.get("prev_week_high", 0)
            _pw_l = _sr.get("prev_week_low",  0)
            _pd_h = _sr.get("prev_day_high",  0)
            _pd_l = _sr.get("prev_day_low",   0)
            if _pw_h or _pw_l:
                print(f"  Prev Week: High ${_pw_h:,.2f} | Low ${_pw_l:,.2f}")
            if _pd_h or _pd_l:
                print(f"  Prev Day:  High ${_pd_h:,.2f} | Low ${_pd_l:,.2f}")
            if _sr.get("at_key_level"):
                print(f"  ⭐ AT KEY LEVEL: {_sr.get('key_level_detail', '')}")
    except Exception:
        pass  # S/R map is non-critical

    # Liquidity map block
    if _LIQ_MB_OK:
        try:
            _live_df_liq = live_data.get("df") if isinstance(live_data, dict) else None
            _live_p_liq  = live_data.get("price", 0) if isinstance(live_data, dict) else 0
            if _live_df_liq is not None and _live_p_liq and float(_live_p_liq) > 0:
                _liq_mb = _build_liq_mb(_live_df_liq, float(_live_p_liq))
                if _liq_mb.get("available"):
                    _liq_ca  = _liq_mb.get("clusters_above", [])
                    _liq_cb  = _liq_mb.get("clusters_below", [])
                    _liq_poc = _liq_mb.get("poc", 0.0)
                    _liq_mv  = _liq_mb.get("likely_move", "NEUTRAL")
                    _liq_rsn = _liq_mb.get("likely_reason", "")
                    print(f"\n  LIQUIDITY MAP:")
                    print(f"  {'─'*38}")
                    for _lc in _liq_ca[:3]:
                        print(f"  🔴 BSL above: ${_lc['price']:>10,.2f}  (+${_lc['distance_usd']:,.1f})  [{_lc['count']} swings]")
                    print(f"  → CURRENT: ${float(_live_p_liq):,.2f}")
                    for _lc in _liq_cb[:3]:
                        print(f"  🟢 SSL below: ${_lc['price']:>10,.2f}  (−${_lc['distance_usd']:,.1f})  [{_lc['count']} swings]")
                    if _liq_poc:
                        print(f"  POC: ${_liq_poc:,.2f}  |  VA: ${_liq_mb.get('va_low',0):,.2f}–${_liq_mb.get('va_high',0):,.2f}")
                    _arr = "⬆" if _liq_mv == "UP" else ("⬇" if _liq_mv == "DOWN" else "↔")
                    print(f"  {_arr} Likely move: {_liq_mv} — {_liq_rsn}")
        except Exception:
            pass  # liquidity map is non-critical

    if not isinstance(live_data, dict) or "error" in live_data:
        print(f"  LIVE PRICE        : [SKIPPED - timeout]")
    else:
        print(
            f"  LIVE PRICE        : ${live_data.get('price', '—')}  |  "
            f"RSI: {live_data.get('rsi', '—')} ({live_data.get('rsi_zone', '—')})  |  "
            f"Trend: {live_data.get('trend', '—')}  |  ATR: ${live_data.get('atr', live_data.get('atr_pips', '—'))}"
        )

    # MTF bias block
    if htf_ctx and htf_ctx.get("available"):
        print()
        print_htf_report(htf_ctx)
    elif htf_ctx:
        print(f"\n  MTF BIAS: {htf_ctx.get('bias_line', 'unavailable')}")

    # DXY correlation block
    if dxy_ctx and dxy_ctx.get("available") and _DXY_OK:
        # Determine dominant gold direction from top signal for print
        gold_dir = "long"
        if signals:
            gold_dir = str(signals[0].get("direction", "long")).lower()
        print_dxy_report(gold_dir, dxy_ctx)
    elif dxy_ctx:
        print(f"\n  DXY: {dxy_ctx.get('display_line', 'unavailable')}")

    # COT block
    if _COT_MB_OK:
        try:
            _cot_mb = _fetch_cot_mb()
            if _cot_mb.get("available"):
                print(f"\n  COT POSITIONING (Gold Futures):")
                print(f"  {_cot_mb.get('display_line', '')}")
                _cot_hed = _cot_mb.get("hedger_note", "")
                if _cot_hed:
                    print(f"  Hedger signal: {_cot_hed}")
        except Exception:
            pass

    # COT block
    if _COT_MB_OK:
        try:
            _cot_mb = _fetch_cot_mb()
            if _cot_mb.get("available"):
                print(f"\n  COT POSITIONING (Gold Futures):")
                print(f"  {_cot_mb.get('display_line', '')}")
                _cot_hed = _cot_mb.get("hedger_note", "")
                if _cot_hed:
                    print(f"  Hedger signal: {_cot_hed}")
        except Exception:
            pass
        print(f"\n  {WARN}  HIGH-IMPACT EVENT: {key_event}")
        print("     Avoid entering trades 30 min before/after.")

    # Top setups
    # ── SIGNALS TODAY header ──────────────────────────────────────────────────
    pb_found  = scan_meta.get("playbook_found",  0)
    pb_passed = scan_meta.get("playbook_passed", 0)
    rl_passed = scan_meta.get("rules_passed",    0)
    active_pb = scan_meta.get("active_playbooks", [])
    strongest = scan_meta.get("strongest", None)
    total_checked = pb_found + len(rules_data.get("rules", []))

    # ── Risk of Ruin summary ─────────────────────────────────────────────────
    try:
        from trade_manager import get_current_risk_profile as _get_ror, format_ror_report as _fmt_ror
        _ror_brief = _get_ror()
        _ror_rating = _ror_brief.get("risk_rating", "UNKNOWN")
        _ror_prob   = _ror_brief.get("ruin_probability", 0.0)
        _ror_safe   = _ror_brief.get("recommended_risk_pct", _ror_brief.get("risk_pct", 10))
        print(
            f"\n  RISK RATING: {_ror_rating} "
            f"({_ror_prob:.1f}% ruin | "
            f"{_ror_safe}% safe risk)"
        )
        if _ror_rating == "DANGER":
            print()
            for _ror_line in _fmt_ror(_ror_brief).split("\n"):
                print(f"  {_ror_line}")
            print()
        elif _ror_rating == "HIGH":
            print(f"  → {_ror_brief.get('recommendation', '')}")
    except Exception:
        pass  # trade_manager not available — skip RoR block

    print("\n  SIGNALS TODAY:")
    print(f"  Playbook signals found   : {pb_found}")
    print(f"  Rules signals found      : {rl_passed}")
    print(f"  Passed checklist         : {len(signals)}")
    print(f"  Shown to you             : top {min(3, len(signals))} only")

    print("\n  STRATEGY COVERAGE:")
    if active_pb:
        print(f"  Active playbooks : {', '.join(active_pb[:4])}{'...' if len(active_pb) > 4 else ''}")
    else:
        print(f"  Active playbooks : none triggered")
    if strongest:
        print(f"  Strongest setup  : {strongest}")
    else:
        print(f"  Strongest setup  : —")

    # ── Signal cards ─────────────────────────────────────────────────────────
    print("\n  TOP SETUPS TODAY:")
    if signals:
        for i, sig in enumerate(signals, start=1):
            _print_signal_card(i, sig, patterns_data)
    else:
        print("\n  No setups meet quality thresholds today.")
        print("  Possible reasons:")
        print("  • Gold bias is 'wait' (no directional sentiment)")
        print("  • Checklist blocked all signals (e.g. weekend or poor confluence)")
        print("  • Rules database empty — run setup.py --refresh")

    # ── Reversal opportunities block ────────────────────────────────────
    _rev_signals = [s for s in signals if s.get("source") == "reversal_hunter"]
    if _rev_signals:
        print("\n  🔄 REVERSAL OPPORTUNITIES:")
        for rev in _rev_signals:
            print(f"  {rev['reversal_strength']}: {rev['direction'].upper()} "
                  f"from ${rev['entry']:,.2f} | Score {rev['score']}/11")
            print(f"  Reason: {rev['key_reason']}")

    # Recommended action
    print("\n  RECOMMENDED ACTION:")
    if safe_to_trade == "NO":
        print("  Stay cautious. High-risk event or high-impact news today.")
        print("  Wait for the event to pass before entering trades.")
    elif not signals:
        print("  No clear setups. Monitor charts and wait for confirmation.")
    else:
        top    = signals[0]
        action = "BUY" if top["direction"] == "long" else "SELL"
        ck     = top.get("checklist_results") or {}
        ck_n   = ck.get("checks_passed", "?")
        src    = "playbook" if top.get("source") == "playbook" else f"Tier {top.get('tier','?')} rule"

        # ── Determine primary bias label ──────────────────────────────────────
        _htf_meta  = (scan_meta or {})
        _htf_ov    = str(_htf_meta.get("htf_overall", "")).lower()
        _htf_str   = str(_htf_meta.get("htf_strength", "")).lower()
        if _htf_ov == "bearish" and _htf_str == "strong":
            _bias_line = "Primary bias: SHORT (D1+H4 bearish)"
        elif _htf_ov == "bullish" and _htf_str == "strong":
            _bias_line = "Primary bias: LONG (D1+H4 bullish)"
        else:
            _bias_line = f"Primary bias: {_htf_ov.upper() or 'MIXED'}"

        print(f"  {_bias_line}")

        # ── Primary trend setup ───────────────────────────────────────────────
        print(
            f"  {action} {top['asset']} — {top['pattern_name']} "
            f"(conf {top['confidence']}/10, {src}).\n"
            f"  Checklist: {ck_n}/5. Confirm on chart before entry."
        )
        if len(signals) > 1:
            s2  = signals[1]
            ck2 = s2.get("checklist_results") or {}
            print(f"  Alt: {'BUY' if s2['direction']=='long' else 'SELL'} "
                  f"{s2['asset']} — {s2['pattern_name']} (conf {s2['confidence']}/10, {ck2.get('checks_passed','?')}/5).")

        # ── Reversal opportunities note ───────────────────────────────────────
        _rev_sigs_ra = [s for s in signals if s.get("source") == "reversal_hunter"]
        if _rev_sigs_ra:
            print(f"\n  Reversal longs: monitored separately")
        elif _htf_ov == "bearish" and _htf_str == "strong":
            print(f"\n  Reversal longs: monitored separately")

    # Weekly stats
    print("\n  YOUR STATS THIS WEEK:")
    if week_total > 0:
        print(f"  Win rate: {week_wr}%  |  Pips: {'+' if week_pips >= 0 else ''}{week_pips:.1f}  |  "
              f"Trades: {week_wins}W/{week_total - week_wins}L  |  Best pattern: {best_pat}")
    else:
        print("  No closed trades this week yet.")

    # Global news context block
    _gnc = global_news_ctx or {}
    _gnc_bias = _gnc.get("gold_bias_from_news", "neutral").upper()
    _gnc_conf = _gnc.get("confidence", 5.0)
    _gnc_key  = _gnc.get("key_event", "—")
    _gnc_rec  = _gnc.get("recommendation", "")
    _gnc_hl   = _gnc.get("all_headlines", [])
    print("\n  GLOBAL NEWS:")
    print(f"  {_gnc_bias} ({_gnc_conf:.1f}/10)")
    print(f"  Key event   : {_gnc_key}")
    if _gnc.get("volatility_warning"):
        print(f"  ⚠ VOLATILITY triggers: {', '.join(_gnc.get('volatility_triggers', [])[:3])}")
    if _gnc.get("bullish_triggers"):
        print(f"  Bullish triggers : {', '.join(_gnc['bullish_triggers'][:2])}")
    if _gnc.get("bearish_triggers"):
        print(f"  Bearish triggers : {', '.join(_gnc['bearish_triggers'][:2])}")
    print(f"  {_gnc_rec}")
    if _gnc_hl:
        print("  Top headlines:")
        for _h in _gnc_hl[:3]:
            print(f"    · {str(_h)[:90]}")

    # News summary
    print("\n  NEWS SUMMARY:")
    if isinstance(news_data, dict) and "error" not in news_data:
        print(f"  Key event today   : {key_event}")
        gold_summ = str((sentiment.get("gold") or {}).get("summary", ""))[:100]
        print(f"  Gold sentiment    : {gold_bias}  —  {gold_summ}")
        print(f"  Headlines scanned : {len(news_data.get('items', []))}")
    else:
        print(f"  News: [SKIPPED - timeout]")

    # Open trades check
    open_trades = [t for t in _load_trade_log() if t.get("status") == "open"]
    if open_trades:
        print(f"\n  {WARN}  You have {len(open_trades)} open trade(s). Run learning.py to close them.")

    # ── Signal quality summary ────────────────────────────────────────────────
    _s = _load_settings()
    _b      = _s.get("balance",        300)
    _rp     = _s.get("risk_pct",        10)
    _lev    = _s.get("leverage",        20)
    _rr     = _s.get("implied_rr",       3)
    _mc     = _s.get("min_confidence", 7.5)
    _sl_rej  = (scan_meta or {}).get("sl_rejected",   0)
    _cf_rej  = (scan_meta or {}).get("conf_rejected",  0)
    _vol_rej = (scan_meta or {}).get("vol_rejected",   0)
    _total_checked = (
        (scan_meta or {}).get("playbook_found", 0) +
        (scan_meta or {}).get("rules_found",    0)
    )
    print("\n  SIGNAL QUALITY AT YOUR SETTINGS:")
    print(f"  Balance: ${_b}  |  Risk: {_rp}%  |  RR: 1:{_rr}  |  Leverage: {_lev}x")
    print()
    print(f"  Signals scanned today        : {_total_checked}")
    print(f"  Rejected (SL too tight)      : {_sl_rej}")
    print(f"  Rejected (low confidence)    : {_cf_rej}")
    print(f"  Rejected (volume)            : {_vol_rej}")
    print(f"  Valid signals shown          : {len(signals)}")
    print()
    print("  Fewer signals = higher quality.")
    print(f"  Every signal shown has passed SL structure check +")
    print(f"  {_mc} confidence threshold + 1:{_rr} minimum RR.")

    # WFO summary
    if _WFO_MB_OK:
        try:
            _wfo_txt = _get_wfo_summary_mb()
            if _wfo_txt:
                print()
                for _wfo_line in _wfo_txt.split("\n"):
                    print(f"  {_wfo_line}")
        except Exception:
            pass

    print("\n" + "═" * 58)
    print("  Bot ready. Good luck with your session.")
    print("═" * 58)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    print()
    print("  ╔" + "═" * 44 + "╗")
    print("  ║" + "      TRADING BOT - DAILY BRIEFING      ".center(44) + "║")
    print("  ║" + f"  {_day_str()} | {datetime.now(GST).strftime('%H:%M')} GST  ".center(44) + "║")
    print("  ╚" + "═" * 44 + "╝")
    print()

    # ── Sunday Walk-Forward Optimization (auto) ───────────────────────────────
    if _WFO_MB_OK:
        try:
            if _check_wfo_mb():
                wfo_result = _run_wfo_mb()
                if wfo_result.get("changes_made"):
                    print("🔄 AUTO-OPTIMIZATION COMPLETE")
                    print(f"   {len(wfo_result['changes'])} settings updated")
                    for c in wfo_result["changes"]:
                        print(f"   • {c}")
                    print()
        except Exception:
            pass

    # ── MT5 Account & Trade Sync ──────────────────────────────────────────────
    if _MT5_OK:
        _mt5_sp = Spinner("[MT5] Syncing account + trade history...", indent=2).start()
        try:
            acct = _mt5_account()
            if acct:
                pnl_data  = _mt5_today_pnl()
                positions = _mt5_positions()
                new_n, total_j = _mt5_sync_journal(days_back=30)
                auto_notes     = _mt5_auto_match(days_back=2)
                _mt5_sp.stop(success=True)

                print(f"       Account #{acct['account']}  {acct['currency']}  1:{acct['leverage']}")
                print(f"       Balance: ${acct['balance']:,.2f}   Equity: ${acct['equity']:,.2f}   "
                      f"Free: ${acct['margin_free']:,.2f}")
                pnl_val  = pnl_data.get("pnl", 0)
                pnl_sign = "+" if pnl_val >= 0 else ""
                print(f"       Today P&L: {pnl_sign}${pnl_val:,.2f}  "
                      f"({pnl_data.get('wins',0)}W / {pnl_data.get('losses',0)}L  "
                      f"{pnl_data.get('trades',0)} trades)")
                if positions:
                    print(f"       Open positions: {len(positions)}")
                    for pos in positions:
                        p_sign = "+" if pos["pnl_usd"] >= 0 else ""
                        print(f"         {pos['direction'].upper()} {pos['symbol']}  "
                              f"{pos['lots']} lots @ ${pos['entry']:,.2f}  "
                              f"P&L: {p_sign}${pos['pnl_usd']:,.2f}")
                if new_n:
                    print(f"       Journal: +{new_n} new trades synced ({total_j} total)")
                if auto_notes:
                    for n in auto_notes:
                        print(f"       ★ Auto-matched: {n['symbol']} {n['direction']} "
                              f"→ {n['outcome']} {n['pnl']}  (pattern memory updated)")
            else:
                _mt5_sp.stop(success=False, suffix="MT5 offline — open MT5 & log in to sync")
        except Exception as _mt5_ex:
            _mt5_sp.stop(success=False, suffix=f"MT5 error: {_mt5_ex}")
    print()


    rules_data    = {}
    news_data     = {}
    live_data     = {}
    patterns_data = {}
    signals:list  = []
    htf_ctx:dict  = {}
    dxy_ctx:dict  = {}

    # ── [1/5] Load rules database ─────────────────────────────────────────────
    t_step = time.time()
    sp = Spinner("[1/5] Loading rules database...", indent=2).start()
    rules_data, elapsed, timed_out = _run_with_timeout(_step1_load_rules, STEP_TIMEOUT, "rules")
    if timed_out or not rules_data:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        rules_data = {"rules": [], "above_threshold": 0}
    else:
        sp.stop(success=True)
    n_rules = len(rules_data.get("rules", []))
    n_above = rules_data.get("above_threshold", 0)
    tc      = rules_data.get("tier_counts", {})
    print(f"       {_bar(100)}  ({_fmt_time(elapsed)})")
    print(f"       {n_rules} rules loaded")
    print(f"       Tier A: {tc.get('A',0)}  |  Tier B: {tc.get('B',0)}  |  Tier C: {tc.get('C',0)}  |  Tier D: {tc.get('D',0)}")
    print()

    # ── [2/5] Fetch market news ───────────────────────────────────────────────
    sp = Spinner("[2/5] Fetching market news...", indent=2).start()
    news_data, elapsed, timed_out = _run_with_timeout(_step2_fetch_news, 30, "news")
    if timed_out:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        news_data = {"error": "timeout"}
    elif not news_data or "error" in news_data:
        sp.stop(success=False, suffix=str((news_data or {}).get("error", "failed")))
        news_data = {"error": "failed"}
    else:
        sp.stop(success=True)
    if isinstance(news_data, dict) and "error" not in news_data:
        total_h = len(news_data.get("items", []))
        gold_h  = news_data.get("gold_count", 0)
        print(f"       {_bar(100)}  ({_fmt_time(elapsed)})")
        print(f"       {total_h} headlines  |  {gold_h} gold relevant")
        print(f"       Sentiment: {news_data.get('gold_bias','?')} ({news_data.get('conf','—')}/10)")
    else:
        print(f"       {_bar(0)}  [SKIPPED - timeout]")
    print()

    # ── [3/5] Live market data ────────────────────────────────────────────────
    sp = Spinner("[3/5] Getting live market data...", indent=2).start()
    live_data, elapsed, timed_out = _run_with_timeout(_step3_live_data, STEP_TIMEOUT, "live")
    if timed_out:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        live_data = {"error": "timeout"}
    elif not live_data or "error" in live_data:
        sp.stop(success=False, suffix=str((live_data or {}).get("error", "failed")))
        live_data = {"error": "failed"}
    else:
        sp.stop(success=True)
    if isinstance(live_data, dict) and "error" not in live_data:
        print(f"       {_bar(100)}  ({_fmt_time(elapsed)})")
        print(f"       XAUUSD: ${live_data.get('price', '—')}  |  RSI: {live_data.get('rsi', '—')}")
        print(f"       Trend: {live_data.get('trend', '—')}  |  ATR: ${live_data.get('atr', live_data.get('atr_pips', '—'))}")
    else:
        print(f"       {_bar(0)}  [SKIPPED - timeout]")
    print()
    # ── [MTF] Multi-Timeframe Bias ───────────────────────────────────────────
    h1_price_for_mtf = live_data.get("price") if isinstance(live_data, dict) else None
    sp = Spinner("[MTF] Fetching D1 + H4 bias...", indent=2).start()
    htf_ctx, elapsed_mtf, timed_out_mtf = _run_with_timeout(
        lambda: _step_mtf_bias("GC=F", h1_price=h1_price_for_mtf),
        15, "mtf"
    )
    if timed_out_mtf or not htf_ctx:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        htf_ctx = {"available": False, "bias_line": "MTF timeout",
                   "htf_bias": {}, "htf_levels": {}}
    else:
        sp.stop(success=True)
    if htf_ctx.get("available"):
        bias   = htf_ctx["htf_bias"]
        print(f"       {_bar(100)}  ({_fmt_time(elapsed_mtf)})")
        print(f"       D1: {str(bias.get('d1_trend','?')).upper():8} | H4: {str(bias.get('h4_trend','?')).upper():8} | "
              f"Overall: {str(bias.get('overall_bias','?')).upper()} ({str(bias.get('bias_strength','?')).upper()})")
    else:
        print(f"       {_bar(0)}  [{htf_ctx.get('bias_line','MTF unavailable')}]")
    print()
    # ── [MTF-4] 4-TF confluence score ─────────────────────────────────────────
    sp = Spinner("[MTF-4] Scoring D1/H4/H1/M15 confluence...", indent=2).start()
    mtf4_result, elapsed_mtf4, timed_out_mtf4 = _run_with_timeout(
        lambda: get_mtf_confluence_score("GC=F"), 20, "mtf4"
    )
    if timed_out_mtf4 or not mtf4_result:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        mtf4_result = {
            "d1_bias": "ranging", "h4_bias": "ranging",
            "h1_bias": "ranging", "m15_bias": "ranging",
            "overall_bias": "ranging", "confluence_score": 0,
            "aligned": False, "summary": "MTF-4 timeout",
        }
    else:
        sp.stop(success=True)
    _m4 = mtf4_result
    print(f"       {_bar(100)}  ({_fmt_time(elapsed_mtf4)})")
    print(f"       D1: {_m4['d1_bias'].upper():8} | H4: {_m4['h4_bias'].upper():8} | "
          f"H1: {_m4['h1_bias'].upper():8} | M15: {_m4['m15_bias'].upper()}")
    _score_bar = '#' * _m4['confluence_score'] + '.' * (4 - _m4['confluence_score'])
    print(f"       Score: [{_score_bar}] {_m4['confluence_score']}/4 — {_m4['summary']}")
    print()
    # ── [DXY+YIELDS] Macro context (DXY + US 10Y Treasury) ──────────────────
    # Derive gold direction from news sentiment for macro scoring
    _news_gold_bias  = str((news_data.get("sentiment") or {}).get("gold", {}).get("bias", "")).lower()
    _macro_gold_dir  = {"buy": "long", "sell": "short"}.get(_news_gold_bias, "long")
    sp = Spinner("[DXY+YIELDS] Fetching macro context (DXY + US10Y)...", indent=2).start()
    dxy_ctx, elapsed_dxy, timed_out_dxy = _run_with_timeout(
        lambda: _step_dxy(_macro_gold_dir), 20, "dxy"
    )
    if timed_out_dxy or not dxy_ctx:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        dxy_ctx = {"available": False, "display_line": "DXY/Macro timeout",
                   "dxy_trend": "sideways", "dxy_rsi": 50.0, "momentum_strength": "weak",
                   "macro_score": 0.0, "macro_bias": "neutral",
                   "macro_confirmed": False, "macro_opposed": False,
                   "confidence_adjustment": 0.0, "summary": "",
                   "dxy": {}, "yields": {}}
    else:
        sp.stop(success=True)
    if dxy_ctx.get("available"):
        _dxy_sub    = dxy_ctx.get("dxy") or dxy_ctx
        _yld_sub    = dxy_ctx.get("yields") or {}
        _dxy_trend  = dxy_ctx.get("dxy_trend", "sideways")
        _dxy_rsi    = dxy_ctx.get("dxy_rsi", "—")
        _yld_cur    = _yld_sub.get("current_yield")
        _yld_trend  = _yld_sub.get("yield_trend", "sideways")
        _yld_arr    = {"rising": "↑", "falling": "↓", "sideways": "→"}.get(_yld_trend, "→")
        _dxy_word   = {"up": "Rising ▲", "down": "Falling ▼", "sideways": "Ranging ─"}.get(_dxy_trend, "—")
        _yld_str    = f"{_yld_cur:.2f}%" if _yld_cur else "N/A"
        _mbias      = dxy_ctx.get("macro_bias", "neutral").replace("_", " ").title()
        _msummary   = dxy_ctx.get("summary", "")
        print(f"       {_bar(100)}  ({_fmt_time(elapsed_dxy)})")
        print(f"       DXY: {_dxy_word}  |  RSI: {_dxy_rsi}  |  US10Y: {_yld_str} {_yld_arr} {_yld_trend.capitalize()}")
        print(f"       Macro bias: {_mbias}  →  {_msummary}")
    else:
        print(f"       {_bar(0)}  [{dxy_ctx.get('display_line','DXY/Macro unavailable')}]")
    print()
    # ── [4/5] Historical patterns ─────────────────────────────────────────────
    sp = Spinner("[4/5] Loading historical patterns...", indent=2).start()
    patterns_data, elapsed, timed_out = _run_with_timeout(_step4_patterns, STEP_TIMEOUT, "patterns")
    if timed_out:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        patterns_data = {}
    elif not patterns_data or "error" in (patterns_data or {}):
        sp.stop(success=False, suffix=str((patterns_data or {}).get("error", "not found")))
        patterns_data = {}
    else:
        sp.stop(success=True)
    if patterns_data and "error" not in patterns_data:
        outcomes = patterns_data.get("outcomes", {})
        n_match  = outcomes.get("total_matches", 0)
        verdict  = outcomes.get("verdict", "—")
        up_pct   = outcomes.get("up_pct", 0)
        print(f"       {_bar(100)}  ({_fmt_time(elapsed)})")
        print(f"       {n_match} similar setups found")
        print(f"       Historical bias: {verdict.split()[0] if verdict else '—'} {up_pct}%")
    else:
        print(f"       {_bar(0)}  [SKIPPED - not found — run setup.py]")
    print()

    # ── [NEWS] Global news context engine ───────────────────────────────────────
    sp = Spinner("[NEWS] Building global news context...", indent=2).start()
    global_news_ctx, elapsed_gnc, timed_out_gnc = _run_with_timeout(
        get_global_news_context, 30, "global_news"
    )
    if timed_out_gnc or not global_news_ctx:
        sp.stop(success=False, suffix="[SKIPPED - timeout]")
        global_news_ctx = {
            "gold_bias_from_news": "neutral", "volatility_warning": False,
            "volatility_triggers": [], "bullish_triggers": [], "bearish_triggers": [],
            "scheduled_bias": "neutral", "headline_bias": "neutral",
            "key_event": "timeout", "all_headlines": [],
            "recommendation": "News context unavailable — rely on technicals.",
            "trade_with_caution": False, "confidence": 5.0,
        }
    else:
        sp.stop(success=True)
    _gnc = global_news_ctx
    _bias_icon = {"bullish": "↑", "bearish": "↓", "volatile": "!", "conflicted": "~", "neutral": "─"}
    print(f"       {_bar(100)}  ({_fmt_time(elapsed_gnc)})")
    print(f"       Bias: {_gnc['gold_bias_from_news'].upper():12}  "
          f"[sched: {_gnc['scheduled_bias']:8} | headlines: {_gnc['headline_bias']}]  "
          f"conf: {_gnc['confidence']:.1f}/10")
    if _gnc["volatility_warning"]:
        print(f"       ⚠ VOLATILITY: {', '.join(_gnc['volatility_triggers'][:2])}")
    print(f"       Key event: {_gnc['key_event']}")
    print()

    # ── [FUNDAMENTAL] Macro fundamental bias ──────────────────────────────────
    fund_ctx: dict = {"fundamental_bias": "NEUTRAL", "total_score": 0,
                      "available": False, "confidence": 5.0, "factors": {}}
    if _FB_MB_OK:
        sp = Spinner("[FUNDAMENTAL] Scoring macro fundamentals...", indent=2).start()
        fund_ctx, _elapsed_fund, _timed_out_fund = _run_with_timeout(_get_fund_bias, 30, "fundamental")
        if not isinstance(fund_ctx, dict) or _timed_out_fund:
            sp.stop(success=False, suffix="[SKIPPED - timeout]")
            fund_ctx = {"fundamental_bias": "NEUTRAL", "total_score": 0,
                        "available": False, "confidence": 5.0, "factors": {}}
        else:
            sp.stop(success=True)
            _fbias  = fund_ctx.get("fundamental_bias", "NEUTRAL")
            _fscore = fund_ctx.get("total_score", 0)
            _fconf  = fund_ctx.get("confidence", 5.0)
            print(f"       {_bar(100)}  ({_fmt_time(_elapsed_fund)})")
            print(f"       Bias: {_fbias:20}  score: {_fscore:+d}  conf: {_fconf:.1f}/10")
            _facs = fund_ctx.get("factors", {})
            _fac_names = [("inflation", "📈 Inflation"), ("oil", "🛢 Oil     "),
                          ("fed", "🏦 Fed     "), ("dxy", "💵 DXY     "),
                          ("geopolitical", "🌍 Geo     ")]
            for _fk, _fn in _fac_names:
                _f  = _facs.get(_fk, {})
                _fs = _f.get("score", 0)
                _fn2 = _f.get("note", "—")
                print(f"       {_fn}  {_fs:+d}  {_fn2}")
    print()

    # ── [GEO] Geopolitical risk scoring ───────────────────────────────────────
    geo_ctx: dict = {"available": False, "geo_risk_level": "normal",
                     "sl_atr_multiplier": 0.0, "confidence_adjustment": 0.0}
    if _GEO_OK:
        sp = Spinner("[GEO] Scoring geopolitical risk...", indent=2).start()
        try:
            _geo_result, _elapsed_geo, _timed_out_geo = _run_with_timeout(_get_geo_score, 20, "geo")
            if isinstance(_geo_result, dict) and _geo_result.get("available"):
                geo_ctx = _geo_result
                sp.stop(success=True)
                _glevel = geo_ctx.get("geo_risk_level", "normal").upper()
                _gscore = geo_ctx.get("geo_score", 0)
                _gtops  = geo_ctx.get("top_headlines", [])
                print(f"       {_bar(100)}  ({_fmt_time(_elapsed_geo)})")
                print(f"       🌍 GEO RISK: {_glevel} (score {_gscore}/10)")
                print(f"       {geo_ctx.get('recommendation', '')}")
                if _gtops:
                    print(f"       Top event: {_gtops[0][:90]}")
            else:
                sp.stop(success=False, suffix="unavailable")
        except Exception as _geo_err:
            sp.stop(success=False, suffix=str(_geo_err))
    print()


    # [HANDOFF] Session handoff analysis
    if _SH_MB_OK and 'live_data' in dir() and live_data and isinstance(live_data, dict):
        try:
            _df_handoff = live_data.get("df")
            if _df_handoff is not None and not _df_handoff.empty:
                _handoff_main = _get_ny_bias_mb(_df_handoff)
                print()
                print(_format_handoff_mb(_handoff_main))
                print()
        except Exception:
            pass

    # ── [5/5] Scan trade setups ────────────────────────────────────────────────
    sp = Spinner("[5/5] Scanning for trade setups...", indent=2).start()
    t5 = time.time()
    scan_meta: dict = {}
    try:
        sentiment = (news_data.get("sentiment", {}) if isinstance(news_data, dict) else {}) or {}
        live_df   = live_data.get("df") if isinstance(live_data, dict) else None

        # ── Log session start before scanning ─────────────────────────────────
        _ld   = live_data if isinstance(live_data, dict) and "error" not in live_data else {}
        _bias = (htf_ctx or {}).get("htf_bias", {}) or {}
        log_session_start(
            gold_price   = float(_ld.get("price", 0) or 0),
            session      = _current_session_str(),
            d1_bias      = str(_bias.get("d1_trend", "—")).upper(),
            h4_bias      = str(_bias.get("h4_trend", "—")).upper(),
            dxy          = str((dxy_ctx or {}).get("dxy_trend", "—")),
            regime       = "—",
            n_rules      = len(rules_data.get("rules", [])),
            n_playbooks  = 12,
        )

        scan_meta["mtf_confluence"]   = mtf4_result
        scan_meta["global_news_ctx"]   = global_news_ctx
        scan_meta["geo_ctx"]           = geo_ctx
        result5   = _step5_scan_signals(
            rules_data.get("rules", []), sentiment,
            df=live_df, htf_ctx=htf_ctx, dxy_ctx=dxy_ctx,
            mtf_confluence=mtf4_result, global_news_ctx=global_news_ctx,
            macro_ctx=dxy_ctx, geo_ctx=geo_ctx,
        )
        if isinstance(result5, tuple):
            signals, scan_meta = result5
        else:
            signals = result5   # backwards-compat fallback
        e5 = time.time() - t5
        sp.stop(success=True)

        # ── Log session end summary ────────────────────────────────────────────
        log_session_end(
            pb_checked          = scan_meta.get("playbook_found",  0),
            pb_triggered        = scan_meta.get("playbook_passed", 0),
            confluence_rejections = 0,
            checklist_rejections  = scan_meta.get("rules_found", 0) - scan_meta.get("rules_passed", 0),
            sl_rejections         = scan_meta.get("sl_rejected",  0),
            conf_rejections       = scan_meta.get("conf_rejected", 0),
            fatigue_rejections    = scan_meta.get("fatigue_rejected", 0),
            valid_signals         = len(signals),
        )
    except Exception as exc:
        e5 = time.time() - t5
        sp.stop(success=False, suffix=str(exc))
        log_error("morning_briefing", "_step5_scan_signals", str(exc),
                  "signals returned empty", "no trade setups shown")
        signals = []
    print(f"       {_bar(100)}  ({_fmt_time(e5)})")
    pb_n  = scan_meta.get("playbook_found",  0)
    rl_n  = scan_meta.get("rules_passed",    0)
    print(f"       Playbook hits: {pb_n}  |  Rules hits: {rl_n}  |  Shown: {len(signals)}")
    print()

    # ── Total time ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"  Total briefing time: {_fmt_time(total_elapsed)}")
    if total_elapsed > TOTAL_TIMEOUT:
        print(f"  {WARN} Exceeded {TOTAL_TIMEOUT}s target")
    print("  " + "═" * 52)

    # ── Session summary ────────────────────────────────────────────────────────
    _print_session_summary(
        rules_data      = rules_data,
        news_data       = news_data,
        live_data       = live_data,
        patterns_data   = patterns_data,
        signals         = signals,
        scan_meta       = scan_meta,
        htf_ctx         = htf_ctx,
        dxy_ctx         = dxy_ctx,
        global_news_ctx = global_news_ctx,
    )

