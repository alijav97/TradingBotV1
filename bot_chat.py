"""
bot_chat.py — TradingBotV1 Streamlit Chat Interface
────────────────────────────────────────────────────
Analyst-only bot.  All trades are placed MANUALLY on MT5.
No order execution code exists in this file.

Run with:
    streamlit run bot_chat.py
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradingBotV1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RULES_FILE    = os.path.join(BASE_DIR, "data", "rules.json")
HIST_CSV      = os.path.join(BASE_DIR, "data", "historical_xauusd.csv")
DATA_DIR      = os.path.join(BASE_DIR, "data")
SETTINGS_FILE = os.path.join(DATA_DIR, "user_settings.json")
GST           = timezone(timedelta(hours=4))
import platform as _platform
IS_CLOUD      = _platform.system() != "Windows"  # Railway/Linux = cloud

# ── Debug logger ──────────────────────────────────────────────────────────────
try:
    from debug_logger import build_export, log_info, log_error
    _DBG_OK = True
except ImportError:
    _DBG_OK = False
    def build_export(*a, **kw): return ("", "Debug logger not available.")
    def log_info(m): pass
    def log_error(**kw): pass

# ── Pattern fatigue ───────────────────────────────────────────────────────────
try:
    from pattern_fatigue import (
        predict_next_outcome, check_strategy_fatigue,
        detect_regime_shift, analyze_failed_trade,
        SequenceTracker,
    )
    _PF_OK = True
except ImportError:
    _PF_OK = False
    def predict_next_outcome(*a, **kw): return {}      # type: ignore[misc]
    def check_strategy_fatigue(*a, **kw): return {}    # type: ignore[misc]
    def detect_regime_shift(*a, **kw): return {}       # type: ignore[misc]
    def analyze_failed_trade(*a, **kw): return {}      # type: ignore[misc]

try:
    from volume_analyzer import VolumeAnalyzer, check_volume_confluence
    _VA_OK = True
except ImportError:
    _VA_OK = False
    def check_volume_confluence(*a, **kw): return {}   # type: ignore[misc]

try:
    from spread_monitor import check_spread as _check_spread_live
    _SM_OK = True
except ImportError:
    _SM_OK = False
    def _check_spread_live(symbol="XAUUSD"):  # type: ignore[misc]
        return {"spread_usd": None, "status": "unavailable", "blocked": False,
                "reason": "", "recommendation": "", "bid": None, "ask": None}

try:
    from signal_tracker import (
        register_signal         as _st_register,
        update_signal_prices    as _st_update,
        get_signal_performance_report as _st_report,
        mark_user_traded        as _st_mark_traded,
    )
    _ST_OK = True
except ImportError:
    _ST_OK = False
    def _st_register(*a, **kw): return ""        # type: ignore[misc]
    def _st_update(*a, **kw): return []           # type: ignore[misc]
    def _st_report(*a, **kw): return {}           # type: ignore[misc]
    def _st_mark_traded(*a, **kw): pass           # type: ignore[misc]

try:
    from geo_filter import get_geopolitical_score as _get_geo
    _GEO_OK = True
except ImportError:
    _GEO_OK = False
    def _get_geo(*a, **kw): return {"available": False, "geo_risk_level": "normal", "geo_score": 0, "sl_atr_multiplier": 0.0, "confidence_adjustment": 0.0, "colour": "#2ecc71", "recommendation": ""}  # type: ignore[misc]

try:
    from world_sessions import (
        get_active_sessions,
        get_session_summary_line,
        get_full_session_board,
    )
    _WS_OK = True
except ImportError:
    _WS_OK = False
    def get_active_sessions(*a, **kw): return []           # type: ignore[misc]
    def get_session_summary_line(*a, **kw): return "—"     # type: ignore[misc]
    def get_full_session_board(*a, **kw): return "Session data unavailable."  # type: ignore[misc]

try:
    from mt5_sync import get_live_price as _get_live_price
    _LP_OK = True
except ImportError:
    _LP_OK = False
    def _get_live_price(symbol="XAUUSD"):  # type: ignore[misc]
        return {"price": None, "source": "unavailable", "is_live": False,
                "stale_warning": "mt5_sync not available", "timestamp_uae": "—",
                "bid": None, "ask": None, "spread": None, "age_seconds": 9999}

try:
    from reversal_hunter import hunt_reversals as _hunt_reversals
    _RH_OK = True
except ImportError:
    _RH_OK = False
    def _hunt_reversals(*a, **kw): return []  # type: ignore[misc]

try:
    from fundamental_bias import (
        get_fundamental_bias as _get_fundamental_bias,
        check_fundamental_conflict as _check_fund_conflict,
    )
    _FB_OK = True
except ImportError:
    _FB_OK = False
    def _get_fundamental_bias(*a, **kw): return {"fundamental_bias": "NEUTRAL", "total_score": 0, "display_line": "📊 Fundamental: Unavailable", "available": False, "confidence": 5.0}  # type: ignore[misc]
    def _check_fund_conflict(*a, **kw): return {"conflict": False, "severity": "NONE", "message": ""}  # type: ignore[misc]

try:
    from trade_manager import (
        calculate_partial_tp_plan  as _calc_tp_plan,
        format_trade_instructions  as _format_trade,
        get_current_risk_profile   as _get_ror_profile,
        format_ror_report          as _format_ror,
    )
    _TM_OK = True
except ImportError:
    _TM_OK = False
    def _calc_tp_plan(*a, **kw):    return {"valid": False}   # type: ignore[misc]
    def _format_trade(*a, **kw):    return ""                 # type: ignore[misc]
    def _get_ror_profile(*a, **kw): return {}                 # type: ignore[misc]
    def _format_ror(*a, **kw):      return ""                 # type: ignore[misc]

try:
    from cot_analyzer import fetch_cot_data as _fetch_cot, get_cot_signal as _get_cot_signal
    _COT_OK = True
except ImportError:
    _COT_OK = False
    def _fetch_cot(): return {"available": False, "bias": "NEUTRAL", "boost": 0.0, "display_line": "COT: unavailable"}  # type: ignore[misc]
    def _get_cot_signal(d, c=None): return {"boost": 0.0, "aligned": False, "opposed": False, "bias": "NEUTRAL", "note": "unavailable"}  # type: ignore[misc]

try:
    from liquidity_map import build_liquidity_map as _build_liq_map, format_liquidity_map as _fmt_liq_map
    _LIQ_OK = True
except ImportError:
    _LIQ_OK = False
    def _build_liq_map(df, p): return {"available": False}   # type: ignore[misc]
    def _fmt_liq_map(l, p): return "Liquidity map unavailable."  # type: ignore[misc]

try:
    from walk_forward import (
        run_walk_forward_optimization as _run_wfo,
        check_if_sunday_run_needed    as _check_wfo,
        get_wfo_summary               as _get_wfo_summary,
    )
    _WFO_OK = True
except ImportError:
    _WFO_OK = False
    def _run_wfo():         return {"optimized": False, "reason": "walk_forward not available"}  # type: ignore[misc]
    def _check_wfo():       return False   # type: ignore[misc]
    def _get_wfo_summary(): return "Walk-forward optimizer not available."  # type: ignore[misc]

try:
    from session_handoff import (
        get_ny_session_bias    as _get_ny_bias,
        format_session_handoff as _format_handoff,
    )
    _SH_OK = True
except ImportError:
    _SH_OK = False
    def _get_ny_bias(df):    return {"ny_bias": "NEUTRAL", "confidence": "LOW", "fake_break_alert": False, "summary": "", "recommendation": "unavailable"}  # type: ignore[misc]
    def _format_handoff(h): return "Session handoff not available."  # type: ignore[misc]

try:
    from indicators import (
        get_all_indicators as _get_all_indicators,
        get_ict_killzones  as _get_killzones,
    )
    _INDICATORS_OK = True
except Exception:
    _INDICATORS_OK = False
    def _get_all_indicators(df): return {}  # type: ignore[misc]
    def _get_killzones():        return {"in_killzone": False, "active_zones": [], "high_quality": False, "next_killzone": None, "bias": "normal"}  # type: ignore[misc]

try:
    from paper_trader import (
        open_paper_trade              as _open_paper,
        update_paper_trades           as _update_paper,
        get_paper_summary             as _paper_summary,
        close_paper_trade_manually    as _close_paper,
        get_paper_performance_report  as _paper_report,
    )
    _PAPER_OK = True
except Exception:
    _PAPER_OK = False
    def _open_paper(s, p):   return {}   # type: ignore[misc]
    def _update_paper(p):    return []   # type: ignore[misc]
    def _paper_summary():    return {"total": 0, "open": 0, "closed": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "open_trades": [], "recent": []}  # type: ignore[misc]
    def _close_paper(i, p):  return {}   # type: ignore[misc]
    def _paper_report():     return "Paper trader not available."  # type: ignore[misc]

try:
    from ml_engine import (
        run_ml_training        as _run_ml,
        generate_ml_insights   as _ml_insights,
        get_ml_confidence_adjustment as _ml_adj,
        get_best_yield_strategies as _best_yield,
        MLEngine               as _MLEngine,
    )
    _ML_OK     = True
    _ml_engine = _MLEngine()   # default XAUUSD; re-instantiated per-instrument in main()
except Exception:
    _ML_OK     = False
    _ml_engine = None  # type: ignore[assignment]
    def _run_ml():       return "ML engine not available."   # type: ignore[misc]
    def _ml_insights():  return {"available": False}          # type: ignore[misc]
    def _ml_adj(**kw):   return {"adjustment": 0.0, "available": False}  # type: ignore[misc]
    def _best_yield(**kw): return "ML engine not available."  # type: ignore[misc]

try:
    from macro_scorer    import MacroScorer as _MacroScorer
    from instrument_data import get_instrument_summary as _get_instr_summary, \
                                 get_market_context    as _get_mkt_ctx
    _macro_scorer = _MacroScorer()
    _MACRO_OK     = True
except Exception:
    _MACRO_OK     = False
    _macro_scorer = None  # type: ignore[assignment]
    def _get_instr_summary(i):  return {"instrument": i, "source": "unavailable"}  # type: ignore[misc]
    def _get_mkt_ctx(i):        return {}   # type: ignore[misc]

try:
    from sector_rotation import SectorRotation as _SectorRotation
    _sector_rotation = _SectorRotation()
    _SR_OK = True
except Exception:
    _SR_OK           = False
    _sector_rotation = None  # type: ignore[assignment]

try:
    from open_interest import OpenInterestAnalyzer as _OIAnalyzer
    _oi_analyzer = _OIAnalyzer()
    _OI_OK = True
except Exception:
    _OI_OK       = False
    _oi_analyzer = None  # type: ignore[assignment]

try:
    from instrument_confluence import InstrumentConfluence as _InstrumentConfluence
    _IC_OK = True
except Exception:
    _IC_OK = False
    _InstrumentConfluence = None  # type: ignore[assignment]

# ── Startup verification print (runs once when Streamlit loads) ───────────────
try:
    _gst_check   = timezone(timedelta(hours=4))
    _start_time  = datetime.now(_gst_check).strftime("%I:%M %p UAE | %A %d %B %Y")
    _start_price = _get_live_price()
    print(f"\n{'='*50}")
    print(f"TradingBotV1 STARTED")
    print(f"UAE Time : {_start_time}")
    if _start_price.get("price") and _start_price["price"] > 0:
        print(f"XAUUSD   : ${_start_price['price']:,.2f} via {_start_price['source']}")
        print(f"Age      : {_start_price['age_seconds']}s")
    else:
        print(f"XAUUSD   : unavailable")
    print(f"{'='*50}\n")
except Exception as _e:
    print(f"Startup check failed: {_e}")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #12151e !important;
    border-right: 1px solid #2a2d3e;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    color: #e8e8e8;
    border-radius: 6px;
    margin-bottom: 3px;
    padding: 0.32rem 0.5rem;
    font-size: 0.80rem;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1D9E75;
    border-color: #1D9E75;
    color: #fff;
}
div[data-testid="stChatInput"] > div {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 10px;
}
/* chip buttons */
div[data-testid="column"] .stButton > button {
    font-size: 0.76rem;
    padding: 0.22rem 0.4rem;
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    color: #aaa;
    border-radius: 20px;
}
div[data-testid="column"] .stButton > button:hover {
    background: #1D9E75;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Session state bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "messages":        [],
        "rules_count":     0,
        "playbooks_count": 12,
        "last_refresh":    None,
        "live_price":      None,
        "live_source":     "—",
        "price_stale":     False,
        "live_rsi":        None,
        "live_trend":      None,
        "live_atr":        None,
        "live_df":         None,
        "d1_bias":         "—",
        "h4_bias":         "—",
        "dxy_status":      "—",
        "session_name":    "—",
        "sentiment":       {},
        "is_live":         False,
        "trigger_cmd":     None,
        "last_signals":    [],
        # MT5 live data
        "mt5_connected":   False,
        "mt5_account":     None,
        "mt5_positions":   [],
        "mt5_today_pnl":   None,
        "mt5_last_sync":   None,
        "mt5_error":       None,
        "mt5_sync_notifications": [],
        "_last_auto_rerun":       0.0,
        # Analysis context
        "mtf_confluence":  {},
        "global_news_ctx": {},
        "spread_check":    {},
        # Brain state
        "brain1_filters":  [],
        "brain2_signals":  {},
        # Macro / DXY
        "macro_bias":      "neutral",
        "yields_context":  {},
        # Geopolitical risk
        "geo_risk_level":  "normal",
        "geo_ctx":         {},
        # UI state
        "account_balance":   300.0,
        "current_session":   "—",
        # Paper trading
        "last_signal":       {},
        "paper_open_count":  0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  Data-file bootstrap — runs once per startup, creates missing files/folders
# ══════════════════════════════════════════════════════════════════════════════

_SETTINGS_DEFAULTS: dict = {
    "balance":               1000,
    "risk_pct":              2.0,
    "risk_per_trade":        0.02,
    "reward_pct":            6.0,
    "min_rr":                3.0,
    "leverage":              10,
    "partial_tp":            True,
    "min_confidence":        6.0,
    "daily_loss_limit_pct":  10,
    "weekly_loss_limit_pct": 15,
    "max_open_trades":       3,
    "sessions":              ["London", "NewYork", "Overlap"],
    "min_volume_ratio":      0.5,
}

INSTRUMENT_RISK_CONFIG: dict = {
    "XAUUSD":  {"grade": "A", "priority": 1, "sl_pct": 0.8,  "tp_pct": 2.4,  "leverage": 10},
    "WTI":     {"grade": "B", "priority": 2, "sl_pct": 1.2,  "tp_pct": 3.6,  "leverage": 10},
    "US30":    {"grade": "B", "priority": 3, "sl_pct": 0.71, "tp_pct": 2.13, "leverage": 10},
    "NAS100":  {"grade": "B", "priority": 4, "sl_pct": 0.65, "tp_pct": 1.94, "leverage": 10},
    "GBPUSD":  {"grade": "C", "priority": 5, "sl_pct": 0.2,  "tp_pct": 0.6,  "leverage": 10},
    "EURUSD":  {"grade": "C", "priority": 6, "sl_pct": 0.2,  "tp_pct": 0.6,  "leverage": 10},
}

# Keyword → canonical instrument ID (used by analyze command router)
_ANALYZE_ALIASES: dict[str, str] = {
    # Gold
    "gold":    "XAUUSD",
    "xauusd":  "XAUUSD",
    "xau":     "XAUUSD",
    # Oil
    "wti":     "WTI",
    "oil":     "WTI",
    "crude":   "WTI",
    "spotcrude": "WTI",
    # Nasdaq
    "nas100":  "NAS100",
    "nasdaq":  "NAS100",
    "tech":    "NAS100",
    "ndx":     "NAS100",
    # Dow
    "us30":    "US30",
    "dow":     "US30",
    "dji":     "US30",
    "djia":    "US30",
    # GBP
    "gbpusd":  "GBPUSD",
    "gbp":     "GBPUSD",
    "cable":   "GBPUSD",
    "pound":   "GBPUSD",
    # EUR
    "eurusd":  "EURUSD",
    "eur":     "EURUSD",
    "euro":    "EURUSD",
    "fiber":   "EURUSD",
}


def _bootstrap_data_files() -> None:
    """
    Ensure every required data file and folder exists.
    Called once at the top of main() before anything else touches disk.
    """
    # ── Folders ───────────────────────────────────────────────────────────────
    for folder in [
        DATA_DIR,
        os.path.join(DATA_DIR, "logs"),
        os.path.join(DATA_DIR, "backtest_results"),
    ]:
        os.makedirs(folder, exist_ok=True)

    # ── JSON files with typed defaults ───────────────────────────────────────
    _file_defaults: list[tuple[str, Any]] = [
        (RULES_FILE,                               []),
        (os.path.join(DATA_DIR, "trade_journal.json"),      []),
        (os.path.join(DATA_DIR, "pattern_memory.json"),     []),
        (os.path.join(DATA_DIR, "bot_signals_log.json"),    []),
        (os.path.join(DATA_DIR, "signal_performance.json"), []),
        (os.path.join(DATA_DIR, "auto_filters.json"),       []),
        (os.path.join(DATA_DIR, "price_cache.json"),        {"ask": 0, "bid": 0}),
    ]
    for fpath, default in _file_defaults:
        if not os.path.exists(fpath):
            try:
                with open(fpath, "w", encoding="utf-8") as _fh:
                    json.dump(default, _fh, indent=2)
            except Exception:
                pass

    # ── user_settings.json — merge defaults without overwriting existing keys ─
    settings_file = SETTINGS_FILE
    if not os.path.exists(settings_file):
        try:
            with open(settings_file, "w", encoding="utf-8") as _fh:
                json.dump(_SETTINGS_DEFAULTS, _fh, indent=2)
        except Exception:
            pass
    else:
        # Add any keys that are in DEFAULTS but missing from the saved file
        try:
            with open(settings_file, "r", encoding="utf-8") as _fh:
                _saved = json.load(_fh)
            _changed = False
            for _k, _v in _SETTINGS_DEFAULTS.items():
                if _k not in _saved:
                    _saved[_k] = _v
                    _changed = True
            if _changed:
                with open(settings_file, "w", encoding="utf-8") as _fh:
                    json.dump(_saved, _fh, indent=2)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Lazy module imports
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_modules() -> dict[str, Any]:
    mods: dict[str, Any] = {}
    for alias, mod_name in [
        ("confluence", "confluence_engine"),
        ("playbooks",  "strategy_playbooks"),
        ("checklist",  "entry_checklist"),
        ("mtf",        "mtf_analyzer"),
        ("dxy",        "dxy_correlation"),
        ("news_mon",   "news_monitor"),
        ("news_fil",   "news_filter"),
        ("smart",      "smart_money"),
    ]:
        try:
            mods[alias] = __import__(mod_name)
        except Exception:
            mods[alias] = None
    return mods

MODS = _load_modules()

# ── MT5 sync (optional — graceful if MT5 not installed / offline) ─────────────
try:
    from mt5_sync import (
        get_account_info,
        get_open_positions,
        get_today_pnl,
        sync_to_journal,
        auto_match_and_update,
        log_bot_signal,
        get_mt5_status_label,
        _load_journal,
        track_open_trades,
        _load_auto_filters,
    )
    _MT5_SYNC_OK = True
except ImportError:
    _MT5_SYNC_OK = False
    def get_account_info(*a, **kw): return None, "mt5_sync not available"  # type: ignore[misc]
    def get_open_positions(*a, **kw): return []   # type: ignore[misc]
    def get_today_pnl(*a, **kw): return None      # type: ignore[misc]
    def sync_to_journal(*a, **kw): return []      # type: ignore[misc]
    def auto_match_and_update(*a, **kw): return []  # type: ignore[misc]
    def log_bot_signal(*a, **kw): pass            # type: ignore[misc]
    def get_mt5_status_label(*a, **kw): return "offline"  # type: ignore[misc]
    def _load_journal(*a, **kw): return []        # type: ignore[misc]
    def track_open_trades(): return []            # type: ignore[misc]
    def _load_auto_filters(): return []           # type: ignore[misc]


def _refresh_mt5_data() -> None:
    """Pull fresh account/position data from MT5 into session_state.
    Safe to call when MT5 is offline — just sets mt5_connected=False.
    Uses a 10-second timeout to prevent hanging startup.
    """
    if not _MT5_SYNC_OK:
        st.session_state["mt5_connected"] = False
        st.session_state["mt5_error"]     = "mt5_sync module not available"
        return
    import threading as _threading

    _result: dict = {}

    def _do_refresh() -> None:
        try:
            acct, err = get_account_info()
            if acct:
                _result["acct"]      = acct
                _result["positions"] = get_open_positions()
                _result["pnl"]       = get_today_pnl()
                _result["notes"]     = auto_match_and_update(days_back=2) or []
            else:
                _result["err"] = err
        except Exception as _ex:
            _result["err"] = str(_ex)

    _t = _threading.Thread(target=_do_refresh, daemon=True)
    _t.start()
    _t.join(timeout=10)   # 10 s hard cap — MT5 connect includes up to 5 s sleep

    if _result.get("acct"):
        st.session_state["mt5_connected"]  = True
        st.session_state["mt5_account"]    = _result["acct"]
        st.session_state["mt5_positions"]  = _result.get("positions", [])
        st.session_state["mt5_today_pnl"]  = _result.get("pnl")
        st.session_state["mt5_last_sync"]  = datetime.now(GST)
        st.session_state["mt5_error"]      = None
        notes = _result.get("notes", [])
        if notes:
            existing = st.session_state.get("mt5_sync_notifications", [])
            st.session_state["mt5_sync_notifications"] = existing + notes
    else:
        st.session_state["mt5_connected"] = False
        if _t.is_alive():
            st.session_state["mt5_error"] = "MT5 offline (timeout)"
        else:
            st.session_state["mt5_error"] = _result.get("err", "MT5 offline")


def check_mt5_status() -> bool:
    """Refresh MT5 data and return whether we are connected."""
    _refresh_mt5_data()
    return bool(st.session_state.get("mt5_connected", False))


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _gst_now() -> str:
    return datetime.now(GST).strftime("%H:%M GST")

def _current_session() -> str:
    """Return a one-line session summary using world_sessions (UAE-aware)."""
    if _WS_OK:
        return get_session_summary_line()
    # Fallback: UTC-hour approximation
    h = datetime.now(timezone.utc).hour
    if 0  <= h < 7:  return "Asian"
    if 7  <= h < 12: return "London"
    if 12 <= h < 16: return "London/NY Overlap"
    if 16 <= h < 21: return "New York"
    return "Off-Hours"

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _load_rules() -> list[dict]:
    rules = _load_json(RULES_FILE, [])
    st.session_state["rules_count"] = len(rules)
    return rules

def _load_df():
    if st.session_state["live_df"] is not None:
        return st.session_state["live_df"]
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(HIST_CSV, index_col=0)
        df.columns = [c.lower() for c in df.columns]
        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        e12 = df["close"].ewm(span=12, adjust=False).mean()
        e26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df = df.dropna(subset=["ema200", "rsi", "atr"])
        st.session_state["live_df"]    = df
        row = df.iloc[-1]
        # Seed from CSV first, then try to get a live price
        st.session_state["live_price"] = round(float(row["close"]), 2)
        st.session_state["live_rsi"]   = round(float(row["rsi"]),   1)
        st.session_state["live_atr"]   = round(float(row["atr"]),   2)
        st.session_state["live_trend"] = "BEARISH" if row["close"] < row["ema200"] else "BULLISH"
        st.session_state["is_live"]    = True
        # ── Upgrade to live price ─────────────────────────────────────────────
        try:
            lp = _get_live_price(st.session_state.get("instrument", "XAUUSD"))
            if lp.get("price") and lp["price"] > 0:
                st.session_state["live_price"]  = round(lp["price"], 2)
                st.session_state["live_source"] = lp.get("source", "—")
                st.session_state["price_stale"] = not lp.get("is_live", False)
                # Overwrite last CSV row so downstream calcs see live price
                df.iloc[-1, df.columns.get_loc("close")] = lp["price"]
        except Exception:
            st.session_state["live_source"] = "CSV"
            st.session_state["price_stale"] = True
        return df
    except Exception:
        return None

def _load_m15_df():
    """Load 5-day M15 XAUUSD data with indicators via yfinance."""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        ticker = yf.Ticker("GC=F")
        df = ticker.history(interval="15m", period="5d", auto_adjust=True)
        if df is None or df.empty:
            return None
        df.index   = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        # EMA
        df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        # ATR
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        df = df.dropna(subset=["ema50", "rsi", "atr"])
        return df if not df.empty else None
    except Exception:
        return None


def _fmt_price(p) -> str:
    try:
        return f"${float(p):,.2f}"
    except Exception:
        return str(p)

def _load_settings() -> dict:
    """Load user_settings.json via central settings module."""
    try:
        from settings import load_settings as _ls
        return _ls()
    except Exception:
        return {
            "balance": 1000, "risk_pct": 2.0, "risk_per_trade": 0.02,
            "risk_usd": 20, "reward_pct": 6.0, "reward_usd": 60, "min_rr": 3.0,
            "leverage": 10, "partial_tp": True, "min_confidence": 7.5,
            "max_risk_usd": 20,
        }

def _lot_size(account: float, risk_pct: float, sl_dollar: float) -> float:
    """Legacy shim — kept for any callers outside the trade card."""
    risk_dollar = account * (risk_pct / 100)
    if sl_dollar <= 0:
        return 0.0
    return round(risk_dollar / (sl_dollar * 100), 2)

def calculate_lot_size(
    entry: float,
    sl: float,
    balance: float,
    risk_pct: float,
    leverage: float,
) -> dict:
    """
    Delegate to central calculate_position(). Returns legacy-compatible dict.
    1 lot = 100 oz  →  $1 price move = $100 P&L per lot.
    """
    try:
        from settings import calculate_position
        settings = _load_settings()
        settings["balance"]     = balance
        settings["risk_pct"]    = risk_pct
        settings["risk_usd"]    = round(balance * risk_pct / 100, 2)
        settings["max_risk_usd"] = settings["risk_usd"]
        settings["leverage"]    = leverage
        pos = calculate_position(entry, sl, settings)
        if not pos.get("tradeable", True):
            return {
                "lots": 0.01, "sl_distance": abs(entry - sl), "tp_distance": 0,
                "risk_usd": abs(entry - sl) * 0.01 * 100,
                "reward_usd": 0, "reward_pct": 0, "risk_pct": 0,
                "wide_sl": True, "not_tradeable": True, "reason": pos.get("reason", ""),
            }
        return {
            "lots":        pos["lots"],
            "sl_distance": pos["sl_distance"],
            "tp_distance": pos["tp2_distance"],
            "risk_usd":    pos["actual_risk_usd"],
            "reward_usd":  pos["target_reward_usd"],
            "reward_pct":  pos["target_reward_pct"],
            "risk_pct":    pos["actual_risk_pct"],
            "wide_sl":     False,
        }
    except Exception:
        # Pure fallback without settings module
        risk_usd    = balance * (risk_pct / 100)
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return {"lots": 0.01, "sl_distance": 0, "tp_distance": 0,
                    "risk_usd": 0, "reward_usd": 0, "reward_pct": 0,
                    "risk_pct": 0, "wide_sl": False}
        base_lots    = risk_usd / (sl_distance * 100)
        max_lots     = (balance * leverage) / (entry * 100)
        final_lots   = max(0.01, round(min(base_lots, max_lots), 2))
        wide_sl      = final_lots <= 0.01 and base_lots > 0.01
        actual_risk  = sl_distance * final_lots * 100
        return {
            "lots": final_lots, "sl_distance": sl_distance,
            "tp_distance": sl_distance * 3,
            "risk_usd": round(actual_risk, 2),
            "reward_usd": round(actual_risk * 3, 2),
            "reward_pct": round(actual_risk * 3 / balance * 100, 1),
            "risk_pct": round(actual_risk / balance * 100, 1),
            "wide_sl": wide_sl,
        }

def _sl_filter(sig: dict, df, settings: dict) -> tuple:
    """
    Returns ('ok'|'reject'|'warn', extra_data_dict).
    'reject' → SL tighter than ATR×0.5 (likely noise)
    'warn'   → SL so wide that lots drop to 0.01 floor
    """
    entry    = float(sig.get("entry",     0) or 0)
    sl       = float(sig.get("stop_loss", 0) or 0)
    if not entry or not sl:
        return "ok", {}
    try:
        atr = float(df["atr"].iloc[-1])
    except Exception:
        atr = 5.0
    min_sl_dist = atr * 0.5
    sl_dist     = abs(entry - sl)
    balance     = settings.get("balance",  1000)
    leverage    = settings.get("leverage",  10)
    risk_pct    = settings.get("risk_pct",  2.0)

    if sl_dist < min_sl_dist:
        return "reject", {"reason": f"SL too tight (${sl_dist:.2f} < min ${min_sl_dist:.2f})"}

    sz = calculate_lot_size(entry, sl, balance, risk_pct, leverage)
    if sz["wide_sl"]:
        return "warn", {"sz": sz}

    return "ok", sz

def _add_user_msg(text: str) -> None:
    st.session_state["messages"].append({"role": "user",      "content": text, "ts": _gst_now()})

def _add_bot_msg(text: str) -> None:
    st.session_state["messages"].append({"role": "assistant", "content": text, "ts": _gst_now()})


# ══════════════════════════════════════════════════════════════════════════════
#  Trade card renderer — matches full spec format
# ══════════════════════════════════════════════════════════════════════════════

def _render_trade_card(sig: dict, idx: int = 1, account: float = 0.0) -> str:
    """Build the full trade card string for a signal."""
    from settings import calculate_position as _calc_pos

    # ── Load settings ────────────────────────────────────────────────────────
    settings   = _load_settings()
    if account <= 0:
        account = float(settings.get("balance", 1000))
    # Always enforce settings values — ignore passed account if settings loaded OK
    account    = float(settings.get("balance", account))
    risk_pct   = float(settings.get("risk_pct",  2.0))
    leverage   = float(settings.get("leverage",  10))
    partial_tp = bool(settings.get("partial_tp", True))
    min_rr     = float(settings.get("min_rr",    3.0))

    # ── Signal fields ────────────────────────────────────────────────────────
    direction = str(sig.get("direction", "SHORT")).upper()
    asset     = sig.get("asset", "XAUUSD")
    pattern   = sig.get("pattern_name", sig.get("name", "Strategy"))
    pb_id     = sig.get("playbook_id", "")
    source    = sig.get("source", "rules")
    conf      = sig.get("confidence", sig.get("score", 0)) or 0
    entry     = float(sig.get("entry",       0) or 0)
    sl        = float(sig.get("stop_loss",   0) or 0)
    tp        = float(sig.get("take_profit", 0) or 0)

    # ── Source tag ───────────────────────────────────────────────────────────
    pb_num = ""
    pb_mod = MODS.get("playbooks")
    if source == "playbook" and pb_mod and hasattr(pb_mod, "PLAYBOOKS"):
        keys = list(pb_mod.PLAYBOOKS.keys())
        if pb_id in keys:
            pb_num = f" {keys.index(pb_id)+1}"
    src_tag = f"Playbook{pb_num}" if source == "playbook" else f"Rules · Tier {sig.get('tier','?')}"

    sl_dist = abs(entry - sl)   if sl else 0.0
    tp_dist = abs(entry - tp)   if tp else 0.0
    sl_sign = "+" if direction == "SHORT" else "−"
    tp_sign = "−" if direction == "SHORT" else "+"
    conf_f  = f"{float(conf):.1f}/10"

    # ── Geo risk SL adjustment ───────────────────────────────────────────────
    _geo_sl_mult  = float(sig.get("sl_atr_multiplier", 0.0) or 0.0)
    _atr_val      = float(sig.get("atr", 0.0) or 0.0)
    if _atr_val == 0.0:
        # Try to read ATR from df if available
        try:
            _df_live = st.session_state.get("live_df")
            if _df_live is not None and not _df_live.empty:
                _atr_val = float(_df_live["atr"].iloc[-1])
        except Exception:
            _atr_val = 20.0  # safe fallback
    if _atr_val == 0.0:
        _atr_val = 20.0
    _geo_sl_adj   = _geo_sl_mult * _atr_val   # extra SL buffer in $

    SEP  = "═" * 47
    DASH = "─" * 47

    # ── Lot sizing ───────────────────────────────────────────────────────────
    pos       = _calc_pos(entry, sl, settings)
    size_mult = float(sig.get("size_multiplier", 1.0))
    not_tradeable = not pos.get("tradeable", True)
    if not_tradeable:
        final_lots        = 0.01
        actual_risk       = sl_dist * 0.01 * 100
        actual_reward     = actual_risk * min_rr
        risk_actual_pct   = round(actual_risk   / account * 100, 1)
        reward_actual_pct = round(actual_reward / account * 100, 1)
        _reject_reason    = pos.get("reason", "Setup not tradeable with current balance")
    else:
        base_lots         = float(pos.get("lots", 0.01))
        final_lots        = max(0.01, round(base_lots * size_mult, 2))
        actual_risk       = sl_dist * final_lots * 100
        actual_reward     = float(pos.get("reward_usd", actual_risk * min_rr)) * size_mult
        risk_actual_pct   = round(actual_risk   / account * 100, 1)
        reward_actual_pct = round(actual_reward / account * 100, 1)
        _reject_reason    = ""

    # ── Session adjustment info from calculate_position() ────────────────────
    _sess_adj   = pos.get("session_adjustment", {}) or {}
    _sess_name  = _sess_adj.get("session", "")
    _sess_grade = _sess_adj.get("grade", "")
    _sess_lot_c = _sess_adj.get("lot_change", "")
    _sess_sl_c  = _sess_adj.get("sl_change",  "")
    _sess_note  = _sess_adj.get("session_note", "")
    _sess_rec   = _sess_adj.get("trading_recommended", True)

    # ── TP1 / TP2 (partial take-profit) ─────────────────────────────────────
    is_long = direction == "LONG"
    if partial_tp and tp:
        tp1 = tp          # 50% position
        # TP2 is the full 1:3 from SL distance
        rr3_dist = sl_dist * 3
        tp2 = round(entry + rr3_dist if is_long else entry - rr3_dist, 2)
        tp1_dist = abs(entry - tp1)
        tp2_dist = abs(entry - tp2)
    else:
        tp1 = tp2 = tp
        tp1_dist = tp2_dist = tp_dist

    rr_actual = f"1:{tp_dist/sl_dist:.1f}" if sl_dist > 0 else "—"

    # ── SL quality block ─────────────────────────────────────────────────────
    sl_q         = sig.get("checklist_results", {}) or {}
    sl_quality   = sl_q.get("sl_quality", {}) or {}
    q_checks     = sl_quality.get("checks", {})
    q_warnings   = sl_quality.get("warnings", [])
    adj_sl       = sl_quality.get("adjusted_sl", sl)

    def _sl_icon(key: str) -> str:
        c = q_checks.get(key, {})
        if not c:
            return "?"
        return "✓" if c.get("passed") else "✗"

    noise_check     = _sl_icon("noise")
    structure_check = _sl_icon("structure")
    spread_check    = _sl_icon("spread")

    sl_check_lines = (
        f"  SL CHECK:   {noise_check} Outside market noise (ATR×0.3 = ${sl_dist * 0.3 / max(sl_dist, 0.01):.2f})\n"
        f"              {structure_check} At swing high/low level\n"
        f"              {spread_check} Spread buffer included\n"
    )
    if q_warnings:
        for w in q_warnings:
            sl_check_lines += f"  ⚠ {w}\n"
    if adj_sl and adj_sl != sl:
        sl_check_lines += f"  → SL adjusted to ${adj_sl:,.2f}\n"

    # ── Dynamic SL breakdown display ─────────────────────────────────────────
    _sl_breakdown  = sig.get("sl_breakdown",    "") or sl_quality.get("sl_breakdown", "")
    _vol_state     = sig.get("volatility_state", "") or sl_quality.get("volatility_state", "")
    _atr_pct_val   = sig.get("atr_percentile",  None)
    if _atr_pct_val is None:
        _atr_pct_val = sl_quality.get("atr_percentile", None)
    if _sl_breakdown:
        sl_check_lines += f"  SL METHOD:  {_sl_breakdown}\n"
    if _vol_state:
        _vol_label = _vol_state.replace("_", " ").upper()
        _pct_str   = f" ({_atr_pct_val:.0f}th percentile)" if _atr_pct_val is not None else ""
        sl_check_lines += f"  Volatility: {_vol_label}{_pct_str}\n"

    # ── Checklist ────────────────────────────────────────────────────────────
    ck_passed = sl_q.get("checks_passed", "?")
    ck_total  = sl_q.get("total_checks", 5)
    if isinstance(ck_passed, int) and ck_passed >= 4:
        ck_line = f"CHECKLIST: {ck_passed}/{ck_total} PASSED ✓"
    elif isinstance(ck_passed, int):
        ck_line = f"CHECKLIST: {ck_passed}/{ck_total} FAILED ✗"
    else:
        ck_line = "CHECKLIST: not run"

    # ── Confluence block ─────────────────────────────────────────────────────
    detail_lines = sig.get("detail_lines", [])
    if detail_lines:
        conf_block = "\n".join(f"  {ln}" for ln in detail_lines)
    else:
        weights = {
            "HTF": 2.5, "SMC": 2.0, "Trend": 1.5, "Structure": 1.5,
            "Momentum": 1.0, "DXY": 1.0, "Candle": 0.5, "Session": 0.5, "Volatility": 0.5,
        }
        c_met    = sig.get("confluence_met",    [])
        c_missed = sig.get("confluence_missed", [])
        conf_rows = []
        for label, w in weights.items():
            if label in c_met:
                conf_rows.append(f"  ✓ {label:<14}  +{w:.1f}")
            elif label in c_missed:
                conf_rows.append(f"  ✗ {label:<14}   0.0")
        conf_block = "\n".join(conf_rows) if conf_rows else "  (no confluence detail)"

    # ── Regime / history ─────────────────────────────────────────────────────
    regime_lbl = sig.get("regime_label", "")
    hist_wr    = sig.get("hist_win_rate")
    hist_n     = sig.get("hist_sample_size", 0)
    boost      = sig.get("confidence_boost", 0.0)
    regime_line = f"  REGIME:     {regime_lbl}  (×{size_mult:.1f} size)" if regime_lbl else ""
    hist_line   = (
        f"  HISTORY:    {hist_n} trades → {hist_wr:.0%} win rate"
        + (f"  (+{boost:.1f} boost)" if boost else "")
    ) if hist_n else ""
    extra_block = "\n".join(ln for ln in [regime_line, hist_line] if ln)

    # ── Pattern analysis block ───────────────────────────────────────────────
    pattern_block = ""
    if _PF_OK:
        try:
            df_live = st.session_state.get("df")
            pred    = predict_next_outcome(pattern)
            fatigue = check_strategy_fatigue(pattern, df_live, direction)
            regime_chk = detect_regime_shift(df_live) if df_live is not None else {}

            if pred:
                _streak   = pred.get("current_streak", 0)
                _stype    = pred.get("streak_type", "NONE")
                _wlevel   = pred.get("warning_level", "low")
                _seq      = pred.get("sequence", [])
                _rec_pred = pred.get("recommendation", "trade_normal")
                _reason   = pred.get("reason", "")
                _conf_pct = pred.get("confidence", 0.0)
                _hist_pat = pred.get("historical_pattern", "")

                _fat_level = fatigue.get("fatigue_level", "none")
                _fat_rec   = fatigue.get("recommendation", "")
                _fat_sigs  = fatigue.get("signals_triggered", [])

                _rg_same   = regime_chk.get("regime_same", True)
                _rg_chgs   = regime_chk.get("changes_detected", [])
                _rg_risk   = regime_chk.get("risk_level", "low")

                # ── Sequence display ─────────────────────────────────────
                if _seq:
                    seq_str = " ".join(_seq[:-1]) + (" ← you are here" if _seq else "")
                else:
                    seq_str = "No history yet"

                # ── Fatigue warning line ─────────────────────────────────
                fat_icon = {
                    "none":     "✓",
                    "moderate": "⚠",
                    "high":     "⚠",
                    "critical": "🚨",
                }.get(_fat_level, "?")

                fat_header = {
                    "none":     "✓ No fatigue signals",
                    "moderate": "⚠ FATIGUE WARNING — MODERATE",
                    "high":     "⚠ FATIGUE WARNING — HIGH RISK",
                    "critical": "🚨 CRITICAL FATIGUE — DO NOT TRADE",
                }.get(_fat_level, "")

                # ── Regime check lines ────────────────────────────────────
                if _rg_same:
                    regime_status = "✓ Same regime as winning trades"
                else:
                    regime_status = "✗ REGIME SHIFT DETECTED"

                # ── Session line ─────────────────────────────────────────
                current_sess = st.session_state.get("current_session", "London")

                # ── Recommendation ────────────────────────────────────────
                if _wlevel == "critical" or _fat_level == "critical":
                    rec_line = "⛔ SKIP THIS TRADE — High reversal risk"
                    extra_note = "\n  Type OVERRIDE to force-trade at your own risk"
                elif _rec_pred == "reduce_size" or _fat_level in ("high", "moderate"):
                    half_lots = max(0.01, round(final_lots * 0.5, 2))
                    rec_line = f"Take at HALF position size ({half_lots:.2f} lots)"
                    extra_note = "  Move SL to breakeven at TP1"
                else:
                    rec_line  = "Normal position size — no fatigue signals"
                    extra_note = ""

                # ── ATR / volatility line ─────────────────────────────────
                atr_chg = next((c for c in _rg_chgs if "ATR" in c), None)
                vol_status = f"⚠ {atr_chg}" if atr_chg else "✓ Similar ATR conditions"
                sess_chg   = next((c for c in _rg_chgs if "SESSION" in c.upper()), None)
                sess_status = f"⚠ {sess_chg}" if sess_chg else f"✓ {current_sess} session"

                # ── Build block ───────────────────────────────────────────
                pa_lines  = f"\n  {DASH}\n"
                pa_lines += f"  PATTERN ANALYSIS:\n"
                pa_lines += f"  {DASH}\n"
                pa_lines += f"  Strategy sequence: {seq_str}\n"
                if _hist_pat and _hist_pat != "Not enough data":
                    pa_lines += f"  {_hist_pat}\n"
                if _streak > 0:
                    pa_lines += f"  Current streak: {_streak} {'win' if _stype == 'WIN' else 'loss'}{'s' if _streak > 1 else ''}\n"
                pa_lines += f"\n"
                pa_lines += f"  {fat_header}\n"
                if _fat_sigs:
                    for sig_label in _fat_sigs:
                        pa_lines += f"  · {sig_label}\n"
                if _reason:
                    pa_lines += f"  {_reason}\n"
                pa_lines += f"\n"
                pa_lines += f"  REGIME CHECK:\n"
                pa_lines += f"  {regime_status}\n"
                pa_lines += f"  Session:     {sess_status}\n"
                pa_lines += f"  Volatility:  {vol_status}\n"
                pa_lines += f"\n"
                pa_lines += f"  RECOMMENDATION:\n"
                pa_lines += f"  {rec_line}\n"
                if extra_note:
                    pa_lines += f"  {extra_note}\n"
                pa_lines += f"  {DASH}\n"

                pattern_block = pa_lines
        except Exception:
            pattern_block = ""

    # ── Volume analysis block ────────────────────────────────────────────────
    volume_block = ""
    if _VA_OK:
        try:
            df_live = st.session_state.get("live_df") or st.session_state.get("df")
            if df_live is not None:
                vol = check_volume_confluence(df_live, direction, pattern)
                v_ratio   = vol.get("volume_ratio",   0)
                v_class   = vol.get("volume_class",   "?")
                v_score   = vol.get("score",          0)
                v_climax  = vol.get("climax",         False)
                v_optimal = vol.get("strategy_optimal", True)
                v_conf_sc = vol.get("confirmation_score", 0)

                vb  = f"\n  {DASH}\n"
                vb += f"  VOLUME ANALYSIS:\n"
                vb += f"  Ratio:       {v_ratio:.2f}x avg  [{v_class.upper()}]\n"
                vb += f"  Score:       {v_score:+.2f}  |  Confirmed: {v_conf_sc}/3 candles\n"
                if v_climax:
                    ct = vol.get("climax_type", "climax")
                    vb += f"  ⚠  VOLUME CLIMAX ({ct.replace('_', ' ').upper()}) — Exhaustion signal!\n"
                if not v_optimal:
                    vb += f"  ✗  Volume suboptimal for this strategy\n"
                for vln in vol.get("details", [])[:3]:
                    vb += f"  {vln}\n"
                vb += f"  {DASH}\n"
                volume_block = vb
        except Exception:
            volume_block = ""

    # ── SMC analysis block ───────────────────────────────────────────────────
    smc_block = ""
    try:
        _smc_ctx = sig.get("smc_context") or {}
        if not _smc_ctx:
            # Try from confluence raw_checks if available
            _conf_raw = sig.get("_confluence_raw") or {}
            _smc_ctx  = (_conf_raw.get("raw_checks") or {}).get("smc") or {}
        if _smc_ctx and isinstance(_smc_ctx, dict) and "entry_quality" in _smc_ctx:
            _sq_grade = _smc_ctx.get("entry_quality", "?")
            _sq_label = _smc_ctx.get("entry_quality_label", f"Grade {_sq_grade}")
            _sq_zone  = (_smc_ctx.get("premium_discount") or {}).get("current_zone", "?")
            _sq_eq    = (_smc_ctx.get("premium_discount") or {}).get("equilibrium", 0)
            _sq_str   = (_smc_ctx.get("structure") or {}).get("structure", "?")
            _sq_str_b = (_smc_ctx.get("structure") or {}).get("bias", "?")
            _sq_bos   = (_smc_ctx.get("structure") or {}).get("last_bos")
            _sq_choch = (_smc_ctx.get("structure") or {}).get("last_choch")
            _sq_obs   = _smc_ctx.get("order_blocks") or []
            _sq_fvgs  = _smc_ctx.get("fair_value_gaps") or []
            _sq_conf  = _smc_ctx.get("confidence_adjustment", 0.0)

            sb  = f"\n  {DASH}\n"
            sb += f"  SMC ANALYSIS:\n"
            sb += f"  Entry quality: Grade {_sq_grade}  {_sq_label}\n"
            sb += f"  Zone:          {_sq_zone.upper()}  (eq=${_sq_eq:,.2f})"
            sb += f"  [adj {_sq_conf:+.1f}]\n" if _sq_conf != 0 else "\n"

            if _sq_obs:
                _ob0 = _sq_obs[0]
                sb += (
                    f"  Order Block:   ${_ob0['ob_level']:,.2f} "
                    f"[{_ob0['ob_low']:,.2f}–{_ob0['ob_high']:,.2f}]  "
                    f"({'untested' if _ob0.get('untested') else 'tested'})\n"
                )
            else:
                sb += f"  Order Block:   none active\n"

            if _sq_fvgs:
                _fv0 = _sq_fvgs[0]
                sb += (
                    f"  FVG:           ${_fv0['fvg_bottom']:,.2f}–${_fv0['fvg_top']:,.2f} "
                    f"({'unfilled'})\n"
                )
            else:
                sb += f"  FVG:           no unfilled FVG nearby\n"

            _bos_str = f"BOS ${_sq_bos:,.2f}" if _sq_bos else ""
            _choch_str = f"CHoCH ${_sq_choch:,.2f}" if _sq_choch else ""
            _struct_extra = "  |  ".join(s for s in [_bos_str, _choch_str] if s)
            sb += (
                f"  Structure:     {_sq_str.replace('_', ' ').upper()} — "
                f"{_sq_str_b.upper()}"
                + (f"  |  {_struct_extra}" if _struct_extra else "")
                + "\n"
            )
            sb += f"  {DASH}\n"
            smc_block = sb
    except Exception:
        smc_block = ""

    # ── S/R Map block ────────────────────────────────────────────────────────
    sr_block = ""
    try:
        _conf_raw   = sig.get("_confluence_raw") or {}
        _raw_checks = (_conf_raw.get("raw_checks") or {})
        _sr         = _raw_checks.get("sr_levels") or {}

        # Fall back to live computation if not cached
        if not _sr:
            _df_sr = st.session_state.get("live_df") or st.session_state.get("df")
            _sp    = st.session_state.get("live_price", 0)
            if _df_sr is not None and _sp:
                from sr_mapper import get_sr_levels as _gsr
                _sr = _gsr(_df_sr, float(_sp))

        if _sr and (_sr.get("resistance_levels") or _sr.get("support_levels")):
            _res_lvls = _sr.get("resistance_levels", [])[:3]
            _sup_lvls = _sr.get("support_levels",    [])[:3]

            _PROX_ICONS = {
                "IMMEDIATE": "⚠️",
                "NEAR":      "🔶",
                "WATCH":     "🔷",
                "DISTANT":   "",
            }

            srb  = f"\n  {DASH}\n"
            srb += f"  📍 KEY LEVELS:\n"

            if _res_lvls:
                srb += f"  Resistance:\n"
                for _rl in _res_lvls:
                    _icon = _PROX_ICONS.get(_rl.get("proximity", ""), "")
                    srb += (
                        f"    ${_rl['price']:>10,.2f}  {_rl['label'][:28]:<28}  "
                        f"[{_rl['strength']}] {_icon}\n"
                    )
            if _sup_lvls:
                srb += f"  Support:\n"
                for _sl in _sup_lvls:
                    _icon = _PROX_ICONS.get(_sl.get("proximity", ""), "")
                    srb += (
                        f"    ${_sl['price']:>10,.2f}  {_sl['label'][:28]:<28}  "
                        f"[{_sl['strength']}] {_icon}\n"
                    )

            if _sr.get("at_key_level"):
                srb += (
                    f"  ⭐ PRICE AT KEY LEVEL: {_sr.get('key_level_detail', '')}\n"
                    f"  This increases setup quality significantly\n"
                )

            if _sr.get("prev_day_high") or _sr.get("prev_day_low"):
                srb += (
                    f"  Prev Day  H: ${_sr.get('prev_day_high', 0):,.2f}  "
                    f"L: ${_sr.get('prev_day_low', 0):,.2f}\n"
                )
            if _sr.get("prev_week_high") or _sr.get("prev_week_low"):
                srb += (
                    f"  Prev Week H: ${_sr.get('prev_week_high', 0):,.2f}  "
                    f"L: ${_sr.get('prev_week_low', 0):,.2f}\n"
                )
            srb += f"  {DASH}\n"
            sr_block = srb
    except Exception:
        sr_block = ""

    # ── Liquidity map block ──────────────────────────────────────────────────
    liq_block = ""
    if _LIQ_OK:
        try:
            _liq_df = st.session_state.get("live_df") or st.session_state.get("df")
            if _liq_df is not None and len(_liq_df) >= 20:
                _liq_price = float(_liq_df["close"].iloc[-1])
                _liq = _build_liq_map(_liq_df, _liq_price)
                if _liq.get("available"):
                    _ca = _liq.get("clusters_above", [])
                    _cb = _liq.get("clusters_below", [])
                    _poc = _liq.get("poc", 0.0)
                    _lmove = _liq.get("likely_move", "NEUTRAL")
                    _arrow = "⬆" if _lmove == "UP" else ("⬇" if _lmove == "DOWN" else "↔")
                    liq_block = f"  {DASH}\n  LIQUIDITY MAP\n  {DASH}\n"
                    if _ca:
                        liq_block += f"  🔴 BSL above: ${_ca[0]['price']:,.2f}  (${_ca[0]['distance_usd']:,.1f} away)\n"
                    if _cb:
                        liq_block += f"  🟢 SSL below: ${_cb[0]['price']:,.2f}  (${_cb[0]['distance_usd']:,.1f} away)\n"
                    if _poc:
                        liq_block += f"  POC: ${_poc:,.2f}  |  VA: ${_liq.get('va_low',0):,.2f}–${_liq.get('va_high',0):,.2f}\n"
                    liq_block += f"  {_arrow} Likely move: {_lmove}\n"
                    liq_block += f"  {DASH}\n"
        except Exception:
            liq_block = ""

    # ── RSI analysis block ───────────────────────────────────────────────────
    rsi_block = ""
    try:
        df_live = st.session_state.get("live_df") or st.session_state.get("df")
        if df_live is not None and "RSI" in df_live.columns:
            rsi_now  = float(df_live["RSI"].iloc[-1])
            rsi_prev = float(df_live["RSI"].iloc[-5]) if len(df_live) >= 5 else rsi_now
            rsi_zone = (
                "OVERSOLD  ← bounce watch" if rsi_now < 30
                else "near oversold"        if rsi_now < 40
                else "OVERBOUGHT ← short watch" if rsi_now > 70
                else "near overbought"      if rsi_now > 60
                else "NEUTRAL"
            )
            rsi_trend = "Rising ↑" if rsi_now > rsi_prev else "Falling ↓"
            rsi_b  = f"\n  {DASH}\n"
            rsi_b += f"  RSI ANALYSIS:\n"
            rsi_b += f"  RSI:         {rsi_now:.1f}  [{rsi_zone}]\n"
            rsi_b += f"  Trend:       {rsi_trend} (was {rsi_prev:.1f} 5 candles ago)\n"
            # Divergence
            price_up = float(df_live["close"].iloc[-1]) > float(df_live["close"].iloc[-5])
            if not price_up and rsi_now > rsi_prev:
                rsi_b += f"  ⚠ BULLISH DIVERGENCE — price falling, RSI rising\n"
            elif price_up and rsi_now < rsi_prev:
                rsi_b += f"  ⚠ BEARISH DIVERGENCE — price rising, RSI falling\n"
            rsi_b += f"  {DASH}\n"
            rsi_block = rsi_b
    except Exception:
        rsi_block = ""

    # ── RSI divergence block (swing-based, from confluence raw_checks) ───────
    div_block = ""
    try:
        _conf_raw   = sig.get("_confluence_raw") or {}
        _raw_checks = (_conf_raw.get("raw_checks") or {})
        _div_result = _raw_checks.get("rsi_divergence") or {}

        # Fall back to live computation if not in cache
        if not _div_result.get("divergence_found") and _div_result == {}:
            _df_div = st.session_state.get("live_df") or st.session_state.get("df")
            if _df_div is not None:
                from confluence_engine import detect_rsi_divergence as _drd
                _div_result = _drd(_df_div)

        if _div_result.get("divergence_found"):
            _div_type  = _div_result["divergence_type"].replace("_", " ").title()
            _div_str   = _div_result.get("strength", "MODERATE")
            _div_note  = _div_result.get("note", "")
            _div_boost = _div_result.get("confidence_boost", 0.0)
            _div_dir   = _div_result.get("signal_direction", "")
            _ps1       = _div_result.get("price_swing1", 0.0)
            _ps2       = _div_result.get("price_swing2", 0.0)
            _rs1       = _div_result.get("rsi_swing1", 0.0)
            _rs2       = _div_result.get("rsi_swing2", 0.0)
            _sig_dir   = str(sig.get("direction", "")).lower()

            db  = f"\n  {DASH}\n"
            if _div_dir == _sig_dir or not _sig_dir:
                # Divergence agrees with signal
                _str_icon = "🟢🟢" if _div_str == "STRONG" else "🟢"
                db += f"  📊 RSI DIVERGENCE: {_div_type} ({_div_str}) {_str_icon}\n"
                db += f"  {_div_note}\n"
                db += f"  Price:  ${_ps1:,.2f} → ${_ps2:,.2f}  "
                db += f"({'lower low' if _ps2 < _ps1 else 'higher low'})\n"
                db += f"  RSI:    {_rs1:.1f} → {_rs2:.1f}  "
                db += f"({'higher low' if _rs2 > _rs1 else 'lower high'})\n"
                db += f"  Confidence boost: +{_div_boost:.1f}\n"
            else:
                # Divergence opposes signal — warning
                db += f"  ⚠ RSI DIVERGENCE WARNING: {_div_type} detected\n"
                db += f"  This opposes your {_sig_dir.upper()} signal\n"
                db += f"  ({_div_note})\n"
                db += f"  Consider waiting for divergence to resolve\n"
            db += f"  {DASH}\n"
            div_block = db
    except Exception:
        div_block = ""

    # ── Market regime block ──────────────────────────────────────────────────
    regime_block = ""
    try:
        df_live = st.session_state.get("live_df") or st.session_state.get("df")
        if df_live is not None:
            from market_context import detect_gold_regime as _dgr
            _reg = _dgr(df_live)
            reg_name  = _reg.get("regime", "unknown").upper()
            reg_mult  = _reg.get("position_size_multiplier", 1.0)
            best_pbs  = _reg.get("best_playbooks", [])
            avoid_pbs = _reg.get("avoid_playbooks", [])
            pb_name   = sig.get("pattern_name", "")
            reg_fit   = "IDEAL ✓" if pb_name in best_pbs else ("AVOID ✗" if pb_name in avoid_pbs else "OK")
            rb  = f"\n  {DASH}\n"
            rb += f"  MARKET REGIME:  {reg_name}\n"
            rb += f"  Size multiplier: {reg_mult:.1f}x  |  Strategy fit: {reg_fit}\n"
            if best_pbs:
                rb += f"  Best now:  {', '.join(best_pbs[:3])}\n"
            if avoid_pbs and pb_name in avoid_pbs:
                rb += f"  ⚠ {pb_name} not recommended in {reg_name} regime!\n"
            rb += f"  {DASH}\n"
            regime_block = rb
    except Exception:
        regime_block = ""

    # ── Trade management plan block (partial TP + trailing SL) ──────────────
    tm_block = ""
    if _TM_OK:
        try:
            _df_tm = st.session_state.get("live_df") or st.session_state.get("df")
            # Inject computed lot size back so plan uses correct sizing
            _sig_for_plan = dict(sig)
            _sig_for_plan["lots"] = final_lots
            _plan = _calc_tp_plan(_sig_for_plan, _df_tm)
            if _plan.get("valid"):
                _p_dir    = str(_plan["direction"]).upper()
                _p_total  = _plan["total_lots"]
                _p_tp1    = _plan["tp1_price"]
                _p_lot1   = _plan["tp1_lots"]
                _p_tp1p   = _plan["tp1_profit_usd"]
                _p_tp2    = _plan["tp2_price"]
                _p_lot2   = _plan["tp2_lots"]
                _p_tp2p   = _plan["tp2_profit_usd"]
                _p_be     = _plan["breakeven_sl"]
                _p_trail  = _plan["trail_step_usd"]
                _p_best   = _plan["best_case_usd"]
                _p_worst  = _plan["worst_case_usd"]

                tmb  = f"\n  {DASH}\n"
                tmb += f"  📋 TRADE MANAGEMENT PLAN\n"
                tmb += f"  {DASH}\n"
                tmb += f"  Total: {_p_total:.2f} lots\n"
                tmb += f"\n"
                tmb += f"  TP1 → ${_p_tp1:,.2f}  [1:2 RR]\n"
                tmb += f"  → Close {_p_lot1:.2f} lots  (+${_p_tp1p:,.2f})\n"
                tmb += f"  → Move SL to breakeven (${_p_be:,.2f}) — RISK ZERO\n"
                tmb += f"\n"
                tmb += f"  TP2 → ${_p_tp2:,.2f}  [1:3 RR]\n"
                tmb += f"  → Trail remaining {_p_lot2:.2f} lots\n"
                tmb += f"  → Trail SL: 1× ATR (${_p_trail:.2f}) behind price\n"
                tmb += f"  → Target +${_p_tp2p:,.2f}\n"
                tmb += f"\n"
                tmb += f"  BEST CASE:  +${_p_best:,.2f}  (both TPs hit)\n"
                tmb += f"  WORST CASE: −${_p_worst:,.2f}  (SL before TP1)\n"
                tmb += f"  {DASH}\n"
                tm_block = tmb
        except Exception:
            tm_block = ""


    # ── Risk of Ruin warning (HIGH or DANGER) ─────────────────────────────────
    ror_warning_block = ""
    if _TM_OK:
        try:
            _ror_card = _get_ror_profile()
            if _ror_card.get("risk_rating") in ("HIGH", "DANGER"):
                ror_warning_block = (
                    f"\n  {DASH}\n"
                    f"  ⚠ RISK WARNING: {_ror_card['risk_rating']}\n"
                    f"  Ruin probability: {_ror_card['ruin_probability']:.1f}%\n"
                    f"  Consider reducing to {_ror_card.get('recommended_risk_pct', risk_pct)}% risk\n"
                    f"  {DASH}\n"
                )
        except Exception:
            pass

    # ── Build per-gate checklist debug lines ────────────────────────────────
    _gate_names = {
        1: "Trend Align", 2: "Confluence", 3: "Risk/Reward",
        4: "News Safety", 5: "Session",
    }
    _ck_detail_lines = ""
    _ck_results_dict = sl_q.get("check_results", {})
    if _ck_results_dict:
        _gate_parts = []
        for _gi in range(1, 6):
            _gr = _ck_results_dict.get(_gi, {})
            _gicon = "✓" if _gr.get("passed") else "✗"
            _gname = _gate_names.get(_gi, f"Check {_gi}")
            _gate_parts.append(f"{_gicon}{_gname}")
        _ck_detail_lines = "  GATES:     " + "  ".join(_gate_parts) + "\n"
        # Add first failure reason if any gate failed
        for _gi in range(1, 6):
            _gr = _ck_results_dict.get(_gi, {})
            if not _gr.get("passed"):
                _fail_detail = _gr.get("detail", "").split("\n")[0]
                _ck_detail_lines += f"  ✗ Gate {_gi}: {_fail_detail}\n"
                break
    _risk_note_line = ""
    if sl_q.get("risk_note"):
        _risk_note_line = f"  ⚠ Risk: {sl_q['risk_note']}\n"
    _grade_line = (
        f"  Grade: {_instr_grade} ({int(_grade_mult*100)}% size) "
        f"| {asset} instrument tier\n"
    )

    wide_sl_block = ""
    if not_tradeable:
        _needed = pos.get("account_needed", 0)
        wide_sl_block = (
            f"  ⛔ SETUP NOT TRADEABLE\n"
            f"  {_reject_reason}\n"
            + (f"  Account needed: ${_needed:,.0f}\n" if _needed else "")
            + f"\n"
            f"  OPTIONS:\n"
            f"  A) Skip — wait for a tighter SL setup\n"
            f"  B) Increase account to ${_needed:,.0f} if shown above\n"
            f"  {DASH}\n"
        )
    elif pos.get("wide_sl"):
        wide_sl_block = (
            f"  ⚠ SL WIDE — REDUCED POSITION\n"
            f"  Technical SL: ${sl_dist:,.2f} away\n"
            f"  At 0.01 lots: Risk = ${actual_risk:.2f} ({risk_actual_pct:.1f}%)\n"
            f"\n"
            f"  TWO OPTIONS:\n"
            f"  A) Take 0.01 lots — risk only ${actual_risk:.2f} (lower than {risk_pct:.0f}% target)\n"
            f"  B) Skip — wait for a tighter setup\n"
            f"  {DASH}\n"
        )

    direction_word = "BUY" if is_long else "SELL"

    # ── Fresh live price + clock (never cached — always computed now) ─────────
    _gst_tz    = timezone(timedelta(hours=4))
    _now_uae   = datetime.now(_gst_tz)
    _live      = _get_live_price()
    _uae_time  = _now_uae.strftime("%I:%M %p")
    _uae_date  = _now_uae.strftime("%A %d %B %Y")
    _price     = _live["price"] if (_live.get("price") and _live["price"] > 0) else entry
    _source    = _live.get("source", "—")
    _spread    = _live.get("spread") or 0.0
    _price_warn = "  ⚠ STALE PRICE — verify on MT5\n" if not _live.get("is_live") and _source not in ("MT5", "yfinance_1m", "yfinance") else ""
    _sess_line = get_session_summary_line() if _WS_OK else _current_session()
    _HDR       = "━" * 43

    # ── Fundamental bias line ────────────────────────────────────────────────
    _fund_line = ""
    if _FB_OK:
        try:
            _fb = _get_fundamental_bias()
            _fund_line = f"  {_fb.get('display_line', '')}\n"
        except Exception:
            pass

    # ── Regime header line (one-liner for the card header) ───────────────────
    try:
        from market_context import detect_gold_regime as _dgr_hdr
        _df_hdr = st.session_state.get("live_df") or st.session_state.get("df")
        _rdata_hdr = _dgr_hdr(_df_hdr) if _df_hdr is not None else {}
        _regime_hdr       = _rdata_hdr.get("regime", "UNKNOWN")
        _regime_label_hdr = _rdata_hdr.get("regime_label", _regime_hdr)
        _regime_mult_hdr  = _rdata_hdr.get("position_size_multiplier", 1.0)
        _regime_note_hdr  = _rdata_hdr.get("regime_note", "")
        _regime_header_line = (
            f"  📈 REGIME: {_regime_label_hdr}"
            + (f" | ×{_regime_mult_hdr}" if _regime_mult_hdr != 1.0 else "")
            + (f" | {_regime_note_hdr}" if _regime_note_hdr else "")
            + "\n"
        )
    except Exception:
        _regime_header_line = ""
    # ── Session handoff block for trade card ─────────────────────────────────
    _sh_card_block = ""
    try:
        _ny_card = st.session_state.get("ny_bias") or {}
        if _ny_card and _ny_card.get("ny_bias", "NEUTRAL") != "NEUTRAL":
            _nyk  = _ny_card.get("ny_bias", "NEUTRAL")
            _cnk  = _ny_card.get("confidence", "")
            _reck = _ny_card.get("recommendation", "")
            _asn  = _ny_card.get("asian_range", {})
            _ldn  = _ny_card.get("london_break", {})
            _bk_t = _ldn.get("break_type", "")
            _aslo = _asn.get("asian_low", 0)
            _ashi = _asn.get("asian_high", 0)
            _sh_b  = f"\n  {DASH}\n"
            _sh_b += f"  \U0001f4ca SESSION HANDOFF\n"
            _sh_b += f"  {DASH}\n"
            if _aslo and _ashi:
                _sh_b += f"  Asian: ${_aslo:,.2f}\u2013${_ashi:,.2f}\n"
            if _bk_t:
                _sh_b += f"  London: {_bk_t}\n"
            _sh_b += f"  NY Bias: {_nyk} ({_cnk})\n"
            if _reck:
                _sh_b += f"  {_reck}\n"
            _sh_b += f"  {DASH}\n"
            _sh_card_block = _sh_b
    except Exception:
        _sh_card_block = ""

    # ── m15_block fallback (safe — may not be populated yet) ─────────────────
    try:
        _raw_m15 = sig.get("raw_checks", {}) if isinstance(sig, dict) else {}
        _raw_m15 = _raw_m15 if isinstance(_raw_m15, dict) else {}
        m15_block = _raw_m15.get("m15", "")
        if not isinstance(m15_block, str):
            m15_block = ""
    except Exception:
        m15_block = ""

    # ── Build card ───────────────────────────────────────────────────────────
    card = (
        f"```\n"
        f"{_HDR}\n"
        f"  🕐 {_uae_date}  |  {_uae_time} UAE\n"
        f"  💰 XAUUSD: ${_price:,.2f}  [{_source}]\n"
        f"  📊 Spread: ${_spread:.2f}\n"
        f"  🌍 {_sess_line}\n"
        f"{_fund_line}"
        f"{_regime_header_line}"
        f"{_price_warn}"
        f"{_HDR}\n"
        f"{SEP}\n"
        f"  SETUP {idx} — {asset} {direction}\n"
        f"  Strategy: {pattern} · {src_tag}\n"
        f"  Confidence: {conf_f}\n"
        + (
            f"  ML Grade:   {sig.get('ml_grade','?')} — {sig.get('ml_verdict','')}\n"
            f"  Blended:    {sig.get('blended_confidence','—')}%  (ML 40% + Technical 60%)\n"
            if sig.get('ml_enhanced') else ""
        )
        + f"{SEP}\n"
        f"\n"
        f"  Entry:      {_fmt_price(entry)}\n"
        f"  SL:         {_fmt_price(sl)}  ({sl_sign}${sl_dist:,.2f})\n"
        + (
            f"  ⚠ Geo risk SL adjustment: +${_geo_sl_adj:,.2f}  ({_geo_sl_mult:.1f}× ATR)"
            f"  → total buffer: ${sl_dist + _geo_sl_adj:,.2f}\n"
            if _geo_sl_adj > 0 else ""
        )
        + f"\n"
        f"{sl_check_lines}"
        f"  {DASH}\n"
        f"  {ck_line}\n"
        f"{_ck_detail_lines}"
        f"{_risk_note_line}"
        f"\n"
        f"  CONFLUENCE:\n"
        f"{conf_block}\n"
        + (f"\n{extra_block}\n" if extra_block else "")
        + pattern_block
        + volume_block
        + smc_block
        + sr_block
        + liq_block
        + rsi_block
        + div_block
        + regime_block
        + tm_block
        + ror_warning_block
        + _sh_card_block
        + m15_block
        + f"\n"
        f"  {DASH}\n"
        f"  YOUR POSITION ({risk_pct:.0f}% risk · {leverage:.0f}x leverage)\n"
        f"  {DASH}\n"
        f"{_grade_line}"
        + wide_sl_block
        + (
            f"  Session:     [{_sess_name}] Grade [{_sess_grade}]"
            + (f" \u26a0 thin liquidity" if not _sess_rec else "")
            + f" \u2014 lot {_sess_lot_c}, SL {_sess_sl_c}\n"
            if _sess_name else ""
        )
        + f"  Lot size:    {final_lots:.2f} lots\n"
        f"  Risk:        ${actual_risk:.2f} ({risk_actual_pct:.1f}% of ${account:,.0f})\n"
        f"\n"
        + (
            f"  TP1 (50%):  {_fmt_price(tp1)}  (+${tp1_dist:,.0f}) → {rr_actual}\n"
            f"  TP2 (50%):  {_fmt_price(tp2)}  (+${tp2_dist:,.0f}) → 1:{tp2_dist/sl_dist:.1f}\n"
            if partial_tp and tp else
            f"  Take Profit: {_fmt_price(tp)}  (+${tp_dist:,.0f}) → {rr_actual}\n"
        )
        + f"\n"
        f"  If both TPs hit:\n"
        f"  Reward:      +${actual_reward:.2f} ({reward_actual_pct:.1f}% of account)\n"
        f"  Account:     ${account:,.0f} → ${account + actual_reward:,.0f}\n"
        f"\n"
        f"  If SL hit:\n"
        f"  Loss:        −${actual_risk:.2f} ({risk_actual_pct:.1f}% of account)\n"
        f"  Account:     ${account:,.0f} → ${account - actual_risk:,.0f}\n"
        f"  {DASH}\n"
        f"  ENTER ON MT5:\n"
        f"  {direction_word} {asset} · {final_lots:.2f} lots\n"
        f"  SL: {sl:.2f}\n"
        f"  TP: {tp:.2f}\n"
        f"{SEP}\n"
        f"```"
    )
    return card


# ══════════════════════════════════════════════════════════════════════════════
#  Command handlers
# ══════════════════════════════════════════════════════════════════════════════

def _handle_setup(_msg: str) -> str:
    instrument = st.session_state.get("instrument", "XAUUSD")
    # 1. Rules
    rules  = _load_rules()
    n_rules = len(rules)

    # 2. Playbooks
    pb_mod = MODS.get("playbooks")
    n_pb   = len(pb_mod.PLAYBOOKS) if pb_mod and hasattr(pb_mod, "PLAYBOOKS") else 12
    st.session_state["playbooks_count"] = n_pb

    # 3. Live data
    _load_df()
    trend = st.session_state.get("live_trend", "—")

    # 4. MTF bias
    d1_bias = "—"; h4_bias = "—"; overall = "—"; overall_note = ""
    mtf_mod = MODS.get("mtf")
    if mtf_mod and hasattr(mtf_mod, "get_htf_context"):
        try:
            ctx = mtf_mod.get_htf_context()
            if ctx:
                d1_bias = str(ctx.get("d1_bias", "—")).upper()
                h4_bias = str(ctx.get("h4_bias", "—")).upper()
        except Exception:
            pass
    if d1_bias == "—":
        # Derive from EMA200
        d1_bias = trend if trend != "—" else "—"
        h4_bias = d1_bias
    if d1_bias == h4_bias == "BEARISH":
        overall = "STRONGLY BEARISH"; overall_note = "Primary bias: SHORT (D1+H4 bearish) | Reversal longs: monitored separately"
    elif d1_bias == h4_bias == "BULLISH":
        overall = "STRONGLY BULLISH"; overall_note = "Primary bias: LONG (D1+H4 bullish) | Reversal shorts: monitored separately"
    else:
        overall = "CONFLICTED"; overall_note = "Wait for alignment before entering"
    st.session_state["d1_bias"] = d1_bias
    st.session_state["h4_bias"] = h4_bias

    # 5. DXY + US10Y Macro context
    dxy_line = "—"; gold_corr = "—"
    dxy_mod  = MODS.get("dxy")
    if dxy_mod and hasattr(dxy_mod, "get_macro_context"):
        try:
            # Use last-known trend direction for initial macro call
            _last_sent  = st.session_state.get("sentiment") or {}
            _gold_bias  = str((_last_sent.get("gold") or {}).get("bias", "buy")).lower()
            _macro_dir  = {"buy": "long", "sell": "short"}.get(_gold_bias, "long")
            mctx        = dxy_mod.get_macro_context(_macro_dir)
            _dxy_trend  = mctx.get("dxy_trend", "sideways")
            _dxy_rsi_v  = mctx.get("dxy_rsi", "—")
            _yctx       = mctx.get("yields") or {}
            _yld        = _yctx.get("current_yield")
            _yld_str    = f"{_yld:.2f}%" if _yld else "N/A"
            _yld_arr    = {"↑": "rising", "↓": "falling"}.get("→", "→")  # placeholder
            _yld_trend  = _yctx.get("yield_trend", "sideways")
            _y_arr      = {"rising": "↑", "falling": "↓", "sideways": "→"}.get(_yld_trend, "→")
            _mbias      = mctx.get("macro_bias", "neutral")
            dxy_line    = f"{_dxy_trend.capitalize()} (RSI {_dxy_rsi_v}) | US10Y: {_yld_str} {_y_arr}"
            gold_corr   = mctx.get("macro_bias", "neutral").replace("_", " ").title()
            st.session_state["dxy_status"]    = _dxy_trend
            st.session_state["macro_bias"]    = _mbias
            st.session_state["yields_context"] = _yctx
        except Exception as e:
            try:
                dctx      = dxy_mod.get_dxy_context()
                dxy_dir   = str(dctx.get("dxy_trend", "—")).capitalize()
                dxy_rsi_v = dctx.get("dxy_rsi", "—")
                dxy_line  = f"{dxy_dir} (RSI {dxy_rsi_v})"
                gold_corr = "aligned" if dctx.get("gold_aligned") else "diverging"
                st.session_state["dxy_status"] = dxy_dir
            except Exception:
                dxy_line = f"Error: {e}"
    elif dxy_mod and hasattr(dxy_mod, "get_dxy_context"):
        try:
            dctx      = dxy_mod.get_dxy_context()
            dxy_dir   = str(dctx.get("dxy_trend", "—")).capitalize()
            dxy_rsi_v = dctx.get("dxy_rsi", "—")
            dxy_line  = f"{dxy_dir} (RSI {dxy_rsi_v})"
            gold_corr = "Aligned ✓" if dctx.get("gold_aligned") else "Diverging ⚠️"
            st.session_state["dxy_status"] = dxy_dir
        except Exception as e:
            dxy_line = f"Error: {e}"
    else:
        dxy_line = "not available"; gold_corr = "—"

    # 6. Session
    sess = _current_session()
    st.session_state["session_name"] = sess

    # 7. News
    news_line = "—"
    nm = MODS.get("news_mon")
    if nm and hasattr(nm, "fetch_news"):
        try:
            items     = nm.fetch_news()
            sentiment = nm.get_market_sentiment(items) or {}
            st.session_state["sentiment"] = sentiment
            news_line = f"{len(items)} headlines fetched"
        except Exception as e:
            news_line = f"Error: {e}"
    else:
        news_line = "not available"

    st.session_state["last_refresh"] = datetime.now(GST)

    online = lambda m: "online ✓" if MODS.get(m) else "not available ⚠️"

    SEP  = "═" * 35
    DASH = "─" * 35

    out = (
        f"```\n"
        f"{SEP}\n"
        f"  SYSTEM STATUS\n"
        f"{SEP}\n"
        f"  Instrument:   {instrument} H1\n"
        f"  Rules:        {n_rules} loaded ✓\n"
        f"  Playbooks:    {n_pb} active ✓\n"
        f"  SMC engine:   {online('smart')}\n"
        f"  MTF analyzer: {online('mtf')}\n"
        f"  DXY tracker:  {online('dxy')}\n"
        f"  News filter:  {online('news_mon')}\n"
        f"{DASH}\n"
        f"  D1 BIAS:  {d1_bias}\n"
        f"  H4 BIAS:  {h4_bias}\n"
        f"  OVERALL:  {overall}\n"
        f"  {overall_note}\n"
        f"{DASH}\n"
        f"  DXY: {dxy_line}\n"
    )
    if instrument == "XAUUSD":
        out += f"  Gold correlation: {gold_corr}\n"
    out += (
        f"{DASH}\n"
        f"  SESSION: {sess}\n"
        f"  News: {news_line}\n"
        f"{SEP}\n"
        f"```\n"
        f"✅ Setup complete for **{instrument}**. "
        f"Type `analyze {instrument.lower()}` or `signals` for all setups."
    )
    return out


_INSTRUMENT_DISPLAY: dict[str, str] = {
    "XAUUSD": "XAUUSD (Gold)",
    "WTI":    "WTI (Crude Oil)",
    "US30":   "US30 (Dow Jones)",
    "NAS100": "NAS100 (Nasdaq)",
    "GBPUSD": "GBPUSD (Cable)",
    "EURUSD": "EURUSD (Euro)",
}


def _handle_analyze_instrument(instr: str, msg: str, account: float = 300.0) -> str:
    """
    Switch the active instrument to `instr`, then run the appropriate analysis.
    Updates st.session_state["instrument"] so the sidebar dropdown reflects the change.
    """
    # Switch instrument
    st.session_state["instrument"] = instr
    display = _INSTRUMENT_DISPLAY.get(instr, instr)

    # For XAUUSD delegate to the existing gold handler
    if instr == "XAUUSD":
        return _handle_gold(msg, account)

    # For all other instruments: fetch live price then run signals handler
    # which already reads st.session_state["instrument"]
    _lp = _get_live_price(instr)
    if _lp and _lp > 0:
        st.session_state["live_price"] = _lp

    # Route to the generic signals/market-read pipeline with the new instrument set
    return (
        f"### 📊 {display} Analysis\n\n"
        f"*Switched to **{display}** — running full analysis…*\n\n"
    ) + _handle_signals(msg, account)


def _handle_gold(_msg: str, account: float = 300.0) -> str:
    lines = ["### 📊 XAUUSD Analysis\n"]

    # ── Session gate — block low-quality signal generation off-hours ─────────
    # ── Session quality flag (no hard block except Fri 9 PM+ / weekend) ─────
    from datetime import timezone, timedelta
    _GST = timezone(timedelta(hours=4))
    _now_gst   = datetime.now(_GST)
    _uae_hr    = _now_gst.hour
    _uae_dow   = _now_gst.weekday()  # 0=Mon … 6=Sun
    _hard_block = (
        (_uae_dow == 4 and _uae_hr >= 21) or  # Friday after 9 PM UAE
        _uae_dow == 5 or                        # Saturday
        _uae_dow == 6                           # Sunday
    )
    _force = any(k in _msg.lower() for k in
                 ["force", "show anyway", "override", "ignore session", "any signals"])
    if _hard_block and not _force:
        _price_g = st.session_state.get("live_price", 0)
        return (
            f"⏸ **Market Closed — Weekend / Friday close**\n\n"
            f"Forex/Gold markets are closed.\n\n"
            f"**Reopens:** Sunday 10:00 PM UAE\n"
            f"**Best window:** Monday–Friday 4:00 PM – 7:00 PM UAE\n\n"
            f"Current price: **${_price_g:,.2f}**\n\n"
            f"*Type 'show anyway' to force scan.*"
        )
    _optimal_window = (16 <= _uae_hr < 19)  # 4–7 PM UAE preferred
    _off_optimal    = not _optimal_window

    df = _load_df()
    if df is None:
        return "❌ Cannot load price data. Run `setup` first or check `data/historical_xauusd.csv`."

    _window_note = ""
    if _off_optimal:
        _window_note = (
            f"\n> ⚠️ **Outside optimal window** (4–7 PM UAE) — "
            f"signals are valid but confidence may be lower. "
            f"Best entries occur during London/NY overlap.\n"
        )

    price = st.session_state["live_price"]
    rsi   = st.session_state["live_rsi"]
    trend = st.session_state["live_trend"]
    atr   = st.session_state["live_atr"]
    if _window_note:
        lines.append(_window_note)
    lines.append(
        f"**Price:** ${price:,}  |  **RSI:** {rsi}  |  "
        f"**Trend:** {trend}  |  **ATR:** ${atr}"
    )
    sentiment = st.session_state.get("sentiment", {})
    gold_bias = str(sentiment.get("gold", {}).get("bias", "")).lower()
    direction = {"buy": "long", "sell": "short"}.get(
        gold_bias, "short" if trend == "BEARISH" else "long"
    )
    lines.append(
        f"**Bias:** {'🟢 LONG' if direction == 'long' else '🔴 SHORT'}  |  "
        f"**Session:** {_current_session()}\n"
    )

    pb_mod = MODS.get("playbooks")
    ck_mod = MODS.get("checklist")
    signals: list[dict] = []
    fallback: list[dict] = []

    if pb_mod and hasattr(pb_mod, "get_active_playbooks"):
        try:
            hits = pb_mod.get_active_playbooks(df, sentiment, top_n=6)
            lines.append(f"**Playbooks triggered:** {len(hits)}")
            for hit in hits:
                pb = hit["playbook"]
                try:
                    entry, sl, tp = pb_mod.format_playbook_signal(pb, df, hit["direction"])
                except Exception:
                    row   = df.iloc[-1]
                    atv   = float(row.get("atr", 20))
                    entry = float(row["close"])
                    sl    = entry + atv * 1.5 if hit["direction"] == "short" else entry - atv * 1.5
                    tp    = entry - atv * 3.0 if hit["direction"] == "short" else entry + atv * 3.0
                sig: dict = {
                    "source":            "playbook",
                    "asset":             "XAUUSD",
                    "direction":         hit["direction"].upper(),
                    "confidence":        hit["score"],
                    "pattern_name":      pb["name"],
                    "playbook_id":       pb.get("id", ""),
                    "entry":             entry,
                    "stop_loss":         sl,
                    "take_profit":       tp,
                    "confluence_met":    hit.get("met_list",    []),
                    "confluence_missed": hit.get("missed_list", []),
                    "checklist_results": None,
                    "tier":              "B",
                }
                # ── Run full confluence scoring and store detail_lines ────
                try:
                    from confluence_engine import score_confluences as _sc
                    _conf_result = _sc(df, hit["direction"], symbol="XAUUSD")
                    sig["detail_lines"] = _conf_result.get("detail_lines", [])
                    sig["confluence_result"] = _conf_result
                    # Use confluence confidence if better than playbook score
                    _ce_conf = _conf_result.get("confidence", 0)
                    if _ce_conf > 0:
                        sig["confidence"] = _ce_conf
                except Exception:
                    sig["detail_lines"] = []

                if ck_mod and hasattr(ck_mod, "validate_entry"):
                    try:
                        ck = ck_mod.validate_entry(sig, df)
                        sig["checklist_results"] = ck
                        sig["confidence"]        = ck["final_confidence"]
                        if ck["checks_passed"] >= 4:
                            signals.append(sig)
                        else:
                            fallback.append(sig)
                    except Exception:
                        fallback.append(sig)
                else:
                    fallback.append(sig)
        except Exception as e:
            lines.append(f"⚠️ Playbook error: {e}")

    lines.append("")
    show = signals if signals else fallback[:3]
    st.session_state["last_signals"] = show  # save for win/loss recording

    # ── Brain 1: apply auto-filters ─────────────────────────────────────────────
    auto_filters = _load_auto_filters()
    if auto_filters and show:
        for s in show:
            pname = str(s.get("pattern_name", "")).lower()
            for af in auto_filters:
                if pname == str(af.get("pattern", "")).lower():
                    n_times = int(af.get("times_triggered", 0))
                    s["confidence"] = max(1.0, s["confidence"] - 1.0)
                    flag = "\ud83d\udeab Pattern flagged — review" if n_times >= 5 else ""
                    old_note = s.get("note", "")
                    s["note"] = (
                        (old_note + " | " if old_note else "") +
                        f"⚠ Auto-filter: failed {n_times}x for {af.get('filter_reason','?')}"
                        + (f"  {flag}" if flag else "")
                    )

    # ── Brain 2: register signals with signal_tracker ────────────────────────
    if _ST_OK and show:
        try:
            _spread = _check_spread_live("XAUUSD")
            for s in show:
                s["spread_at_signal"] = _spread.get("spread_usd") or 0.0
                s["signal_id"] = _st_register(s)
        except Exception:
            pass

    # ── Geo risk — fetch and apply to signals ─────────────────────────────────
    if _GEO_OK and show:
        try:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _gex:
                _gfut = _gex.submit(_get_geo)
                try:
                    _geo_ctx = _gfut.result(timeout=20)
                except Exception:
                    _geo_ctx = {}
            if _geo_ctx.get("available"):
                st.session_state["geo_ctx"]        = _geo_ctx
                st.session_state["geo_risk_level"] = _geo_ctx.get("geo_risk_level", "normal")
                _g_sl_mult = float(_geo_ctx.get("sl_atr_multiplier", 0.0))
                _g_conf    = float(_geo_ctx.get("confidence_adjustment", 0.0))
                _g_level   = str(_geo_ctx.get("geo_risk_level", "normal"))
                for _s in show:
                    _s["sl_atr_multiplier"] = _g_sl_mult
                    _s["geo_risk_level"]    = _g_level
                    if _g_conf != 0.0:
                        _s["confidence"] = round(min(10.0, max(1.0, float(_s.get("confidence", 5)) + _g_conf)), 1)
        except Exception:
            pass

    # Log to mt5_sync signal file so auto-match can link them later
    if _MT5_SYNC_OK and show:
        try:
            sess_name = st.session_state.get("session_name") or _current_session()
            for s in show:
                log_bot_signal({
                    **s,
                    "session":      s.get("session", sess_name),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            pass
    # ── Fundamental conflict check ──────────────────────────────────────────
    if _FB_OK and show:
        try:
            _fund_ctx = _get_fundamental_bias()
            for _s in show:
                _tech_dir = _s.get("direction", "LONG")
                _conf_chk = _check_fund_conflict(_tech_dir)
                if _conf_chk.get("conflict"):
                    _sev = _conf_chk["severity"]
                    _adj = -2 if _sev == "HIGH" else -1
                    _s["confidence"] = round(max(1.0, float(_s.get("confidence", 5)) + _adj), 1)
                    _old = _s.get("note", "")
                    _s["note"] = (_old + " | " if _old else "") + f"⚠ FUND CONFLICT ({_sev})"
                    if _sev == "HIGH" and float(_s.get("confidence", 5)) < 4:
                        _s["note"] += " — LOW CONF SKIP"
        except Exception:
            pass

    # ── Regime strategy filter ────────────────────────────────────────────
    if show:
        try:
            from market_context import detect_gold_regime as _dgr_g, get_regime_strategy_config as _grsc_g
            _df_rg  = st.session_state.get("live_df") or st.session_state.get("df")
            if _df_rg is not None:
                _reg_g  = _dgr_g(_df_rg)
                _rcfg_g = _grsc_g(_reg_g["regime"])
                _rname_g = _reg_g["regime"]
                _rmult_g = _reg_g["position_size_multiplier"]
                for _sg in show:
                    _pn_g   = str(_sg.get("pattern_name", "")).lower()
                    _src_g  = str(_sg.get("source", ""))
                    _ef_g   = _rcfg_g["entry_filter"]
                    _rn_g   = _sg.get("note", "")
                    if _ef_g == "sr_bounce_only" and not any(
                            p in _pn_g for p in ["bounce", "double", "fibonacci", "s/r", "support"]):
                        _sg["confidence"] = max(1.0, float(_sg["confidence"]) - 1.0)
                        _sg["note"] = (_rn_g + " | " if _rn_g else "") + "⚠ Not ideal for ranging market"
                    elif _ef_g == "breakout_only" and "breakout" not in _pn_g:
                        _sg["confidence"] = max(1.0, float(_sg["confidence"]) - 1.5)
                        _sg["note"] = (_rn_g + " | " if _rn_g else "") + "⚠ Wait for breakout in squeeze"
                    elif _ef_g == "news_fade_only" and "news" not in _pn_g:
                        _sg["confidence"] = max(1.0, float(_sg["confidence"]) - 1.0)
                        _sg["note"] = (_rn_g + " | " if _rn_g else "") + "⚠ High vol — news fade preferred"
                    if _src_g == "reversal_hunter" and _rname_g == "TRENDING_STRONG":
                        _sg["confidence"] = max(1.0, float(_sg["confidence"]) - 0.5)
                        _sg["note"] = (_rn_g + " | " if _rn_g else "") + \
                            f"⚠ COUNTER-TREND — dominant trend is {_rname_g} — smaller size recommended"
                    _sg["size_multiplier"] = _rmult_g
                    _sg["regime_config"]   = _rcfg_g
        except Exception:
            pass

    # ── Sort all signals — NO blocking, all shown ────────────────────────────
    ranked = sorted(show, key=lambda s: s.get("confidence", 0), reverse=True)

    # Collect fresh reversal signals and merge
    _rev_fresh: list[dict] = []
    if _RH_OK:
        try:
            _rh_df    = st.session_state.get("live_df")
            _rh_price = st.session_state.get("live_price")
            _rev_fresh = _hunt_reversals(_rh_df, _rh_price)
        except Exception:
            pass

    _trend_sigs    = [s for s in ranked if s.get("source") != "reversal_hunter"]
    _reversal_sigs = [s for s in ranked if s.get("source") == "reversal_hunter"]
    # Merge fresh reversals not already present
    _rev_ids = {(r.get("direction"), r.get("entry")) for r in _reversal_sigs}
    for _rf in _rev_fresh:
        if (_rf.get("direction"), _rf.get("entry")) not in _rev_ids:
            _reversal_sigs.append(_rf)

    all_signals = _trend_sigs[:3] + _reversal_sigs  # top 3 trend + all reversals

    # ── M15 entry confirmation for trend signals ──────────────────────────────
    if _trend_sigs:
        try:
            _df_m15 = _load_m15_df()
            if _df_m15 is not None:
                _m15_close = float(_df_m15["close"].iloc[-1])
                _m15_ema50 = float(_df_m15["ema50"].iloc[-1])
                _m15_rsi   = float(_df_m15["rsi"].iloc[-1])
                _m15_atr   = float(_df_m15["atr"].iloc[-1])
                _m15_last3 = _df_m15["close"].tail(3).tolist()
                _m15_mom   = "up" if _m15_last3[-1] > _m15_last3[0] else "down"
                for _sig_m in _trend_sigs[:3]:
                    _dir_m = str(_sig_m.get("direction", "")).lower()
                    if _dir_m == "long":
                        _m15_sl_v = float(_df_m15["low"].tail(5).min()) - (_m15_atr * 0.3)
                        _m15_conf = (_m15_close > _m15_ema50) or (_m15_rsi > 45)
                        _m15_note = (
                            f"M15 entry: {chr(9989) + ' confirmed' if _m15_conf else chr(9203) + ' wait for bounce'}"
                            f" | M15 SL: ${_m15_sl_v:,.2f} (tighter)"
                            f" | M15 momentum: {_m15_mom}"
                        )
                    else:
                        _m15_sl_v = float(_df_m15["high"].tail(5).max()) + (_m15_atr * 0.3)
                        _m15_conf = (_m15_close < _m15_ema50) or (_m15_rsi < 55)
                        _m15_note = (
                            f"M15 entry: {chr(9989) + ' confirmed' if _m15_conf else chr(9203) + ' wait for rejection'}"
                            f" | M15 SL: ${_m15_sl_v:,.2f} (tighter)"
                            f" | M15 momentum: {_m15_mom}"
                        )
                    _sig_m["m15_sl"]        = round(_m15_sl_v, 2)
                    _sig_m["m15_confirmed"] = _m15_conf
                    _sig_m["m15_note"]      = _m15_note
                    _sig_m["m15_rsi"]       = round(_m15_rsi, 1)
                    _sig_m["m15_momentum"]  = _m15_mom
        except Exception:
            pass

    # ── Quality label helper ──────────────────────────────────────────────────
    def _quality_label(sig: dict) -> tuple[str, str]:
        _conf = float(sig.get("confidence", 0) or 0)
        _ck   = (sig.get("checklist_results") or {}).get("checks_passed", 0)
        if _conf >= 7.0 and _ck >= 4:
            return "🟢 HIGH QUALITY", "Full size"
        elif _conf >= 5.5 and _ck >= 2:
            return "🟡 MODERATE", "50% size"
        elif _conf >= 4.0:
            return "🔴 LOW QUALITY", "25% size — tight SL"
        else:
            return "⚫ SPECULATIVE", "10% size — monitor only"

    # ── HTF bias header ───────────────────────────────────────────────────────
    _d1_g = str(st.session_state.get("d1_bias", "—")).upper()
    _h4_g = str(st.session_state.get("h4_bias", "—")).upper()
    if _d1_g == _h4_g == "BEARISH":
        _overall_g = "STRONGLY BEARISH"
        _trend_emoji = "📉"
    elif _d1_g == _h4_g == "BULLISH":
        _overall_g = "STRONGLY BULLISH"
        _trend_emoji = "📈"
    else:
        _overall_g = ""
        _trend_emoji = "↔"
    if _overall_g:
        lines.append(f"\n{_trend_emoji} **Overall bias: {_overall_g}**\n")

    # ── PART 3 — Scenario block ───────────────────────────────────────────────
    if _reversal_sigs and _trend_sigs:
        _sc_rev   = _reversal_sigs[0]
        _sc_trend = _trend_sigs[0]
        _sc_rv_dir = str(_sc_rev.get("direction", "")).lower()
        _sc_tr_dir = str(_sc_trend.get("direction", "")).lower()
        _sc_rv_entry = float(_sc_rev.get("entry", 0) or 0)
        _sc_rv_sl    = float(_sc_rev.get("stop_loss", 0) or 0)
        _sc_rv_tp    = float(_sc_rev.get("take_profit", 0) or 0)
        _sc_tr_tp    = float(_sc_trend.get("take_profit", 0) or 0)
        _sc_rv_move  = abs(_sc_rv_tp - _sc_rv_entry)
        _sc_tr_move  = abs(_sc_tr_tp - _sc_rv_tp)
        if _sc_rv_dir != _sc_tr_dir:
            lines.append(
                f"## 📋 FULL TRADE SCENARIO\n\n"
                f"**Market context:** {_sc_tr_dir.upper()} trend "
                f"with short-term {_sc_rv_dir.upper()} reversal\n\n"
                f"**STEP 1 — Reversal first:**\n"
                f"{_sc_rv_dir.upper()} from ${_sc_rv_entry:,.2f} → ${_sc_rv_tp:,.2f}\n"
                f"SL: ${_sc_rv_sl:,.2f} | Expected move: +${_sc_rv_move:.2f}\n"
                f"*Take profit at TP, then watch for Step 2*\n\n"
                f"**STEP 2 — After reversal completes:**\n"
                f"Watch for {_sc_tr_dir.upper()} rejection at ${_sc_rv_tp:,.2f}\n"
                f"{_sc_tr_dir.upper()} entry ~${_sc_rv_tp:,.2f} → ${_sc_tr_tp:,.2f}\n"
                f"SL: above ${_sc_rv_tp + 5:,.2f} | "
                f"Expected move: +${_sc_tr_move:.2f}\n\n"
                f"**Combined potential: +${_sc_rv_move + _sc_tr_move:.2f}**\n\n"
                f"─────────────────────────────\n"
                f"*Step 1 is optional — skip to Step 2 "
                f"if you only want trend-following trades.*\n"
            )
    elif _trend_sigs and not _reversal_sigs:
        _sc_trend = _trend_sigs[0]
        lines.append(
            f"## 📋 TRADE SCENARIO\n\n"
            f"**Trend:** {_sc_trend.get('direction','').upper()} continuation\n"
            f"**No reversal detected** — trend is clear\n"
            f"**Entry:** ${float(_sc_trend.get('entry',0) or 0):,.2f} | "
            f"**Target:** ${float(_sc_trend.get('take_profit',0) or 0):,.2f}\n\n"
        )
    elif _reversal_sigs and not _trend_sigs:
        _sc_rev = _reversal_sigs[0]
        lines.append(
            f"## 📋 TRADE SCENARIO\n\n"
            f"**Counter-trend reversal only**\n"
            f"Main trend: {trend.upper() if trend else 'BEARISH'}\n"
            f"Short-term bounce detected: "
            f"{_sc_rev.get('direction','').upper()} → "
            f"${float(_sc_rev.get('take_profit',0) or 0):,.2f}\n"
            f"After bounce: expect main trend to RESUME\n"
            f"Use 25-50% size only\n\n"
        )

    # ── PART 4 — Show ALL trend signals (top 3) ───────────────────────────────
    if _trend_sigs:
        lines.append(f"─────────────────────────────")
        lines.append(f"### 📊 TREND SIGNALS ({len(_trend_sigs[:3])} shown)\n")
        for _ti, _ts in enumerate(_trend_sigs[:3], 1):
            _ql, _sa = _quality_label(_ts)
            _tc   = float(_ts.get("confidence", 0) or 0)
            _tck  = ((_ts.get("checklist_results") or {}).get("checks_passed", 0))
            _tdir = str(_ts.get("direction", "")).upper()
            _tpat = _ts.get("pattern_name", "Strategy")
            _tent = float(_ts.get("entry", 0) or 0)
            _tsl  = float(_ts.get("stop_loss", 0) or 0)
            _ttp  = float(_ts.get("take_profit", 0) or 0)
            _tsl_d = abs(_tent - _tsl)
            _ttp_d = abs(_ttp - _tent)
            _trr   = _ttp_d / _tsl_d if _tsl_d > 0 else 0
            _ttp2  = round(_tent + _tsl_d * 3 if _tdir == "LONG" else _tent - _tsl_d * 3, 2)
            _ttp2_d = abs(_ttp2 - _tent)
            _conf_met = (_ts.get("confluence_met") or [])[:3]
            _lots_rough = max(0.01, round(account * 0.015 / max(_tsl_d * 100, 1), 2))
            lines.append(
                f"\n{_ql} **SETUP {_ti} — XAUUSD {_tdir}**\n"
                f"Strategy: {_tpat} | Confidence: {_tc:.1f}/10 | "
                f"Checklist: {_tck}/5 | Size: {_sa}\n\n"
                f"Entry:  **${_tent:,.2f}**\n"
                f"SL:     ${_tsl:,.2f} (-${_tsl_d:.2f})\n"
                f"TP1:    ${_ttp:,.2f} (+${_ttp_d:.2f}) → 1:2\n"
                f"TP2:    ${_ttp2:,.2f} (+${_ttp2_d:.2f}) → 1:3\n\n"
                + (f"Why: {', '.join(_conf_met)}\n\n" if _conf_met else "")
                + (f"M15: {_ts.get('m15_note','')}\n\n" if _ts.get("m15_note") else "")
                + f"**Enter on MT5:** {'BUY' if _tdir == 'LONG' else 'SELL'} XAUUSD "
                f"{_lots_rough:.2f} lots | SL: {_tsl:.2f} | TP: {_ttp:.2f}\n"
            )
    else:
        lines.append(
            "\nNo trend signals found right now.\n"
            "Try again during London/NY session (12pm–9pm UAE)."
        )

    # ── Show ALL reversal signals ─────────────────────────────────────────────
    if _reversal_sigs:
        lines.append(f"\n─────────────────────────────")
        lines.append(f"### 🔄 REVERSAL SIGNALS ({len(_reversal_sigs)} found)\n")
        for _rev in _reversal_sigs:
            _rl, _rsa   = _quality_label(_rev)
            _rv_str     = _rev.get("reversal_strength", "MODERATE")
            _rv_dir     = str(_rev.get("direction", "")).upper()
            _rv_entry   = float(_rev.get("entry",       0) or 0)
            _rv_sl      = float(_rev.get("stop_loss",   0) or 0)
            _rv_tp      = float(_rev.get("take_profit", 0) or 0)
            _rv_score   = _rev.get("score", 0)
            _rv_sl_d    = abs(_rv_entry - _rv_sl)
            _rv_tp_d    = abs(_rv_tp - _rv_entry)
            _rv_rr      = _rv_tp_d / _rv_sl_d if _rv_sl_d > 0 else 0
            _rv_conds   = (_rev.get("conditions_met") or [])[:3]
            _rv_reason  = _rev.get("key_reason", "")
            _rv_lots    = max(0.01, round(account * 0.015 / max(_rv_sl_d * 100, 1) * 0.5, 2))
            lines.append(
                f"\n{_rl} **REVERSAL {_rv_str} — {_rv_dir}**\n"
                f"Score: {_rv_score}/13 | {_rv_reason} | Size: {_rsa}\n\n"
                f"Entry:  **${_rv_entry:,.2f}**\n"
                f"SL:     ${_rv_sl:,.2f} (-${_rv_sl_d:.2f})\n"
                f"TP:     ${_rv_tp:,.2f} (+${_rv_tp_d:.2f}) → 1:{_rv_rr:.1f}\n\n"
                + (f"Conditions: {', '.join(_rv_conds)}\n\n" if _rv_conds else "")
                + f"⚠ Counter-trend — use 25-50% size. Tight SL.\n"
                f"**Enter on MT5:** {'BUY' if _rv_dir == 'LONG' else 'SELL'} XAUUSD "
                f"{_rv_lots:.2f} lots | SL: {_rv_sl:.2f} | TP: {_rv_tp:.2f}\n"
            )
        lines.append(
            f"\n> ⚠️ *Reversals go against the dominant trend. "
            f"Use 25–50% normal size. Tight SL only.*"
        )

    # ── Store best signal for paper trading ──────────────────────────────────
    if _trend_sigs:
        st.session_state["last_signal"] = _trend_sigs[0]
    elif _reversal_sigs:
        st.session_state["last_signal"] = _reversal_sigs[0]

    # ── Safety fallback — never return None ──────────────────────────────────
    result = "\n".join(lines) if lines else ""
    if not result.strip():
        result = (
            f"⏳ Scanning for setups...\n\n"
            f"No signals generated right now.\n"
            f"Current XAUUSD: **${st.session_state.get('live_price',0):,.2f}** [MT5]\n\n"
            f"Try: 'market read' for plain English analysis."
        )
    result += (
        "\n\n──────────────────────────────\n"
        "*Type **'trade plan'** for all pending orders, limit entries,\n"
        "and full scenario breakdown (current + pending, both directions).*"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ML SUGGEST command handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_ml_suggest(_msg: str, account: float = 300.0) -> str:
    """Fetch top signal, run ML enhancement, display dedicated ML suggestion card."""
    try:
        if not _ML_OK or _ml_engine is None:
            return "⚠️ ML engine is not available yet. Try `train ml` first."

        df = _load_df()
        if df is None:
            return "❌ Cannot load price data. Run `setup` first."

        sentiment = st.session_state.get("sentiment", {})
        pb_mod = MODS.get("playbooks")
        ck_mod = MODS.get("checklist")
        candidates: list = []

        if pb_mod and hasattr(pb_mod, "get_active_playbooks"):
            hits = pb_mod.get_active_playbooks(df, sentiment, top_n=5)
            for hit in hits:
                pb = hit["playbook"]
                try:
                    entry, sl, tp = pb_mod.format_playbook_signal(pb, df, hit["direction"])
                except Exception:
                    row   = df.iloc[-1]
                    atv   = float(row.get("atr", 20))
                    entry = float(row["close"])
                    sl    = entry + atv * 1.5 if hit["direction"] == "short" else entry - atv * 1.5
                    tp    = entry - atv * 3.0 if hit["direction"] == "short" else entry + atv * 3.0
                sig: dict = {
                    "source":       "playbook",
                    "asset":        "XAUUSD",
                    "direction":    hit["direction"].upper(),
                    "confidence":   hit["score"],
                    "pattern_name": pb["name"],
                    "entry": entry, "stop_loss": sl, "take_profit": tp,
                    "checklist_results": None,
                }
                if ck_mod and hasattr(ck_mod, "validate_entry"):
                    try:
                        ck = ck_mod.validate_entry(sig, df)
                        sig["checklist_results"] = ck
                        sig["confidence"]        = ck["final_confidence"]
                    except Exception:
                        pass
                candidates.append(sig)

        if not candidates:
            return "⚠️ No signals available right now. Try again during London/NY session."

        # Pick highest confidence, enhance with ML
        candidates.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        sig = _ml_engine.enhance_signal(candidates[0])

        direction = str(sig.get("direction", "LONG")).upper()
        entry     = float(sig.get("entry", 0) or 0)
        pattern   = sig.get("pattern_name", "Strategy")
        conf      = sig.get("confidence", 0)
        ml_grade  = sig.get("ml_grade", "?")
        ml_score  = sig.get("ml_score", 50)
        blended   = sig.get("blended_confidence", conf)
        verdict   = sig.get("ml_verdict", "")
        red_flags = sig.get("ml_red_flags", [])
        grn_flags = sig.get("ml_green_flags", [])

        grade_icons = {"A": "🟢", "B": "🔵", "C": "🟠", "D": "🔴"}
        g_icon = grade_icons.get(ml_grade, "⚪")

        if ml_grade in ("A", "B"):
            final_call = "✅ TAKE THIS TRADE"
        elif ml_grade == "C":
            final_call = "⏳ WAIT — conditions marginal, watch for improvement"
        else:
            final_call = "❌ SKIP — ML advises against this setup"

        lines = [f"### 🤖 ML Suggestion\n"]
        lines.append(f"**{direction}** {sig.get('asset','XAUUSD')} — _{pattern}_")
        lines.append(f"**Entry:** ${entry:,.2f}  |  **Confidence:** {conf:.1f}/10")
        lines.append(f"**{g_icon} ML Grade: `{ml_grade}`** — {verdict}")
        lines.append(f"**⚡ Blended Confidence: `{blended}%`** _(ML 40% + Technical 60%)_")

        if red_flags:
            lines.append("\n**⚠️ Red Flags:**")
            for f in red_flags:
                lines.append(f"🔴 {f}")

        if grn_flags:
            lines.append("\n**✅ Green Flags:**")
            for f in grn_flags:
                lines.append(f"🟢 {f}")

        lines.append(f"\n---\n## {final_call}")
        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ ML Suggest error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# BEST SIGNAL command handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_best_signal(_msg: str, account: float = 300.0) -> str:
    """Fetch LONG and SHORT signals, ML-enhance both, return the better one."""
    try:
        if not _ML_OK or _ml_engine is None:
            return "⚠️ ML engine is not available yet. Try `train ml` first."

        df = _load_df()
        if df is None:
            return "❌ Cannot load price data. Run `setup` first."

        sentiment = st.session_state.get("sentiment", {})
        pb_mod = MODS.get("playbooks")
        ck_mod = MODS.get("checklist")
        candidates: list = []

        if pb_mod and hasattr(pb_mod, "get_active_playbooks"):
            hits = pb_mod.get_active_playbooks(df, sentiment, top_n=10)
            for hit in hits:
                pb = hit["playbook"]
                try:
                    entry, sl, tp = pb_mod.format_playbook_signal(pb, df, hit["direction"])
                except Exception:
                    row   = df.iloc[-1]
                    atv   = float(row.get("atr", 20))
                    entry = float(row["close"])
                    sl    = entry + atv * 1.5 if hit["direction"] == "short" else entry - atv * 1.5
                    tp    = entry - atv * 3.0 if hit["direction"] == "short" else entry + atv * 3.0
                sig: dict = {
                    "source":       "playbook",
                    "asset":        "XAUUSD",
                    "direction":    hit["direction"].upper(),
                    "confidence":   hit["score"],
                    "pattern_name": pb["name"],
                    "entry": entry, "stop_loss": sl, "take_profit": tp,
                    "checklist_results": None,
                }
                if ck_mod and hasattr(ck_mod, "validate_entry"):
                    try:
                        ck = ck_mod.validate_entry(sig, df)
                        sig["checklist_results"] = ck
                        sig["confidence"]        = ck["final_confidence"]
                    except Exception:
                        pass
                candidates.append(sig)

        if not candidates:
            return "⚠️ No signals available right now."

        # ML-enhance all candidates, pick highest blended_confidence
        enhanced = [_ml_engine.enhance_signal(s) for s in candidates]
        best = max(enhanced, key=lambda s: float(s.get("blended_confidence", s.get("confidence", 0))))

        ml_grade  = best.get("ml_grade", "?")
        blended   = best.get("blended_confidence", best.get("confidence", 0))
        verdict   = best.get("ml_verdict", "")
        red_flags = best.get("ml_red_flags", [])
        grn_flags = best.get("ml_green_flags", [])

        grade_icons = {"A": "🟢", "B": "🔵", "C": "🟠", "D": "🔴"}
        g_icon = grade_icons.get(ml_grade, "⚪")

        if ml_grade in ("A", "B"):
            final_call = "✅ TAKE THIS TRADE"
        elif ml_grade == "C":
            final_call = "⏳ WAIT — conditions marginal"
        else:
            final_call = "❌ SKIP — ML advises against"

        direction = str(best.get("direction", "LONG")).upper()
        entry     = float(best.get("entry", 0) or 0)
        pattern   = best.get("pattern_name", "Strategy")
        conf      = best.get("confidence", 0)

        lines = ["### 🏆 Best Setup Right Now — ML Selected\n"]
        lines.append(f"**{direction}** {best.get('asset','XAUUSD')} — _{pattern}_")
        lines.append(f"**Entry:** ${entry:,.2f}  |  **Confidence:** {conf:.1f}/10")
        lines.append(f"**{g_icon} ML Grade: `{ml_grade}`** — {verdict}")
        lines.append(f"**⚡ Blended Confidence: `{blended}%`** _(ML 40% + Technical 60%)_")

        if red_flags:
            lines.append("\n**⚠️ Red Flags:**")
            for f in red_flags:
                lines.append(f"🔴 {f}")

        if grn_flags:
            lines.append("\n**✅ Green Flags:**")
            for f in grn_flags:
                lines.append(f"🟢 {f}")

        lines.append(f"\n---\n## {final_call}")
        lines.append("\n" + _render_trade_card(best, 1, account))
        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ Best Signal error: {e}"


def _handle_signals(_msg: str, account: float = 300.0) -> str:
    _sig_instr = st.session_state.get("instrument", "XAUUSD")
    lines = [f"### 📡 Signal Scan — {_sig_instr}\n"]

    # ── Session quality flag (no hard block except Fri 9 PM+ / weekend) ─────
    from datetime import timezone, timedelta
    _GST2      = timezone(timedelta(hours=4))
    _now_gst2  = datetime.now(_GST2)
    _uae_hr2   = _now_gst2.hour
    _uae_dow2  = _now_gst2.weekday()  # 0=Mon … 6=Sun
    _hard_block2 = (
        (_uae_dow2 == 4 and _uae_hr2 >= 21) or  # Friday after 9 PM UAE
        _uae_dow2 == 5 or                          # Saturday
        _uae_dow2 == 6                             # Sunday
    )
    _force2 = any(k in _msg.lower() for k in
                  ["force", "show anyway", "override", "ignore session", "any signals"])
    if _hard_block2 and not _force2:
        _price_s = st.session_state.get("live_price", 0)
        return (
            f"⏸ **Market Closed — Weekend / Friday close**\n\n"
            f"Forex/Gold markets are closed.\n\n"
            f"**Reopens:** Sunday 10:00 PM UAE\n"
            f"**Best window:** Monday–Friday 4:00 PM – 7:00 PM UAE\n\n"
            f"Current {_sig_instr}: **${_price_s:,.2f}**\n\n"
            f"*Type 'show anyway' to force scan.*"
        )
    _optimal_window2 = (16 <= _uae_hr2 < 19)
    _off_optimal2    = not _optimal_window2

    df = _load_df()
    if df is None:
        return "❌ Cannot load price data. Run `setup` first."

    _window_note2 = ""
    if _off_optimal2:
        _window_note2 = (
            f"\n> ⚠️ **Outside optimal window** (4–7 PM UAE) — "
            f"signals are valid but confidence may be lower. "
            f"Best entries occur during London/NY overlap.\n"
        )

    sentiment = st.session_state.get("sentiment", {})
    pb_mod = MODS.get("playbooks")
    ck_mod = MODS.get("checklist")
    passed: list[dict] = []

    if pb_mod and hasattr(pb_mod, "get_active_playbooks"):
        try:
            hits = pb_mod.get_active_playbooks(df, sentiment, top_n=10)
            for hit in hits:
                pb = hit["playbook"]
                try:
                    entry, sl, tp = pb_mod.format_playbook_signal(pb, df, hit["direction"])
                except Exception:
                    row   = df.iloc[-1]
                    atv   = float(row.get("atr", 20))
                    entry = float(row["close"])
                    sl    = entry + atv * 1.5 if hit["direction"] == "short" else entry - atv * 1.5
                    tp    = entry - atv * 3.0 if hit["direction"] == "short" else entry + atv * 3.0
                sig: dict = {
                    "source":            "playbook",
                    "asset":             _sig_instr,
                    "direction":         hit["direction"].upper(),
                    "confidence":        hit["score"],
                    "pattern_name":      pb["name"],
                    "playbook_id":       pb.get("id", ""),
                    "entry":             entry,
                    "stop_loss":         sl,
                    "take_profit":       tp,
                    "confluence_met":    hit.get("met_list",    []),
                    "confluence_missed": hit.get("missed_list", []),
                    "checklist_results": None,
                    "tier":              "B",
                }
                # ── Run full confluence scoring and store detail_lines ────
                try:
                    from confluence_engine import score_confluences as _sc
                    _conf_result = _sc(df, hit["direction"], symbol=_sig_instr)
                    sig["detail_lines"] = _conf_result.get("detail_lines", [])
                    sig["confluence_result"] = _conf_result
                    _ce_conf = _conf_result.get("confidence", 0)
                    if _ce_conf > 0:
                        sig["confidence"] = _ce_conf
                except Exception:
                    sig["detail_lines"] = []

                if ck_mod and hasattr(ck_mod, "validate_entry"):
                    try:
                        ck = ck_mod.validate_entry(sig, df)
                        sig["checklist_results"] = ck
                        sig["confidence"]        = ck["final_confidence"]
                        if ck["checks_passed"] >= 4:
                            passed.append(sig)
                    except Exception:
                        pass
                else:
                    passed.append(sig)
        except Exception as e:
            lines.append(f"⚠️ Scan error: {e}")

    passed.sort(key=lambda s: s.get("confidence", 0), reverse=True)
    st.session_state["last_signals"] = passed  # save for win/loss recording

    # ── ML enhance all signals ────────────────────────────────────────────────
    if _ML_OK and _ml_engine is not None and passed:
        try:
            passed = [_ml_engine.enhance_signal(s) for s in passed]
        except Exception:
            pass

    # ── Instrument confluence hard rules + extra score (non-XAUUSD) ───────────
    _ic_instr = st.session_state.get("instrument", "XAUUSD")
    if _ic_instr != "XAUUSD" and _IC_OK and _InstrumentConfluence is not None and passed:
        try:
            _ic_obj  = _InstrumentConfluence(_ic_instr)
            _ic_ctx  = _ic_obj.get_full_signal_context()
            if _ic_ctx.get("blocked"):
                _ic_viols = _ic_ctx.get("violations", [])
                _viol_md  = "\n".join(f"🔴 {v}" for v in _ic_viols)
                return (
                    f"⛔ **{_ic_instr} signal blocked by instrument rules:**\n\n"
                    f"{_viol_md}\n\n"
                    f"_Wait for conditions to be met before trading this instrument._"
                )
            _ic_extra   = _ic_ctx.get("extra_score", 0)
            _ic_factors = _ic_ctx.get("extra_factors", [])
            if _ic_extra != 0:
                for _s in passed:
                    _s["confidence"] = min(100, max(0,
                        float(_s.get("confidence", 50)) + _ic_extra))
                    _s["extra_confluence_factors"] = _ic_factors
        except Exception:
            pass

    # ── Brain 1: apply auto-filters ─────────────────────────────────────────────
    auto_filters = _load_auto_filters()
    if auto_filters and passed:
        for s in passed:
            pname = str(s.get("pattern_name", "")).lower()
            for af in auto_filters:
                if pname == str(af.get("pattern", "")).lower():
                    n_times = int(af.get("times_triggered", 0))
                    s["confidence"] = max(1.0, s["confidence"] - 1.0)
                    flag = "\ud83d\udeab Pattern flagged — review" if n_times >= 5 else ""
                    old_note = s.get("note", "")
                    s["note"] = (
                        (old_note + " | " if old_note else "") +
                        f"⚠ Auto-filter: failed {n_times}x for {af.get('filter_reason','?')}"
                        + (f"  {flag}" if flag else "")
                    )

    # ── Brain 2: register signals with signal_tracker ────────────────────────
    if _ST_OK and passed:
        try:
            _spread = _check_spread_live("XAUUSD")
            for s in passed:
                s["spread_at_signal"] = _spread.get("spread_usd") or 0.0
                s["signal_id"] = _st_register(s)
        except Exception:
            pass

    # ── Geo risk — fetch and apply to signals ────────────────────────────────
    if _GEO_OK and passed:
        try:
            import concurrent.futures as _cf2
            with _cf2.ThreadPoolExecutor(max_workers=1) as _gex2:
                _gfut2 = _gex2.submit(_get_geo)
                try:
                    _geo_ctx2 = _gfut2.result(timeout=20)
                except Exception:
                    _geo_ctx2 = {}
            if _geo_ctx2.get("available"):
                st.session_state["geo_ctx"]        = _geo_ctx2
                st.session_state["geo_risk_level"] = _geo_ctx2.get("geo_risk_level", "normal")
                _g_sl_mult2 = float(_geo_ctx2.get("sl_atr_multiplier", 0.0))
                _g_conf2    = float(_geo_ctx2.get("confidence_adjustment", 0.0))
                _g_level2   = str(_geo_ctx2.get("geo_risk_level", "normal"))
                for _s2 in passed:
                    _s2["sl_atr_multiplier"] = _g_sl_mult2
                    _s2["geo_risk_level"]    = _g_level2
                    if _g_conf2 != 0.0:
                        _s2["confidence"] = round(min(10.0, max(1.0, float(_s2.get("confidence", 5)) + _g_conf2)), 1)
        except Exception:
            pass

    # ── Fundamental conflict check ──────────────────────────────────────────
    if _FB_OK and passed:
        try:
            for _sp in passed:
                _tech_dir_s = _sp.get("direction", "LONG")
                _conf_chk_s = _check_fund_conflict(_tech_dir_s)
                if _conf_chk_s.get("conflict"):
                    _sev_s = _conf_chk_s["severity"]
                    _adj_s = -2 if _sev_s == "HIGH" else -1
                    _sp["confidence"] = round(max(1.0, float(_sp.get("confidence", 5)) + _adj_s), 1)
                    _old_s = _sp.get("note", "")
                    _sp["note"] = (_old_s + " | " if _old_s else "") + f"⚠ FUND CONFLICT ({_sev_s})"
        except Exception:
            pass

    # ── Regime strategy filter ────────────────────────────────────────────
    if passed:
        try:
            from market_context import detect_gold_regime as _dgr_s, get_regime_strategy_config as _grsc_s
            _df_rs  = st.session_state.get("live_df") or st.session_state.get("df")
            if _df_rs is not None:
                _reg_s  = _dgr_s(_df_rs)
                _rcfg_s = _grsc_s(_reg_s["regime"])
                _rname_s = _reg_s["regime"]
                _rmult_s = _reg_s["position_size_multiplier"]
                for _sps in passed:
                    _pn_s  = str(_sps.get("pattern_name", "")).lower()
                    _srcs  = str(_sps.get("source", ""))
                    _ef_s  = _rcfg_s["entry_filter"]
                    _rns   = _sps.get("note", "")
                    if _ef_s == "sr_bounce_only" and not any(
                            p in _pn_s for p in ["bounce", "double", "fibonacci", "s/r", "support"]):
                        _sps["confidence"] = max(1.0, float(_sps["confidence"]) - 1.0)
                        _sps["note"] = (_rns + " | " if _rns else "") + "⚠ Not ideal for ranging market"
                    elif _ef_s == "breakout_only" and "breakout" not in _pn_s:
                        _sps["confidence"] = max(1.0, float(_sps["confidence"]) - 1.5)
                        _sps["note"] = (_rns + " | " if _rns else "") + "⚠ Wait for breakout in squeeze"
                    elif _ef_s == "news_fade_only" and "news" not in _pn_s:
                        _sps["confidence"] = max(1.0, float(_sps["confidence"]) - 1.0)
                        _sps["note"] = (_rns + " | " if _rns else "") + "⚠ High vol — news fade preferred"
                    if _srcs == "reversal_hunter" and _rname_s == "TRENDING_STRONG":
                        _sps["confidence"] = max(1.0, float(_sps["confidence"]) - 0.5)
                        _sps["note"] = (_rns + " | " if _rns else "") + \
                            f"⚠ COUNTER-TREND — dominant trend is {_rname_s} — smaller size recommended"
                    _sps["size_multiplier"] = _rmult_s
                    _sps["regime_config"]   = _rcfg_s
        except Exception:
            pass

    # ── Session handoff — NY bias filter ───────────────────────────────────
    if _SH_OK and df is not None and passed:
        try:
            _handoff_s = st.session_state.get("ny_bias") or _get_ny_bias(df)
            st.session_state["ny_bias"] = _handoff_s
            _ny_s   = _handoff_s.get("ny_bias", "NEUTRAL")
            _conf_s = _handoff_s.get("confidence", "LOW")
            for _sps2 in passed:
                _dir_s = str(_sps2.get("direction", "")).lower()
                _note_s = _sps2.get("note", "")
                if _ny_s == "BULLISH" and _dir_s == "long" and _conf_s == "HIGH":
                    _sps2["confidence"] = min(10.0, float(_sps2.get("confidence", 5)) + 0.5)
                    _sps2["note"] = (_note_s + " | " if _note_s else "") + "✓ Session handoff confirms LONG"
                elif _ny_s == "BEARISH" and _dir_s == "short" and _conf_s == "HIGH":
                    _sps2["confidence"] = min(10.0, float(_sps2.get("confidence", 5)) + 0.5)
                    _sps2["note"] = (_note_s + " | " if _note_s else "") + "✓ Session handoff confirms SHORT"
                elif (_ny_s == "BULLISH" and _dir_s == "short") or (_ny_s == "BEARISH" and _dir_s == "long"):
                    _sps2["confidence"] = max(0.0, float(_sps2.get("confidence", 5)) - 0.5)
                    _sps2["note"] = (_note_s + " | " if _note_s else "") + f"⚠ Session handoff opposes {_dir_s.upper()}"
            if _handoff_s.get("fake_break_alert") and "fake_break_notified" not in st.session_state:
                st.session_state["fake_break_notified"] = True
                _sh_notifs2 = st.session_state.get("mt5_sync_notifications", [])
                _sh_notifs2.append(
                    f"⚠ FAKE BREAK DETECTED — {_handoff_s['london_break']['note']}"
                )
                st.session_state["mt5_sync_notifications"] = _sh_notifs2
        except Exception:
            pass

    if not passed:
        lines.append("⚠️ No signals passed checklist 4/5+ right now.\n")
        if _window_note2:
            lines.append(_window_note2)
        lines.append("Possible reasons:")
        lines.append("- Confluence below threshold (< 3 factors aligned)")
        lines.append("- R:R below 2.0 minimum\n")
        lines.append("Try `gold` to see best available setups without gate filter.")
    else:
        if _window_note2:
            lines.append(_window_note2)
        lines.append(f"**{len(passed)} signal(s) passed — sorted by confidence:**\n")
        for i, sig in enumerate(passed[:5], 1):
            lines.append(_render_trade_card(sig, i, account))

    # ── Reversal Hunter ───────────────────────────────────────────────────────
    if _RH_OK:
        try:
            df_live    = st.session_state.get("live_df")
            live_price = st.session_state.get("live_price")
            reversals  = _hunt_reversals(df_live, live_price)
            for rev in reversals:
                if rev["reversal_strength"] == "STRONG":
                    lines.append(
                        f"\n🔄 **REVERSAL SIGNAL — {rev['reversal_strength']}**\n"
                        f"Score: {rev['score']}/11\n"
                        f"Key reason: {rev['key_reason']}\n"
                        + _render_trade_card(rev, 99, account)
                    )
                elif rev["reversal_strength"] == "MODERATE":
                    lines.append(
                        f"\n🔄 **REVERSAL OPPORTUNITY — {rev['reversal_strength']}**\n"
                        f"Score: {rev['score']}/11 — {rev['key_reason']}\n"
                        f"Direction: {rev['direction'].upper()} | "
                        f"Entry: ${rev['entry']:,.2f} | "
                        f"SL: ${rev['stop_loss']:,.2f} | "
                        f"TP: ${rev['take_profit']:,.2f}\n"
                        f"*Moderate confidence — smaller size recommended*"
                    )
        except Exception:
            pass

    # ── Safety fallback — never return None ──────────────────────────────────
    result = "\n".join(lines) if lines else ""
    if not result or result.strip() == "":
        result = (
            "⏳ Scanning for setups...\n\n"
            "No signals met the minimum quality threshold right now.\n"
            f"- Volume may be below threshold\n"
            f"- Try again in 15-30 minutes\n\n"
            f"Current XAUUSD: "
            f"**${st.session_state.get('live_price',0):,.2f}** [MT5]\n\n"
            f"Or type **'show anyway'** to force scan with all signals."
        )
    return result


def _handle_news(_msg: str) -> str:
    lines = ["### 📰 Economic Calendar — Today\n"]

    nf = MODS.get("news_fil")
    if nf and hasattr(nf, "get_todays_events"):
        try:
            events = nf.get_todays_events(impact_filter={"High", "Medium"})
            if not events:
                lines.append("✅ No high/medium impact events today.")
            else:
                lines.append("| Time (GST) | Impact | Event | Country |")
                lines.append("|---|---|---|---|")
                icons = {"High": "🔴", "Medium": "🟡"}
                for ev in events:
                    impact = ev.get("impact", "")
                    icon   = icons.get(impact, "⚪")
                    title  = ev.get("title", "—")[:42]
                    cty    = ev.get("country", "—")
                    t_gst  = ev.get("time_gst", ev.get("time_utc", "—"))
                    warn   = " ⚠️" if impact == "High" else ""
                    lines.append(f"| {t_gst}{warn} | {icon} {impact} | {title} | {cty} |")
        except Exception as e:
            lines.append(f"⚠️ Calendar error: {e}")
    else:
        lines.append("⚠️ `news_filter` module not available.")

    nm = MODS.get("news_mon")
    if nm and hasattr(nm, "fetch_news"):
        try:
            items     = nm.fetch_news()
            sentiment = nm.get_market_sentiment(items) or {}
            gold      = sentiment.get("gold", {})
            bias      = str(gold.get("bias", "wait")).upper()
            conf_v    = gold.get("confidence", "—")
            summary   = str(gold.get("summary", ""))[:130]
            lines.append(f"\n**Gold headline sentiment:** {bias} ({conf_v}/10)")
            if summary:
                lines.append(f"_{summary}_")
            st.session_state["sentiment"] = sentiment
        except Exception as e:
            lines.append(f"\n⚠️ Headline error: {e}")

    return "\n".join(lines)


def _handle_risk(_msg: str, account: float = 1000.0) -> str:
    risk_pct    = 1.5
    risk_dollar = round(account * risk_pct / 100, 2)
    daily_limit = round(account * 0.05, 0)
    SEP = "═" * 45

    rows = ""
    for sl_d in [10, 15, 20, 25, 30, 35, 40, 50]:
        lots   = _lot_size(account, risk_pct, float(sl_d))
        actual = round(lots * sl_d * 100, 2)
        rows  += f"  SL ${sl_d:<5}→  {lots:.2f} lots  →  risk ${actual:.0f}\n"

    out = (
        f"```\n"
        f"{SEP}\n"
        f"  POSITION SIZE GUIDE\n"
        f"  Account: ${account:,.0f}  |  Risk: {risk_pct}% = ${risk_dollar:.2f}\n"
        f"{SEP}\n"
        f"{rows}"
        f"{SEP}\n"
        f"  Rule: Never risk more than ${risk_dollar:.2f} per trade\n"
        f"  Rule: Max 3 trades open at once\n"
        f"  Rule: Stop trading if down ${daily_limit:.0f} today\n"
        f"{SEP}\n"
        f"```\n"
        f"\n**Formula:** `Lots = Risk$ ÷ (SL distance × 100)`\n"
        f"_(Gold: 1 lot = 100 oz · $1 move per oz = $100 P&L per lot)_"
    )
    return out


def _handle_backtest(msg: str) -> str:
    lower    = msg.lower()
    lines    = ["### 🧪 Backtest Engine\n"]
    name_hint = ""
    for kw in ["backtest ", "test "]:
        if kw in lower:
            name_hint = lower.split(kw, 1)[1].strip()
            break

    # Try new pipeline first
    try:
        from backtest import run_backtest, generate_backtest_report
        settings = _load_settings()
        playbook = name_hint or "London Breakout"
        _bt_instr = st.session_state.get("instrument", "XAUUSD")
        with st.spinner(f"Running pipeline backtest for '{playbook}' [{_bt_instr}]…"):
            results = run_backtest(playbook, settings=settings,
                                   instrument=_bt_instr)
        if not results:
            lines.append("⚠️ No historical data found. Try `run setup` first.")
            return "\n".join(lines)
        _path, report = generate_backtest_report(results, playbook, settings)
        lines.append(f"```\n{report}\n```")
        traded   = [r for r in results if r.get("stage_rejected") is None]
        rejected = [r for r in results if r.get("stage_rejected") is not None]
        lines.append(f"\n✅ **{len(traded)} trades** executed | **{len(rejected)} rejected** by pipeline")
        lines.append(f"📁 Full report saved to: `{_path}`")
        return "\n".join(lines)
    except Exception as _bt_err:
        pass   # fall back to legacy table

    # Legacy fallback
    rules = _load_rules()
    if not rules:
        return "❌ No rules loaded. Run `setup` first."
    matched = (
        [r for r in rules if
         name_hint in str(r.get("name", "")).lower() or
         name_hint in str(r.get("pattern_name", "")).lower()]
        if name_hint else
        [r for r in rules if r.get("tier") in ("A", "B")]
    )
    if not matched:
        lines.append(f"⚠️ No rule matching **'{name_hint}'**. Showing Tier A/B.\n")
        matched = [r for r in rules if r.get("tier") in ("A", "B")]
    lines.append(f"**{len(matched)} rule(s) found:**\n")
    lines.append("| Rule | Tier | Win Rate | Prof Factor | Trades |")
    lines.append("|---|---|---|---|---|")
    for r in matched[:15]:
        name = (r.get("name") or r.get("pattern_name", "—"))[:38]
        tier = r.get("tier", "?")
        bt   = r.get("backtest", {}) or {}
        wr   = bt.get("win_rate",      r.get("win_rate",      0))
        pf   = bt.get("profit_factor", r.get("profit_factor", 0))
        n    = bt.get("total_trades",  r.get("total_trades",  "—"))
        try:
            wr_s = f"{float(wr)*100:.1f}%" if float(wr) <= 1 else f"{float(wr):.1f}%"
        except Exception:
            wr_s = str(wr)
        lines.append(f"| {name} | **{tier}** | {wr_s} | {pf} | {n} |")
    if len(matched) > 15:
        lines.append(f"_...and {len(matched)-15} more_")
    return "\n".join(lines)


def _handle_post_loss(msg: str) -> str:
    """Analyze recent losses with the full post-loss pipeline."""
    try:
        from backtest import run_post_loss_analysis
    except ImportError:
        return "❌ Backtest module not available."

    journal = _load_journal() or []
    losses  = [t for t in journal if str(t.get("outcome", "")).lower() == "loss"]
    if not losses:
        return "✅ No losses found in your journal to analyze."

    losses.sort(key=lambda t: str(t.get("closed_at") or t.get("open_time") or ""), reverse=True)
    recent = losses[:5]

    lines = ["### 🔬 Post-Loss Analysis\n"]
    for i, trade in enumerate(recent, 1):
        try:
            result = run_post_loss_analysis(trade)
            lines.append(
                f"**Trade {i} — {trade.get('pattern_name','?')} ({trade.get('direction','?').upper()})**"
            )
            lines.append(f"- Similar setups found: {result['similar_setups_found']}")
            lines.append(f"- Historical win rate: {result['win_rate']}%")
            lines.append(f"- Primary failure reason: `{result['primary_failure_reason']}`")
            lines.append(f"- Fix: {result['fix_description']}")
            lines.append(f"- Expected: {result['expected_improvement']}")
            lines.append(f"- Confidence: {result['confidence_in_fix']*100:.0f}%")
            if result["fix_applied"]:
                lines.append("✅ Fix auto-applied to `auto_fixes.json`")
            else:
                lines.append("📋 Saved to `pending_fixes.json` for review")
            lines.append("")
        except Exception as exc:
            lines.append(f"⚠️ Trade {i}: {exc}\n")

    return "\n".join(lines)


def _handle_learning_report(_msg: str) -> str:
    """Brain 1 — what has the bot learned from real trades."""
    lines = ["### 🧠 Brain 1 — Learning Report\n"]

    # ── Pattern memory stats ──────────────────────────────────────────────
    memory: list[dict] = []
    if _MT5_SYNC_OK:
        try:
            memory = _load_json(
                os.path.join(DATA_DIR, "pattern_memory.json"), []
            )
        except Exception:
            pass

    if not memory:
        lines.append("⚠️ No real trade data yet. Trades are learned after MT5 auto-match runs.")
    else:
        from collections import defaultdict
        pb_groups: dict = defaultdict(list)
        for m in memory:
            pb = m.get("playbook", "unknown")
            pb_groups[pb].append(m)

        lines.append("#### Patterns with 3+ real trades:\n")
        lines.append("| Pattern | Trades | Win% | Avg RR | Best Regime | Best Session | Top Failure |")
        lines.append("|---|---|---|---|---|---|---|")
        for pb, trades in sorted(pb_groups.items(), key=lambda x: len(x[1]), reverse=True):
            if len(trades) < 3:
                continue
            wins     = sum(1 for t in trades if str(t.get("outcome", "")).lower() == "win")
            wr       = round(wins / len(trades) * 100, 1) if trades else 0.0
            rr_vals  = [float(t["rr_achieved"]) for t in trades if t.get("rr_achieved") is not None]
            avg_rr   = round(sum(rr_vals) / len(rr_vals), 2) if rr_vals else 0.0
            # best regime (highest win rate per regime)
            reg_wins: dict = defaultdict(lambda: [0, 0])
            for t in trades:
                rg = t.get("regime", "unknown")
                reg_wins[rg][1] += 1
                if str(t.get("outcome", "")).lower() == "win":
                    reg_wins[rg][0] += 1
            best_rg = max(reg_wins, key=lambda k: reg_wins[k][0] / max(reg_wins[k][1], 1), default="—")
            # best session
            sess_wins: dict = defaultdict(lambda: [0, 0])
            for t in trades:
                ss = t.get("session", "unknown")
                sess_wins[ss][1] += 1
                if str(t.get("outcome", "")).lower() == "win":
                    sess_wins[ss][0] += 1
            best_ss = max(sess_wins, key=lambda k: sess_wins[k][0] / max(sess_wins[k][1], 1), default="—")
            # top failure reason from failed_trade log
            top_fail = "—"
            try:
                import os as _os
                fail_log_path = _os.path.join(DATA_DIR, "logs", "failed_trade_analysis.json")
                fail_log = _load_json(fail_log_path, [])
                fail_reasons = [
                    f["primary_reason"]
                    for f in fail_log
                    if f.get("playbook") == pb and f.get("primary_reason")
                ]
                if fail_reasons:
                    top_fail = max(set(fail_reasons), key=fail_reasons.count)
            except Exception:
                pass
            lines.append(f"| {pb[:30]} | {len(trades)} | {wr}% | {avg_rr} | {best_rg} | {best_ss} | {top_fail} |")

    # ── Auto-filters active ───────────────────────────────────────────────
    filters = _load_auto_filters()
    if filters:
        lines.append("\n#### ⚠ Active Auto-Filters\n")
        lines.append("| Pattern | Reason | Triggered | Action |")
        lines.append("|---|---|---|---|")
        for f in filters:
            lines.append(
                f"| {f.get('pattern','?')} | {f.get('filter_reason','?')} "
                f"| {f.get('times_triggered','?')}x | {f.get('action','?')} |"
            )
    else:
        lines.append("\n✅ No auto-filters active yet.")

    # ── Live vs backtest divergence ───────────────────────────────────────
    try:
        rules = _load_json(RULES_FILE, [])
        diffs = []
        for r in rules:
            live_wr = r.get("live_win_rate")
            bt      = r.get("backtest", {}) or {}
            bt_wr   = bt.get("win_rate", r.get("win_rate"))
            if live_wr is None or bt_wr is None:
                continue
            bt_wr_f   = float(bt_wr) * 100 if float(bt_wr) <= 1 else float(bt_wr)
            live_wr_f = float(live_wr)
            diffs.append((abs(live_wr_f - bt_wr_f), r.get("name", "?"), live_wr_f, bt_wr_f))
        diffs.sort(reverse=True)
        if diffs:
            lines.append("\n#### Top 5 Rules: Live vs Backtest Win Rate\n")
            lines.append("| Rule | Live WR | Backtest WR | Δ |")
            lines.append("|---|---|---|---|")
            for diff, name, lwr, bwr in diffs[:5]:
                arrow = "↑" if lwr > bwr else "↓"
                lines.append(f"| {name[:38]} | {lwr:.1f}% | {bwr:.1f}% | {arrow} {diff:.1f}% |")
    except Exception:
        pass

    return "\n".join(lines)


def _handle_signal_performance(_msg: str) -> str:
    """Brain 2 — bot signal accuracy report."""
    if not _ST_OK:
        return "⚠️ `signal_tracker` module not available."

    rep = _st_report()
    if not rep:
        return "⚠️ No signal performance data yet."

    ov  = rep.get("overall", {})
    uc  = rep.get("user_comparison", {})
    pb  = rep.get("by_playbook",  {})
    ses = rep.get("by_session",   {})
    reg = rep.get("by_regime",    {})
    verdict = rep.get("verdict", "")

    lines = ["### 📡 Brain 2 — Bot Signal Accuracy\n"]
    lines.append(
        f"**Total signals tracked:** {ov.get('total_signals', 0)}  "
        f"| **Resolved:** {ov.get('resolved', 0)}  "
        f"| **Open:** {ov.get('open', 0)}"
    )
    lines.append(
        f"**Win rate:** {ov.get('win_rate', 0)}%  "
        f"| **Avg win:** +{ov.get('avg_win_pips', 0)} pips  "
        f"| **Avg loss:** -{ov.get('avg_loss_pips', 0)} pips  "
        f"| **Profit factor:** {ov.get('profit_factor', 0)}"
    )
    lines.append(
        f"**Best playbook:** {ov.get('best_playbook', '—')}  "
        f"| **Worst:** {ov.get('worst_playbook', '—')}  "
        f"| **Best session:** {ov.get('best_session', '—')}\n"
    )

    if pb:
        lines.append("#### By Playbook\n")
        lines.append("| Playbook | Signals | Win% | Avg Pips | Real RR |")
        lines.append("|---|---|---|---|---|")
        for name, d in sorted(pb.items(), key=lambda x: -x[1].get("win_rate", 0)):
            rr_str = f"{d['avg_real_rr']:.2f}" if d.get("avg_real_rr") is not None else "—"
            lines.append(
                f"| {name[:32]} | {d.get('signals',0)} | {d.get('win_rate',0)}% "
                f"| {d.get('avg_pips',0):+.1f} | {rr_str} |"
            )

    if ses:
        lines.append("\n#### By Session\n")
        lines.append("| Session | Signals | Win% |")
        lines.append("|---|---|---|")
        for s, d in sorted(ses.items(), key=lambda x: -x[1].get("win_rate", 0)):
            lines.append(f"| {s} | {d.get('total',0)} | {d.get('win_rate',0)}% |")

    if reg:
        lines.append("\n#### By Regime\n")
        lines.append("| Regime | Signals | Win% |")
        lines.append("|---|---|---|")
        for r, d in sorted(reg.items(), key=lambda x: -x[1].get("win_rate", 0)):
            lines.append(f"| {r} | {d.get('total',0)} | {d.get('win_rate',0)}% |")

    lines.append("\n#### 🔍 You vs The Bot\n")
    lines.append(
        f"Signals you took: **{uc.get('signals_user_took',0)}**  "
        f"→ your win rate: **{uc.get('user_win_rate',0)}%**"
    )
    lines.append(
        f"Signals you skipped: **{uc.get('signals_user_skipped',0)}**  "
        f"→ skipped win rate: **{uc.get('skipped_win_rate',0)}%**"
    )
    lines.append(f"\n**{verdict}**")

    return "\n".join(lines)


def _handle_full_brain_report(_msg: str) -> str:
    """Combined Brain 1 + Brain 2 report."""
    b1 = _handle_learning_report(_msg)
    b2 = _handle_signal_performance(_msg)
    uc_lines = []
    if _ST_OK:
        try:
            rep    = _st_report()
            uc     = rep.get("user_comparison", {})
            verdict = rep.get("verdict", "")
            uc_lines = [
                "### 🔍 You vs The Bot\n",
                f"Your win rate:  **{uc.get('user_win_rate',0)}%** ({uc.get('signals_user_took',0)} trades taken)",
                f"Bot win rate:   **{uc.get('overall_bot_win_rate',0)}%** ({rep.get('overall',{}).get('resolved',0)} signals resolved)",
                f"Skipped signal win rate: **{uc.get('skipped_win_rate',0)}%**",
                f"\n**{verdict}**",
            ]
        except Exception:
            pass
    separator = "\n\n---\n"
    return b1 + separator + b2 + (separator + "\n".join(uc_lines) if uc_lines else "")


def _handle_help(_msg: str) -> str:
    _instr = st.session_state.get("instrument", "XAUUSD")
    return f"""### 📋 TradingBotV1 — Command Reference

**Current instrument:** `{_instr}` (change in sidebar)

#### 📊 Analysis
| Command | What it does |
|---|---|
| `analyze gold` / `analyze xauusd` | Full XAUUSD analysis with trade cards |
| `analyze nas100` / `analyze nasdaq` | Full NAS100 analysis |
| `analyze us30` / `analyze dow` | Full US30 analysis |
| `analyze eurusd` / `analyze euro` | Full EURUSD analysis |
| `analyze gbpusd` / `analyze cable` | Full GBPUSD analysis |
| `analyze wti` / `analyze oil` | Full WTI Crude analysis |
| `show signals` | All signals that passed checklist 4/5+ |
| `market read` | Current market structure + bias summary |
| `indicators` | Full 14-indicator technical scan |
| `scalp setup` | M15 scalp signal for current session |

#### 🌍 Multi-Instrument (new)
| Command | What it does |
|---|---|
| `price check` | Live price + context for selected instrument |
| `macro analysis` / `macro bias` | Forex economic health score (GBPUSD/EURUSD) |
| `sector rotation` / `money flow` | Sector ETF flows driving NAS100/US30/WTI |
| `open interest` / `volume analysis` | Volume signal — institutional buying/selling |
| `instrument rules` | Hard entry rules for selected instrument |

#### 🤖 ML & Learning
| Command | What it does |
|---|---|
| `ml suggest` | ML quality assessment for next trade |
| `best signal` | ML best setup picker |
| `best yield` | Top strategies ranked by pips/hour |
| `ml insights` / `why analysis` | ML pattern learning report |
| `train ml` | Retrain ML on paper trade history |

#### 💰 Paper Trading
| Command | What it does |
|---|---|
| `paper long` / `paper buy` | Open a mock long trade |
| `paper short` / `paper sell` | Open a mock short trade |
| `paper trades` | Show open paper positions |
| `close paper` | Close a paper trade manually |
| `paper results` | Paper trade P&L stats |

#### 🧪 Backtest & Research
| Command | What it does |
|---|---|
| `backtest [name]` | Backtest stats for a named strategy |
| `cot report` | Commitment of Traders data |
| `liquidity map` | Order flow heatmap |
| `why did it fail` | Post-loss pattern analysis |
| `weekly review` | This week's trade summary |

#### ℹ️ Utilities
| Command | What it does |
|---|---|
| `run setup` / `refresh` | Load rules, live price, MTF bias |
| `sessions` / `market hours` | Session board + current active session |
| `news today` / `calendar` | Economic calendar + sentiment |
| `risk guide` / `position size` | Position sizing for your balance |
| `risk of ruin` | RoR probability check |
| `export logs` | Full session export for analysis |
| `help` / `?` | This list |

**Account:** Pepperstone #51486884 · `{_instr}` H1
**Note:** This bot analyses only — all trades placed manually on MT5.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Export handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_export(_msg: str) -> str:
    """Build and return a full session export."""
    if not _DBG_OK:
        return (
            "⚠️ **Debug logger not available.**\n\n"
            "Make sure `debug_logger.py` is in the project folder and restart the bot."
        )
    try:
        settings = _load_settings()
        acct     = st.session_state.get("mt5_account")
        acct_name = (
            f"Pepperstone #{acct['account']}"
            if acct else "Pepperstone #51486884"
        )
        filepath, _ = build_export(
            account_name=acct_name,
            settings=settings,
        )
        # Extract just the filename for display
        fname = os.path.basename(filepath)
        log_info(f"Session export requested by user → {fname}")
        return (
            f"✅ **Export ready.**\n\n"
            f"📁 Find it at:\n"
            f"```\ndata/logs/{fname}\n```\n\n"
            f"Copy the full contents of that file and paste to Claude for:\n"
            f"- Full analysis of every signal and rejection\n"
            f"- Why signals were filtered out\n"
            f"- Error diagnosis\n"
            f"- Improvements to the bot\n\n"
            f"_Session: {settings.get('balance','?')}$ · {settings.get('risk_pct','?')}% risk · "
            f"{settings.get('leverage','?')}x · 1:{settings.get('implied_rr','?')} RR_"
        )
    except Exception as e:
        return f"⚠️ Export failed: {e}"


def _handle_general(msg: str) -> str:
    lower = msg.lower()

    if any(w in lower for w in ["ema", "moving average", "golden cross", "death cross"]):
        return (
            "**EMA Trend Logic in TradingBotV1:**\n\n"
            "- **EMA50** = short-term momentum\n"
            "- **EMA200** = long-term trend direction\n"
            "- Price **above EMA200** → Bullish → look for longs\n"
            "- Price **below EMA200** → Bearish → look for shorts\n"
            "- **Golden Cross** (EMA50 crosses above EMA200) = strong buy signal\n"
            "- **Death Cross** (EMA50 crosses below EMA200) = strong sell signal\n\n"
            "Checklist Gate 1 verifies your trade direction aligns with H4 EMA200."
        )
    if any(w in lower for w in ["rsi", "oversold", "overbought"]):
        return (
            "**RSI Signals in TradingBotV1:**\n\n"
            "- RSI < 30 → Oversold → long bounce setups\n"
            "- RSI > 70 → Overbought → short reversal setups\n"
            "- RSI 40–60 → Neutral → trend continuation\n"
            "- **Divergence:** Price lower lows + RSI higher lows = bullish divergence\n\n"
            "RSI feeds the Momentum check (+1.0 weight in the 9-factor confidence score)."
        )
    if any(w in lower for w in ["smc", "smart money", "order block", "fvg", "liquidity", "bos", "choch"]):
        return (
            "**Smart Money Concepts (SMC):**\n\n"
            "- **Order Blocks (OB):** Last bearish/bullish candle before strong impulse — key S/R\n"
            "- **Fair Value Gaps (FVG):** Imbalance zones price tends to revisit\n"
            "- **Liquidity Sweeps:** Stop hunts above highs / below lows before reversal\n"
            "- **BOS (Break of Structure):** Price breaks swing high/low confirming new trend\n"
            "- **CHoCH (Change of Character):** First BOS against current trend = reversal signal\n\n"
            "SMC contributes **+2.0 weight** to confidence score (2nd after HTF +2.5)."
        )
    if any(w in lower for w in ["confluence", "confidence", "weighted score", "9 factor"]):
        return (
            "**9-Factor Weighted Confidence Scoring:**\n\n"
            "| Factor | Weight |\n|---|---|\n"
            "| HTF (D1+H4 aligned) | +2.5 |\n"
            "| SMC (order block/FVG) | +2.0 |\n"
            "| Trend (EMA200) | +1.5 |\n"
            "| Structure (key level) | +1.5 |\n"
            "| Momentum (RSI) | +1.0 |\n"
            "| DXY correlation | +1.0 |\n"
            "| Candle pattern | +0.5 |\n"
            "| Session (London/NY) | +0.5 |\n"
            "| Volatility (ATR) | +0.5 |\n\n"
            "Max: 10.0. Signals show when score ≥ 6.0 and checklist ≥ 4/5."
        )
    if any(w in lower for w in ["playbook", "strategy", "strategies", "12 play"]):
        pb_mod = MODS.get("playbooks")
        if pb_mod and hasattr(pb_mod, "PLAYBOOKS"):
            names = [pb["name"] for pb in pb_mod.PLAYBOOKS.values()]
            return (
                f"**{len(names)} Strategy Playbooks loaded:**\n\n"
                + "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
                + "\n\nType `gold` to see which ones are triggering right now."
            )
        return "12 strategy playbooks are loaded. Type `gold` to see active ones."
    if any(w in lower for w in ["session", "london", "new york", "asian", "overlap"]):
        return (
            "**Trading Sessions (UTC):**\n\n"
            "| Session | UTC | Quality |\n|---|---|---|\n"
            "| Asian | 00:00–07:00 | ⚠️ Low |\n"
            "| London Open | 07:00–12:00 | ✅ HIGH |\n"
            "| London/NY Overlap | 12:00–16:00 | ✅ HIGHEST |\n"
            "| New York | 16:00–21:00 | ✅ HIGH |\n"
            "| Off-Hours | 21:00–00:00 | ❌ Skip |\n\n"
            "Gold moves most at **London Open 07:00–09:00 UTC**. "
            "Session is one of the 5 entry checklist gates."
        )

    # Fallback: search rules database
    rules = _load_json(RULES_FILE, [])
    matched = [
        r for r in rules
        if any(
            w in str(r.get("name", "")).lower() or w in str(r.get("description", "")).lower()
            for w in lower.split() if len(w) > 3
        )
    ]
    if matched:
        r    = matched[0]
        name = r.get("name") or r.get("pattern_name", "—")
        desc = str(r.get("description", ""))[:200]
        tier = r.get("tier", "?")
        bt   = r.get("backtest", {}) or {}
        wr   = bt.get("win_rate", "—")
        return f"**Found in database:** {name} (Tier {tier})\n\n{desc}\n\nBacktest win rate: {wr}"

    _cur_instr   = st.session_state.get("instrument", "XAUUSD")
    _cur_display = _INSTRUMENT_DISPLAY.get(_cur_instr, _cur_instr)
    return (
        f"I'm **TradingBotV1**, specialised in **{_cur_display}** analysis.\n\n"
        "Quick commands:\n"
        f"- `analyze {_cur_instr.lower()}` — current {_cur_display} setups\n"
        "- `run setup` — load all data\n"
        "- `show signals` — all trade signals\n"
        "- `risk guide` — position sizing\n"
        "- `news today` — economic calendar\n"
        "- `help` — full command list\n\n"
        "Or ask about EMA, RSI, SMC, confluence, sessions, or any playbook."
    )


def _handle_journal(msg: str) -> str:
    """show journal / my trades — syncs MT5 first, then displays journal."""
    lines = ["### 📒 Trade Journal\n"]
    new_n = 0
    total_n = 0
    if _MT5_SYNC_OK:
        try:
            _refresh_mt5_data()
            new_n, total_n = sync_to_journal(days_back=30)
            if new_n:
                lines.append(f"✅ **Synced {new_n} new trade(s) from MT5** ({total_n} total in journal)\n")
            else:
                lines.append(f"ℹ️ Journal up to date — {total_n} trade(s) on file.\n")
        except Exception as e:
            lines.append(f"⚠️ MT5 sync error: {e}\n")
    else:
        lines.append("⚠️ MT5 module not available — showing offline journal.\n")

    journal = []
    if _MT5_SYNC_OK:
        try:
            journal = _load_journal()
        except Exception:
            pass
    if not journal:
        lines.append("No trades in journal yet. Connect MT5 and click **Sync MT5**.")
        return "\n".join(lines)

    # Sort newest first, show last 20
    journal_sorted = sorted(journal, key=lambda x: x.get("closed_at", ""), reverse=True)
    lines.append(f"**Last {min(20, len(journal_sorted))} closed trades:**\n")
    lines.append("| Date | Symbol | Dir | Lots | P&L | Outcome |")
    lines.append("|---|---|---|---|---|---|")
    for t in journal_sorted[:20]:
        pnl     = t.get("pnl_usd", 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        outcome = t.get("outcome", "—").upper()
        icon    = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "➖")
        lines.append(
            f"| {t.get('closed_at','—')[:10]} "
            f"| {t.get('symbol','—')} "
            f"| {t.get('direction','—').upper()} "
            f"| {t.get('lots','—')} "
            f"| {pnl_str} "
            f"| {icon} {outcome} |"
        )

    total_pnl  = sum(t.get("pnl_usd", 0) for t in journal_sorted)
    n_wins     = sum(1 for t in journal_sorted if t.get("outcome") == "win")
    n_losses   = sum(1 for t in journal_sorted if t.get("outcome") == "loss")
    wr         = n_wins / (n_wins + n_losses) * 100 if (n_wins + n_losses) > 0 else 0
    pnl_total  = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
    lines.append(f"\n**Summary ({len(journal_sorted)} trades):** {n_wins}W / {n_losses}L  "
                 f"Win rate: **{wr:.0f}%**  Total P&L: **{pnl_total}**")
    return "\n".join(lines)


def _handle_trade_outcome(msg: str) -> str:
    """Handle 'that trade won' / 'that trade lost' to save to pattern memory."""
    try:
        from market_context import save_trade_outcome
    except ImportError:
        return "⚠️ market_context module not available."

    lower = msg.lower()
    outcome = "win" if any(w in lower for w in ["won", "win", "profit", "tp hit"]) else "loss"

    # Try to pull the last signal from session state
    signals = st.session_state.get("last_signals", [])
    if not signals:
        return (
            f"✅ Outcome recorded as **{outcome.upper()}**.\n\n"
            "_(No active signal in session — outcome saved without signal context.)_"
        )

    sig = signals[0]
    try:
        save_trade_outcome(
            playbook=   sig.get("pattern_name", "unknown"),
            regime=     sig.get("regime", "RANGING"),
            session=    sig.get("session", "London"),
            direction=  sig.get("direction", "long"),
            entry=      float(sig.get("entry",     0.0)),
            sl=         float(sig.get("stop_loss", 0.0)),
            tp=         float(sig.get("take_profit",0.0)),
            outcome=    outcome,
        )
        pname = sig.get("pattern_name", "?")
        rgm   = sig.get("regime", "?")
        base_response = (
            f"✅ **Trade outcome saved:** {outcome.upper()}\n\n"
            f"- Pattern: **{pname}**\n"
            f"- Regime: **{rgm}**\n\n"
            "_Pattern memory updated. Future win rates will reflect this trade._"
        )

        # ── Auto-analyze failures ─────────────────────────────────────────
        if outcome == "loss" and _PF_OK:
            try:
                trade_record = {
                    "symbol":       sig.get("asset", "XAUUSD"),
                    "direction":    sig.get("direction", ""),
                    "entry":        float(sig.get("entry", 0.0)),
                    "stop_loss":    float(sig.get("stop_loss", 0.0)),
                    "pattern_name": pname,
                    "regime":       rgm,
                    "session":      sig.get("session", ""),
                    "pnl_usd":      0.0,
                }
                analysis = analyze_failed_trade(trade_record)
                reason    = analysis.get("primary_reason", "UNKNOWN")
                all_r     = analysis.get("all_reasons", [])
                next_time = analysis.get("what_to_do_next", "")
                updated   = analysis.get("strategy_updated", False)

                reason_labels = {
                    "NEWS_SPIKE":      "📰 News spike near SL hit",
                    "WRONG_REGIME":    "📊 Wrong market regime",
                    "SL_TOO_TIGHT":    "📏 SL was too tight",
                    "FAKEOUT_SWEEP":   "🪤 Fakeout / liquidity sweep",
                    "TREND_EXHAUSTION":"⚡ Trend exhaustion at level",
                    "SESSION_CHANGE":  "🕐 Session changed during trade",
                    "VOLUME_MISMATCH": "📊 Volume too low for this strategy",
                    "CLIMAX_MISSED":   "🌊 Volume climax near entry — exhaustion",
                    "UNKNOWN":         "❓ Unknown — review manually",
                }
                reason_display = reason_labels.get(reason, reason)

                fail_block = (
                    f"\n\n---\n"
                    f"**WHY THIS TRADE FAILED:**\n\n"
                    f"**Primary:** {reason_display}\n"
                )
                if len(all_r) > 1:
                    others = [reason_labels.get(r, r) for r in all_r if r != reason]
                    fail_block += f"**Also:** {', '.join(others)}\n"
                fail_block += f"\n**Next time:** {next_time}"
                if updated:
                    fail_block += f"\n\n_⚙️ Strategy rule auto-updated to filter this failure pattern._"

                # ── Also run pipeline post-loss analysis ──────────────────
                try:
                    from backtest import run_post_loss_analysis
                    _pla = run_post_loss_analysis(trade_record)
                    if _pla.get("primary_failure_reason") != "UNKNOWN":
                        if _pla["fix_applied"]:
                            fail_block += f"\n\n_⚙️ Pipeline fix auto-applied: {_pla['fix_description']}_"
                        else:
                            fail_block += f"\n\n_📋 Pipeline suggestion logged: {_pla['fix_description']}_"
                except Exception:
                    pass

                return base_response + fail_block
            except Exception:
                pass

        return base_response
    except Exception as e:
        return f"⚠️ Failed to save outcome: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  Sessions command handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_price_check(msg=""):
    try:
        import yfinance as yf
        instr = st.session_state.get(
            "instrument", "XAUUSD")
        YF = {
            "XAUUSD": "GC=F",
            "NAS100": "NQ=F",
            "US30":   "YM=F",
            "GBPUSD": "GBPUSD=X",
            "EURUSD": "EURUSD=X",
            "WTI":    "CL=F",
        }
        ticker = YF.get(instr, "GC=F")
        tk     = yf.Ticker(ticker)
        hist   = tk.history(
            period="5d", interval="5m")

        if hist.empty:
            # Hard fallback: try 1mo daily
            hist = tk.history(period="1mo", interval="1d")
        if hist.empty:
            st.warning(
                f"⏳ {instr} — price data temporarily unavailable")
            return

        price  = float(hist["Close"].iloc[-1])
        open_p = float(hist["Open"].iloc[0])
        high   = float(hist["High"].max())
        low    = float(hist["Low"].min())
        change = price - open_p
        pct    = (change / open_p) * 100
        arrow  = "🟢 ▲" if change >= 0 else "🔴 ▼"

        st.markdown(
            f"## 💰 {instr} Live Price")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Current Price",
            f"${price:,.4f}",
            delta=f"{change:+.4f} "
                  f"({pct:+.2f}%)")
        c2.metric("Day High",
                  f"${high:,.4f}")
        c3.metric("Day Low",
                  f"${low:,.4f}")

        st.markdown(
            f"**Source:** yfinance "
            f"({ticker}) | "
            f"{arrow} {abs(pct):.2f}% today")

        # Adaptive SL/TP display
        try:
            from auto_trader import (
                calculate_adaptive_sl_tp,
                INSTRUMENTS)
            if instr in INSTRUMENTS:
                sl, tp, sl_pct, tp_pct = (
                    calculate_adaptive_sl_tp(
                        instr, price, "LONG"))
                acc_r = round(
                    sl_pct * 10 * 100, 1)
                acc_p = round(
                    tp_pct * 10 * 100, 1)
                st.markdown("---")
                st.markdown(
                    "**📊 Adaptive SL/TP "
                    "at 10x leverage:**")
                c4, c5, c6 = st.columns(3)
                c4.metric(
                    "🛑 Stop Loss",
                    f"${sl:,.4f}",
                    delta=f"-{acc_r}% account",
                    delta_color="inverse")
                c5.metric(
                    "🎯 Take Profit",
                    f"${tp:,.4f}",
                    delta=f"+{acc_p}% account")
                c6.metric(
                    "⚖️ RR Ratio", "1:3")
        except:
            pass

    except Exception as e:
        st.error(
            f"Price check error: {str(e)}")


def _handle_macro_analysis(_msg: str) -> str:
    """Full macro currency score breakdown for the selected instrument."""
    _instr = st.session_state.get("instrument", "XAUUSD")

    if not _MACRO_OK or _macro_scorer is None:
        return "⚠️ Macro scorer not available — `macro_scorer.py` not loaded."

    # Index / commodity instruments — no per-currency macro score
    if _instr not in ("GBPUSD", "EURUSD"):
        # Show simple DXY / context snapshot
        ctx = _get_mkt_ctx(_instr)
        lines = [
            f"```",
            f"{'═'*44}",
            f"  MACRO CONTEXT — {_instr}",
            f"{'═'*44}",
        ]
        if ctx.get("dxy"):
            lines.append(f"  DXY (USD Index) : {ctx['dxy']:.3f}")
        if ctx.get("vix"):
            _vix_lbl = "🔴 HIGH RISK" if ctx["vix"] > 25 else ("🟡 ELEVATED" if ctx["vix"] > 18 else "🟢 LOW")
            lines.append(f"  VIX (Fear)      : {ctx['vix']:.2f}  {_vix_lbl}")
        if ctx.get("us10y"):
            lines.append(f"  US 10Y Yield    : {ctx['us10y']:.3f}%")
        if not ctx:
            lines.append(f"  No market context available for {_instr}")
        lines += [f"{'─'*44}", f"  Macro scoring only available for GBPUSD / EURUSD", f"{'═'*44}", "```"]
        return "\n".join(lines)

    # Forex pair — full currency vs currency breakdown
    pair = _macro_scorer.score_pair(_instr)
    base_d  = pair.get("base_details",  {})
    quote_d = pair.get("quote_details", {})
    base_c  = pair.get("base_currency",  "?")
    quote_c = pair.get("quote_currency", "?")

    def _grade_bar(score: int) -> str:
        filled = max(0, min(10, score // 10))
        return "█" * filled + "░" * (10 - filled)

    lines = [
        f"```",
        f"{'═'*50}",
        f"  MACRO ANALYSIS — {_instr}",
        f"{'═'*50}",
        f"  Pair Bias   : {pair.get('bias','N/A')}",
        f"  Confidence  : {pair.get('confidence', 0)}%",
        f"  Reason      : {pair.get('reason', '—')}",
        f"  Rate Diff   : {pair.get('interest_rate_diff', 0):+.2f}% ({base_c} vs {quote_c})",
        f"{'─'*50}",
        f"  {base_c} SCORE : {base_d.get('score', 0)}/100   [{_grade_bar(base_d.get('score', 0))}]   Grade: {base_d.get('grade','?')}",
        f"    • Bias         : {base_d.get('bias','—')}",
        f"    • Interest Rate: {base_d.get('interest_rate', 0):.2f}%",
        f"    • GDP Growth   : {base_d.get('gdp_growth', 0):.1f}%",
        f"    • Unemployment : {base_d.get('unemployment', 0):.1f}%",
        f"    • CPI Inflation: {base_d.get('cpi', 0):.1f}%",
        f"    • Retail Sales : {base_d.get('retail_sales', 0):+.1f}%",
        f"{'─'*50}",
        f"  {quote_c} SCORE: {quote_d.get('score', 0)}/100   [{_grade_bar(quote_d.get('score', 0))}]   Grade: {quote_d.get('grade','?')}",
        f"    • Bias         : {quote_d.get('bias','—')}",
        f"    • Interest Rate: {quote_d.get('interest_rate', 0):.2f}%",
        f"    • GDP Growth   : {quote_d.get('gdp_growth', 0):.1f}%",
        f"    • Unemployment : {quote_d.get('unemployment', 0):.1f}%",
        f"    • CPI Inflation: {quote_d.get('cpi', 0):.1f}%",
        f"    • Retail Sales : {quote_d.get('retail_sales', 0):+.1f}%",
        f"{'─'*50}",
        f"  Score Diff  : {pair.get('score_diff', 0):+d}  ({base_c} vs {quote_c})",
        f"{'═'*50}",
        f"  ⚠ Data is manually updated — not real-time",
        f"{'═'*50}",
        "```",
    ]
    return "\n".join(lines)


def _handle_instrument_rules(_msg: str) -> str:
    """Show hard trading rules for the current instrument with live pass/fail status."""
    _instr = st.session_state.get("instrument", "XAUUSD")
    if not _IC_OK or _InstrumentConfluence is None:
        return "⚠️ Instrument confluence module not available — `instrument_confluence.py` not loaded."
    try:
        _ic  = _InstrumentConfluence(_instr)
        _ctx = _ic.get_full_signal_context()
    except Exception as _e:
        return f"⚠️ Rule check failed: {_e}"

    _violations = _ctx.get("violations", [])
    _blocked    = _ctx.get("blocked", False)
    _weights    = _ctx.get("weights", {})
    _extra      = _ctx.get("extra_confluence", {})
    _factors    = _extra.get("factors", [])
    _ex_score   = _extra.get("extra_score", 0)

    # Pull INSTRUMENT_HARD_RULES from module
    try:
        from instrument_confluence import INSTRUMENT_HARD_RULES as _IHR
        _rules_list = _IHR.get(_instr, [])
    except Exception:
        _rules_list = []

    _block_line = "⛔ SIGNAL BLOCKED" if _blocked else "✅ ALL RULES PASS"
    _viol_lines = ""
    if _violations:
        _viol_lines = "\n".join(f"  🔴 {v}" for v in _violations)
    else:
        _viol_lines = "  All rule checks passed"

    _rules_str = ("\n".join(f"  • {r.replace('_', ' ')}" for r in _rules_list)
                  if _rules_list else "  No hard rules for XAUUSD")

    _weights_str = "\n".join(
        f"  {k:<14}: {int(v*100)}%" for k, v in _weights.items()
    ) if _weights else "  Default weights"

    _factors_str = ("\n".join(f"  {f}" for f in _factors)
                    if _factors else "  No extra confluence data yet")

    lines = [
        f"```",
        f"{'═'*50}",
        f"  INSTRUMENT RULES — {_instr}",
        f"{'═'*50}",
        f"  Status      : {_block_line}",
        f"{'─'*50}",
        f"  HARD RULES:",
        _rules_str,
        f"{'─'*50}",
        f"  CURRENT VIOLATIONS:",
        _viol_lines,
        f"{'─'*50}",
        f"  CONFLUENCE WEIGHTS:",
        _weights_str,
        f"{'─'*50}",
        f"  EXTRA CONFLUENCE SCORE: {_ex_score:+.1f} pts",
        _factors_str,
        f"{'═'*50}",
        "```",
    ]
    return "\n".join(lines)


def _handle_sector_rotation(_msg: str) -> str:
    """Full sector rotation report for the selected instrument."""
    _instr = st.session_state.get("instrument", "XAUUSD")
    if not _SR_OK or _sector_rotation is None:
        return "⚠️ Sector rotation module not available — `sector_rotation.py` not loaded."
    try:
        report = _sector_rotation.get_full_report(_instr)
    except Exception as _e:
        return f"⚠️ Sector rotation fetch failed: {_e}"

    if report.get("error"):
        return (
            f"```\n{'═'*46}\n  SECTOR ROTATION — {_instr}\n{'═'*46}\n"
            f"  ⚠ {report['error']}\n"
            f"  Sector data unavailable — markets may be closed\n{'═'*46}\n```"
        )

    _bias   = report.get("instrument_bias", {})
    _risk   = report.get("risk_appetite",   {})
    _top3in = report.get("top_3_inflow",   [])
    _top3out= report.get("top_3_outflow",  [])
    _sects  = _bias.get("sectors_tracked", [])

    _b      = _bias.get("bias", "NEUTRAL")
    _conf   = _bias.get("confidence", 0)
    _regime = _risk.get("regime", "NEUTRAL")
    _rdiff  = _risk.get("diff", 0.0)
    _rmean  = _risk.get("meaning", "")

    _b_icon = ("🟢" if "BULL" in _b else
               "🔴" if "BEAR" in _b or "CAUTION" in _b else
               "🟡")
    _r_icon = ("🟢" if _regime == "RISK_ON" else
               "🔴" if _regime == "RISK_OFF" else
               "🟡")

    _in_lines  = "  " + "\n  ".join(
        f"🟢 {s:<14} {chg:+.2f}%" for s, chg in _top3in)
    _out_lines = "  " + "\n  ".join(
        f"🔴 {s:<14} {chg:+.2f}%" for s, chg in _top3out)
    _sect_lines = "  " + "\n  ".join(_sects) if _sects else "  None"

    _avg = _bias.get("avg_sector_flow", 0.0)

    lines = [
        f"```",
        f"{'═'*48}",
        f"  SECTOR ROTATION — {_instr}",
        f"{'═'*48}",
        f"  Instrument Bias : {_b_icon} {_b}  ({_conf}% confidence)",
        f"  Avg Sector Flow : {_avg:+.2f}%",
        f"{'─'*48}",
        f"  Market Regime   : {_r_icon} {_regime}",
        f"  Risk Diff       : {_rdiff:+.2f}  (risk-on minus risk-off)",
        f"  Meaning         : {_rmean}",
        f"{'─'*48}",
        f"  TOP 3 INFLOWS (strongest sectors):",
        _in_lines,
        f"{'─'*48}",
        f"  TOP 3 OUTFLOWS (weakest sectors):",
        _out_lines,
        f"{'─'*48}",
        f"  SECTORS DRIVING {_instr}:",
        _sect_lines,
        f"{'─'*48}",
        f"  VERDICT: {_b} with {_conf}% confidence",
        f"  Market risk posture: {_regime}",
        f"{'═'*48}",
        "```",
    ]
    return "\n".join(lines)


def _handle_open_interest(_msg: str) -> str:
    """Volume / open-interest proxy analysis for the selected instrument."""
    _instr = st.session_state.get("instrument", "XAUUSD")
    if not _OI_OK or _oi_analyzer is None:
        return "⚠️ Open Interest module not available — `open_interest.py` not loaded."
    try:
        d = _oi_analyzer.get_volume_analysis(_instr)
    except Exception as _e:
        return f"⚠️ Volume analysis failed: {_e}"

    _sig  = d.get("signal", "UNKNOWN")
    _bias = d.get("bias",   "NEUTRAL")
    _rsn  = d.get("reason", "—")
    _conf = d.get("confidence", 0)
    _pd   = d.get("price_direction",  "—")
    _vd   = d.get("volume_direction", "—")
    _vr   = d.get("volume_ratio", 0.0)
    _pc   = d.get("price_change_5d", 0.0)
    _av   = d.get("avg_volume", 0)
    _rv   = d.get("recent_volume", 0)
    _err  = d.get("error", "")

    _b_icon = ("🟢" if _bias == "BULLISH" else
               "🔴" if _bias in ("BEARISH", "CAUTION") else
               "🟡")
    _v_icon = "⬆" if _vd == "RISING" else ("⬇" if _vd == "FALLING" else "➡")
    _p_icon = "⬆" if _pd == "UP" else "⬇"

    # Plain-English interpretation
    _interp = {
        "STRONG TREND":    ("Institutions are actively buying. "
                            "Trust this move — follow the trend."),
        "WEAK MOVE":       ("Price rose but volume dried up. "
                            "This is likely short-covering, not real demand. "
                            "Be cautious — the move may fail."),
        "STRONG DOWNTREND":("Institutions are actively selling. "
                            "Avoid longs — follow the downtrend."),
        "TREND EXHAUSTION":("Price falling but sellers are leaving. "
                            "Watch for a reversal setup forming soon."),
    }.get(_sig, "No strong volume signal at this time.")

    lines = [
        f"```",
        f"{'═'*46}",
        f"  VOLUME / OPEN INTEREST — {_instr}",
        f"{'═'*46}",
        f"  Signal     : {_sig}",
        f"  Bias       : {_b_icon} {_bias}  ({_conf}% confidence)",
        f"{'─'*46}",
        f"  Price Dir  : {_p_icon} {_pd}  (5d chg: {_pc:+.4f})",
        f"  Volume Dir : {_v_icon} {_vd}  (ratio: {_vr:.2f}x avg)",
        f"  Avg Vol    : {int(_av):,}",
        f"  Recent Vol : {int(_rv):,}",
        f"{'─'*46}",
        f"  REASON: {_rsn}",
        f"{'─'*46}",
        f"  INTERPRETATION:",
        f"  {_interp}",
    ]
    if _err:
        lines.append(f"  ⚠ Error: {_err}")
    lines += [f"{'═'*46}", "```"]
    return "\n".join(lines)


def _handle_sessions(_msg: str) -> str:
    board = get_full_session_board() if _WS_OK else "Session data unavailable — world_sessions module not loaded."
    lp = _get_live_price(st.session_state.get("instrument", "XAUUSD"))
    price_line = ""
    if lp.get("price") and lp["price"] > 0:
        warn = f"\n⚠️ {lp['stale_warning']}" if lp.get("stale_warning") else ""
        price_line = (
            f"\n💰 **XAUUSD Live:** ${lp['price']:,.2f}"
            f"  |  Source: `{lp['source']}`"
            f"  |  {lp.get('timestamp_uae','—')}"
            f"{warn}"
        )
    return f"```\n{board}\n```{price_line}"


def _handle_cot(_msg: str) -> str:
    """Return a formatted COT (Commitment of Traders) report for Gold."""
    if not _COT_OK:
        return "⚠️ COT Analyzer not available — `cot_analyzer.py` not loaded."
    try:
        cot = _fetch_cot()
        if not cot.get("available"):
            return "⚠️ COT data unavailable."
        bias       = cot.get("bias", "NEUTRAL").replace("_", " ")
        spec_net   = cot.get("spec_net", 0)
        spec_pct   = cot.get("spec_net_pct", 0.0)
        hedger     = cot.get("hedger", "NEUTRAL").replace("_", " ")
        hedger_n   = cot.get("hedger_note", "")
        report_d   = cot.get("report_date", "unknown")
        source     = cot.get("source", "unknown")
        total_oi   = cot.get("total_oi", 0)
        # Directional boosts
        long_sig   = _get_cot_signal("long",  cot)
        short_sig  = _get_cot_signal("short", cot)
        colour_map = {
            "STRONGLY BULLISH": "🟢🟢", "BULLISH": "🟢",
            "NEUTRAL": "⚪",
            "BEARISH": "🔴", "STRONGLY BEARISH": "🔴🔴",
        }
        emoji = colour_map.get(bias, "⚪")
        lines = [
            f"### {emoji} Commitment of Traders — Gold Futures",
            f"> Report: {report_d}  |  Source: `{source}`",
            "",
            f"**Speculator bias:** `{bias}` ({spec_pct:+.1f}% net, {spec_net:+,} contracts)",
            f"**Open interest:** {total_oi:,} contracts",
            f"**Hedger signal:** {hedger} — {hedger_n}",
            "",
            f"**Trade alignment:**",
            f"  • LONG trade:  boost `{long_sig['boost']:+.1f}`  — {long_sig['note']}",
            f"  • SHORT trade: boost `{short_sig['boost']:+.1f}`  — {short_sig['note']}",
            "",
            f"> COT data is updated weekly (Fridays). Use as a directional filter, not a timing tool.",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ COT report error: {e}"


def _handle_liquidity(_msg: str) -> str:
    """Return a formatted liquidity heatmap for the current Gold price."""
    if not _LIQ_OK:
        return "⚠️ Liquidity Map not available — `liquidity_map.py` not loaded."
    try:
        df = _load_df()
        if df is None or len(df) < 20:
            return "⚠️ Not enough price data for liquidity map."
        price = float(df["close"].iloc[-1])
        liq   = _build_liq_map(df, price)
        if not liq.get("available"):
            return f"⚠️ Liquidity map failed: {liq.get('likely_reason', 'unknown error')}"
        header = f"### 💧 Liquidity Map — XAUUSD @ ${price:,.2f}\n\n"
        return header + f"```\n{_fmt_liq_map(liq, price)}\n```"
    except Exception as e:
        return f"⚠️ Liquidity map error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  Message router
# ══════════════════════════════════════════════════════════════════════════════

def _handle_ror(_msg: str) -> str:
    """Return a full Risk-of-Ruin analysis for the current settings."""
    try:
        ror  = _get_ror_profile()
        report = _format_ror(ror)
        _RATING_EMOJI = {
            "SAFE":     "✅",
            "MODERATE": "🟡",
            "HIGH":     "🟠",
            "DANGER":   "🔴",
        }
        emoji = _RATING_EMOJI.get(ror.get("risk_rating", ""), "⚠️")
        header = (
            f"### {emoji} Risk of Ruin Analysis\n\n"
            f"> {ror.get('summary', '')}\n\n"
        )
        return header + f"```\n{report}\n```"
    except Exception as _e:
        return f"⚠️ Risk of Ruin unavailable: {_e}"


def _handle_handoff(_msg: str) -> str:
    """Return the session handoff analysis (Asian → London → NY bias)."""
    if not _SH_OK:
        return "Session handoff analyzer not available."
    df = _load_df()
    if df is None:
        return "❌ No data loaded. Run `setup` first."
    try:
        handoff = _get_ny_bias(df)
        st.session_state["ny_bias"] = handoff
        header = (
            f"### 📊 Session Handoff Analysis\n\n"
            f"> {handoff.get('summary', '')}\n\n"
        )
        return header + f"```\n{_format_handoff(handoff)}\n```"
    except Exception as _e:
        return f"⚠️ Session handoff error: {_e}"


def _handle_wfo(msg: str) -> str:
    """Return walk-forward optimization summary or run it manually."""
    if not _WFO_OK:
        return "Walk-forward optimizer not available."
    try:
        if "run now" in msg.lower() or "force" in msg.lower():
            result = _run_wfo()
            if not result.get("optimized", True):
                return (
                    f"### 🔄 Walk-Forward Optimization\n\n"
                    f"⚠ Not optimized: {result.get('reason', 'unknown')}\n\n"
                    f"Need at least 10 resolved signals in the last 30 days."
                )
            n_changes = len(result.get("changes", []))
            header = (
                f"### 🔄 Walk-Forward Optimization Complete\n\n"
                f"> {n_changes} setting(s) updated based on last 30 days.\n\n"
            )
            return header + f"```\n{_get_wfo_summary()}\n```"
        return f"### 🔄 Walk-Forward Optimization\n\n```\n{_get_wfo_summary()}\n```"
    except Exception as _e:
        return f"⚠️ Walk-forward optimizer error: {_e}"


# ══════════════════════════════════════════════════════════════════════════════
#  Indicator Dashboard Handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_indicators(_msg: str) -> str:
    if not _INDICATORS_OK:
        return "Indicators module not available."
    df = _load_df()
    if df is None:
        return "Run setup first."

    price = st.session_state.get("live_price", 0) or 0
    inds  = _get_all_indicators(df)

    def _vote_emoji(bias: str) -> str:
        b = bias.lower()
        if "strongly_bull" in b: return "🟢🟢"
        if "bull" in b:          return "🟢"
        if "strongly_bear" in b: return "🔴🔴"
        if "bear" in b:          return "🔴"
        if "squeeze" in b:       return "🟡"
        return "⚪"

    al  = inds.get("alligator",     {})
    adx = inds.get("adx",           {})
    mac = inds.get("macd",          {})
    st_ = inds.get("stoch_rsi",     {})
    ich = inds.get("ichimoku",      {})
    vw  = inds.get("vwap",          {})
    sq  = inds.get("squeeze",       {})
    su  = inds.get("supertrend",    {})
    km  = inds.get("kama",          {})
    kz  = inds.get("killzones",     {})
    wy  = inds.get("wyckoff",       {})
    rr  = inds.get("real_rate",     {})
    mc  = inds.get("market_cipher", {})
    ob  = inds.get("obv",           {})

    bull_votes = sum(1 for i in inds.values() if "bull" in str(i.get("bias", "")))
    bear_votes = sum(1 for i in inds.values() if "bear" in str(i.get("bias", "")))
    verdict    = (
        "BULLISH" if bull_votes > bear_votes
        else "BEARISH" if bear_votes > bull_votes
        else "NEUTRAL"
    )
    verdict_emoji = "🟢" if verdict == "BULLISH" else "🔴" if verdict == "BEARISH" else "⚪"

    kz_active = ", ".join(kz.get("active_zones", [])) if kz.get("in_killzone") else "Not in kill zone"
    nk        = kz.get("next_killzone")
    kz_next   = f"{nk[0]} in {nk[1]}min" if nk else "N/A"

    lines = [
        f"## 📊 ALL 14 INDICATORS — XAUUSD\n",
        f"Price: **${price:,.2f}** | 🟢 Bull votes: {bull_votes} | 🔴 Bear votes: {bear_votes}\n",
        f"**Overall: {verdict_emoji} {verdict}**\n",
        "─" * 45 + "\n",

        f"{_vote_emoji(al.get('bias',''))} **Alligator:** "
        f"{al.get('state','')} | {al.get('bias','')}",

        f"{_vote_emoji(adx.get('bias',''))} **ADX:** {adx.get('adx', 0):.0f} "
        f"({adx.get('strength', '')}) | {adx.get('bias','')}",

        f"{_vote_emoji(mac.get('bias',''))} **MACD:** hist {mac.get('histogram', 0):.3f} "
        f"| {mac.get('bias','')}"
        + (" ← CROSSOVER" if mac.get("bearish_cross") or mac.get("bullish_cross") else ""),

        f"{_vote_emoji(st_.get('bias',''))} **StochRSI:** "
        f"K={st_.get('k', 0):.0f} D={st_.get('d', 0):.0f} | {st_.get('bias','')}",

        f"{_vote_emoji(ich.get('bias',''))} **Ichimoku:** "
        f"{'Above cloud ✓' if ich.get('above_cloud') else 'Below cloud' if ich.get('below_cloud') else 'In cloud'} "
        f"| {ich.get('bias','')}",

        f"{_vote_emoji(vw.get('bias',''))} **VWAP:** ${vw.get('vwap', 0):,.2f} "
        f"({'above' if vw.get('above') else 'below'}) | {vw.get('bias','')}",

        f"{_vote_emoji(sq.get('bias',''))} **Squeeze:** "
        + ("🔥 FIRED" if sq.get('squeeze_off') else "⚡ BUILDING" if sq.get('squeeze_on') else "Normal")
        + f" | {sq.get('bias','')}",

        f"{_vote_emoji(su.get('bias',''))} **Supertrend:** ${su.get('supertrend', 0):,.2f} "
        f"| {su.get('trend','')}"
        + (" ← JUST FLIPPED!" if su.get("just_flipped") else ""),

        f"{_vote_emoji(km.get('bias',''))} **Adaptive MA (KAMA):** "
        f"${km.get('kama', 0):,.2f} | {km.get('bias','')}",

        ("🎯" if kz.get('in_killzone') else "⚪") + f" **ICT Kill Zone:** {kz_active}\n"
        f"   Next: {kz_next}",

        f"{_vote_emoji(wy.get('bias',''))} **Wyckoff:** "
        f"{wy.get('phase', '')} | {wy.get('bias','')}",

        f"{_vote_emoji(rr.get('bias',''))} **Real Rate:** "
        f"{rr.get('real_rate', 0):.2f}% | {rr.get('bias','')}",

        f"{_vote_emoji(mc.get('bias',''))} **Market Cipher:** WT={mc.get('wt1', 0):.0f} "
        f"| {mc.get('bias','')}"
        + (" ← CROSS" if mc.get("bullish_cross") or mc.get("bearish_cross") else ""),

        f"{_vote_emoji(ob.get('bias',''))} **OBV:** "
        f"{ob.get('divergence', '') or 'aligned'} | {ob.get('bias','')}",

        "\n" + "─" * 45,
        f"**{bull_votes} BULLISH votes vs {bear_votes} BEARISH votes**",
        f"**Verdict: {verdict_emoji} {verdict}**",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Paper Trading Handlers
# ══════════════════════════════════════════════════════════════════════════════

def _handle_paper_status(_msg: str) -> str:
    if not _PAPER_OK:
        return "Paper trader not available."

    # Get FRESH live price right now
    try:
        from mt5_sync import get_live_price as _glp_ps
        _live_ps = _glp_ps()
        price    = _live_ps["price"]
        source   = _live_ps["source"]
        st.session_state["live_price"]  = price
        st.session_state["live_source"] = source
    except Exception:
        price  = st.session_state.get("live_price", 0) or 0
        source = "cached"

    s = _paper_summary(st.session_state.get("instrument", "XAUUSD"))

    lines = [
        f"## \U0001f4cb PAPER TRADING JOURNAL\n\n"
        f"Live price: **${price:,.2f}** [{source}]\n\n"
        f"Total: {s['total']} trades | "
        f"Open: {s['open']} | "
        f"Closed: {s['closed']}\n"
        f"Win rate: {s['win_rate']}% | "
        f"P&L: {'+' if s['total_pnl'] > 0 else ''}"
        f"{s['total_pnl']} pips\n\n"
    ]

    if s["open_trades"]:
        lines.append("## \U0001f534 OPEN TRADES (LIVE):\n")
        lines.append("\u2500" * 50)

        for t in s["open_trades"]:
            d     = t["direction"]
            entry = t["entry"]
            sl    = t["sl"]
            tp    = t["tp1"]

            if d == "short":
                fp_pips    = round((entry - price) / 0.1, 1)
                dist_to_sl = round(sl - price, 2)
                dist_to_tp = round(price - tp, 2)
                pct_to_tp  = round((entry - price) / (entry - tp) * 100, 1) if entry != tp else 0.0
                pct_to_sl  = round((price - entry) / (sl - entry) * 100, 1) if entry != sl else 0.0
            else:
                fp_pips    = round((price - entry) / 0.1, 1)
                dist_to_sl = round(entry - price, 2)
                dist_to_tp = round(tp - price, 2)
                pct_to_tp  = round((price - entry) / (tp - entry) * 100, 1) if entry != tp else 0.0
                pct_to_sl  = round((entry - price) / (entry - sl) * 100, 1) if entry != sl else 0.0

            if fp_pips > 0:
                status_emoji = "\u2705"
                status_text  = "IN PROFIT"
                pnl_prefix   = "+"
            elif fp_pips < 0:
                status_emoji = "\u26a0\ufe0f"
                status_text  = "IN LOSS"
                pnl_prefix   = ""
            else:
                status_emoji = "\u27a1\ufe0f"
                status_text  = "BREAKEVEN"
                pnl_prefix   = ""

            # Text-based progress bar toward TP
            if pct_to_tp > 0:
                filled       = min(int(pct_to_tp / 5), 20)
                bar          = "\u2588" * filled + "\u2591" * (20 - filled)
                progress_line = f"TP Progress: [{bar}] {pct_to_tp:.0f}%"
            else:
                progress_line = "TP Progress: Moving away from TP"

            sl_note = "away" if dist_to_sl > 0 else "BREACHED"

            lines.append(
                f"\n{status_emoji} **{t['trade_id']} \u2014 "
                f"{d.upper()} {status_text}**\n"
                f"Strategy: {t['strategy']}\n"
                f"Opened:   {t['opened_at']}\n\n"
                f"Entry:    ${entry:,.2f}\n"
                f"Current:  ${price:,.2f}\n"
                f"SL:       ${sl:,.2f} "
                f"(${abs(dist_to_sl):.2f} {sl_note})\n"
                f"TP:       ${tp:,.2f} "
                f"(${abs(dist_to_tp):.2f} to go)\n\n"
                f"Floating: {pnl_prefix}{fp_pips} pips\n"
                f"Max profit seen: +{t.get('max_profit_pips', 0)} pips\n"
                f"Max loss seen:   {t.get('max_loss_pips', 0)} pips\n\n"
                f"{progress_line}\n"
            )
            try:
                _entry  = float(t.get("entry", 0))
                _cur    = float(price)
                _sl_raw = float(t.get("sl", _entry))
                _tp_raw = float(t.get("tp1", _entry))
                if _entry == 0: raise ValueError

                _dir = t.get("direction", "LONG").upper()
                _fp_sign = 1 if (
                    (_dir == "LONG"  and _cur > _entry) or
                    (_dir == "SHORT" and _cur < _entry)
                ) else -1

                _lot = 6000 / (_entry * 100)
                _pv  = _lot * 100 * 0.10

                _now_usd = round((abs(_cur - _entry) / 0.10) * _pv * _fp_sign, 2)
                _tp_usd  = round((abs(_tp_raw - _cur) / 0.10) * _pv, 2)
                _sl_usd  = round((abs(_sl_raw - _cur) / 0.10) * _pv, 2)

                _now_pct = round((_now_usd / 300) * 100, 1)
                _tp_pct  = round((_tp_usd  / 300) * 100, 1)
                _sl_pct  = round((_sl_usd  / 300) * 100, 1)
                _nc = "#00ff88" if _now_usd >= 0 else "#ff4444"
                _ns = "+" if _now_usd >= 0 else ""

                st.markdown(
                    f"<div style='font-family:monospace;margin:6px 0;"
                    f"padding:8px;border-left:3px solid #444'>"
                    f"<span style='color:#888'>💰 $300@20x | "
                    f"lot={round(_lot,4)} | pip=${round(_pv,4)}</span><br>"
                    f"<span style='color:#aaa'>📍 Now &nbsp;: </span>"
                    f"<span style='color:{_nc}'>{_ns}${abs(_now_usd)} "
                    f"({_ns}{_now_pct}%)</span><br>"
                    f"<span style='color:#aaa'>🎯 If TP : </span>"
                    f"<span style='color:#00ff88'>+${_tp_usd} "
                    f"(+{_tp_pct}%)</span><br>"
                    f"<span style='color:#aaa'>🛑 If SL : </span>"
                    f"<span style='color:#ff4444'>-${_sl_usd} "
                    f"(-{_sl_pct}%)</span></div>",
                    unsafe_allow_html=True
                )
            except Exception:
                pass
            lines.append(f"{chr(8212) * 50}")

        lines.append(
            "\n*Refreshes every 60s automatically.*\n"
            "*Type 'paper trades' anytime for latest status.*\n"
            "*Type 'close paper [ID]' to close manually.*"
        )
    else:
        lines.append(
            "No open paper trades.\n\n"
            "Type **'paper short'** or **'paper long'** to open one."
        )

    if s.get("recent"):
        lines.append(f"\n## \U0001f4ca RECENT CLOSED TRADES:\n")
        for t in s["recent"][-10:]:
            emoji        = "\u2705" if t.get("outcome") == "WIN" else "\u274c"
            pips_sign    = "+" if t.get("pnl_pips", 0) > 0 else ""
            close_reason = t.get("close_reason", "")
            close_detail = t.get("close_detail", "")
            time_held    = t.get("time_held", "?")
            lines.append(
                f"\n{emoji} **{t['trade_id']} \u2014 "
                f"{t['direction'].upper()} {t.get('outcome', '')}**\n"
                f"Entry:    ${t['entry']:,.2f} | "
                f"Close:    ${t.get('close_price', 0):,.2f}\n"
                f"P&L:      {pips_sign}{t.get('pnl_pips', 0)} pips\n"
                f"Reason:   {close_reason}\n"
                f"Detail:   {close_detail}\n"
                f"Opened:   {t.get('opened_at', '')}\n"
                f"Closed:   {t.get('closed_at', '')}\n"
                f"Held for: {time_held}\n"
                f"Strategy: {t.get('strategy', '')}\n"
            )

    return "\n".join(lines)


def _handle_open_paper(_msg: str, direction: str) -> str:
    if not _PAPER_OK:
        return "Paper trader not available."
    df = _load_df()

    # Always fetch fresh live price at moment of opening
    _instr_pt = st.session_state.get("instrument", "XAUUSD")
    try:
        from mt5_sync import get_live_price as _glp_fresh
        _live_fresh  = _glp_fresh(_instr_pt)
        price        = _live_fresh["price"]
        price_source = _live_fresh["source"]
        # Update session state with fresh price too
        st.session_state["live_price"]  = price
        st.session_state["live_source"] = price_source
    except Exception:
        price        = st.session_state.get("live_price", 0) or 0
        price_source = "cached"

    if not price or price == 0:
        return "Cannot open paper trade — no live price available. Run setup first."
    if df is None:
        return "Run setup first."

    atr = float(df["atr"].iloc[-1]) if df is not None else 20.0

    # Use last signal if direction matches, otherwise build ATR-based signal
    last_sig = st.session_state.get("last_signal", {})
    if last_sig.get("direction") == direction:
        signal = last_sig
    else:
        if direction == "short":
            sl_calc = price + (atr * 2)
            tp_calc = price - (atr * 3)
        else:
            sl_calc = price - (atr * 2)
            tp_calc = price + (atr * 3)
        signal = {
            "direction":    direction,
            "stop_loss":    sl_calc,
            "take_profit":  tp_calc,
            "pattern_name": "Manual Paper Trade",
            "confidence":   0,
        }

    trade = _open_paper(signal, price, _instr_pt)
    st.session_state["paper_open_count"] = _paper_summary(_instr_pt).get("open", 0)
    return (
        f"## \U0001f4cb PAPER TRADE OPENED \u2705\n\n"
        f"ID:        **{trade['trade_id']}**\n"
        f"Direction: **{direction.upper()}**\n"
        f"Entry:     ${trade['entry']:,.2f} [{price_source}] \u2190 live price at this moment\n"
        f"SL:        ${trade['sl']:,.2f} "
        f"(-${trade['sl_distance']:.2f})\n"
        f"TP:        ${trade['tp1']:,.2f} "
        f"(+${trade['tp_distance']:.2f})\n"
        f"Strategy:  {trade['strategy']}\n"
        f"Opened:    {trade['opened_at']}\n\n"
        f"Bot will notify when TP or SL is hit.\n"
        f"Type **'paper trades'** to check status anytime."
    )


def _handle_close_paper(_msg: str) -> str:
    if not _PAPER_OK:
        return "Paper trader not available."
    price = st.session_state.get("live_price", 0) or 0

    # Extract 8-character alphanumeric trade ID from message
    words    = _msg.upper().split()
    trade_id = None
    for w in words:
        if len(w) == 8 and w.isalnum():
            trade_id = w
            break
    if not trade_id:
        return "Please specify a trade ID. Example: **close paper A3F2B7C1**"

    t = _close_paper(trade_id, price, st.session_state.get("instrument", "XAUUSD"))
    if not t:
        return f"Trade **{trade_id}** not found or already closed."

    emoji    = "\u2705" if t["outcome"] == "WIN" else "\u274c"
    pip_sign = "+" if t["pnl_pips"] > 0 else ""
    return (
        f"{emoji} **PAPER TRADE CLOSED MANUALLY**\n\n"
        f"ID:        {t['trade_id']}\n"
        f"Direction: {t['direction'].upper()}\n"
        f"Entry:     ${t['entry']:,.2f}\n"
        f"Close:     ${t['close_price']:,.2f}\n"
        f"P&L:       {pip_sign}{t['pnl_pips']} pips\n"
        f"Result:    **{t['outcome']}**"
    )


def _analyze_price_targets(df, current_price, df_m15=None) -> dict:
    """Build every plausible trade scenario from S/R + indicators."""
    atr    = float(df["atr"].iloc[-1])
    rsi    = float(df["rsi"].iloc[-1])
    ema50  = float(df["ema50"].iloc[-1])
    ema200 = float(df["ema200"].iloc[-1])

    trend_bearish = current_price < ema200
    trend_bullish = current_price > ema200

    try:
        from sr_mapper import get_sr_levels as _gsr_pt
        sr = _gsr_pt(df, current_price)
        res_levels = [r["price"] for r in sr.get("resistance_levels", [])[:4]]
        sup_levels = [s["price"] for s in sr.get("support_levels",    [])[:4]]
    except Exception:
        res_levels = []
        sup_levels = []
    if not res_levels:
        res_levels = [round(current_price + atr * m, 2) for m in (1, 2, 3)]
    if not sup_levels:
        sup_levels = [round(current_price - atr * m, 2) for m in (1, 2, 3)]

    nearest_res = min(res_levels, key=lambda x: abs(x - current_price))
    nearest_sup = min(sup_levels, key=lambda x: abs(x - current_price))
    _above_res  = sorted([r for r in res_levels if r > nearest_res])
    _below_sup  = sorted([s for s in sup_levels if s < nearest_sup], reverse=True)
    next_res = _above_res[0]  if _above_res  else round(nearest_res + atr * 2, 2)
    next_sup = _below_sup[0]  if _below_sup  else round(nearest_sup - atr * 2, 2)

    rsi_oversold   = rsi < 35
    rsi_overbought = rsi > 65
    rsi_bearish    = rsi < 50
    rsi_bullish    = rsi > 50

    all_scenarios: list[dict] = []

    # SCENARIO 1 — SHORT NOW
    if trend_bearish or rsi_bearish:
        _e = current_price
        _s = round(nearest_res + atr * 0.5, 2)
        _t1 = round(nearest_sup, 2)
        _t2 = round(next_sup, 2)
        _rr = round(abs(_t2 - _e) / max(abs(_s - _e), 0.01), 1)
        all_scenarios.append({
            "id": "SHORT_NOW", "direction": "SHORT", "entry_type": "MARKET",
            "title": "SHORT — Enter Now (Trend Following)",
            "timing": "NOW",
            "entry": round(_e, 2), "sl": _s, "tp1": _t1, "tp2": _t2,
            "rr": _rr, "sl_dist": round(abs(_s - _e), 2),
            "tp2_dist": round(abs(_t2 - _e), 2),
            "quality": "MODERATE" if (rsi_bearish and trend_bearish) else "LOW",
            "why": (f"Trend bearish (below EMA200). RSI {rsi:.0f}"
                    f"{' — bearish momentum' if rsi_bearish else ''}. "
                    f"Short now, target support ${_t1:,.2f} then ${_t2:,.2f}."),
            "mt5": f"SELL XAUUSD | SL: {_s:.2f} | TP: {_t1:.2f}"
        })

    # SCENARIO 2 — PENDING SHORT (bounce to resistance)
    _pe = round(nearest_res - 0.50, 2)
    _ps = round(nearest_res + atr * 0.5, 2)
    _pt1 = round(nearest_sup, 2)
    _pt2 = round(next_sup, 2)
    _prr = round(abs(_pt2 - _pe) / max(abs(_ps - _pe), 0.01), 1)
    all_scenarios.append({
        "id": "SHORT_PENDING", "direction": "SHORT", "entry_type": "LIMIT SELL",
        "title": "SHORT — Pending (Wait for Bounce to Resistance)",
        "timing": f"When price reaches ${nearest_res:,.2f}",
        "entry": _pe, "sl": _ps, "tp1": _pt1, "tp2": _pt2,
        "rr": _prr, "sl_dist": round(abs(_ps - _pe), 2),
        "tp2_dist": round(abs(_pt2 - _pe), 2),
        "quality": "HIGH" if trend_bearish else "MODERATE",
        "why": (f"Price expected to bounce to resistance ${nearest_res:,.2f} before dropping. "
                f"Set limit sell order there. Tighter SL (${abs(_ps - _pe):.2f}) than shorting now. Better RR."),
        "mt5": f"LIMIT SELL XAUUSD at {_pe:.2f} | SL: {_ps:.2f} | TP: {_pt1:.2f}"
    })

    # SCENARIO 3 — LONG REVERSAL NOW (counter-trend)
    if rsi_oversold or (trend_bearish and abs(current_price - nearest_sup) < atr * 0.5):
        _re = current_price
        _rs = round(nearest_sup - atr * 0.5, 2)
        _rt1 = round(nearest_res, 2)
        _rt2 = round(next_res, 2)
        _rrr = round(abs(_rt1 - _re) / max(abs(_re - _rs), 0.01), 1)
        all_scenarios.append({
            "id": "LONG_REVERSAL_NOW", "direction": "LONG", "entry_type": "MARKET",
            "title": "LONG — Reversal Now (Counter-trend Bounce)",
            "timing": "NOW",
            "entry": round(_re, 2), "sl": _rs, "tp1": _rt1, "tp2": _rt2,
            "rr": _rrr, "sl_dist": round(abs(_re - _rs), 2),
            "tp2_dist": round(abs(_rt2 - _re), 2),
            "quality": "MODERATE" if rsi_oversold else "LOW",
            "why": (
                (f"RSI oversold at {rsi:.0f} — bounce likely. " if rsi_oversold else "")
                + f"Counter-trend bounce to resistance ${_rt1:,.2f}. "
                  f"After bounce completes SHORT resumes. Use 25-50% size only."
            ),
            "mt5": f"BUY XAUUSD | SL: {_rs:.2f} | TP: {_rt1:.2f}"
        })

    # SCENARIO 4 — PENDING LONG (drop to support)
    _ple = round(nearest_sup + 0.50, 2)
    _pls = round(nearest_sup - atr * 0.5, 2)
    _plt1 = round(nearest_res, 2)
    _plt2 = round(next_res, 2)
    _plrr = round(abs(_plt1 - _ple) / max(abs(_ple - _pls), 0.01), 1)
    all_scenarios.append({
        "id": "LONG_PENDING", "direction": "LONG", "entry_type": "LIMIT BUY",
        "title": "LONG — Pending (Wait for Drop to Support)",
        "timing": f"When price drops to ${nearest_sup:,.2f}",
        "entry": _ple, "sl": _pls, "tp1": _plt1, "tp2": _plt2,
        "rr": _plrr, "sl_dist": round(abs(_ple - _pls), 2),
        "tp2_dist": round(abs(_plt2 - _ple), 2),
        "quality": "MODERATE",
        "why": (f"If price drops to support ${nearest_sup:,.2f} → "
                f"expect bounce to resistance ${nearest_res:,.2f}. "
                f"Set limit buy order. After bounce → SHORT resumes."),
        "mt5": f"LIMIT BUY XAUUSD at {_ple:.2f} | SL: {_pls:.2f} | TP: {_plt1:.2f}"
    })

    # SCENARIO 5 — SHORT AFTER REVERSAL COMPLETES (best setup)
    _rse = round(nearest_res - 0.50, 2)
    _rss = round(nearest_res + atr * 0.5, 2)
    _rst1 = round(current_price, 2)
    _rst2 = round(next_sup, 2)
    _rsrr = round(abs(_rst2 - _rse) / max(abs(_rss - _rse), 0.01), 1)
    all_scenarios.append({
        "id": "SHORT_AFTER_REVERSAL", "direction": "SHORT", "entry_type": "LIMIT SELL",
        "title": "SHORT — After Reversal Completes (Best Setup)",
        "timing": f"After price bounces to ${nearest_res:,.2f}",
        "entry": _rse, "sl": _rss, "tp1": _rst1, "tp2": _rst2,
        "rr": _rsrr, "sl_dist": round(abs(_rss - _rse), 2),
        "tp2_dist": round(abs(_rst2 - _rse), 2),
        "quality": "HIGH" if trend_bearish else "MODERATE",
        "why": (f"Best scenario: wait for reversal bounce to ${nearest_res:,.2f}, "
                f"then short when it rejects. Tight SL above resistance. "
                f"Full trend continuation target ${_rst2:,.2f}. Cleanest setup."),
        "mt5": f"LIMIT SELL XAUUSD at {_rse:.2f} | SL: {_rss:.2f} | TP: {_rst2:.2f}"
    })

    # Build expected sequence
    if trend_bearish:
        sequence = [
            f"1. Price may bounce to ${nearest_res:,.2f} (short-term reversal)",
            f"2. SHORT at ${nearest_res:,.2f} rejection \u2192 ${next_sup:,.2f}",
            f"   OR drop to ${nearest_sup:,.2f} first \u2192 LONG \u2192 ${nearest_res:,.2f}",
            f"3. SHORT resumes after bounce completes",
            f"Best pending order: LIMIT SELL at ${_rse:,.2f}",
        ]
    else:
        sequence = [
            f"1. Price may pull back to ${nearest_sup:,.2f} support",
            f"2. LONG at ${nearest_sup:,.2f} bounce \u2192 ${next_res:,.2f}",
            f"3. Watch for reversal SHORT if trend weakens",
        ]

    return {
        "current_price":      current_price,
        "nearest_resistance": nearest_res,
        "nearest_support":    nearest_sup,
        "next_resistance":    next_res,
        "next_support":       next_sup,
        "trend_bearish":      trend_bearish,
        "rsi":                rsi,
        "atr":                atr,
        "all_scenarios":      all_scenarios,
        "sequence":           sequence,
    }


def _handle_pending_orders(_msg: str) -> str:
    df     = _load_df()
    price  = st.session_state.get("live_price", 0)
    df_m15 = _load_m15_df()

    if df is None or not price:
        return "Run setup first."

    data      = _analyze_price_targets(df, price, df_m15)
    scenarios = data["all_scenarios"]
    sequence  = data["sequence"]

    _Q = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴", "SPECULATIVE": "⚫"}
    SEP = "─" * 50

    lines = [
        f"## 🎯 ALL TRADE SCENARIOS — XAUUSD\n",
        f"Price: **${price:,.2f}** | "
        f"ATR: ${data['atr']:.2f} | "
        f"Trend: {'BEARISH 📉' if data['trend_bearish'] else 'BULLISH 📈'} | "
        f"RSI: {data['rsi']:.1f}\n",
        f"Resistance: **${data['nearest_resistance']:,.2f}** | "
        f"Support: **${data['nearest_support']:,.2f}**\n",
        f"\n{SEP}",
        f"## 📋 EXPECTED SEQUENCE:\n",
    ]
    for step in sequence:
        lines.append(step)

    lines.append(f"\n{SEP}")
    lines.append(f"\n## 📊 ALL SCENARIOS ({len(scenarios)} found):\n")

    for i, s in enumerate(scenarios, 1):
        q_emoji = _Q.get(s["quality"], "⚫")
        lines.append(
            f"\n{q_emoji} **SCENARIO {i}: {s['title']}**\n"
            f"Quality: {s['quality']} | Timing: {s['timing']}\n\n"
            f"Entry:  **${s['entry']:,.2f}** ({s['entry_type']})\n"
            f"SL:     ${s['sl']:,.2f} (-${s['sl_dist']:.2f})\n"
            f"TP1:    ${s['tp1']:,.2f}\n"
            f"TP2:    ${s['tp2']:,.2f} (+${s['tp2_dist']:.2f})\n"
            f"R:R:    1:{s['rr']}\n\n"
            f"Why: {s['why']}\n\n"
            f"📱 MT5: `{s['mt5']}`\n"
            f"{SEP}"
        )

    _best = next((s["title"] for s in scenarios if s["quality"] == "HIGH"), scenarios[0]["title"])
    lines.append(
        f"\n## 💡 RECOMMENDATION:\n\n"
        f"Best setup = **{_best}**\n\n"
        f"Set MULTIPLE pending orders on MT5 now.\n"
        f"Let price come to your orders — don't chase.\n"
        f"Cancel unused orders after 24 hours."
    )

    return "\n".join(lines)


def _handle_scalp(_msg: str) -> str:
    df_m15 = _load_m15_df()
    price  = st.session_state.get("live_price", 0)

    if df_m15 is None:
        return "M15 data unavailable — try again in a moment."

    m15_close = float(df_m15["close"].iloc[-1])
    m15_ema50 = float(df_m15["ema50"].iloc[-1])
    m15_rsi   = float(df_m15["rsi"].iloc[-1])
    m15_atr   = float(df_m15["atr"].iloc[-1])

    last5_closes = df_m15["close"].tail(5).tolist()
    momentum_up  = last5_closes[-1] > last5_closes[0]
    momentum_str = "BULLISH" if momentum_up else "BEARISH"

    scalps = []

    # SHORT scalp
    if m15_close < m15_ema50 and m15_rsi < 55 and not momentum_up:
        entry_p = price or m15_close
        sl_p    = float(df_m15["high"].tail(3).max()) + (m15_atr * 0.3)
        target  = entry_p - (m15_atr * 2.0)
        denom   = sl_p - entry_p
        if denom > 0:
            rr = (entry_p - target) / denom
            if rr >= 1.5:
                scalps.append({
                    "direction": "SHORT",
                    "entry":   round(entry_p, 2),
                    "sl":      round(sl_p, 2),
                    "tp":      round(target, 2),
                    "rr":      round(rr, 1),
                    "sl_dist": round(sl_p - entry_p, 2),
                    "tp_dist": round(entry_p - target, 2),
                    "reason":  f"M15 bearish momentum | RSI {m15_rsi:.0f} | below EMA50",
                })

    # LONG scalp
    if m15_close > m15_ema50 and m15_rsi > 45 and momentum_up:
        entry_p = price or m15_close
        sl_p    = float(df_m15["low"].tail(3).min()) - (m15_atr * 0.3)
        target  = entry_p + (m15_atr * 2.0)
        denom   = entry_p - sl_p
        if denom > 0:
            rr = (target - entry_p) / denom
            if rr >= 1.5:
                scalps.append({
                    "direction": "LONG",
                    "entry":   round(entry_p, 2),
                    "sl":      round(sl_p, 2),
                    "tp":      round(target, 2),
                    "rr":      round(rr, 1),
                    "sl_dist": round(entry_p - sl_p, 2),
                    "tp_dist": round(target - entry_p, 2),
                    "reason":  f"M15 bullish momentum | RSI {m15_rsi:.0f} | above EMA50",
                })

    lines = [
        f"## ⚡ M15 SCALP SCANNER\n",
        f"**Price:** ${price:,.2f} | **M15 RSI:** {m15_rsi:.1f} | **Momentum:** {momentum_str}\n",
        f"**M15 EMA50:** ${m15_ema50:,.2f} | **ATR:** ${m15_atr:.2f}\n\n",
    ]

    if scalps:
        for s in scalps:
            lines.append(
                f"### ⚡ SCALP {s['direction']}\n"
                f"Entry:  **${s['entry']:,.2f}**\n"
                f"SL:     ${s['sl']:,.2f} (-${s['sl_dist']:.2f})\n"
                f"TP:     ${s['tp']:,.2f} (+${s['tp_dist']:.2f})\n"
                f"R:R:    1:{s['rr']}\n"
                f"Reason: {s['reason']}\n\n"
                f"⚠ Scalp trade — target 30-60 min hold\n"
                f"Use 25-50% normal size\n"
            )
    else:
        lines.append(
            "**No scalp setup right now.**\n\n"
            f"M15 momentum: {momentum_str}\n"
            + ("Price vs EMA50: Above ✅\n\n" if m15_close > m15_ema50 else "Price vs EMA50: Below 📉\n\n")
            + "Wait for momentum to confirm direction.\n"
        )

    lines.append(
        "─────────────────────────────\n"
        "**M15 context:**\n"
        + ("Last 5 candles: Rising 📈\n" if momentum_up else "Last 5 candles: Falling 📉\n")
        + f"M15 ATR: ${m15_atr:.2f} (expected move per 15 min)\n"
        "*Type 'market read' for full H1 analysis*"
    )

    return "\n".join(lines)


def _handle_market_read(_msg: str) -> str:
    df    = _load_df()
    price = st.session_state.get("live_price", 0)
    if df is None or not price:
        return "Run setup first."

    close = float(df["close"].iloc[-1])
    high  = float(df["high"].iloc[-1])
    low   = float(df["low"].iloc[-1])

    try:
        ema50  = float(df["ema50"].iloc[-1])
        ema200 = float(df["ema200"].iloc[-1])
    except KeyError:
        ema50  = float(df["close"].rolling(50).mean().iloc[-1])
        ema200 = float(df["close"].rolling(200).mean().iloc[-1])

    try:
        rsi = float(df["rsi"].iloc[-1])
    except KeyError:
        rsi = 50.0

    try:
        atr = float(df["atr"].iloc[-1])
    except KeyError:
        atr = close * 0.005

    above_ema50  = close > ema50
    above_ema200 = close > ema200

    if rsi > 70:   rsi_read = "overbought — watch for reversal"
    elif rsi > 60: rsi_read = "bullish momentum"
    elif rsi > 50: rsi_read = "neutral leaning bullish"
    elif rsi > 40: rsi_read = "neutral leaning bearish"
    elif rsi > 30: rsi_read = "bearish momentum"
    else:          rsi_read = "oversold — bounce risk"

    last_candle_bull = close > float(df["open"].iloc[-1])

    try:
        from sr_mapper import get_sr_levels
        sr = get_sr_levels(df, close)
        nearest_res = sr["nearest_resistance"]["price"]
        nearest_sup = sr["nearest_support"]["price"]
        res_dist    = sr["nearest_resistance"]["distance_usd"]
        sup_dist    = sr["nearest_support"]["distance_usd"]
    except Exception:
        nearest_res = close + atr
        nearest_sup = close - atr
        res_dist    = atr
        sup_dist    = atr

    ema50_slope = float(df["ema50"].iloc[-1]) - float(df["ema50"].iloc[-5]) \
        if "ema50" in df.columns and len(df) >= 5 \
        else float(df["close"].rolling(50).mean().iloc[-1]) - float(df["close"].rolling(50).mean().iloc[-5])
    if ema50_slope > 0.5:   momentum = "rising"
    elif ema50_slope < -0.5: momentum = "falling"
    else:                    momentum = "flat"

    if not above_ema200 and rsi < 50 and momentum == "falling":
        likely = "DOWN"
        reason = f"Price below EMA200, RSI {rsi:.0f}, momentum falling"
        watch  = f"Watch ${nearest_sup:,.2f} support — break confirms SHORT"
        entry  = f"SHORT entry if price rejects ${nearest_res:,.2f} resistance"
    elif above_ema200 and rsi > 50 and momentum == "rising":
        likely = "UP"
        reason = f"Price above EMA200, RSI {rsi:.0f}, momentum rising"
        watch  = f"Watch ${nearest_res:,.2f} resistance — break confirms LONG"
        entry  = f"LONG entry if price holds ${nearest_sup:,.2f} support"
    elif rsi < 35:
        likely = "BOUNCE"
        reason = f"RSI {rsi:.0f} oversold — bounce likely before continuation"
        watch  = "Watch for RSI turning up above 35"
        entry  = f"Small LONG from ${close:,.2f} targeting ${nearest_res:,.2f}"
    elif rsi > 65:
        likely = "PULLBACK"
        reason = f"RSI {rsi:.0f} overbought — pullback likely"
        watch  = "Watch for RSI turning down below 65"
        entry  = f"Small SHORT from ${close:,.2f} targeting ${nearest_sup:,.2f}"
    else:
        likely = "RANGE"
        reason = f"RSI {rsi:.0f} neutral, EMA flat — ranging conditions"
        watch  = f"Wait for break of ${nearest_res:,.2f} or ${nearest_sup:,.2f}"
        entry  = "No clear directional edge right now — wait"

    trend_emoji  = "📈" if above_ema200 else "📉"
    candle_emoji = "🟢" if last_candle_bull else "🔴"

    return (
        f"## 👁 MARKET READ — XAUUSD\n\n"
        f"**Price:** ${price:,.2f} | "
        f"**RSI:** {rsi:.1f} ({rsi_read})\n\n"
        f"**Trend:** {trend_emoji} "
        f"{'Above' if above_ema200 else 'Below'} EMA200 | "
        f"EMA50 momentum: {momentum}\n\n"
        f"**Last candle:** {candle_emoji} "
        f"{'Bullish' if last_candle_bull else 'Bearish'}\n\n"
        f"─────────────────────────────\n"
        f"**KEY LEVELS:**\n"
        f"🔴 Resistance: ${nearest_res:,.2f} ({res_dist:.1f} away)\n"
        f"→  Current:    ${price:,.2f}\n"
        f"🟢 Support:    ${nearest_sup:,.2f} ({sup_dist:.1f} away)\n\n"
        f"─────────────────────────────\n"
        f"**LIKELY NEXT MOVE: {likely}**\n"
        f"Reason: {reason}\n\n"
        f"**Watch:** {watch}\n"
        f"**Entry idea:** {entry}\n\n"
        f"─────────────────────────────\n"
        f"*This is market context, not a trade signal.\n"
        f"Type 'analyze gold' for full signal with SL/TP.*"
    )


def _route(msg: str, account: float = 1000.0) -> str:
    lower = msg.strip().lower()
    # Internal sidebar triggers
    if lower.startswith("__mt5_sync_done_"):
        parts = lower.replace("__mt5_sync_done_", "").split("_")
        new_n   = parts[0] if parts else "?"
        total_n = parts[1] if len(parts) > 1 else "?"
        return f"✅ **MT5 sync complete** — {new_n} new trade(s) added ({total_n} total in journal)."
    if lower.startswith("__mark_signal_"):
        parts = lower.replace("__mark_signal_", "").split("_")
        ticket  = parts[0] if parts else "?"
        symbol  = parts[1].upper() if len(parts) > 1 else "XAUUSD"
        direct  = parts[2]         if len(parts) > 2 else "?"
        entry   = parts[3]         if len(parts) > 3 else "0"
        return (
            f"🤖 Position #{ticket} ({symbol} {direct.upper()} @ ${entry}) "
            f"marked as a bot signal.\n\n"
            f"When this trade closes, MT5 auto-sync will match it to the last "
            f"signal and update pattern memory automatically."
        )
    if any(k in lower for k in ["that trade won", "trade won", "tp hit", "that trade lost", "trade lost", "sl hit", "stopped out"]):
        return _handle_trade_outcome(msg)
    if any(k in lower for k in ["export logs", "export session", "show me the logs", "show logs", "download logs"]):
        return _handle_export(msg)
    if any(k in lower for k in ["journal", "show journal", "my trades", "trade history"]):
        return _handle_journal(msg)
    if any(k in lower for k in ["setup", "refresh", "run setup"]):
        return _handle_setup(msg)
    if any(k in lower for k in [
            "session handoff", "london break", "ny bias",
            "asian range", "handoff analysis", "fake break",
            "session bias"]):
        return _handle_handoff(msg)
    if any(k in lower for k in [
            "optimization report", "wfo", "weekly optimization",
            "how is bot learning", "walk forward", "auto optimize"]):
        return _handle_wfo(msg)
    if any(k in lower for k in ["cot data", "cot report", "commitment of traders", "show cot", "cot bias"]):
        return _handle_cot(msg)
    if any(k in lower for k in ["liquidity", "heatmap", "where is liquidity", "liq map", "stop clusters", "liquidity map"]):
        return _handle_liquidity(msg)
    if any(k in lower for k in ["pending orders", "where to enter", "price targets",
                                   "all trades", "all scenarios", "show all setups",
                                   "trade plan", "game plan"]):
        return _handle_pending_orders(msg)
    if any(k in lower for k in ["paper short", "mock short", "paper sell", "mock sell"]):
        return _handle_open_paper(msg, "short")
    if any(k in lower for k in ["paper long", "mock long", "paper buy", "mock buy"]):
        return _handle_open_paper(msg, "long")
    if any(k in lower for k in ["close paper", "close mock", "close trade"]):
        return _handle_close_paper(msg)
    if any(k in lower for k in ["paper results", "paper performance", "paper journal",
                                   "paper stats", "mock results"]):
        return _paper_report()
    if any(k in lower for k in ["paper trades", "paper trade", "my paper trades",
                                   "open paper trades", "mock trades"]):
        return _handle_paper_status(msg)
    if any(k in lower for k in ["train ml", "train model", "ml training",
                                   "update ml", "retrain", "ml learn"]):
        if _ML_OK:
            result = _run_ml()
            return f"## \U0001f916 ML TRAINING\n\n{result}"
        return "ML engine not available."
    if any(k in lower for k in ["ml insights", "ml report", "what did ml learn",
                                   "ml performance", "ai insights"]):
        if _ML_OK:
            insights = _ml_insights()
            if not insights.get("available"):
                return ("No ML insights yet.\nPaper trade 5+ signals first, "
                        "then type 'train ml'.")
            lines = [
                "## \U0001f916 ML INSIGHTS\n\n"
                f"Trained on: {insights['total_trades']} trades\n"
                f"Overall WR: {insights['overall_wr']}%\n"
                f"Last trained: {insights['trained_at']}\n\n"
                "**What the ML learned:**\n"
            ]
            for insight in insights.get("insights", []):
                lines.append(f"\u2022 {insight}")
            return "\n".join(lines)
        return "ML engine not available."
    if any(k in lower for k in ["indicators", "show indicators", "technical analysis",
                                   "all indicators", "indicator scan", "14 indicators"]):
        return _handle_indicators(msg)
    if any(k in lower for k in ["scalp", "m15", "15 min", "quick trade",
                                   "scalp setup", "fast trade"]):
        return _handle_scalp(msg)
    if any(k in lower for k in ["market read", "read market", "what do you see",
                                   "trader view", "read chart"]):
        return _handle_market_read(msg)
    # ── analyze <instrument> — switch instrument then analyze ────────────────
    _analyze_instr: str | None = None
    if lower.startswith("analyze "):
        _keyword = lower[len("analyze "):].strip().split()[0] if lower[len("analyze "):].strip() else ""
        _analyze_instr = _ANALYZE_ALIASES.get(_keyword)
    if _analyze_instr is None:
        # Also match bare instrument names / aliases used without "analyze" prefix
        # but only when the whole message is just the keyword
        _bare = lower.strip().split()[0]
        if lower.strip() in _ANALYZE_ALIASES or (lower.startswith("analyze") and _bare == "analyze"):
            _analyze_instr = _ANALYZE_ALIASES.get(lower.strip())
    if _analyze_instr is not None:
        return _handle_analyze_instrument(_analyze_instr, msg, account)
    if any(k in lower for k in ["gold", "xauusd", "analyze gold", "xau", "gold setup", "dig into gold"]):
        return _handle_gold(msg, account)
    if any(k in lower for k in ["ml suggest", "ml suggestion", "should i trade", "trade now?"]):
        return _handle_ml_suggest(msg, account)
    if any(k in lower for k in ["best yield", "yield strategy", "fastest strategy", "yield per hour", "top yield"]):
        return _best_yield()
    if any(k in lower for k in ["best signal", "best setup", "top signal", "ml pick", "ml best"]):
        return _handle_best_signal(msg, account)
    if any(k in lower for k in ["signals", "show signals", "what's the trade", "any setups", "trade setup"]):
        return _handle_signals(msg, account)
    if any(k in lower for k in ["news", "calendar", "economic", "news today"]):
        return _handle_news(msg)
    if any(k in lower for k in ["instrument rules", "trading rules", "what rules", "hard rules", "entry rules", "rule check"]):
        return _handle_instrument_rules(msg)
    if any(k in lower for k in ["sector rotation", "sector flow", "money flow", "sectors", "sector analysis", "which sectors"]):
        return _handle_sector_rotation(msg)
    if any(k in lower for k in ["open interest", "volume analysis", "oi", "volume signal", "volume check", "oi analysis"]):
        return _handle_open_interest(msg)
    if any(k in lower for k in ["macro bias", "economic score", "country score", "macro analysis", "currency score", "forex macro"]):
        return _handle_macro_analysis(msg)
    if any(k in lower for k in ["price check", "check price", "live price", "current price"]):
        return _handle_price_check(msg)
    if any(k in lower for k in ["sessions", "market hours", "world markets", "trading hours", "when is london", "when is ny", "when is new york"]):
        return _handle_sessions(msg)
    if any(k in lower for k in ["daily summary", "today results", "how did bot do",
                                   "day summary", "today's results", "bot results today"]):
        return _handle_daily_summary()
    if any(k in lower for k in ["risk of ruin", "ror", "am i safe", "risk check"]):
        return _handle_ror(msg)
    if any(k in lower for k in ["risk", "position size", "risk guide", "lot size", "how much"]):
        return _handle_risk(msg, account)
    if any(k in lower for k in ["backtest", "test "]):
        return _handle_backtest(msg)
    if any(k in lower for k in ["why did it fail", "analyze losses", "post loss", "loss analysis", "failed trades"]):
        return _handle_post_loss(msg)
    if any(k in lower for k in ["weekly review", "week summary", "this week"]):
        return _handle_post_loss("analyze losses")
    if any(k in lower for k in ["full learning report", "brain report", "full brain"]):
        return _handle_full_brain_report(msg)
    if any(k in lower for k in ["learning report", "what has the bot learned", "what did the bot learn"]):
        return _handle_learning_report(msg)
    if any(k in lower for k in ["signal performance", "how accurate is the bot", "bot accuracy", "signal accuracy"]):
        return _handle_signal_performance(msg)
    if any(k in lower for k in ["fundamental bias", "macro bias", "why is gold moving",
                                   "fundamentals", "macro factors"]):
        return _handle_fundamental(msg)
    if any(k in lower for k in ["regime history", "regime changes", "market regime history"]):
        return _handle_regime_history(msg)
    if any(k in lower for k in ["reversal", "bounce", "show reversals",
                                   "reversal hunt", "bounce setup"]):
        return _handle_reversals(msg)
    if any(k in lower for k in [
            "key levels", "support resistance", "sr levels",
            "where is resistance", "where is support", "levels today"]):
        return _handle_key_levels(msg)
    if any(k in lower for k in ["help", "how do i use", "commands", "what can you do", "?"]):
        return _handle_help(msg)
    return _handle_general(msg)


def _handle_daily_summary() -> str:
    """Return a markdown daily P&L summary across all instruments."""
    try:
        from auto_trader import get_at_status, get_daily_summary, INSTRUMENTS as _AT_INSTRUMENTS
        _AT_OK = True
    except ImportError:
        _AT_OK = False

    if not _AT_OK:
        return (
            "### 📊 Daily Summary\n\n"
            "⚠️ Auto trader module not available. "
            "Make sure `auto_trader.py` is present."
        )

    try:
        day = get_daily_summary()

        if "error" in day:
            return f"⚠️ Error loading daily summary: {day['error']}"

        today     = day.get("date", "?")
        n_trades  = day.get("total_trades_today", 0)
        n_wins    = day.get("total_wins", 0)
        n_losses  = day.get("total_losses", 0)
        dpnl      = day.get("total_pnl_pct", 0.0)
        dollar_pnl = round(1000 * dpnl / 100, 2) if dpnl else 0.0

        pnl_emoji = "🟢" if dpnl > 0 else ("🔴" if dpnl < 0 else "⚪")
        wr_str    = (f"{n_wins}/{n_trades} ({100*n_wins//n_trades}%)"
                     if n_trades > 0 else "—")

        lines = [
            f"## 📊 Daily Summary — {today}\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| 📅 Date | {today} |",
            f"| 📈 Trades Today | {n_trades} |",
            f"| ✅ Wins | {n_wins} |",
            f"| ❌ Losses | {n_losses} |",
            f"| 🎯 Win Rate | {wr_str} |",
            f"| {pnl_emoji} Day P&L | {dpnl:+.1f}% (${dollar_pnl:+.2f}) |",
            "",
            "### By Instrument",
        ]

        instr_data = day.get("instruments", {})
        any_trades = False
        for instr, d in sorted(
                instr_data.items(),
                key=lambda x: _AT_INSTRUMENTS.get(x[0], {}).get("priority", 99)):
            if d.get("trades", 0) == 0:
                continue
            any_trades = True
            grade = _AT_INSTRUMENTS.get(instr, {}).get("grade", "?")
            pnl   = d.get("pnl_pct", 0.0)
            col   = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            lines.append(
                f"- **{instr}** [{grade}]: "
                f"{d['wins']}W {d['losses']}L | "
                f"P&L: {col} {pnl:+.1f}%"
            )

        if not any_trades:
            lines.append("*No auto trades closed today yet.*")

        best  = day.get("best_trade")
        worst = day.get("worst_trade")
        if best:
            lines.append(
                f"\n🏆 **Best trade:** {best['instr']} {best['direction']} "
                f"→ {best['outcome']} ({best['pnl_pct']:+.2f}%)"
            )
        if worst and worst != best:
            lines.append(
                f"💔 **Worst trade:** {worst['instr']} {worst['direction']} "
                f"→ {worst['outcome']} ({worst['pnl_pct']:+.2f}%)"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ Daily summary error: {e}"


def _handle_regime_history(_msg: str) -> str:
    """Show last 24 entries from regime_history.json."""
    try:
        from market_context import get_regime_history as _grh
        history = _grh(last_n=24)
        if not history:
            return (
                "### 📈 Regime History\n\n"
                "No regime history yet.  History is recorded every 60 seconds during active sessions.  "
                "Leave the bot open and check back later."
            )
        _REGIME_ICONS = {
            "TRENDING_STRONG":    "🟢🟢",
            "TRENDING_WEAK":      "🟢",
            "RANGING":            "🔵",
            "VOLATILE_EXPANDING": "🔴",
            "SQUEEZE_BUILDING":   "⚪",
        }
        lines = ["### 📈 Regime History (last 24 entries)\n"]
        lines.append("| Time (UAE) | Regime | ATR | EMA50 Slope | Price |")
        lines.append("|-----------|--------|-----|-------------|-------|")
        for snap in reversed(history):
            ts     = str(snap.get("timestamp", "—"))  # full timestamp
            regime = snap.get("regime", "RANGING")
            icon   = _REGIME_ICONS.get(regime, "⚪")
            atr    = snap.get("atr_now", 0.0)
            slope  = snap.get("ema50_slope", 0.0)
            price  = snap.get("price", 0.0)
            label  = regime.replace("_", " ").title()
            price_s = f"${price:,.0f}" if price > 0 else "—"
            lines.append(f"| {ts} | {icon} {label} | ${atr:.1f} | {slope:+.3f} | {price_s} |")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Regime history unavailable: {e}"


def _handle_key_levels(_msg: str) -> str:
    """Return a markdown S/R key-levels map for XAUUSD."""
    try:
        from sr_mapper import get_sr_levels as _gsr
    except ImportError:
        return "⚠️ S/R mapper module not available."

    df_live = st.session_state.get("live_df") or st.session_state.get("df")
    price   = st.session_state.get("live_price", 0)

    if df_live is None or not price:
        return (
            "Run **setup** first to load price data, then ask for key levels again."
        )

    try:
        sr = _gsr(df_live, float(price))
    except Exception as e:
        return f"⚠️ S/R calculation error: {e}"

    lines = ["### 📍 Key Levels — XAUUSD\n"]
    lines.append(f"Current price: **${price:,.2f}**\n")

    res_lvls = sr.get("resistance_levels", [])[:5]
    sup_lvls = sr.get("support_levels",    [])[:5]

    lines.append("**RESISTANCE:**")
    if res_lvls:
        for r in res_lvls:
            dist = r["distance_usd"]
            prox = f"[{r['proximity']}]" if r["proximity"] != "DISTANT" else ""
            lines.append(
                f"- `${r['price']:,.2f}` — {r['label']}  "
                f"**[{r['strength']}]** {prox} ({dist:.1f} away)"
            )
    else:
        lines.append("- No resistance levels identified")

    lines.append(f"\n**→ CURRENT: ${price:,.2f}**\n")

    lines.append("**SUPPORT:**")
    if sup_lvls:
        for s in sup_lvls:
            dist = s["distance_usd"]
            prox = f"[{s['proximity']}]" if s["proximity"] != "DISTANT" else ""
            lines.append(
                f"- `${s['price']:,.2f}` — {s['label']}  "
                f"**[{s['strength']}]** {prox} ({dist:.1f} away)"
            )
    else:
        lines.append("- No support levels identified")

    if sr.get("at_key_level"):
        lines.append(
            f"\n⭐ **PRICE AT KEY LEVEL**\n{sr.get('key_level_detail', '')}\n"
            f"Setup quality is higher when entering near a key level."
        )

    pw_h = sr.get("prev_week_high", 0)
    pw_l = sr.get("prev_week_low", 0)
    pd_h = sr.get("prev_day_high", 0)
    pd_l = sr.get("prev_day_low", 0)

    if pw_h or pw_l:
        lines.append(
            f"\nPrev Week: High **${pw_h:,.2f}** | Low **${pw_l:,.2f}**"
        )
    if pd_h or pd_l:
        lines.append(
            f"Prev Day:  High **${pd_h:,.2f}** | Low **${pd_l:,.2f}**"
        )

    return "\n".join(lines)


def _handle_fundamental(_msg: str) -> str:
    """Return a markdown fundamental bias report for gold."""
    if not _FB_OK:
        return "⚠️ Fundamental bias module not available."
    try:
        fb = _get_fundamental_bias()
        bias   = fb.get("fundamental_bias", "NEUTRAL")
        score  = fb.get("total_score", 0)
        summ   = fb.get("summary", "")
        conf   = fb.get("confidence", 5.0)
        tf     = fb.get("timeframe", "")
        facs   = fb.get("factors", {})
        bias_emoji = {
            "STRONGLY_BULLISH": "🟢🟢",
            "BULLISH":          "🟢",
            "NEUTRAL":          "⚪",
            "BEARISH":          "🔴",
            "STRONGLY_BEARISH": "🔴🔴",
        }.get(bias, "⚪")
        lines = [
            f"### {bias_emoji} Macro Fundamental Bias — XAUUSD",
            f"",
            f"**Bias:** {bias.replace('_',' ').title()}  |  **Score:** {score:+d}/9  |  **Confidence:** {conf:.1f}/10",
            f"**Summary:** {summ}",
            f"**Timeframe:** {tf}",
            f"",
            f"| Factor | Score | Notes |",
            f"|--------|-------|-------|",
        ]
        factor_icons = {
            "inflation":    "📈 Inflation",
            "oil":          "🛢 Oil",
            "fed":          "🏦 Fed",
            "dxy":          "💵 DXY",
            "geopolitical": "🌍 Geo Risk",
        }
        for key, icon in factor_icons.items():
            f = facs.get(key, {})
            s = f.get('score', 0)
            n = f.get('note', '—')
            sign = f"{s:+d}" if s != 0 else "0"
            lines.append(f"| {icon} | {sign} | {n} |")
        if not fb.get("available"):
            lines.append(f"\n⚠️ *Some data sources unavailable — result may be partial.*")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Fundamental bias error: {e}"


def _handle_reversals(_msg: str) -> str:
    df = _load_df()
    if df is None:
        return "❌ No data available — run `setup` first."
    reversals = _hunt_reversals(df, st.session_state.get("live_price"))
    if not reversals:
        return "No reversal signals detected right now."
    lines = ["### 🔄 Reversal Opportunities\n"]
    for rev in reversals:
        lines.append(
            f"**{rev['pattern_name']}** — {rev['direction'].upper()}\n"
            f"Score: {rev['score']}/11 | Confidence: {rev['confidence']}/10\n"
            f"Conditions: {', '.join(rev['conditions_met'][:3])}\n"
            f"Entry: ${rev['entry']:,.2f} | SL: ${rev['stop_loss']:,.2f} | "
            f"TP: ${rev['take_profit']:,.2f}\n"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_home() -> None:
    """Render the home screen with quick-action buttons and glossary."""
    _active_instr = st.session_state.get("instrument", "XAUUSD")
    st.markdown(
        f"**Trading {_active_instr}** — signals, strategies, risk"
    )

    # Live price card — same source priority as sidebar (MT5 → yfinance)
    try:
        _instr    = st.session_state.get("instrument", "XAUUSD")
        _hprice   = 0.0
        _hsrc     = ""
        # Priority 1: MT5 / get_price_for_instrument
        try:
            from mt5_sync import get_price_for_instrument as _gpfi_home
            _hprice = float(_gpfi_home(_instr) or 0)
            _hsrc   = "MT5" if _hprice > 0 else ""
        except Exception:
            pass
        # Priority 2: yfinance (5d window so weekends always have data)
        if _hprice <= 0:
            import yfinance as yf
            _YF_HOME = {
                "XAUUSD": "GC=F", "NAS100": "NQ=F", "US30": "YM=F",
                "GBPUSD": "GBPUSD=X", "EURUSD": "EURUSD=X", "WTI": "CL=F",
            }
            _tk   = yf.Ticker(_YF_HOME.get(_instr, "GC=F"))
            _hist = _tk.history(period="5d", interval="5m")
            if _hist.empty:
                _hist = _tk.history(period="1mo", interval="1d")
            if not _hist.empty:
                _hprice = float(_hist["Close"].iloc[-1])
                _hsrc   = "yfinance"
        # Also sync session state so header and sidebar show identical price
        if _hprice > 0:
            st.session_state["live_price"]  = _hprice
            st.session_state["live_source"] = _hsrc
        # Compute day change from session state history if available
        _prev = st.session_state.get("_home_open_price", _hprice)
        if "_home_open_price" not in st.session_state:
            st.session_state["_home_open_price"] = _hprice
        _hchange = _hprice - _prev
        _hpct    = (_hchange / _prev * 100) if _prev else 0.0
        _hcolor  = "🟢" if _hchange >= 0 else "🔴"
        if _hprice > 0:
            st.success(
                f"💰 **{_instr}**: "
                f"${_hprice:,.4f} "
                f"{_hcolor} {_hchange:+.2f} "
                f"({_hpct:+.2f}%) "
                f"<small>[{_hsrc}]</small>",
                # unsafe_allow_html only available in st.markdown — use caption for source
            )
            st.caption(f"Source: {_hsrc} | Updates every 60s")
        else:
            st.warning(f"⏳ {_instr} price updating...")
    except Exception as _he:
        st.warning(f"⏳ Price loading ({str(_he)[:60]})") 

    st.markdown("---")

    # ── Row 1: Quick Actions ─────────────────────────────────────────────────
    st.markdown("### ⚡ Quick Actions")
    _hc1, _hc2, _hc3 = st.columns(3)
    with _hc1:
        if st.button("🔧 Setup", use_container_width=True, key="home_setup"):
            st.session_state["pending_cmd"] = "run setup"
    with _hc2:
        if st.button("📊 Analyze", use_container_width=True, key="home_analyze"):
            st.session_state["pending_cmd"] = f"analyze {_active_instr.lower()}"
    with _hc3:
        if st.button("🎯 Signals", use_container_width=True, key="home_signals"):
            st.session_state["pending_cmd"] = "show signals"

    _hc4, _hc5, _hc6 = st.columns(3)
    with _hc4:
        if st.button("📰 News", use_container_width=True, key="home_news"):
            st.session_state["pending_cmd"] = "news today"
    with _hc5:
        if st.button("💰 Risk Guide", use_container_width=True, key="home_risk"):
            st.session_state["pending_cmd"] = "risk guide"
    with _hc6:
        if st.button("📈 Backtest", use_container_width=True, key="home_bt"):
            st.session_state["pending_cmd"] = "backtest"

    # ── Row 2: ML Brain ───────────────────────────────────────────────────────
    st.markdown("### 🧠 ML Brain")
    _hc7, _hc8, _hc9 = st.columns(3)
    with _hc7:
        if st.button("🤖 ML Suggest", use_container_width=True, key="home_ml"):
            st.session_state["pending_cmd"] = "ml suggest"
    with _hc8:
        if st.button("🏆 Best Signal", use_container_width=True, key="home_best"):
            st.session_state["pending_cmd"] = "best signal"
    with _hc9:
        if st.button("📋 Why Analysis", use_container_width=True, key="home_why"):
            st.session_state["pending_cmd"] = "why analysis"

    # ── Row 3: Paper Trading ───────────────────────────────────────────────
    st.markdown("### 📝 Paper Trading")
    _hc10, _hc11, _hc12 = st.columns(3)
    with _hc10:
        if st.button("📄 My Trades", use_container_width=True, key="home_trades"):
            st.session_state["pending_cmd"] = "paper trades"
    with _hc11:
        if st.button("📊 Results", use_container_width=True, key="home_results"):
            st.session_state["pending_cmd"] = "paper results"
    with _hc12:
        if st.button("🤖 Auto Status", use_container_width=True, key="home_auto"):
            st.session_state["pending_cmd"] = "auto status"

    # ── Row 4: Market Context ──────────────────────────────────────────────
    st.markdown("### 🌍 Market Context")
    _hc13, _hc14, _hc15 = st.columns(3)
    with _hc13:
        if st.button("💹 Sector Flow", use_container_width=True, key="home_sector"):
            st.session_state["pending_cmd"] = "sector rotation"
    with _hc14:
        if st.button("📊 Open Interest", use_container_width=True, key="home_oi"):
            st.session_state["pending_cmd"] = "open interest"
    with _hc15:
        if st.button("🌐 Macro Bias", use_container_width=True, key="home_macro"):
            st.session_state["pending_cmd"] = "macro analysis"

    st.markdown("---")

    # ── Glossary ───────────────────────────────────────────────────────────────
    st.markdown("### 📖 Command Glossary")
    with st.expander("📊 Analysis Commands", expanded=False):
        st.markdown("""
| Command | What it does |
|---|---|
| `analyze gold` | Full XAUUSD analysis |
| `analyze nas100` | Full NAS100 analysis |
| `analyze us30` | Full US30 analysis |
| `analyze eurusd` | Full EURUSD analysis |
| `analyze gbpusd` | Full GBPUSD analysis |
| `analyze wti` | Full WTI Crude analysis |
| `show signals` | All passing signals |
| `market read` | Plain English market view |
| `key levels` | Support & resistance |
| `reversal` | Reversal setups |
| `indicators` | 14 technical indicators |
""")
    with st.expander("🧠 ML Commands", expanded=False):
        st.markdown("""
| Command | What it does |
|---|---|
| `ml suggest` | ML quality assessment |
| `best signal` | ML picks best setup |
| `ml insights` | Win/loss patterns |
| `train ml` | Retrain ML model |
| `full brain report` | Brain 1 + 2 combined |
""")
    with st.expander("📝 Paper Trading", expanded=False):
        st.markdown("""
| Command | What it does |
|---|---|
| `paper long` | Open LONG paper trade |
| `paper short` | Open SHORT paper trade |
| `paper trades` | See all open trades |
| `close paper [ID]` | Close a trade |
| `paper results` | Performance report |
| `daily summary` | Today's P&L |
""")
    with st.expander("🌍 Market Context", expanded=False):
        st.markdown("""
| Command | What it does |
|---|---|
| `sector rotation` | Money flow by sector |
| `open interest` | Volume signal |
| `macro analysis` | Economic health score |
| `cot data` | Institutional positioning |
| `liquidity` | BSL/SSL levels |
| `session handoff` | Asian/London/NY bias |
| `news today` | Economic calendar |
| `fundamental bias` | Why gold is moving |
""")
    with st.expander("⚙️ System Commands", expanded=False):
        st.markdown("""
| Command | What it does |
|---|---|
| `run setup` | Load all data |
| `price check` | Live price |
| `risk guide` | Position sizing |
| `backtest` | Run strategy backtest |
| `wfo` | Walk-forward optimization |
| `instrument rules` | Hard rules check |
| `help` | Full command list |
""")


def _render_sidebar(account: float) -> None:
    import time as _time
    with st.sidebar:
        # ── Instrument selector (top of sidebar) ─────────────────────────────
        _INSTRUMENT_INFO = {
            "XAUUSD": "🭇 Gold — Session: London/NY overlap",
            "NAS100": "💻 Nasdaq 100 — Session: NY 13:30-22:00 GST",
            "US30":   "🏭 Dow Jones — Session: NY 13:30-22:00 GST",
            "GBPUSD": "💷 GBP/USD — Session: London 08:00-17:00 GST",
            "EURUSD": "💶 EUR/USD — Session: Frankfurt+London 07:00-17:00 GST",
            "WTI":    "🛢️ Crude Oil — Session: NY 13:30-22:00 GST + EIA Wed",
        }
        selected_instrument = st.selectbox(
            "🎯 Instrument",
            ["XAUUSD", "NAS100", "US30", "GBPUSD", "EURUSD", "WTI"],
            key="instrument_selector",
        )
        st.session_state["instrument"] = selected_instrument
        st.caption(_INSTRUMENT_INFO.get(selected_instrument, selected_instrument))

        # ── Macro bias card (forex pairs only) ───────────────────────────────
        if selected_instrument in ("GBPUSD", "EURUSD") and _MACRO_OK and _macro_scorer is not None:
            try:
                _mb = _macro_scorer.score_pair(selected_instrument)
                _mb_bias  = _mb.get("bias", "NEUTRAL")
                _mb_conf  = _mb.get("confidence", 0)
                _mb_color = ("#1a6e34" if "LONG" in _mb_bias
                             else "#6e1a1a" if "SHORT" in _mb_bias
                             else "#3a3a3a")
                _mb_base  = _mb.get("base_currency", "?")
                _mb_quote = _mb.get("quote_currency", "?")
                _mb_diff  = _mb.get("score_diff", 0)
                st.markdown(
                    f"<div style='background:{_mb_color};border-radius:6px;"
                    f"padding:6px 10px;margin-top:4px;font-size:12px;'>"
                    f"<b>📊 Macro Bias</b>&nbsp;&nbsp;"
                    f"<span style='color:#f0f0f0'>{_mb_bias}</span>&nbsp;"
                    f"<small style='color:#aaa'>({_mb_conf}% conf &nbsp;|&nbsp; "
                    f"{_mb_base} {_mb_diff:+d} vs {_mb_quote})</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
        elif selected_instrument in ("NAS100", "US30"):
            # VIX pill
            if _MACRO_OK:
                try:
                    _vx = _get_mkt_ctx(selected_instrument)
                    if _vx.get("vix"):
                        _v  = _vx["vix"]
                        _vc = "#6e1a1a" if _v > 25 else ("#7a6a00" if _v > 18 else "#1a6e34")
                        _vl = "HIGH RISK" if _v > 25 else ("ELEVATED" if _v > 18 else "LOW RISK")
                        st.markdown(
                            f"<div style='background:{_vc};border-radius:6px;"
                            f"padding:6px 10px;margin-top:4px;font-size:12px;'>"
                            f"<b>😨 VIX</b>&nbsp;&nbsp;"
                            f"<span style='color:#f0f0f0'>{_v:.2f} — {_vl}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
            # Sector flow pill
            if _SR_OK and _sector_rotation is not None:
                try:
                    _sb_bias = _sector_rotation.get_instrument_bias(selected_instrument)
                    _sb_b    = _sb_bias.get("bias", "NEUTRAL")
                    _sb_reg  = _sb_bias.get("risk_regime", "NEUTRAL")
                    _sb_col  = ("#1a6e34" if "BULL" in _sb_b else
                                "#6e1a1a" if "BEAR" in _sb_b or "CAUTION" in _sb_b else
                                "#3a3a3a")
                    st.markdown(
                        f"<div style='background:{_sb_col};border-radius:6px;"
                        f"padding:6px 10px;margin-top:4px;font-size:12px;'>"
                        f"<b>🔄 Sector</b>&nbsp;&nbsp;"
                        f"<span style='color:#f0f0f0'>{_sb_b}</span>"
                        f"<small style='color:#aaa'>&nbsp;| Risk: {_sb_reg}</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass
        elif selected_instrument == "XAUUSD":
            # DXY direction pill
            if _MACRO_OK:
                try:
                    _xau_ctx = _get_mkt_ctx("XAUUSD")
                    if _xau_ctx.get("dxy"):
                        _dv = _xau_ctx["dxy"]
                        st.markdown(
                            f"<div style='background:#2a2d3e;border-radius:6px;"
                            f"padding:6px 10px;margin-top:4px;font-size:12px;'>"
                            f"<b>💵 DXY</b>&nbsp;&nbsp;"
                            f"<span style='color:#f0f0f0'>{_dv:.3f}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
        elif selected_instrument == "WTI":
            # EIA report Wednesday warning (GST = UTC+4)
            try:
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _now_gst = _dt.now(_tz((_td(hours=4))))
                if _now_gst.weekday() == 2:  # Wednesday
                    st.markdown(
                        "<div style='background:#6e3a00;border-radius:6px;"
                        "padding:6px 10px;margin-top:4px;font-size:12px;'>"
                        "<b>⚠️ EIA Report</b>&nbsp;&nbsp;"
                        "<span style='color:#f0f0f0'>Today at 18:30 GST</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        st.markdown("---")

        dot = "🟢" if st.session_state["is_live"] else "🔴"
        st.markdown(f"## 📈 TradingBotV1 {dot}")
        st.caption(f"Pepperstone #51486884 · {st.session_state.get('instrument', 'XAUUSD')} H1")

        # ── Live clock + price ────────────────────────────────────────────────
        _gst_sb      = timezone(timedelta(hours=4))
        _sidebar_time = datetime.now(_gst_sb).strftime("%I:%M %p")
        _sidebar_date = datetime.now(_gst_sb).strftime("%a %d %b %Y")
        _sb_instr = st.session_state.get("instrument", "XAUUSD")
        st.markdown(f"**🕐 {_sidebar_time} UAE** | {_sidebar_date}")
        try:
            # Priority 1: MT5 live tick — if it returns a price, market is OPEN
            _sb_price = 0.0
            _sb_src   = ""
            try:
                from mt5_sync import get_price_for_instrument as _gpfi_sb
                _sb_price = float(_gpfi_sb(_sb_instr) or 0)
                _sb_src   = "MT5" if _sb_price > 0 else ""
            except Exception:
                pass
            # Priority 2: yfinance (5-day window so weekends always show last price)
            if _sb_price <= 0:
                import yfinance as yf
                _YF_SB = {
                    "XAUUSD": "GC=F", "NAS100": "NQ=F", "US30": "YM=F",
                    "GBPUSD": "GBPUSD=X", "EURUSD": "EURUSD=X", "WTI": "CL=F",
                }
                _tk_sb = yf.Ticker(_YF_SB.get(_sb_instr, "GC=F"))
                _h_sb  = _tk_sb.history(period="5d", interval="5m")
                if _h_sb.empty:
                    _h_sb = _tk_sb.history(period="1mo", interval="1d")
                if not _h_sb.empty:
                    _sb_price = float(_h_sb["Close"].iloc[-1])
                    _sb_src   = "yfinance"
            if _sb_price > 0:
                st.markdown(
                    f"**💰 {_sb_instr}: ${_sb_price:,.2f}** "
                    f"<small>[{_sb_src}]</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"{_sb_instr}: price unavailable")
        except Exception:
            st.info("Price loading...")
        st.caption(f"🌍 {get_session_summary_line() if _WS_OK else _current_session()}")
        st.caption("Price and time refresh every 60 seconds")

        st.markdown("---")
        if st.button("\U0001f504 Refresh Prices Now",
                     key="manual_refresh_btn",
                     use_container_width=True):
            try:
                from mt5_sync import get_price_for_instrument as _gpfi_rb
                _rb_instr = st.session_state.get("instrument", "XAUUSD")
                _live_rb  = _gpfi_rb(_rb_instr)
                _price_rb = _live_rb.get("price", 0) if isinstance(_live_rb, dict) else float(_live_rb or 0)
                _src_rb   = _live_rb.get("source", "live") if isinstance(_live_rb, dict) else "live"
                st.session_state["live_price"]  = _price_rb
                st.session_state["live_source"] = _src_rb
                # Update all paper trades with new price
                if _PAPER_OK and _price_rb > 0:
                    _closed_rb = _update_paper(_price_rb)
                    for _t_rb in _closed_rb:
                        _e_rb     = "\u2705" if _t_rb["outcome"] == "WIN" else "\u274c"
                        _sign_rb  = "+" if _t_rb["pnl_pips"] > 0 else ""
                        _notif_rb = st.session_state.get("mt5_sync_notifications", [])
                        _notif_rb.append(
                            f"{_e_rb} PAPER TRADE CLOSED \u2014 {_t_rb['trade_id']}\n"
                            f"{_t_rb['direction'].upper()} {_t_rb['outcome']}\n"
                            f"{_t_rb.get('close_reason', '')} at "
                            f"${_t_rb.get('close_price', 0):,.2f}\n"
                            f"P&L: {_sign_rb}{_t_rb['pnl_pips']} pips | "
                            f"Time held: {_t_rb.get('time_held', '?')}\n"
                            f"{_t_rb.get('close_detail', '')}"
                        )
                        st.session_state["mt5_sync_notifications"] = _notif_rb
                # Update UAE time
                _gst_rb = timezone(timedelta(hours=4))
                st.session_state["current_uae_time"] = datetime.now(_gst_rb).strftime("%I:%M %p")
                st.session_state["current_uae_date"] = datetime.now(_gst_rb).strftime("%A %d %B %Y")
                st.success(f"\u2705 Refreshed! ${_price_rb:,.2f} [{_src_rb}]")
                st.rerun()
            except Exception as _e_rb:
                st.error(f"Refresh failed: {str(_e_rb)[:50]}")

        # ── Reversal Hunter sidebar widget ────────────────────────────────────
        try:
            if _RH_OK:
                _df_sb  = st.session_state.get("live_df")
                if _df_sb is not None:
                    _revs_sb = _hunt_reversals(_df_sb, st.session_state.get("live_price"))
                    if _revs_sb:
                        _best_r = _revs_sb[0]
                        _rcol   = "#1D9E75" if _best_r["reversal_strength"] == "STRONG" else "#F4C542"
                        st.markdown(
                            f"<span style='color:{_rcol}'>🔄 Reversal: "
                            f"{_best_r['direction'].upper()} {_best_r['reversal_strength']} "
                            f"({_best_r['score']}/11)</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("🔄 Reversal: None detected")
        except Exception:
            pass

        # ── Risk Rating indicator ─────────────────────────────────────────────
        if _TM_OK:
            try:
                _ror_sb = _get_ror_profile()
                _ror_color_map = {
                    "SAFE":     "#1D9E75",
                    "MODERATE": "#F4C542",
                    "HIGH":     "#E07820",
                    "DANGER":   "#E05555",
                }
                _ror_color = _ror_color_map.get(_ror_sb.get("risk_rating", ""), "grey")
                st.markdown(
                    f"<span style='color:{_ror_color}'>⚠ Risk Rating: "
                    f"**{_ror_sb.get('risk_rating', '?')}** "
                    f"({_ror_sb.get('ruin_probability', 0):.1f}% ruin)</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        # WFO status line
        if _WFO_OK:
            try:
                import json as _wfo_json, os as _wfo_os
                _wfo_hist_path = os.path.join(DATA_DIR, "wfo_history.json")
                _wfo_last_str  = "Never"
                if _wfo_os.path.exists(_wfo_hist_path):
                    with open(_wfo_hist_path) as _wf:
                        _wfo_hist = _wfo_json.load(_wf)
                    if _wfo_hist:
                        _wfo_last_dt = datetime.fromisoformat(
                            _wfo_hist[-1].get("run_at", "")
                        )
                        _wfo_last_str = _wfo_last_dt.strftime("%d %b")
                st.caption(f"🔄 Last optimized: {_wfo_last_str} | Next: Sunday")
            except Exception:
                st.caption("🔄 Last optimized: — | Next: Sunday")

        st.divider()

        # ── MT5 ACCOUNT (live) ────────────────────────────────────────────────
        if IS_CLOUD:
            st.info("🌐 **Cloud Mode** — MT5 not available.  \nPrices via yfinance.", icon="ℹ️")
        else:
            acct      = st.session_state.get("mt5_account")
            connected = st.session_state.get("mt5_connected", False)
            today_pnl = st.session_state.get("mt5_today_pnl")
            sync_time = st.session_state.get("mt5_last_sync")

            if connected and acct:
                pnl_val  = (today_pnl or {}).get("pnl", 0.0)
                pnl_sign = "+" if pnl_val >= 0 else ""
                pnl_col  = "#1D9E75" if pnl_val >= 0 else "#E05555"
                wins     = (today_pnl or {}).get("wins",   0)
                losses   = (today_pnl or {}).get("losses", 0)
                n_trades = (today_pnl or {}).get("trades", 0)
                st.markdown("**ACCOUNT** *(live from MT5)*")
                st.markdown(
                    f"Balance:&nbsp;&nbsp;**${acct['balance']:,.2f}**  "
                    f"\nEquity:&nbsp;&nbsp;&nbsp;&nbsp;**${acct['equity']:,.2f}**  "
                    f"\nFree margin: **${acct['margin_free']:,.2f}**"
                )
                st.markdown(
                    f"Today P&L: <span style='color:{pnl_col};font-weight:700'>"
                    f"{pnl_sign}${pnl_val:,.2f}</span>  "
                    f"({wins}W / {losses}L, {n_trades} trades)",
                    unsafe_allow_html=True,
                )
                sync_str = sync_time.strftime("%H:%M GST") if sync_time else "—"
                st.caption(f"MT5 #{acct['account']} · {acct['currency']} · 1:{acct['leverage']}  · synced {sync_str}")
            else:
                st.warning(
                    "**MT5 offline**  \n"
                    "Open MetaTrader 5 and log in  \n"
                    "to sync live data",
                    icon="⚠️",
                )
                _mt5_err = st.session_state.get("mt5_error")
                if _mt5_err:
                    st.caption(f"Error: {_mt5_err}")
                st.caption(
                    "**To fix:** MT5 → Tools → Options → Expert Advisors "
                    "→ ✅ Allow algorithmic trading"
                )

            # ── Sync MT5 button ───────────────────────────────────────────────
            if _MT5_SYNC_OK:
                sb1, sb2 = st.columns(2)
                with sb1:
                    if st.button("🔁 Sync MT5", key="sb_mt5_sync", use_container_width=True):
                        with st.spinner("Connecting to MT5…"):
                            _refresh_mt5_data()
                        if st.session_state.get("mt5_connected"):
                            try:
                                new_n, total_n = sync_to_journal(days_back=30)
                                if new_n:
                                    st.success(f"✅ Connected! {new_n} new trade(s) synced.")
                                    st.session_state["trigger_cmd"] = f"__mt5_sync_done_{new_n}_{total_n}"
                                else:
                                    st.success("✅ MT5 Connected! Journal up to date.")
                            except Exception as _je:
                                st.success("✅ MT5 Connected!")
                                st.caption(f"Journal sync: {_je}")
                        else:
                            _btn_err = st.session_state.get("mt5_error", "Unknown error")
                            st.error(
                                f"❌ Still offline\n\n"
                                f"`{_btn_err}`\n\n"
                                "Make sure MT5 is running and logged in."
                            )
                        st.rerun()
                with sb2:
                    if st.button("📒 Journal", key="sb_journal", use_container_width=True):
                        st.session_state["trigger_cmd"] = "show journal"

        st.divider()

        # ── OPEN POSITIONS (live) ─────────────────────────────────────────────
        positions = st.session_state.get("mt5_positions", [])
        if positions:
            st.markdown(f"**OPEN POSITIONS** ({len(positions)})")
            for pos in positions:
                pnl_p   = pos["pnl_usd"]
                p_col   = "#1D9E75" if pnl_p >= 0 else "#E05555"
                p_sign  = "+" if pnl_p >= 0 else ""
                dir_ico = "🔼" if pos["direction"] == "long" else "🔽"
                st.markdown(
                    f"{dir_ico} **{pos['symbol']}** {pos['direction'].upper()}  "
                    f"{pos['lots']} lots  \n"
                    f"Entry: ${pos['entry']:,.2f}  →  Now: ${pos['current_price']:,.2f}  \n"
                    f"P&L: <span style='color:{p_col};font-weight:700'>{p_sign}${pnl_p:,.2f}</span>",
                    unsafe_allow_html=True,
                )
                # "Mark as Bot Signal" button
                btn_key = f"mark_signal_{pos['ticket']}"
                if st.button(f"🤖 Mark as bot signal", key=btn_key, use_container_width=True):
                    st.session_state["trigger_cmd"] = (
                        f"__mark_signal_{pos['ticket']}_"
                        f"{pos['symbol']}_{pos['direction']}_"
                        f"{pos['entry']}"
                    )
            st.divider()

        # ── Auto-sync notifications ───────────────────────────────────────────
        notes = st.session_state.get("mt5_sync_notifications", [])
        if notes:
            st.markdown("**🔔 Auto-synced from MT5:**")
            for n in notes[-3:]:
                if not isinstance(n, dict):
                    continue
                pnl_s = n.get("pnl", "")
                st.success(
                    f"{n.get('symbol','')} {n.get('direction','')} closed ${n.get('close_price',0):,} "
                    f"→ **{n.get('outcome','')}** {pnl_s}  \n"
                    f"Pattern memory updated."
                )
            st.session_state["mt5_sync_notifications"] = []
            st.divider()

        # ── Auto Trader panel ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Auto Trader")
        try:
            from auto_trader import (
                start_auto_trader as _at_start,
                stop_auto_trader  as _at_stop,
                get_status        as _at_get_status,
            )
            _at = _at_get_status()
            _at_running = _at.get("enabled", False)
            if _at_running:
                st.success("🟢 ACTIVE — scanning 60s")
            else:
                st.error("🔴 STOPPED")
            _atc1, _atc2 = st.columns(2)
            with _atc1:
                _at_lbl = "⏹️ STOP" if _at_running else "▶️ START"
                if st.button(_at_lbl, key="at_sidebar_btn",
                             use_container_width=True):
                    if _at_running:
                        _at_stop()
                    else:
                        _at_start()
                    st.rerun()
            with _atc2:
                st.metric("Trades", _at.get("total_trades", 0))
            # Per instrument detailed view
            _at_total_pnl = 0.0
            for _at_instr in [
                    "XAUUSD", "WTI", "US30",
                    "NAS100", "GBPUSD", "EURUSD"]:
                _at_d      = _at.get("instruments", {}).get(_at_instr, {})
                _at_done   = _at_d.get("trades_today", 0)
                _at_open   = _at_d.get("has_open", False)
                _at_pnl    = _at_d.get("daily_pnl", 0.0)
                _at_total_pnl += _at_pnl
                if _at_open:
                    _at_status = "🔄"
                elif _at_done >= 2:
                    _at_status = "✅"
                else:
                    _at_status = "⬜"
                _at_pnl_color = (
                    "🟢" if _at_pnl > 0
                    else "🔴" if _at_pnl < 0
                    else "⚪")
                _at_dollar = abs(_at_pnl / 100 * 1000)
                _at_dollar_str = (
                    f"+${_at_dollar:.0f}" if _at_pnl >= 0
                    else f"-${_at_dollar:.0f}")
                st.markdown(
                    f"{_at_status} **{_at_instr}** "
                    f"{_at_done}/2 trades | "
                    f"{_at_pnl_color} {_at_pnl:+.1f}% "
                    f"({_at_dollar_str})")
            st.markdown("---")
            _at_tot_dollar = abs(_at_total_pnl / 100 * 1000)
            _at_tot_color = (
                "🟢" if _at_total_pnl > 0
                else "🔴" if _at_total_pnl < 0
                else "⚪")
            _at_tot_sign = (
                f"+${_at_tot_dollar:.0f}" if _at_total_pnl >= 0
                else f"-${_at_tot_dollar:.0f}")
            st.markdown(
                f"**📊 Today Total:** "
                f"{_at_tot_color} {_at_total_pnl:+.1f}% "
                f"({_at_tot_sign})")
            st.caption(f"Last scan: {_at.get('last_scan', 'Never')}")
        except Exception as _at_e:
            st.warning("Auto trader loading...")
            st.caption(f"Error: {str(_at_e)[:60]}")

        # ── Bot status items ──────────────────────────────────────────────────
        rules_n  = st.session_state["rules_count"]  or 610
        pb_n     = st.session_state["playbooks_count"]
        last     = st.session_state["last_refresh"]
        sess     = st.session_state.get("session_name") or _current_session()
        dxy_s    = st.session_state.get("dxy_status", "—")
        d1_b     = st.session_state.get("d1_bias",    "—")
        h4_b     = st.session_state.get("h4_bias",    "—")
        last_str = last.strftime("%H:%M GST") if last else "Never"

        def _b_icon(b: str) -> str:
            b = str(b).upper()
            return "📈" if "BULL" in b else ("📉" if "BEAR" in b else "➡️")

        def _d_icon(d: str) -> str:
            d = str(d).lower()
            return "📈" if "ris" in d or "up" in d else ("📉" if "fall" in d or "down" in d else "➡️")

        st.markdown(f"📚 **Rules loaded:** {rules_n}")
        st.markdown(f"🎯 **Playbooks:** {pb_n} active")
        st.markdown(f"🕐 **Last refresh:** {last_str}")
        st.markdown(f"🌐 **Session:** {sess}")

        # ── Session profile badge ─────────────────────────────────────────
        try:
            from session_profiler import get_current_session_profile as _gsp
            _sp_now  = _gsp()
            _sp_g    = _sp_now.get("session_grade", "B")
            _sp_lot  = _sp_now.get("lot_multiplier",  1.0)
            _sp_sl   = _sp_now.get("sl_multiplier",   1.0)
            _sp_rec  = _sp_now.get("trading_recommended", True)
            _sp_colour = {
                "A": "#1D9E75",    # green
                "B": "#F4C542",    # yellow
                "C": "#E08020",    # orange
            }.get(_sp_g, "#E05555")  # red for OffHours / unknown
            if sess == "OffHours":
                _sp_colour = "#E05555"
            _sp_warn = " \u26a0" if not _sp_rec else ""
            st.markdown(
                f"📊 <span style='color:{_sp_colour};font-weight:700'>"
                f"{sess} Grade {_sp_g}{_sp_warn}</span>"
                f" | lot\u00d7{_sp_lot:.1f} SL\u00d7{_sp_sl:.1f}",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        # ── ICT Kill Zone indicator ─────────────────────────────────────────────
        if _INDICATORS_OK:
            try:
                _kz_sb = _get_killzones()
                if _kz_sb.get("in_killzone"):
                    _kz_zones = ", ".join(_kz_sb.get("active_zones", []))
                    _kz_col   = "#1D9E75" if _kz_sb.get("high_quality") else "#F4C542"
                    st.markdown(
                        f"<span style='color:{_kz_col}'>🎯 Kill Zone: {_kz_zones}</span>",
                        unsafe_allow_html=True,
                    )
                elif _kz_sb.get("next_killzone"):
                    _nk = _kz_sb["next_killzone"]
                    st.caption(f"⏰ Next KZ: {_nk[0]} in {_nk[1]}min")
            except Exception:
                pass

        # ── ML status widget ────────────────────────────────────────────────
        if _ML_OK:
            try:
                import json as _json_ml
                with open("data/ml_insights.json", encoding="utf-8") as _f_ml:
                    _ml_ins = _json_ml.load(_f_ml)
                if _ml_ins.get("available"):
                    st.markdown(
                        f"\U0001f916 ML: {_ml_ins['total_trades']} trades trained | "
                        f"WR {_ml_ins['overall_wr']}%"
                    )
            except Exception:
                st.caption("\U0001f916 ML: No data yet")

        # ── Paper trades quick-check button ──────────────────────────────────
        if _PAPER_OK:
            if st.button("\U0001f4cb Check Paper Trades",
                         key="paper_check_btn",
                         use_container_width=True):
                _price_pc = st.session_state.get("live_price", 0) or 0
                if _price_pc > 0:
                    try:
                        _closed_pc = _update_paper(_price_pc)
                        _s_pc      = _paper_summary()
                        if _s_pc.get("open", 0) > 0:
                            st.info(
                                f"Open: {_s_pc['open']} trades | "
                                f"P&L: {'+' if _s_pc['total_pnl'] > 0 else ''}"
                                f"{_s_pc['total_pnl']} pips"
                            )
                        for _t_pc in _closed_pc:
                            _e_pc = "\u2705" if _t_pc["outcome"] == "WIN" else "\u274c"
                            st.success(
                                f"{_e_pc} {_t_pc['trade_id']} CLOSED \u2014 "
                                f"{_t_pc['outcome']} "
                                f"{'+' if _t_pc['pnl_pips'] > 0 else ''}"
                                f"{_t_pc['pnl_pips']} pips"
                            )
                        st.rerun()
                    except Exception:
                        pass

        # ── DXY + Yields + Macro bias (colour-coded) ───────────────────────────
        _dxy_s  = st.session_state.get("dxy_status", "—")
        _mbias  = st.session_state.get("macro_bias", "neutral")
        _yctx   = st.session_state.get("yields_context") or {}
        _yld    = _yctx.get("current_yield")
        _yld_tr = _yctx.get("yield_trend", "sideways")
        _y_arr  = {"rising": "↑", "falling": "↓", "sideways": "→"}.get(_yld_tr, "→")
        _yld_str = f"{_yld:.2f}%" if _yld else "N/A"
        _dxy_icon = "📉" if "fall" in str(_dxy_s).lower() or "down" in str(_dxy_s).lower() else (
                    "📈" if "ris" in str(_dxy_s).lower() or "up" in str(_dxy_s).lower() else "➡️")
        _macro_colour = {
            "strongly_bullish": "#1D9E75",
            "bullish":          "#2ecc71",
            "neutral":          "#888888",
            "bearish":          "#E08020",
            "strongly_bearish": "#E05555",
        }.get(_mbias, "#888888")
        _mbias_label = _mbias.replace("_", " ").title()
        st.markdown(
            f"{_dxy_icon} **DXY:** {str(_dxy_s).capitalize()}  |  "
            f"🏦 **US10Y:** {_yld_str} {_y_arr}",
            unsafe_allow_html=False,
        )
        st.markdown(
            f"**Macro:** <span style='color:{_macro_colour};font-weight:700'>{_mbias_label}</span>",
            unsafe_allow_html=True,
        )

        # Fundamental bias line
        if _FB_OK:
            try:
                _fb_s  = _get_fundamental_bias()
                _fb_b  = _fb_s.get("fundamental_bias", "NEUTRAL")
                _fb_sc = _fb_s.get("total_score", 0)
                _fb_colour = {
                    "strongly_bullish": "#1D9E75",
                    "bullish":          "#2ecc71",
                    "neutral":          "#888888",
                    "bearish":          "#F4C542",
                    "strongly_bearish": "#E05555",
                }.get(_fb_b.lower(), "#888888")
                _fb_label = _fb_b.replace("_", " ").title()
                st.markdown(
                    f"📊 **Fundamental:** <span style='color:{_fb_colour};font-weight:700'>{_fb_label} ({_fb_sc:+d})</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        # Geo risk line
        _geo_level  = st.session_state.get("geo_risk_level", "normal")
        _geo_score  = (st.session_state.get("geo_ctx") or {}).get("geo_score", 0)
        _geo_colour = {"extreme": "#E05555", "high": "#E08020", "elevated": "#F4C542"}.get(_geo_level, "#2ecc71")
        st.markdown(
            f"🌍 **Geo Risk:** <span style='color:{_geo_colour}'>{_geo_level.title()} ({_geo_score:.0f}/10)</span>",
            unsafe_allow_html=True,
        )

        # Regime line
        try:
            _df_sb = st.session_state.get("live_df") or st.session_state.get("df")
            if _df_sb is not None:
                from market_context import detect_gold_regime as _dgr_sb
                _reg_sb   = _dgr_sb(_df_sb)
                _rname_sb = _reg_sb.get("regime", "")
                _rlbl_sb  = _reg_sb.get("regime_label", _rname_sb)
                _rmult_sb = _reg_sb.get("position_size_multiplier", 1.0)
                _rcolour_sb = {
                    "TRENDING_STRONG":    "#2ecc71",
                    "TRENDING_WEAK":      "#a8e6a3",
                    "RANGING":            "#5b9bd5",
                    "VOLATILE_EXPANDING": "#E05555",
                    "SQUEEZE_BUILDING":   "#888888",
                }.get(_rname_sb, "#888888")
                st.markdown(
                    f"📈 **Regime:** <span style='color:{_rcolour_sb};font-weight:700'>"
                    f"{_rlbl_sb} (×{_rmult_sb:.1f})</span>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # COT bias line
        if _COT_OK:
            try:
                _cot_sb   = _fetch_cot()
                _cot_bias = _cot_sb.get("bias", "NEUTRAL").replace("_", " ")
                _cot_pct  = _cot_sb.get("spec_net_pct", 0.0)
                _cot_clr  = {
                    "STRONGLY BULLISH": "#1D9E75",
                    "BULLISH":          "#2ecc71",
                    "NEUTRAL":          "#888888",
                    "BEARISH":          "#F4C542",
                    "STRONGLY BEARISH": "#E05555",
                }.get(_cot_bias, "#888888")
                st.markdown(
                    f"📋 **COT:** <span style='color:{_cot_clr};font-weight:700'>"
                    f"{_cot_bias} ({_cot_pct:+.1f}%)</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        # Liquidity map line
        if _LIQ_OK:
            try:
                _liq_df_sb = st.session_state.get("live_df") or st.session_state.get("df")
                if _liq_df_sb is not None and len(_liq_df_sb) >= 20:
                    _liq_p_sb  = float(_liq_df_sb["close"].iloc[-1])
                    _liq_sb    = _build_liq_map(_liq_df_sb, _liq_p_sb)
                    if _liq_sb.get("available"):
                        _liq_mv  = _liq_sb.get("likely_move", "NEUTRAL")
                        _liq_arr = "⬆" if _liq_mv == "UP" else ("⬇" if _liq_mv == "DOWN" else "↔")
                        _liq_ca  = _liq_sb.get("clusters_above", [])
                        _liq_cb  = _liq_sb.get("clusters_below", [])
                        _liq_det = ""
                        if _liq_ca:
                            _liq_det += f"BSL ${_liq_ca[0]['price']:,.0f}"
                        if _liq_cb:
                            _liq_det += (" | " if _liq_det else "") + f"SSL ${_liq_cb[0]['price']:,.0f}"
                        _liq_clr = "#2ecc71" if _liq_mv == "UP" else ("#E05555" if _liq_mv == "DOWN" else "#888888")
                        st.markdown(
                            f"💧 **Liquidity:** <span style='color:{_liq_clr};font-weight:700'>"
                            f"{_liq_arr} {_liq_mv}</span>"
                            + (f"  <span style='color:#aaa;font-size:0.85em'>({_liq_det})</span>" if _liq_det else ""),
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass

        # NY session bias from session handoff
        if _SH_OK:
            try:
                _ny_sb = st.session_state.get("ny_bias") or {}
                _ny_bias_val  = _ny_sb.get("ny_bias", "")
                _ny_conf_val  = _ny_sb.get("confidence", "")
                _ny_fake_val  = _ny_sb.get("fake_break_alert", False)
                if _ny_bias_val and _ny_bias_val != "NEUTRAL":
                    _ny_colour = "#1D9E75" if _ny_bias_val == "BULLISH" else "#E05555"
                    st.markdown(
                        f"<span style='color:{_ny_colour}'>📊 NY Bias: "
                        f"**{_ny_bias_val}** ({_ny_conf_val})</span>",
                        unsafe_allow_html=True,
                    )
                    if _ny_fake_val:
                        st.warning("⚠ Fake break detected!")
            except Exception:
                pass

        st.markdown(f"{_b_icon(d1_b)} **D1 Bias:** {d1_b}")
        st.markdown(f"{_b_icon(h4_b)} **H4 Bias:** {h4_b}")

        st.divider()

        # ── Paper Trading widget ───────────────────────────────────────────────
        if _PAPER_OK:
            st.markdown("**\U0001f4cb Paper Trading**")
            try:
                _pt_s     = _paper_summary()
                _pt_price = float(st.session_state.get("live_price", 0) or 0)

                if _pt_s["open"] > 0:
                    for _pt_t in _pt_s["open_trades"]:
                        _pt_d  = _pt_t["direction"]
                        _pt_fp = (
                            round((_pt_t["entry"] - _pt_price) / 0.1, 1)
                            if _pt_d == "short"
                            else round((_pt_price - _pt_t["entry"]) / 0.1, 1)
                        )
                        _pt_col  = "#1D9E75" if _pt_fp > 0 else "#E05555"
                        _pt_icon = "\U0001f4c8" if _pt_fp > 0 else "\U0001f4c9"
                        _pt_sign = "+" if _pt_fp > 0 else ""
                        st.markdown(
                            f"<span style='color:{_pt_col}'>"
                            f"{_pt_icon} {_pt_t['trade_id']} {_pt_d.upper()} "
                            f"{_pt_sign}{_pt_fp} pips"
                            f"</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No open paper trades")

                if _pt_s["closed"] > 0:
                    _pt_wr  = _pt_s["win_rate"]
                    _pt_pnl = _pt_s["total_pnl"]
                    _pt_col = "#1D9E75" if _pt_pnl > 0 else "#E05555"
                    _pt_psign = "+" if _pt_pnl > 0 else ""
                    st.markdown(
                        f"<span style='color:{_pt_col}'>"
                        f"WR: {_pt_wr}% | {_pt_psign}{_pt_pnl} pips total"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.caption("Paper trader loading...")
            st.divider()

        # Account settings display (loaded from data/user_settings.json)
        _s = _load_settings()
        _bal  = float(_s.get("balance",   1000))
        _rpct = float(_s.get("risk_pct",  2.0))
        _rusd = _bal * _rpct / 100
        _rr   = float(_s.get("min_rr",    3.0))
        st.markdown("**⚙️ Risk Settings** *(from user_settings.json)*")
        st.markdown(f"💰 **Balance:** ${_bal:,.0f}")
        st.markdown(f"🎯 **Risk/trade:** {_rpct:.0f}% = ${_rusd:.0f}")
        st.markdown(f"⚖️ **Min R:R:** 1:{_rr:.0f}")
        # Keep session_state in sync so callers see current balance
        st.session_state["account_balance"] = _bal

        st.markdown("**⚙️ Risk Config by Instrument**")
        for _irc_name, _irc in sorted(
                INSTRUMENT_RISK_CONFIG.items(),
                key=lambda x: x[1]["priority"]):
            _irc_sl_usd = _bal * _irc["sl_pct"] / 100
            _irc_tp_usd = _bal * _irc["tp_pct"] / 100
            _irc_grade  = _irc["grade"]
            _irc_lev    = _irc["leverage"]
            _grade_icon = "🏅" if _irc_grade == "A" else ("🥈" if _irc_grade == "B" else "🥉")
            st.markdown(
                f"{_grade_icon} **{_irc_name}** [Grade {_irc_grade}]  \n"
                f"    SL: {_irc['sl_pct']}% = ${_irc_sl_usd:.0f}  |  "
                f"TP: {_irc['tp_pct']}% = ${_irc_tp_usd:.0f}  |  "
                f"{_irc_lev}x leverage")

        # ── Spread monitor ─────────────────────────────────────────────
        try:
            _sp = _check_spread_live("XAUUSD")
            _sp_usd    = _sp.get("spread_usd")
            _sp_status = _sp.get("status", "unavailable")
            _sp_label  = f"${_sp_usd:.2f}" if _sp_usd is not None else "N/A"
            if _sp_status == "acceptable":
                st.markdown(
                    f"<span style='color:#1D9E75'>Spread: {_sp_label} ✓</span>",
                    unsafe_allow_html=True,
                )
            elif _sp_status == "warning":
                st.markdown(
                    f"<span style='color:#F4C542'>Spread: {_sp_label} ⚠</span>",
                    unsafe_allow_html=True,
                )
            elif _sp_status == "blocked":
                st.markdown(
                    f"<span style='color:#E05555'>Spread: {_sp_label} ⛔ {_sp.get('reason','')}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Spread: unavailable")
        except Exception:
            st.caption("Spread: unavailable")
        st.divider()
        # ── Brain 1 sidebar ───────────────────────────────────────────────────
        try:
            _mem      = _load_json(os.path.join(DATA_DIR, "pattern_memory.json"), [])
            _afs      = _load_auto_filters()
            _real_n   = len(_mem)
            _real_w   = sum(1 for m in _mem if str(m.get("outcome","")).lower() == "win")
            _real_l   = sum(1 for m in _mem if str(m.get("outcome","")).lower() == "loss")
            _real_tot = _real_w + _real_l
            _real_wr  = round(_real_w / _real_tot * 100, 1) if _real_tot else 0.0
            _upd      = _load_json(
                os.path.join(DATA_DIR, "logs", "failed_trade_analysis.json"), []
            )
            _last_upd = ""
            if _upd:
                _last_upd_str = str(_upd[-1].get("saved_at", ""))
                try:
                    _lu = datetime.fromisoformat(_last_upd_str)
                    _last_upd = _lu.strftime("%H:%M %d %b")
                except Exception:
                    _last_upd = _last_upd_str[:16]
            st.markdown("**🧠 Brain 1 — Real Trades**")
            st.markdown(
                f"Trades tracked: **{_real_n}**  \n"
                f"Win rate: **{_real_wr}%** ({_real_w}W / {_real_l}L)  \n"
                f"Auto-filters: **{len(_afs)}** active"
                + (f"  \nLast update: {_last_upd}" if _last_upd else "")
            )
        except Exception:
            st.caption("🧠 Brain 1: loading...")

        # ── Brain 2 sidebar ───────────────────────────────────────────────────
        if _ST_OK:
            try:
                _srep    = _st_report()
                _sov     = _srep.get("overall", {})
                _suc     = _srep.get("user_comparison", {})
                _sn      = _sov.get("total_signals", 0)
                _swr     = _sov.get("win_rate", 0.0)
                _uwr     = _suc.get("user_win_rate", 0.0)
                _utook   = _suc.get("signals_user_took", 0)
                _verdict = _srep.get("verdict", "")
                # one-line verdict stripped of emoji prefix
                _vline   = _verdict[2:].strip() if _verdict and _verdict[0] in ("✅", "⚠", "➡", "⚪") else _verdict
                st.markdown("**📡 Brain 2 — Signal Accuracy**")
                st.markdown(
                    f"Signals tracked: **{_sn}**  \n"
                    f"Bot win rate: **{_swr}%**  \n"
                    f"Your win rate: **{_uwr}%** ({_utook} trades)"
                )
                if _vline:
                    st.caption(_vline)
            except Exception:
                st.caption("📡 Brain 2: loading...")

        st.divider()

        # Quick action buttons — 2 columns
        st.markdown("**Quick Actions**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Run Setup",     key="sb_setup"):
                st.session_state["trigger_cmd"] = "setup"
        with c2:
            if st.button("📊 Gold Analysis", key="sb_gold"):
                st.session_state["trigger_cmd"] = "gold"
        with c1:
            if st.button("📡 Show Signals",  key="sb_signals"):
                st.session_state["trigger_cmd"] = "signals"
        with c2:
            if st.button("📰 News Calendar", key="sb_news"):
                st.session_state["trigger_cmd"] = "news"
        with c1:
            if st.button("💰 Risk Guide",    key="sb_risk"):
                st.session_state["trigger_cmd"] = "risk"
        with c2:
            if st.button("🧪 Backtest",      key="sb_backtest"):
                st.session_state["trigger_cmd"] = "backtest"

        st.markdown("")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("🔬 Analyze Losses", key="sb_losses", use_container_width=True):
                st.session_state["trigger_cmd"] = "analyze losses"
        with c4:
            if st.button("📅 Weekly Review",  key="sb_weekly", use_container_width=True):
                st.session_state["trigger_cmd"] = "weekly review"

        st.markdown("")
        c5, c6 = st.columns(2)
        with c5:
            if st.button("🧠 Brain 1", key="sb_brain1", use_container_width=True):
                st.session_state["trigger_cmd"] = "learning report"
        with c6:
            if st.button("📡 Brain 2", key="sb_brain2", use_container_width=True):
                st.session_state["trigger_cmd"] = "signal performance"
        _bf_col, = st.columns(1)
        with _bf_col:
            if st.button("🧠📡 Full Brain Report", key="sb_brain_full", use_container_width=True):
                st.session_state["trigger_cmd"] = "full brain report"

        st.divider()
        st.markdown("**Debug & Export**")
        if st.button("📋 Export Session Logs", key="sb_export", use_container_width=True):
            st.session_state["trigger_cmd"] = "export logs"
        st.caption("Exports full session log to data/logs/ for Claude analysis.")

        st.divider()
        st.caption("TradingBotV1 · May 2026")
        st.caption("Analyst only — trade manually on MT5")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import time as _time

    # ── 1. Session state ─────────────────────────────────────────────────────
    _init_state()

    # ── 2. Data file bootstrap (creates any missing files/folders) ───────────
    _bootstrap_data_files()

    # ── 3. Lazy module imports (cached — only runs once) ─────────────────────
    # MODS is already loaded at module level via _load_modules(); no-op here.

    # ── 4. 60-second auto-refresh for MT5 data (with graceful fallback) ──────
    now_ts = _time.time()
    last_rerun = st.session_state.get("_last_auto_rerun", 0.0)
    if now_ts - last_rerun >= 60:
        st.session_state["_last_auto_rerun"] = now_ts
        try:
            _refresh_mt5_data()
        except Exception as _mt5_ex:
            st.session_state["mt5_connected"] = False
            st.session_state["mt5_error"]     = str(_mt5_ex)
        # ── Live price refresh ────────────────────────────────────────────────
        try:
            _lp_refresh = _get_live_price(st.session_state.get("instrument", "XAUUSD"))
            _new_price  = _lp_refresh.get("price") or 0
            if _new_price > 0:
                _prev_price = st.session_state.get("live_price") or 0
                st.session_state["live_price"]  = round(_new_price, 2)
                st.session_state["live_source"] = _lp_refresh.get("source", "—")
                st.session_state["price_stale"] = not _lp_refresh.get("is_live", False)
                # Notify on significant price move (≥ $1)
                if _prev_price > 0 and abs(_new_price - _prev_price) >= 1.0:
                    _dir_e = "📈" if _new_price > _prev_price else "📉"
                    _notifs = st.session_state.get("mt5_sync_notifications", [])
                    _notifs.append({
                        "symbol": "XAUUSD", "direction": "",
                        "close_price": round(_new_price, 2),
                        "outcome": f"{_dir_e} Price moved: ${_prev_price:,.2f} → ${_new_price:,.2f}",
                        "pnl": "",
                    })
                    st.session_state["mt5_sync_notifications"] = _notifs
        except Exception:
            pass
        # ── Reversal Hunter — alert on strong signals ─────────────────────────
        if _RH_OK:
            try:
                df_live = st.session_state.get("live_df")
                if df_live is not None:
                    _revs  = _hunt_reversals(df_live, st.session_state.get("live_price"))
                    _strong = [r for r in _revs if r["reversal_strength"] == "STRONG"]
                    if _strong:
                        _rnotifs = st.session_state.get("mt5_sync_notifications", [])
                        for _r in _strong:
                            _rnotifs.append(
                                f"🔄 REVERSAL: {_r['direction'].upper()} "
                                f"${_r['entry']:,.2f} | {_r['key_reason']} "
                                f"(Score {_r['score']}/11) — type 'show reversals'"
                            )
                        st.session_state["mt5_sync_notifications"] = _rnotifs
            except Exception:
                pass
        # ── Update UAE clock in session state ─────────────────────────────────
        _gst_r = timezone(timedelta(hours=4))
        st.session_state["current_uae_time"] = datetime.now(_gst_r).strftime("%I:%M %p")
        st.session_state["current_uae_date"] = datetime.now(_gst_r).strftime("%A %d %B %Y")
        # ── Save regime snapshot for history tracking ─────────────────────────
        try:
            from market_context import detect_gold_regime as _dgr, save_regime_snapshot as _srs
            _df_snap = st.session_state.get("live_df")
            if _df_snap is not None:
                _rdata = _dgr(_df_snap)
                _rprice = st.session_state.get("live_price", 0.0) or 0.0
                _srs(_rdata, price=float(_rprice))
                st.session_state["current_regime"]       = _rdata.get("regime", "RANGING")
                st.session_state["current_regime_label"] = _rdata.get("regime_label", "Ranging")
        except Exception:
            pass
        # Brain 2: update signal prices with current live price
        if _ST_OK:
            try:
                _live_p = st.session_state.get("live_price")
                if _live_p:
                    _new_outcomes = _st_update(float(_live_p))
                    if _new_outcomes:
                        existing = st.session_state.get("mt5_sync_notifications", [])
                        st.session_state["mt5_sync_notifications"] = existing + [
                            {"symbol": r.get("asset", "XAUUSD"),
                             "direction": r.get("direction", ""),
                             "close_price": r.get("last_price", 0),
                             "outcome": r.get("outcome", ""),
                             "pnl": ""}
                            for r in _new_outcomes if r.get("outcome") not in ("open", None)
                        ]
            except Exception:
                pass
        # Brain 1: track open trades
        if _MT5_SYNC_OK:
            try:
                track_open_trades()
            except Exception:
                pass
        # ── Risk of Ruin alert (every 10 minutes) ─────────────────────────────
        if _TM_OK:
            _last_ror = st.session_state.get("_last_ror_check", 0.0)
            if now_ts - _last_ror > 600:
                st.session_state["_last_ror_check"] = now_ts
                try:
                    _ror_alert = _get_ror_profile()
                    st.session_state["risk_rating"] = _ror_alert.get("risk_rating", "SAFE")
                    if _ror_alert.get("risk_rating") in ("HIGH", "DANGER"):
                        _ror_notifs = st.session_state.get("mt5_sync_notifications", [])
                        _ror_notifs.append(
                            f"⚠ RISK ALERT: {_ror_alert['risk_rating']} — "
                            f"{_ror_alert['ruin_probability']:.1f}% ruin probability. "
                            f"Type 'risk check' for details."
                        )
                        st.session_state["mt5_sync_notifications"] = _ror_notifs
                except Exception:
                    pass
        # ── Sunday Walk-Forward Optimization (auto) ───────────────────────────
        if _WFO_OK:
            try:
                if _check_wfo():
                    result = _run_wfo()
                    if result.get("changes_made"):
                        _wfo_notifs = st.session_state.get("mt5_sync_notifications", [])
                        _wfo_notifs.append(
                            f"🔄 AUTO-OPTIMIZED: {len(result['changes'])} "
                            f"settings updated based on last 30 days. "
                            f"Type 'optimization report' to see changes."
                        )
                        st.session_state["mt5_sync_notifications"] = _wfo_notifs
            except Exception:
                pass
        # ── Paper trade auto-update ───────────────────────────────────────────
        if _PAPER_OK:
            try:
                _pt_price = st.session_state.get("live_price", 0)
                if _pt_price and _pt_price > 0:
                    _pt_closed = _update_paper(float(_pt_price))
                    for _pt in _pt_closed:
                        _pt_emoji = "\u2705" if _pt["outcome"] == "WIN" else "\u274c"
                        _pt_sign  = "+" if _pt["pnl_pips"] > 0 else ""
                        _pt_notif = (
                            f"{_pt_emoji} PAPER TRADE CLOSED \u2014 {_pt['trade_id']}\n"
                            f"{_pt['direction'].upper()} {_pt['outcome']}\n"
                            f"{_pt.get('close_reason', '')} at "
                            f"${_pt.get('close_price', 0):,.2f}\n"
                            f"P&L: {_pt_sign}{_pt['pnl_pips']} pips | "
                            f"Time held: {_pt.get('time_held', '?')}\n"
                            f"{_pt.get('close_detail', '')}"
                        )
                        if "mt5_sync_notifications" not in st.session_state:
                            st.session_state["mt5_sync_notifications"] = []
                        st.session_state["mt5_sync_notifications"].append(_pt_notif)
                    # keep open_count in sync
                    st.session_state["paper_open_count"] = _paper_summary().get("open", 0)
            except Exception:
                pass
        # ── ML startup auto-train (once per 60-second cycle) ─────────────────
        if _ML_OK:
            try:
                _run_ml()  # retrain on latest paper trades
            except Exception:
                pass

    # Account balance is read from sidebar widget (or MT5 live balance)
    acct_live = (st.session_state.get("mt5_account") or {}).get("balance")
    account   = float(acct_live if acct_live else st.session_state.get("account_balance", 300.0))

    # ── 5. Sidebar ────────────────────────────────────────────────────────────

    _render_sidebar(account)
    # Re-read after sidebar renders
    account = float(st.session_state.get("account_balance", 300.0))

    # Title
    st.title("💬 TradingBotV1")
    _active_instr = st.session_state.get("instrument", "XAUUSD")
    st.caption(f"Trading {_active_instr} — signals, strategies, risk — or use the sidebar buttons.")

    # Welcome on first load — show home screen with buttons and glossary
    if not st.session_state["messages"]:
        _render_home()

    # Handle pending_cmd from home screen buttons
    if "pending_cmd" in st.session_state:
        _pcmd = st.session_state.pop("pending_cmd")
        if _pcmd:
            _add_user_msg(_pcmd)
            with st.spinner("Analyzing market conditions..."):
                _presp = _route(_pcmd, account)
            _add_bot_msg(_presp)
            st.rerun()

    # Handle sidebar button trigger (set in _render_sidebar)
    trigger = st.session_state.pop("trigger_cmd", None)
    if trigger:
        _add_user_msg(trigger)
        with st.spinner("Analyzing market conditions..."):
            resp = _route(trigger, account)
        _add_bot_msg(resp)
        st.rerun()

    # Render all chat messages
    for msg in st.session_state["messages"]:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Quick-command chips above chat input
    chip_cols = st.columns(5)
    _chip_instr = st.session_state.get("instrument", "XAUUSD")
    _analyze_cmd   = f"analyze {_chip_instr.lower()}"
    _analyze_label = f"📊 analyze {_chip_instr}"
    chips = [
        ("run setup",             "🔄 run setup"),
        (_analyze_cmd,            _analyze_label),
        ("show signals",          "📡 show signals"),
        ("news today",            "📰 news today"),
        ("how do I use this bot", "❓ how to use"),
    ]
    for col, (cmd, label) in zip(chip_cols, chips):
        with col:
            if st.button(label, key=f"chip_{cmd}", use_container_width=True):
                _add_user_msg(cmd)
                with st.spinner("Analyzing market conditions..."):
                    resp = _route(cmd, account)
                _add_bot_msg(resp)
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask me anything or type a command..."):
        _add_user_msg(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing market conditions..."):
                response = _route(prompt, account)
            st.markdown(response, unsafe_allow_html=True)
            _add_bot_msg(response)


# ── Auto-restart auto trader on cloud boot ─────────────────────────────────
try:
    from auto_trader import (
        start_auto_trader as _boot_start,
        load_state        as _boot_load_state,
    )
    _boot_state = _boot_load_state()
    if _boot_state.get("enabled", False) and not _boot_state.get("running", False):
        _boot_start()
        print("[Bot] Auto trader restarted after redeploy")
except Exception:
    pass


if __name__ == "__main__":
    main()
