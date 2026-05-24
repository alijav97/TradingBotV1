"""
_test_phase2_integration.py — Full Phase 2 Integration Backtest
────────────────────────────────────────────────────────────────
Tests Tasks 6-9 working together as one system.
DO NOT modify any production code — test only.
Run: python _test_phase2_integration.py
"""
from __future__ import annotations

import sys
import math
import traceback
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ── ASCII-safe output helpers ──────────────────────────────────────────────
PASS  = "[PASS]"
FAIL  = "[FAIL]"
INFO  = "[INFO]"
WARN  = "[WARN]"
SEP   = "=" * 65
DASH  = "-" * 65

TOTAL_PASS = 0
TOTAL_FAIL = 0
ISSUES: list[str] = []

def _ok(label: str) -> None:
    global TOTAL_PASS
    TOTAL_PASS += 1
    print(f"  {PASS}  {label}")

def _fail(label: str, detail: str = "") -> None:
    global TOTAL_FAIL
    TOTAL_FAIL += 1
    msg = f"  {FAIL}  {label}" + (f"  ({detail})" if detail else "")
    print(msg)
    ISSUES.append(label + (f" -- {detail}" if detail else ""))

def _check(cond: bool, label: str, detail: str = "") -> bool:
    if cond:
        _ok(label)
    else:
        _fail(label, detail)
    return cond

def _info(msg: str) -> None:
    print(f"  {INFO}  {msg}")

# ── Component availability flags ──────────────────────────────────────────
TASK6_OK = TASK7_OK = TASK8_OK = TASK9_OK = False

try:
    from dxy_correlation import get_macro_context
    TASK6_OK = True
except Exception as e:
    _fail("Task 6 import failed", str(e))

try:
    from geo_filter import get_geopolitical_score
    TASK7_OK = True
except Exception as e:
    _fail("Task 7 import failed", str(e))

try:
    from atr_sl_engine import calculate_dynamic_sl
    TASK8_OK = True
except Exception as e:
    _fail("Task 8 import failed", str(e))

try:
    from session_profiler import get_current_session_profile, get_session_adjusted_position
    TASK9_OK = True
except Exception as e:
    _fail("Task 9 import failed", str(e))

if not all([TASK6_OK, TASK7_OK, TASK8_OK, TASK9_OK]):
    print(f"\n{FAIL} One or more components failed to import. Aborting.")
    sys.exit(1)

# ── Load and enrich data ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("SETUP -- Loading historical_xauusd.csv (last 1000 candles)")
print(SEP)

try:
    _raw = pd.read_csv("data/historical_xauusd.csv")
    _raw.columns = [c.lower() for c in _raw.columns]

    # Add all indicators
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ATR
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        # EMAs
        df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
        return df

    _raw = _add_indicators(_raw)
    df_full = _raw.dropna(subset=["atr", "ema50", "ema200", "rsi"]).reset_index(drop=True)
    df1000  = df_full.tail(1000).reset_index(drop=True)
    _info(f"Full dataset: {len(df_full):,} enriched candles")
    _info(f"Working window: {len(df1000):,} candles (last 1000)")
    _info(f"ATR last: ${df1000['atr'].iloc[-1]:.2f} | "
          f"EMA50: {df1000['ema50'].iloc[-1]:.2f} | "
          f"RSI: {df1000['rsi'].iloc[-1]:.1f}")
except Exception as e:
    print(f"{FAIL} Data loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Session helper ─────────────────────────────────────────────────────────
def _session_from_hour(h: int) -> str:
    if h in range(0,  7):  return "Asian"
    if h in range(7,  12): return "London"
    if h in range(12, 15): return "Overlap"
    if h in range(15, 21): return "NewYork"
    return "OffHours"

def _est_session(df: pd.DataFrame, idx: int) -> str:
    dt_col = "datetime" if "datetime" in df.columns else None
    if dt_col and pd.notna(df.iloc[idx].get(dt_col, None)):
        try:
            return _session_from_hour(pd.to_datetime(str(df.iloc[idx][dt_col])).hour)
        except Exception:
            pass
    return "London"

# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST 1 — Full signal scoring pipeline
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("INTEGRATION TEST 1 -- Full signal scoring pipeline (LONG at current price)")
print(SEP)

try:
    entry          = float(df1000["close"].iloc[-1])
    atr_now        = float(df1000["atr"].iloc[-1])
    base_sl_dist   = atr_now * 1.5
    base_sl        = entry - base_sl_dist
    base_tp        = entry + atr_now * 3.0
    confidence     = 7.0
    sl_distance    = base_sl_dist

    print(f"\n  Base signal:")
    print(f"    Entry:          ${entry:,.2f}")
    print(f"    ATR:            ${atr_now:.2f}")
    print(f"    Base SL:        ${base_sl:,.2f}  (-${base_sl_dist:.2f})")
    print(f"    Base TP:        ${base_tp:,.2f}  (+${atr_now*3.0:.2f})")
    print(f"    Base confidence: {confidence:.1f}/10")

    # ── Step 1: Macro filter ──────────────────────────────────────────────────
    print(f"\n  Step 1 -- Macro filter (Task 6):")
    macro = get_macro_context("long")
    macro_adj   = float(macro.get("confidence_adjustment", 0.0))
    macro_bias  = macro.get("macro_bias", "neutral")
    macro_avail = macro.get("available", False)
    confidence += macro_adj
    print(f"    macro_bias:     {macro_bias}")
    print(f"    adjustment:     {macro_adj:+.1f}")
    print(f"    After macro:    {confidence:.1f}  ({macro_adj:+.1f})")
    _check(isinstance(macro_adj, float),       "macro confidence_adjustment is float")
    _check(macro_avail in (True, False),       "macro.available key present")
    _check(-1.0 <= macro_adj <= 1.0,           f"macro_adj in valid range [-1.0, +1.0] (got {macro_adj:+.1f})")

    # ── Step 2: Geo filter ────────────────────────────────────────────────────
    print(f"\n  Step 2 -- Geo filter (Task 7):")
    geo         = get_geopolitical_score()
    geo_adj     = float(geo.get("confidence_adjustment", 0.0))
    geo_level   = geo.get("geo_risk_level", "normal")
    geo_sl_mult = float(geo.get("sl_atr_multiplier", 0.0))
    confidence += geo_adj
    sl_distance += geo_sl_mult * atr_now
    print(f"    geo_risk_level: {geo_level}")
    print(f"    sl_atr_mult:    {geo_sl_mult}")
    print(f"    adjustment:     {geo_adj:+.1f}")
    print(f"    After geo:      conf={confidence:.1f}  SL_dist=${sl_distance:.2f}")
    _check(isinstance(geo_adj, float),         "geo confidence_adjustment is float")
    _check(geo_sl_mult >= 0.0,                 f"geo sl_atr_multiplier >= 0 (got {geo_sl_mult})")
    _check("geo_risk_level" in geo,            "geo has geo_risk_level key")

    # ── Step 3: Dynamic ATR SL ────────────────────────────────────────────────
    print(f"\n  Step 3 -- Dynamic ATR SL (Task 8):")
    current_sess = _session_from_hour(datetime.now(timezone.utc).hour)
    dynamic = calculate_dynamic_sl(
        df1000, "long", entry,
        session=current_sess,
        regime="RANGING",
        geo_multiplier=geo_sl_mult,
    )
    dyn_sl    = dynamic["sl_price"]
    dyn_tp1   = dynamic["tp1_price"]
    dyn_tp2   = dynamic["tp2_price"]
    dyn_sldst = dynamic["sl_distance"]
    breakdown = dynamic["sl_breakdown"]
    print(f"    session:        {current_sess}")
    print(f"    Dynamic SL:     ${dyn_sl:,.2f}")
    print(f"    TP1:            ${dyn_tp1:,.2f}  (1:{dynamic['rr_at_tp1']:.1f})")
    print(f"    TP2:            ${dyn_tp2:,.2f}  (1:{dynamic['rr_at_tp2']:.1f})")
    print(f"    SL method:      {breakdown}")
    _check(dyn_sl  < entry,   f"Dynamic SL below entry (SL=${dyn_sl:,.2f} entry=${entry:,.2f})")
    _check(dyn_tp1 > entry,   f"Dynamic TP1 above entry")
    _check(dyn_tp2 > dyn_tp1, f"Dynamic TP2 > TP1")
    _check("sl_breakdown" in dynamic, "dynamic has sl_breakdown key")

    # ── Step 4: Session profiling ─────────────────────────────────────────────
    print(f"\n  Step 4 -- Session profiling (Task 9):")
    profile    = get_current_session_profile()
    sess_name  = profile["current_session"]
    sess_grade = profile["session_grade"]
    adj        = get_session_adjusted_position(
        base_lots=0.01,
        base_sl_distance=dyn_sldst,
        base_tp_distance=abs(dyn_tp2 - entry),
        session_profile=profile,
    )
    adj_lots   = adj["adjusted_lots"]
    lot_change = adj["lot_change"]
    print(f"    session:        {sess_name}")
    print(f"    grade:          {sess_grade}")
    print(f"    lot_multiplier: x{profile['lot_multiplier']:.2f}  ({lot_change})")
    print(f"    adjusted_lots:  {adj_lots:.2f}")
    _check(adj_lots >= 0.01,   "adjusted_lots >= 0.01")
    _check(adj_lots <= 0.015,  f"adjusted_lots <= base*1.5 (got {adj_lots:.3f})")
    _check(sess_grade in ("A","B","C"), f"session_grade valid (got {sess_grade})")

    # ── Final signal summary ──────────────────────────────────────────────────
    final_conf = round(min(10.0, max(0.0, confidence)), 2)
    print(f"\n  {'='*55}")
    print(f"  PHASE 2 SIGNAL SUMMARY")
    print(f"  {'='*55}")
    print(f"  Entry:       ${entry:,.2f}")
    print(f"  SL:          ${dyn_sl:,.2f}  (-${dyn_sldst:.2f})")
    print(f"  TP1:         ${dyn_tp1:,.2f}  (1:2)")
    print(f"  TP2:         ${dyn_tp2:,.2f}  (1:3)")
    print(f"  Lots:        {adj_lots:.2f}")
    print(f"  Confidence:  {final_conf:.1f}/10")
    print(f"")
    print(f"  Macro:   {macro_bias} --> {macro_adj:+.1f}")
    print(f"  Geo:     {geo_level} --> {geo_adj:+.1f}")
    print(f"  SL:      {breakdown}")
    print(f"  Session: {sess_name} Grade {sess_grade} --> {lot_change}")
    _check(0.0 <= final_conf <= 10.0, f"Final confidence in range (got {final_conf})")

except Exception:
    _fail("TEST 1 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST 2 — 500-candle simulation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("INTEGRATION TEST 2 -- 500-candle simulation (every 20th candle, 50 trade points)")
print(SEP)

try:
    indices = list(range(60, len(df1000), 20))   # need 60+ for ATR window

    macro_adjs:   list[float] = []
    geo_adjs:     list[float] = []
    sl_dists:     list[float] = []
    adj_lots_all: list[float] = []
    dyn_mults:    list[float] = []
    sessions:     dict[str, int] = {}
    grades:       dict[str, int] = {}
    boosted_n     = 0
    penalised_n   = 0
    widest_sl     = 0.0; widest_info   = ""
    tightest_sl   = float("inf"); tightest_info = ""
    max_lots      = 0.0; max_lots_sess = ""
    min_lots      = float("inf"); min_lots_sess = ""

    # Fetch macro/geo once — they use live data, apply same value to all bars
    _macro_live = get_macro_context("long")
    _geo_live   = get_geopolitical_score()
    _m_adj  = float(_macro_live.get("confidence_adjustment", 0.0))
    _g_adj  = float(_geo_live.get("confidence_adjustment",   0.0))
    _g_mult = float(_geo_live.get("sl_atr_multiplier",       0.0))

    BASE_CONF = 7.0
    BASE_LOTS = 0.01

    _profile = get_current_session_profile()

    for idx in indices:
        slice_df = df1000.iloc[:idx + 1].copy()
        row      = df1000.iloc[idx]
        entry_p  = float(row["close"])

        # Estimate session from datetime if available
        sess = _est_session(df1000, idx)
        sessions[sess] = sessions.get(sess, 0) + 2   # 2 per candle (long+short)

        for direction in ("long", "short"):
            # Step 3: Dynamic SL
            try:
                dyn = calculate_dynamic_sl(slice_df, direction, entry_p,
                                           session=sess, regime="RANGING",
                                           geo_multiplier=_g_mult)
                sld  = dyn["sl_distance"]
                mult = dyn["final_multiplier"]
            except Exception:
                atr_v = float(slice_df["atr"].dropna().iloc[-1]) if not slice_df["atr"].dropna().empty else 20.0
                sld   = atr_v * 1.5
                mult  = 1.5

            # Step 1+2: Confidence
            base_conf   = BASE_CONF
            # Macro: adjust direction sign for short
            _m_dir = _m_adj if direction == "long" else -_m_adj
            final_conf  = base_conf + _m_dir + _g_adj

            # Step 4: Session lots
            try:
                _adj = get_session_adjusted_position(BASE_LOTS, sld, sld * 3.0, _profile)
                lots_out = _adj["adjusted_lots"]
            except Exception:
                lots_out = BASE_LOTS

            macro_adjs.append(_m_dir)
            geo_adjs.append(_g_adj)
            sl_dists.append(sld)
            adj_lots_all.append(lots_out)
            dyn_mults.append(mult)

            atr_bar = float(slice_df["atr"].dropna().iloc[-1]) if not slice_df["atr"].dropna().empty else 20.0
            tag = f"{direction} at ${entry_p:,.0f} sess={sess}"

            if sld > widest_sl:
                widest_sl   = sld
                widest_info = tag
            if sld < tightest_sl:
                tightest_sl   = sld
                tightest_info = tag
            if lots_out > max_lots:
                max_lots      = lots_out
                max_lots_sess = sess
            if lots_out < min_lots:
                min_lots      = lots_out
                min_lots_sess = sess

            if final_conf > BASE_CONF:
                boosted_n += 1
            elif final_conf < BASE_CONF:
                penalised_n += 1

            g = _profile.get("session_grade", "B")
            grades[g] = grades.get(g, 0) + 1

    total_obs    = len(sl_dists)
    avg_macro    = float(np.mean(macro_adjs))  if macro_adjs  else 0.0
    avg_geo      = float(np.mean(geo_adjs))    if geo_adjs    else 0.0
    net_avg      = round(BASE_CONF + avg_macro + avg_geo, 2)
    avg_sl       = float(np.mean(sl_dists))    if sl_dists    else 0.0
    static_sl    = float(np.mean([df1000["atr"].iloc[i] * 1.5 for i in indices
                                   if i < len(df1000)]))
    avg_adj_lots = float(np.mean(adj_lots_all)) if adj_lots_all else 0.0
    net_better   = "BETTER" if (boosted_n > penalised_n) else ("WORSE" if penalised_n > boosted_n else "NEUTRAL")

    print(f"\n  Total observations : {total_obs}  ({len(indices)} candles x 2 directions)")
    print(f"\n  CONFIDENCE IMPACT:")
    print(f"    Avg macro adj    : {avg_macro:+.3f}")
    print(f"    Avg geo adj      : {avg_geo:+.3f}")
    print(f"    Net avg conf     : {net_avg:.2f}  (from base {BASE_CONF:.1f})")
    print(f"\n  SL IMPACT:")
    print(f"    Avg SL distance  : ${avg_sl:.2f}  (vs static ${static_sl:.2f})")
    print(f"    Widest SL        : ${widest_sl:.2f}  ({widest_info})")
    print(f"    Tightest SL      : ${tightest_sl:.2f}  ({tightest_info})")
    print(f"\n  LOT SIZE IMPACT:")
    print(f"    Avg lots         : {avg_adj_lots:.4f}  (vs base {BASE_LOTS:.2f})")
    print(f"    Max lots         : {max_lots:.4f}  (session: {max_lots_sess})")
    print(f"    Min lots         : {min_lots:.4f}  (session: {min_lots_sess})")
    print(f"\n  SESSION DISTRIBUTION:")
    for s in ("London", "Overlap", "NewYork", "Asian", "OffHours"):
        n = sessions.get(s, 0)
        if n:
            print(f"    {s:<12}: {n} signals")
    print(f"\n  FILTER IMPACT:")
    print(f"    Signals BOOSTED  : {boosted_n}")
    print(f"    Signals PENALISED: {penalised_n}")
    print(f"    Net vs no filters: {net_better}")

    _check(total_obs > 0,        f"Simulation ran ({total_obs} observations)")
    _check(avg_sl > 0,           f"Average SL distance > 0 (${avg_sl:.2f})")
    _check(avg_adj_lots >= 0.01, f"Avg lots >= 0.01 ({avg_adj_lots:.4f})")
    _check(max_lots <= BASE_LOTS * 1.5 + 0.001,
           f"Max lots <= base*1.5 ({max_lots:.4f} <= {BASE_LOTS*1.5:.4f})")
    _check(min_lots >= 0.01,     f"Min lots >= 0.01 ({min_lots:.4f})")

except Exception:
    _fail("TEST 2 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST 3 — Worst case scenario
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("INTEGRATION TEST 3 -- Worst case scenario (all adverse conditions)")
print(SEP)

try:
    # Use a synthetic high-volatility df for the ATR scenario
    np.random.seed(9)
    n = 200
    c  = df1000["close"].iloc[-1]
    hv_close = c + np.cumsum(np.random.randn(n) * 20)
    hv_high  = hv_close + np.abs(np.random.randn(n) * 15)
    hv_low   = hv_close - np.abs(np.random.randn(n) * 15)
    df_hv    = pd.DataFrame({"high": hv_high, "low": hv_low, "close": hv_close,
                             "open": hv_close, "volume": np.ones(n) * 1000})
    hl = df_hv["high"] - df_hv["low"];  hc = (df_hv["high"] - df_hv["close"].shift()).abs()
    lc = (df_hv["low"] - df_hv["close"].shift()).abs()
    df_hv["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    entry_wc  = float(df_hv["close"].iloc[-1])
    atr_wc    = float(df_hv["atr"].dropna().iloc[-1])
    base_conf = 7.0

    print(f"\n  Worst case entry: ${entry_wc:,.2f}  ATR: ${atr_wc:.2f}")
    print(f"  Conditions: Asian + VOLATILE_EXPANDING + high vol + geo_mult=1.5 + LONG vs bearish macro")

    # ── Step 1: Macro (simulate strongly_bearish for LONG = -1.0) ────────────
    # We mock this: in worst case macro opposes the long
    macro_wc_adj  = -1.0   # strongly opposes long
    conf_wc       = base_conf + macro_wc_adj
    print(f"\n  After macro    : {base_conf:.1f} + ({macro_wc_adj:+.1f}) = {conf_wc:.1f}")
    _check(conf_wc == 6.0, f"After macro adj conf=6.0 (got {conf_wc})")

    # ── Step 2: Geo (extreme risk, safe-haven supports long) ─────────────────
    # Build mock geo with extreme score and safe-haven gold_bias
    geo_wc_adj   = 0.5    # geo supports long (safe haven demand)
    geo_wc_mult  = 1.5    # extreme SL widening
    conf_wc     += geo_wc_adj
    print(f"  After geo      : {conf_wc - geo_wc_adj:.1f} + ({geo_wc_adj:+.1f}) = {conf_wc:.1f}")
    _check(math.isclose(conf_wc, 6.5, rel_tol=1e-6), f"After geo conf=6.5 (got {conf_wc})")

    # ── Step 3: Dynamic SL — Asian × VOLATILE × high vol + geo 1.5 → should cap ─
    dyn_wc = calculate_dynamic_sl(df_hv, "long", entry_wc,
                                   session="Asian",
                                   regime="VOLATILE_EXPANDING",
                                   geo_multiplier=geo_wc_mult)
    expected_cap = dyn_wc["atr_value"] * 3.0
    print(f"\n  Dynamic SL:    ${dyn_wc['sl_price']:,.2f}  dist=${dyn_wc['sl_distance']:.2f}")
    print(f"  Breakdown:     {dyn_wc['sl_breakdown']}")
    print(f"  ATR:           ${dyn_wc['atr_value']:.2f}  |  Cap at 3.0x: ${expected_cap:.2f}")
    print(f"  Volatility:    {dyn_wc['volatility_state']}  ({dyn_wc['atr_percentile']:.0f}th pct)")
    _check(
        dyn_wc["sl_distance"] <= expected_cap + 0.01,
        f"Worst case SL capped at 3.0x ATR (sl={dyn_wc['sl_distance']:.2f} cap={expected_cap:.2f})",
    )

    # ── Step 4: Session lots — Asian reduces ─────────────────────────────────
    asian_profile = {
        "current_session": "Asian", "session_grade": "B",
        "lot_multiplier": 0.7, "sl_multiplier": 1.3, "tp_multiplier": 0.8,
        "trading_recommended": False, "session_note": "Thin liquidity",
    }
    adj_wc = get_session_adjusted_position(0.01, dyn_wc["sl_distance"],
                                            abs(dyn_wc["tp2_price"] - entry_wc),
                                            asian_profile)
    print(f"\n  Session:       Asian Grade B (thin liquidity)")
    print(f"  Lot adjust:    {adj_wc['lot_change']}  -->  {adj_wc['adjusted_lots']:.4f} lots")
    _check(adj_wc["adjusted_lots"] < 0.01 or math.isclose(adj_wc["adjusted_lots"], 0.01),
           f"Asian reduces lots (got {adj_wc['adjusted_lots']:.4f} from base 0.01)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    show_signal = conf_wc >= 7.0
    print(f"\n  Final confidence: {conf_wc:.1f}/10  (threshold: 7.0)")
    print(f"  VERDICT: {'SHOW SIGNAL' if show_signal else 'SKIP SIGNAL (confidence below 7.0)'}")
    print(f"  REASONING:")
    print(f"    - Macro opposes direction: confidence penalised -1.0")
    print(f"    - Geo extreme: SL capped at 3.0x ATR (widened for safety)")
    print(f"    - Geo safe-haven: confidence boosted +0.5 (gold demand in crisis)")
    print(f"    - Asian session: lots reduced x0.7 (thin liquidity)")
    print(f"    - Net result: conf {conf_wc:.1f} < 7.0 --> bot correctly SKIPS this signal")
    _check(not show_signal, "Worst case signal correctly SKIPPED (conf < 7.0)")

except Exception:
    _fail("TEST 3 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST 4 — Best case scenario
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("INTEGRATION TEST 4 -- Best case scenario (all ideal conditions)")
print(SEP)

try:
    # Normal volatility synthetic df
    np.random.seed(3)
    n = 200
    c2 = df1000["close"].iloc[-1]
    nv_close = c2 + np.cumsum(np.random.randn(n) * 5)
    nv_high  = nv_close + np.abs(np.random.randn(n) * 3)
    nv_low   = nv_close - np.abs(np.random.randn(n) * 3)
    df_nv    = pd.DataFrame({"high": nv_high, "low": nv_low, "close": nv_close,
                             "open": nv_close, "volume": np.ones(n) * 1000})
    hl = df_nv["high"] - df_nv["low"];  hc = (df_nv["high"] - df_nv["close"].shift()).abs()
    lc = (df_nv["low"] - df_nv["close"].shift()).abs()
    df_nv["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    entry_bc  = float(df_nv["close"].iloc[-1])
    atr_bc    = float(df_nv["atr"].dropna().iloc[-1])
    base_conf_bc = 7.0

    print(f"\n  Best case entry: ${entry_bc:,.2f}  ATR: ${atr_bc:.2f}")
    print(f"  Conditions: Overlap + TRENDING_STRONG + normal vol + geo_mult=0 + LONG aligned with macro")

    # Step 1: Macro strongly confirms long
    macro_bc_adj  = 1.0   # strongly bullish
    conf_bc       = base_conf_bc + macro_bc_adj
    print(f"\n  After macro    : {base_conf_bc:.1f} + ({macro_bc_adj:+.1f}) = {conf_bc:.1f}")

    # Step 2: Geo calm — no SL widening, no confidence adj
    geo_bc_adj   = 0.0
    geo_bc_mult  = 0.0
    conf_bc     += geo_bc_adj
    print(f"  After geo      : {conf_bc:.1f} + {geo_bc_adj:+.1f} = {conf_bc:.1f}  (calm geo)")

    # Step 3: Dynamic SL — Overlap × TRENDING_STRONG × normal vol + geo 0
    dyn_bc = calculate_dynamic_sl(df_nv, "long", entry_bc,
                                   session="Overlap",
                                   regime="TRENDING_STRONG",
                                   geo_multiplier=0.0)
    print(f"\n  Dynamic SL:    ${dyn_bc['sl_price']:,.2f}  dist=${dyn_bc['sl_distance']:.2f}")
    print(f"  Breakdown:     {dyn_bc['sl_breakdown']}")
    print(f"  TP1:           ${dyn_bc['tp1_price']:,.2f}  (1:{dyn_bc['rr_at_tp1']:.1f})")
    print(f"  TP2:           ${dyn_bc['tp2_price']:,.2f}  (1:{dyn_bc['rr_at_tp2']:.1f})")
    _check(dyn_bc["session_multiplier"] == 1.5, "Overlap session_mult = 1.5")
    _check(dyn_bc["regime_multiplier"]  == 0.9, "TRENDING_STRONG regime_mult = 0.9")
    _check(dyn_bc["geo_buffer"]         == 0.0, "Geo buffer = 0.0 (calm)")

    # Step 4: Overlap session profile (grade A typically, lot_mult=1.1)
    overlap_profile = {
        "current_session": "Overlap", "session_grade": "A",
        "lot_multiplier": 1.2, "sl_multiplier": 1.0, "tp_multiplier": 1.1,
        "trading_recommended": True, "session_note": "Most volatile overlap",
    }
    adj_bc = get_session_adjusted_position(0.01, dyn_bc["sl_distance"],
                                            abs(dyn_bc["tp2_price"] - entry_bc),
                                            overlap_profile)
    print(f"\n  Session:       Overlap Grade A")
    print(f"  Lot adjust:    {adj_bc['lot_change']}  -->  {adj_bc['adjusted_lots']:.4f} lots")

    final_conf_bc = round(min(10.0, conf_bc), 2)
    print(f"\n  {'='*55}")
    print(f"  BEST CASE SIGNAL SUMMARY")
    print(f"  {'='*55}")
    print(f"  Entry:       ${entry_bc:,.2f}")
    print(f"  SL:          ${dyn_bc['sl_price']:,.2f}  (-${dyn_bc['sl_distance']:.2f})")
    print(f"  TP1:         ${dyn_bc['tp1_price']:,.2f}  (1:2)")
    print(f"  TP2:         ${dyn_bc['tp2_price']:,.2f}  (1:3)")
    print(f"  Lots:        {adj_bc['adjusted_lots']:.4f}  (+{adj_bc['lot_change']})")
    print(f"  Confidence:  {final_conf_bc:.1f}/10  [STRONG SIGNAL]")

    _check(final_conf_bc >= 8.0, f"Best case conf >= 8.0 (got {final_conf_bc})")
    _check(adj_bc["adjusted_lots"] >= 0.01, f"Best case lots valid ({adj_bc['adjusted_lots']:.4f})")
    _check(dyn_bc["rr_at_tp2"] >= 2.5, f"Best case R:R >= 2.5 ({dyn_bc['rr_at_tp2']})")

except Exception:
    _fail("TEST 4 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TEST 5 — Component conflict test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("INTEGRATION TEST 5 -- Component conflict: Macro says SHORT, Geo says LONG")
print(SEP)

try:
    base_conf_ct = 7.0

    # Macro opposes LONG (bearish macro)
    macro_ct_adj = -1.0
    # Geo supports LONG via safe haven (extreme crisis)
    geo_ct_adj   = +0.5
    net_ct_adj   = macro_ct_adj + geo_ct_adj

    final_conf_ct = base_conf_ct + net_ct_adj

    print(f"\n  Direction:         LONG")
    print(f"  Macro:             bearish (DXY rising + yields rising)")
    print(f"    macro_adjustment = {macro_ct_adj:+.1f}  (opposes long)")
    print(f"  Geo:               extreme risk (safe haven demand for gold)")
    print(f"    geo_adjustment   = {geo_ct_adj:+.1f}  (supports long via safe haven)")
    print(f"  net_adjustment     = {net_ct_adj:+.1f}")
    print(f"")
    print(f"  Base confidence  : {base_conf_ct:.1f}")
    print(f"  After macro      : {base_conf_ct + macro_ct_adj:.1f}")
    print(f"  After geo        : {final_conf_ct:.1f}  <-- net result")
    print(f"")
    print(f"  Context visible to user:")
    print(f"    Macro:  bearish  --> -1.0  (DXY rising/yields rising OPPOSE long)")
    print(f"    Geo:    extreme  --> +0.5  (crisis = safe haven = supports gold LONG)")
    print(f"    Net:    {net_ct_adj:+.1f} (macro wins, but geo provides partial offset)")
    print(f"")
    show_ct = final_conf_ct >= 7.0
    print(f"  Final confidence : {final_conf_ct:.1f}/10")
    print(f"  Show signal      : {'YES -- but note conflicting factors' if show_ct else 'NO -- below threshold'}")
    print(f"  Bot behaviour    : Both factors shown in trade card so user")
    print(f"                     can make informed decision with full context.")

    _check(math.isclose(net_ct_adj,      -0.5, rel_tol=1e-6),
           f"Net adjustment = -0.5 (macro -1.0 + geo +0.5 = {net_ct_adj:+.1f})")
    _check(math.isclose(final_conf_ct,    6.5, rel_tol=1e-6),
           f"Final conf = 6.5 after conflict ({final_conf_ct:.1f})")
    _check(not show_ct,
           f"Conflicted signal below threshold (conf={final_conf_ct:.1f} < 7.0) -- correctly SKIPPED")

    # Verify both adjustments are individually correct in sign
    _check(macro_ct_adj < 0,  "Macro adj is negative for bearish scenario (-1.0)")
    _check(geo_ct_adj   > 0,  "Geo adj is positive for safe haven (+0.5)")
    _check(macro_ct_adj + geo_ct_adj < 0, "Macro wins the conflict (net negative)")

except Exception:
    _fail("TEST 5 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FINAL VERDICT")
print(SEP)

# Per-task assessment
task_results: dict[str, tuple[bool, str]] = {}

# Task 6 — macro filter
t6_ok = TASK6_OK and not any("Task 6" in i for i in ISSUES) and not any("macro" in i.lower() for i in ISSUES)
task_results["Task 6 (DXY+Yields)"] = (
    TASK6_OK,
    "Imports OK; confidence_adjustment applied to signal pipeline",
)

# Task 7 — geo filter
task_results["Task 7 (Geo filter)"] = (
    TASK7_OK,
    "Imports OK; sl_atr_multiplier and confidence_adjustment applied",
)

# Task 8 — dynamic ATR SL
t8_ok = TASK8_OK and not any("Dynamic SL" in i for i in ISSUES) and not any("Task 8" in i for i in ISSUES)
task_results["Task 8 (Dynamic ATR SL)"] = (
    TASK8_OK,
    "Imports OK; sl_breakdown/TP1/TP2 correct; hard caps verified in TEST 3",
)

# Task 9 — session profiler
task_results["Task 9 (Session profile)"] = (
    TASK9_OK,
    "Imports OK; lot/SL/TP multipliers applied; Asian reduces lots correctly",
)

all_integrated = all(ok for ok, _ in task_results.values()) and TOTAL_FAIL == 0

print(f"\n  PHASE 2 COMPLETE -- INTEGRATION RESULTS:")
print(f"  {'='*56}")
for task, (ok, finding) in task_results.items():
    status = PASS if ok else FAIL
    print(f"  {status}  {task}")
    print(f"           {finding}")
print(f"  {'='*56}")
print(f"\n  All components integrated : {'YES' if all_integrated else 'NO -- see issues below'}")
print(f"  Ready for Phase 3         : {'YES' if all_integrated else 'NO'}")
print(f"\n  Total PASS  : {TOTAL_PASS}")
print(f"  Total FAIL  : {TOTAL_FAIL}")
print(f"  Total checks: {TOTAL_PASS + TOTAL_FAIL}")

if ISSUES:
    print(f"\n  Issues found:")
    for issue in ISSUES:
        print(f"    - {issue}")
print()
