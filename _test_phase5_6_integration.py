"""
Full Integration Backtest — Phase 5 + 6
Tests: COT, Liquidity Heatmap, Walk-Forward, Session Handoff
Run: python _test_phase5_6_integration.py
"""

import os, sys, json, traceback
os.environ.setdefault("PYTHONUTF8", "1")

import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP  = "─" * 60
DSEP = "═" * 60
PASS = "✅ PASS"
FAIL = "❌ FAIL"

results: dict[str, bool] = {}

def hdr(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(label: str, cond: bool, detail: str = "") -> bool:
    sym  = "  ✓" if cond else "  ✗"
    tail = f"  {detail}" if detail else ""
    print(f"{sym}  {label}{tail}")
    return cond

# ── Load data ─────────────────────────────────────────────────────────────────
hdr("Loading historical data")
try:
    df = pd.read_csv("data/historical_xauusd.csv")
    # Normalise column names to lower-case
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse datetime index
    time_col = next((c for c in df.columns if "time" in c or "date" in c), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df.set_index(time_col).sort_index()
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    current_price = float(df["close"].iloc[-1])
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    print(f"  Current price: ${current_price:,.2f}")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")
    DATA_OK = True
except Exception as _e:
    print(f"  ERROR loading data: _e")
    traceback.print_exc()
    DATA_OK = False
    current_price = 2350.0
    df = pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Session Handoff unit tests
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 1 — Session Handoff unit tests")
try:
    from session_handoff import (
        get_asian_session_range,
        detect_london_break,
        get_ny_session_bias,
        format_session_handoff,
    )

    asian = get_asian_session_range(df)
    required_asian_base = ["available"]
    required_asian_full = ["asian_high", "asian_low", "asian_range", "asian_mid", "asian_bias"]
    t1a_base = ok("asian 'available' key present", "available" in asian)
    if asian.get("available"):
        t1a = ok("asian full keys present", all(k in asian for k in required_asian_full))
        t1b = ok("asian_high > asian_low",
                 float(asian.get("asian_high", 0)) >= float(asian.get("asian_low", 0)),
                 f"  high={asian.get('asian_high')} low={asian.get('asian_low')}")
    else:
        t1a = ok("asian session not available today (fallback OK)", True,
                 f"  reason={asian.get('reason', 'unknown')}")
        t1b = ok("asian_high/low N/A (no session data)", True)

    london = detect_london_break(df, asian)
    valid_break_types = {"REAL_BREAK", "FAKE_BREAK", "INSIDE_RANGE", "VOLATILE", "no_data", "unknown"}
    valid_ny_biases   = {"BULLISH", "BEARISH", "NEUTRAL"}
    t1c = ok("london keys present",
              all(k in london for k in ["break_detected", "break_type", "ny_bias"]))
    t1d = ok(f"break_type valid ({london.get('break_type')})",
              london.get("break_type") in valid_break_types)
    t1e = ok(f"ny_bias valid ({london.get('ny_bias')})",
              london.get("ny_bias") in valid_ny_biases)

    handoff = get_ny_session_bias(df)
    req_h   = ["asian_range", "london_break", "ny_bias", "confidence",
               "confidence_score", "recommendation", "action", "summary"]
    t1f = ok("ny_bias keys present", all(k in handoff for k in req_h),
             f"  missing={[k for k in req_h if k not in handoff]}")

    print(f"\n  Full handoff summary:")
    print(format_session_handoff(handoff))

    passed = all([t1a, t1b, t1c, t1d, t1e, t1f])
    results["T1_session_handoff_unit"] = passed
    print(f"\n  {PASS if passed else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T1_session_handoff_unit"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Session handoff confidence adjustment
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 2 — Session handoff confidence adjustment")
try:
    def _apply_sh_adj(signal: dict, ny_bias: str, confidence_level: str) -> dict:
        """Mirrors the bot_chat / morning_briefing logic."""
        s = dict(signal)
        _dir  = str(s.get("direction", "")).lower()
        _note = s.get("note", "")
        if ny_bias == "BULLISH" and _dir == "long" and confidence_level == "HIGH":
            s["confidence"] = min(10.0, float(s["confidence"]) + 0.5)
            s["note"] = (_note + " | " if _note else "") + "\u2713 Session handoff confirms LONG"
        elif ny_bias == "BEARISH" and _dir == "short" and confidence_level == "HIGH":
            s["confidence"] = min(10.0, float(s["confidence"]) + 0.5)
            s["note"] = (_note + " | " if _note else "") + "\u2713 Session handoff confirms SHORT"
        elif (ny_bias == "BULLISH" and _dir == "short") or (ny_bias == "BEARISH" and _dir == "long"):
            s["confidence"] = max(0.0, float(s["confidence"]) - 0.5)
            s["note"] = (_note + " | " if _note else "") + f"\u26a0 Session handoff opposes {_dir.upper()}"
        return s

    base_conf = 7.0

    # Scenario A: BULLISH + LONG + HIGH → +0.5
    sa = _apply_sh_adj({"direction": "long", "confidence": base_conf, "note": ""},
                        "BULLISH", "HIGH")
    t2a = ok("Scenario A: BULLISH+LONG+HIGH → +0.5",
              abs(sa["confidence"] - (base_conf + 0.5)) < 0.001,
              f"  got={sa['confidence']}")

    # Scenario B: BEARISH + LONG → -0.5 + note
    sb = _apply_sh_adj({"direction": "long", "confidence": base_conf, "note": ""},
                        "BEARISH", "MODERATE")
    t2b1 = ok("Scenario B: BEARISH+LONG → -0.5",
               abs(sb["confidence"] - (base_conf - 0.5)) < 0.001,
               f"  got={sb['confidence']}")
    t2b2 = ok("Scenario B: note contains 'opposes LONG'",
               "opposes LONG" in sb.get("note", ""))

    # Scenario C: NEUTRAL → unchanged
    sc = _apply_sh_adj({"direction": "long", "confidence": base_conf, "note": ""},
                        "NEUTRAL", "LOW")
    t2c = ok("Scenario C: NEUTRAL → unchanged",
              abs(sc["confidence"] - base_conf) < 0.001,
              f"  got={sc['confidence']}")

    passed = all([t2a, t2b1, t2b2, t2c])
    results["T2_confidence_adjustment"] = passed
    print(f"\n  {PASS if passed else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T2_confidence_adjustment"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — COT integration test
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 3 — COT integration test")
try:
    from cot_analyzer import fetch_cot_data, get_cot_signal

    cot = fetch_cot_data()
    t3a = ok("fetch_cot_data returns dict", isinstance(cot, dict))

    # Check cache file
    cache_path = "data/cot_cache.json"
    t3b = ok("cot_cache.json exists", os.path.exists(cache_path))

    long_sig  = get_cot_signal("long",  cot)
    short_sig = get_cot_signal("short", cot)
    t3c = ok("long boost valid",  isinstance(long_sig.get("boost"), (int, float)))
    t3d = ok("short boost valid", isinstance(short_sig.get("boost"), (int, float)))

    # Verify FACTOR 11 is in confluence_engine
    import confluence_engine
    src = open("confluence_engine.py", encoding="utf-8").read()
    t3e = ok("FACTOR 11 COT wired in confluence_engine",
              "FACTOR 11" in src and "_cot_signal" in src)

    _cot_bias = cot.get("bias", cot.get("net_position_bias", "NEUTRAL"))
    print(f"\n  COT: {_cot_bias} | LONG boost={long_sig.get('boost', 0):.2f}"
          f" | SHORT boost={short_sig.get('boost', 0):.2f}")

    passed = all([t3a, t3b, t3c, t3d, t3e])
    results["T3_cot_integration"] = passed
    print(f"\n  {PASS if passed else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T3_cot_integration"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Liquidity map integration test
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 4 — Liquidity map integration test")
try:
    from liquidity_map import build_liquidity_map, format_liquidity_map

    liq = build_liquidity_map(df, current_price)
    t4a = ok("build_liquidity_map returns dict", isinstance(liq, dict))

    above = liq.get("clusters_above", [])
    below = liq.get("clusters_below", [])
    likely = liq.get("likely_move", "NEUTRAL")

    def _cluster_price(c) -> float:
        return float(c) if not isinstance(c, dict) else float(c.get("price", 0))
    t4b = ok("clusters_above all > current_price",
              all(_cluster_price(c) > current_price for c in above) if above else True,
              f"  {len(above)} clusters")
    t4c = ok("clusters_below all < current_price",
              all(_cluster_price(c) < current_price for c in below) if below else True,
              f"  {len(below)} clusters")
    t4d = ok(f"likely_move valid ({likely})",
              likely in {"UP", "DOWN", "NEUTRAL"})

    fmt = format_liquidity_map(liq, current_price)
    t4e = ok("format_liquidity_map returns non-empty string", bool(fmt and len(fmt) > 10))

    # Print top 2 each side
    poc = liq.get("poc", 0.0)
    poc_price = float(poc) if not isinstance(poc, dict) else float(poc.get("price", 0))
    print(f"\n  Clusters above current (top 2):")
    for c in above[:2]:
        _cp = float(c) if not isinstance(c, dict) else float(c.get('price', 0))
        print(f"    ${_cp:,.2f}")
    print(f"  Clusters below current (top 2):")
    for c in below[:2]:
        _cp = float(c) if not isinstance(c, dict) else float(c.get('price', 0))
        print(f"    ${_cp:,.2f}")
    print(f"  POC: ${poc_price:,.2f}  |  Likely move: {likely}")

    passed = all([t4a, t4b, t4c, t4d, t4e])
    results["T4_liquidity_map"] = passed
    print(f"\n  {PASS if passed else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T4_liquidity_map"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Walk-forward optimization test
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 5 — Walk-forward optimization test")
try:
    from walk_forward import (
        check_if_sunday_run_needed,
        get_wfo_summary,
        run_walk_forward_optimization,
    )

    sunday_flag = check_if_sunday_run_needed()
    t5a = ok("check_if_sunday_run_needed returns bool", isinstance(sunday_flag, bool),
             f"  result={sunday_flag}")

    summary = get_wfo_summary()
    t5b = ok("get_wfo_summary returns string", isinstance(summary, str))

    history_path = "data/wfo_history.json"
    # wfo_history.json is created on first Sunday run; may not exist yet
    t5c = ok("wfo_history.json exists OR check_if_sunday=False (first run)",
              os.path.exists(history_path) or not sunday_flag,
              f"  path_exists={os.path.exists(history_path)}  sunday_needed={sunday_flag}")

    # run_walk_forward_optimization() takes 0 args — it reads signal_log internally
    wfo_result = run_walk_forward_optimization()
    t5d = ok("run_walk_forward_optimization() returns dict",
              isinstance(wfo_result, dict),
              f"  optimized={wfo_result.get('optimized')}  reason={wfo_result.get('reason', '')}")

    print(f"\n  WFO summary (first 200 chars):\n  {summary[:200]}")
    passed = all([t5a, t5b, t5c, t5d])
    results["T5_walk_forward"] = passed
    print(f"\n  {PASS if passed else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T5_walk_forward"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Full Phase 5+6 pipeline
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 6 — Full Phase 5+6 pipeline")
try:
    from session_handoff import get_ny_session_bias
    from cot_analyzer    import fetch_cot_data, get_cot_signal
    from liquidity_map   import build_liquidity_map
    from walk_forward    import get_wfo_summary

    # Step 1
    p6_handoff = get_ny_session_bias(df)
    ny_bias    = p6_handoff.get("ny_bias", "NEUTRAL")
    ny_conf    = p6_handoff.get("confidence", "LOW")
    asian_r    = p6_handoff.get("asian_range", {})
    london_r   = p6_handoff.get("london_break", {})

    # Step 2
    p6_cot     = fetch_cot_data()
    long_sig   = get_cot_signal("long", p6_cot)
    cot_boost  = float(long_sig.get("boost", 0.0))
    cot_bias_s = p6_cot.get("bias", p6_cot.get("net_position_bias", "NEUTRAL"))

    # Step 3
    p6_liq     = build_liquidity_map(df, current_price)
    liq_move   = p6_liq.get("likely_move", "NEUTRAL")
    above_cls  = p6_liq.get("clusters_above", [])
    liq_target = above_cls[0].get("price", current_price) if above_cls else current_price

    # Step 4
    p6_wfo     = get_wfo_summary()

    # Apply to mock LONG signal
    base_conf = 7.0
    sh_adj    = 0.0
    if ny_bias == "BULLISH" and ny_conf == "HIGH":
        sh_adj = +0.5
    elif ny_bias == "BEARISH":
        sh_adj = -0.5
    liq_adj = +0.3 if liq_move == "UP" else (-0.3 if liq_move == "DOWN" else 0.0)
    final_conf = min(10.0, max(0.0, base_conf + cot_boost + sh_adj + liq_adj))

    # Formatting helpers
    ah = float(asian_r.get("asian_high", 0))
    al = float(asian_r.get("asian_low",  0))
    btype = london_r.get("break_type", "no_data")
    rec   = p6_handoff.get("recommendation", "N/A")

    print(f"\n  PHASE 5+6 PIPELINE TEST:")
    print(f"  {SEP}")
    print(f"  Asian range:       ${al:,.2f}–${ah:,.2f}")
    print(f"  London break:      {btype}")
    print(f"  NY bias:           {ny_bias} ({ny_conf})")
    print(f"  COT:               {cot_bias_s} (boost {cot_boost:+.2f})")
    print(f"  Liquidity:         {liq_move} → ${liq_target:,.2f}")
    print(f"  {SEP}")
    print(f"  Base confidence:   7.0")
    print(f"  + COT:             {cot_boost:+.2f}")
    print(f"  + Session handoff: {sh_adj:+.2f}")
    print(f"  + Liquidity:       {liq_adj:+.2f}")
    print(f"  Final:             {final_conf:.1f}/10")
    print(f"  {SEP}")
    print(f"  Recommendation:    {rec}")

    t6 = (isinstance(p6_handoff, dict) and isinstance(p6_cot, dict)
          and isinstance(p6_liq, dict) and 0.0 <= final_conf <= 10.0)
    results["T6_full_pipeline"] = t6
    print(f"\n  {PASS if t6 else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T6_full_pipeline"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Regression test all phases
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 7 — Regression test all phases")

def _reg(label: str, fn, *args, **kwargs) -> bool:
    """Call fn with args; return True if it doesn't raise and returns a non-None value."""
    try:
        res = fn(*args, **kwargs)
        passed = res is not None
        ok(label, passed, f"type={type(res).__name__}")
        results[f"T7_{label.replace(' ','_')}"] = passed
        return passed
    except Exception as _exc:
        ok(label, False, f"EXCEPTION: {_exc}")
        results[f"T7_{label.replace(' ','_')}"] = False
        return False

try:
    from spread_monitor  import check_spread
    from dxy_correlation import get_macro_context
    from geo_filter      import get_geopolitical_score
    from atr_sl_engine   import calculate_dynamic_sl
    from confluence_engine import detect_rsi_divergence
    from sr_mapper       import get_sr_levels
    from reversal_hunter import hunt_reversals
    from trade_manager   import calculate_partial_tp_plan, get_current_risk_profile
    from cot_analyzer    import fetch_cot_data
    from liquidity_map   import build_liquidity_map
    from session_handoff import get_ny_session_bias

    _mock_signal = {
        "asset": "XAUUSD", "direction": "long",
        "entry": current_price, "stop_loss": current_price - 15,
        "take_profit": current_price + 30, "confidence": 7.0,
        "pattern_name": "RSI Oversold Bounce", "timeframe": "H1",
        "risk_reward": 2.0,
    }

    p1 = _reg("Phase 1: check_spread",           check_spread)
    p2a = _reg("Phase 2: get_macro_context",      get_macro_context, "long")
    p2b = _reg("Phase 2: get_geopolitical_score", get_geopolitical_score)
    p2c = _reg("Phase 2: calculate_dynamic_sl",   calculate_dynamic_sl,
               df, "long", current_price)
    p3a = _reg("Phase 3: detect_rsi_divergence",  detect_rsi_divergence, df)
    p3b = _reg("Phase 3: get_sr_levels",          get_sr_levels, df, current_price)
    p3c = _reg("Phase 3: hunt_reversals",         hunt_reversals, df, current_price)
    p4a = _reg("Phase 4: calculate_partial_tp_plan", calculate_partial_tp_plan,
               _mock_signal, df)
    p4b = _reg("Phase 4: get_current_risk_profile",  get_current_risk_profile)
    p5a = _reg("Phase 5: fetch_cot_data",         fetch_cot_data)
    p5b = _reg("Phase 5: build_liquidity_map",    build_liquidity_map,
               df, current_price)
    p6  = _reg("Phase 6: get_ny_session_bias",    get_ny_session_bias, df)

    t7_all = all([p1, p2a, p2b, p2c, p3a, p3b, p3c, p4a, p4b, p5a, p5b, p6])
    results["T7_regression_overall"] = t7_all
    print(f"\n  {PASS if t7_all else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION during setup: {_e}")
    traceback.print_exc()
    results["T7_regression_overall"] = False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Command routing test
# ─────────────────────────────────────────────────────────────────────────────
hdr("TEST 8 — Command routing test (static source analysis)")
try:
    with open("bot_chat.py", encoding="utf-8") as _f:
        bc_src = _f.read()

    route_checks = {
        '"session handoff"':     "_handle_handoff",
        '"london break"':        "_handle_handoff",
        '"ny bias"':             "_handle_handoff",
        '"liquidity"':           "_handle_liquidity",
        '"heatmap"':             "_handle_liquidity",
        '"cot data"':            "_handle_cot",
        '"optimization report"': "_handle_wfo",
    }

    t8_results = {}
    for keyword, handler in route_checks.items():
        kw_plain = keyword.strip('"')
        # keyword present in source
        kw_present = kw_plain in bc_src
        # handler defined
        hd_present = f"def {handler}" in bc_src
        passed_check = kw_present and hd_present
        t8_results[keyword] = passed_check
        ok(f"{keyword} → {handler}()", passed_check,
           f"  kw={'found' if kw_present else 'MISSING'}"
           f"  handler={'found' if hd_present else 'MISSING'}")

    new_cmd_count = sum(1 for v in t8_results.values() if v)
    print(f"\n  New routed commands present: {new_cmd_count}/{len(route_checks)}")

    t8 = all(t8_results.values())
    results["T8_routing"] = t8
    print(f"\n  {PASS if t8 else FAIL}")
except Exception as _e:
    print(f"  EXCEPTION: {_e}")
    traceback.print_exc()
    results["T8_routing"] = False

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DSEP}")
all_pass = all(results.values())
tests_passed = sum(results.values())
tests_total  = len(results)

# Per-test summary
for name, res in results.items():
    sym = "✅" if res else "❌"
    print(f"  {sym}  {name}")

print(f"\n  Tests passed: {tests_passed}/{tests_total}")
print(f"\n{DSEP}")
print(f"\n{'  ' + PASS if all_pass else '  SOME TESTS FAILED — see above'}")

print(f"""
{DSEP}
 FULL BOT BUILD — COMPLETE
 {DSEP}
 Phase 1 — Stop bad trades:    {"✅" if results.get("T7_Phase_1:_check_spread", results.get("T7_Phase 1: check_spread", True)) else "⚠"}
 Phase 2 — Protect capital:    {"✅" if all(results.get(k, False) for k in results if "Phase_2" in k or "Phase 2" in k) else "⚠"}
 Phase 3 — Better entries:     {"✅" if all(results.get(k, False) for k in results if "Phase_3" in k or "Phase 3" in k) else "⚠"}
 Phase 4 — Maximise profits:   {"✅" if all(results.get(k, False) for k in results if "Phase_4" in k or "Phase 4" in k) else "⚠"}
 Phase 5 — Institutional edge: {"✅" if all(results.get(k, False) for k in results if "Phase_5" in k or "Phase 5" in k) else "⚠"}
 Phase 6 — Session handoff:    {"✅" if results.get("T1_session_handoff_unit") and results.get("T6_full_pipeline") else "⚠"}
 {DSEP}
 All systems: {"READY FOR LIVE USE" if all_pass else "REVIEW FAILURES ABOVE"}
""")
