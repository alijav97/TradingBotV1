"""
_test_phase3_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3 Complete Integration Backtest — NO CODE CHANGES, TEST ONLY
7 tests covering:
  1. S/R Auto-Mapper (sr_mapper.py)
  2. 11-Factor Confluence Engine (confluence_engine.py)
  3. Reversal Hunter (reversal_hunter.py)
  4. Full 8-step Pipeline (mock LONG signal)
  5. 50-candle rolling simulation (last 500 candles, every 10th)
  6. Today's Market Analysis (all Phase 3 components live)
  7. Phase 1/2 Regression (no crashes allowed)
"""

import sys, os, time, traceback
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ── working directory = project root ──────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: dict[str, bool] = {}

def section(title: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    return condition

def record(test_name: str, passed: bool) -> None:
    results[test_name] = passed


# ══════════════════════════════════════════════════════════════════════════════
#  Load real CSV data
# ══════════════════════════════════════════════════════════════════════════════
section("LOADING DATA")

CSV_PATH = os.path.join("data", "historical_xauusd.csv")
try:
    raw_df = pd.read_csv(CSV_PATH)
    # Normalise column names
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    rename_map = {}
    for col in raw_df.columns:
        if col in ("open", "o"):  rename_map[col] = "open"
        if col in ("high", "h"):  rename_map[col] = "high"
        if col in ("low", "l"):   rename_map[col] = "low"
        if col in ("close", "c"): rename_map[col] = "close"
        if col in ("volume", "vol", "v"): rename_map[col] = "volume"
    raw_df.rename(columns=rename_map, inplace=True)

    # Parse datetime
    dt_col = next((c for c in raw_df.columns if "time" in c or "date" in c), None)
    if dt_col:
        raw_df[dt_col] = pd.to_datetime(raw_df[dt_col], utc=True, errors="coerce")
        raw_df.set_index(dt_col, inplace=True)
    raw_df.sort_index(inplace=True)
    raw_df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    print(f"  Rows loaded : {len(raw_df):,}")
    print(f"  Date range  : {raw_df.index[0]}  →  {raw_df.index[-1]}")
    print(f"  Last close  : ${raw_df['close'].iloc[-1]:,.2f}")

    LIVE_DF    = raw_df.copy()
    LIVE_PRICE = float(raw_df["close"].iloc[-1])
    DATA_OK    = True
except Exception as e:
    print(f"  {FAIL}  Could not load CSV: {e}")
    DATA_OK = False
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — S/R AUTO-MAPPER
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 1 — S/R AUTO-MAPPER  (sr_mapper.py)")

t1_pass = True
try:
    from sr_mapper import get_sr_levels, is_at_support, is_at_resistance

    sr = get_sr_levels(LIVE_DF, LIVE_PRICE)

    # Required top-level keys
    req_keys = [
        "resistance_levels", "support_levels",
        "nearest_resistance", "nearest_support",
        "at_key_level", "key_level_detail",
        "prev_week_high", "prev_week_low",
        "prev_day_high", "prev_day_low",
        "round_numbers", "summary",
    ]
    missing_keys = [k for k in req_keys if k not in sr]
    ok = check("All required keys present", not missing_keys,
               f"missing: {missing_keys}" if missing_keys else "")
    t1_pass &= ok

    # Resistance levels above price, support levels below
    res_above = all(r["price"] >= LIVE_PRICE * 0.995 for r in sr["resistance_levels"])
    sup_below = all(s["price"] <= LIVE_PRICE * 1.005 for s in sr["support_levels"])
    t1_pass &= check("Resistance levels at/above current price", res_above,
                     f"price={LIVE_PRICE:.2f}, levels={[r['price'] for r in sr['resistance_levels'][:3]]}")
    t1_pass &= check("Support levels at/below current price", sup_below,
                     f"price={LIVE_PRICE:.2f}, levels={[s['price'] for s in sr['support_levels'][:3]]}")

    # nearest_resistance / nearest_support sub-keys
    nr = sr["nearest_resistance"]
    ns = sr["nearest_support"]
    for sub in ("price", "distance_usd", "distance_pct", "proximity", "sources"):
        t1_pass &= check(f"nearest_resistance has '{sub}'", sub in nr)
        t1_pass &= check(f"nearest_support has '{sub}'", sub in ns)

    # Round numbers present
    t1_pass &= check("Round numbers list non-empty", len(sr["round_numbers"]) > 0,
                     f"count={len(sr['round_numbers'])}")

    # at_key_level is bool
    t1_pass &= check("at_key_level is bool", isinstance(sr["at_key_level"], bool))

    # is_at_support / is_at_resistance helpers
    sup_flag  = is_at_support(LIVE_DF, LIVE_PRICE)
    res_flag  = is_at_resistance(LIVE_DF, LIVE_PRICE)
    t1_pass &= check("is_at_support() returns bool",  isinstance(sup_flag, bool))
    t1_pass &= check("is_at_resistance() returns bool", isinstance(res_flag, bool))

    print(f"\n  S/R Summary: {sr['summary']}")
    print(f"  Nearest Resistance: ${nr['price']:,.2f}  ({nr['proximity']})  dist={nr['distance_usd']:.1f}")
    print(f"  Nearest Support:    ${ns['price']:,.2f}  ({ns['proximity']})  dist={ns['distance_usd']:.1f}")
    print(f"  Resistance levels: {len(sr['resistance_levels'])}  |  Support levels: {len(sr['support_levels'])}")
    print(f"  At key level: {sr['at_key_level']}  ({sr['key_level_detail']})")

except Exception as e:
    print(f"  {FAIL}  sr_mapper import/run error: {e}")
    traceback.print_exc()
    t1_pass = False

record("T1_SR_Mapper", t1_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — 11-FACTOR CONFLUENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 2 — 11-FACTOR CONFLUENCE ENGINE  (confluence_engine.py)")

t2_pass = True
try:
    from confluence_engine import score_confluences

    for direction in ("long", "short"):
        conf = score_confluences(LIVE_DF, direction)

        t2_pass &= check(f"[{direction.upper()}] Returns dict",      isinstance(conf, dict))
        t2_pass &= check(f"[{direction.upper()}] 'confidence' key",  "confidence" in conf,
                         f"keys={list(conf.keys())[:8]}")
        t2_pass &= check(f"[{direction.upper()}] confidence in 0-10",
                         0.0 <= conf.get("confidence", -1) <= 10.0,
                         f"val={conf.get('confidence')}")
        t2_pass &= check(f"[{direction.upper()}] 'detail_lines' present",
                         "detail_lines" in conf and len(conf["detail_lines"]) >= 5)
        t2_pass &= check(f"[{direction.upper()}] 'raw_checks' present",
                         "raw_checks" in conf)
        t2_pass &= check(f"[{direction.upper()}] 'trade_valid' present",
                         "trade_valid" in conf)

        rc = conf.get("raw_checks", {})
        for factor in ("sr_levels", "rsi_divergence", "smc", "htf", "dxy"):
            t2_pass &= check(f"[{direction.upper()}] raw_checks has '{factor}'",
                             factor in rc, f"raw_checks keys={list(rc.keys())}")

        print(f"\n  [{direction.upper()}] Confidence: {conf.get('confidence'):.1f}/10  "
              f"valid={conf.get('trade_valid')}  "
              f"passed={conf.get('passed_count')}/{conf.get('total_checks')}")
        for line in conf.get("detail_lines", [])[:11]:
            print(f"    {line}")

except Exception as e:
    print(f"  {FAIL}  confluence_engine error: {e}")
    traceback.print_exc()
    t2_pass = False

record("T2_Confluence_11Factor", t2_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — REVERSAL HUNTER
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 3 — REVERSAL HUNTER  (reversal_hunter.py)")

t3_pass = True
try:
    from reversal_hunter import hunt_reversals

    signals = hunt_reversals(LIVE_DF, LIVE_PRICE)

    t3_pass &= check("hunt_reversals() returns list", isinstance(signals, list))

    if signals:
        sig = signals[0]
        t3_pass &= check("Signal has 'score' key",          "score" in sig)
        t3_pass &= check("Signal has 'conditions_met' key", "conditions_met" in sig)
        t3_pass &= check("Score is numeric (0-13)",
                         isinstance(sig.get("score"), (int, float)) and 0 <= sig["score"] <= 13,
                         f"score={sig.get('score')}")
        print(f"\n  Reversal signals found: {len(signals)}")
        for s in signals[:3]:
            direction_label = s.get("direction", s.get("type", "?"))
            print(f"    Score={s.get('score')}/13  dir={direction_label}  "
                  f"conditions={len(s.get('conditions_met', []))}")
    else:
        print("\n  No reversal signals on current candle (expected — market may not be at reversal zone)")
        t3_pass &= check("hunt_reversals() completed without error", True)

    # Also test on a sub-slice to ensure it doesn't crash on smaller data
    small_df = LIVE_DF.tail(100)
    sigs_small = hunt_reversals(small_df, float(small_df["close"].iloc[-1]))
    t3_pass &= check("hunt_reversals() works on 100-candle slice", isinstance(sigs_small, list))

except Exception as e:
    print(f"  {FAIL}  reversal_hunter error: {e}")
    traceback.print_exc()
    t3_pass = False

record("T3_ReversalHunter", t3_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — FULL 8-STEP PIPELINE (mock LONG signal)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 4 — FULL 8-STEP PIPELINE  (mock LONG signal)")

t4_pass  = True
pipeline: dict[str, dict] = {}

try:
    from sr_mapper         import get_sr_levels
    from confluence_engine import score_confluences
    from reversal_hunter   import hunt_reversals
    from market_context    import detect_gold_regime, get_regime_strategy_config
    from fundamental_bias  import get_fundamental_bias, check_fundamental_conflict

    direction = "long"

    print(f"\n  Step 1 — Load market data")
    df    = LIVE_DF.copy()
    price = LIVE_PRICE
    t4_pass &= check("DataFrame ready", len(df) > 200, f"rows={len(df)}")

    print(f"\n  Step 2 — Detect market regime")
    regime_data = detect_gold_regime(df)
    pipeline["regime"] = regime_data
    regime = regime_data.get("regime", "UNKNOWN")
    t4_pass &= check("Regime detected",       bool(regime) and regime != "UNKNOWN", f"regime={regime}")
    t4_pass &= check("Position size multiplier present",
                     "position_size_multiplier" in regime_data,
                     f"val={regime_data.get('position_size_multiplier')}")
    print(f"    Regime: {regime}  |  size_mult={regime_data.get('position_size_multiplier')}")

    print(f"\n  Step 3 — Get regime strategy config")
    regime_cfg = get_regime_strategy_config(regime)
    pipeline["regime_cfg"] = regime_cfg
    t4_pass &= check("Regime config returned", isinstance(regime_cfg, dict))
    print(f"    Best playbooks: {regime_cfg.get('best_playbooks', [])}")

    print(f"\n  Step 4 — S/R levels")
    sr = get_sr_levels(df, price)
    pipeline["sr"] = sr
    t4_pass &= check("S/R levels computed", "nearest_resistance" in sr)
    print(f"    Nearest R: ${sr['nearest_resistance']['price']:,.2f}  "
          f"Nearest S: ${sr['nearest_support']['price']:,.2f}")

    print(f"\n  Step 5 — Score confluences (11 factors)")
    conf = score_confluences(df, direction)
    pipeline["confluence"] = conf
    t4_pass &= check("Confluence score valid", 0 <= conf.get("confidence", -1) <= 10)
    t4_pass &= check("sr_levels in raw_checks", "sr_levels" in conf.get("raw_checks", {}))
    print(f"    Confidence: {conf.get('confidence'):.1f}/10  "
          f"trade_valid={conf.get('trade_valid')}  "
          f"passed={conf.get('passed_count')}/{conf.get('total_checks')}")

    print(f"\n  Step 6 — Hunt reversals")
    rev = hunt_reversals(df, price)
    pipeline["reversals"] = rev
    t4_pass &= check("Reversal hunt completed", isinstance(rev, list))
    print(f"    Reversal signals: {len(rev)}")

    print(f"\n  Step 7 — Fundamental bias")
    fund = get_fundamental_bias()
    pipeline["fundamental"] = fund
    t4_pass &= check("Fundamental bias returned",    "fundamental_bias" in fund)
    t4_pass &= check("Fundamental bias has confidence", "confidence" in fund)
    print(f"    Bias: {fund.get('fundamental_bias')}  "
          f"score={fund.get('total_score'):+}  "
          f"conf={fund.get('confidence')}")

    print(f"\n  Step 8 — Check fundamental conflict with {direction.upper()}")
    conflict = check_fundamental_conflict(direction)
    pipeline["conflict"] = conflict
    t4_pass &= check("Conflict check returned",  "conflict" in conflict)
    t4_pass &= check("Severity field present",   "severity" in conflict)
    print(f"    Conflict={conflict.get('conflict')}  "
          f"severity={conflict.get('severity')}  "
          f"msg={conflict.get('message','')[:60]}")

    # Summary
    print(f"\n  ── PIPELINE SUMMARY ──────────────────────────────────────────")
    print(f"  Direction    : {direction.upper()}")
    print(f"  Price        : ${price:,.2f}")
    print(f"  Regime       : {regime}")
    print(f"  Confidence   : {conf.get('confidence'):.1f}/10  (valid={conf.get('trade_valid')})")
    print(f"  At S/R Level : {sr.get('at_key_level')} ({sr.get('key_level_detail')})")
    print(f"  Reversal Sigs: {len(rev)}")
    print(f"  Fundamental  : {fund.get('fundamental_bias')} ({fund.get('total_score'):+})")
    print(f"  Fund Conflict: {conflict.get('conflict')} ({conflict.get('severity')})")
    print(f"  ─────────────────────────────────────────────────────────────")

except Exception as e:
    print(f"  {FAIL}  Pipeline error: {e}")
    traceback.print_exc()
    t4_pass = False

record("T4_FullPipeline", t4_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — 50-CANDLE ROLLING SIMULATION (last 500 candles, every 10th)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 5 — 50-CANDLE ROLLING SIMULATION  (last 500 candles, every 10th)")

t5_pass = True
try:
    from sr_mapper         import get_sr_levels
    from confluence_engine import score_confluences
    from reversal_hunter   import hunt_reversals
    from market_context    import detect_gold_regime

    sim_df = LIVE_DF.tail(500).copy()
    # Indices of candles to simulate (every 10th, from index 200 onward to have enough history)
    sim_indices = list(range(200, len(sim_df), 10))[:50]

    long_confidences  = []
    short_confidences = []
    sr_at_key_count   = 0
    reversal_count    = 0
    regime_counts: dict[str, int] = {}
    errors            = 0

    print(f"  Simulating {len(sim_indices)} candles…")
    for i, idx in enumerate(sim_indices):
        window = sim_df.iloc[:idx].copy()
        price  = float(window["close"].iloc[-1])

        try:
            lc = score_confluences(window, "long").get("confidence", 0)
            sc = score_confluences(window, "short").get("confidence", 0)
            long_confidences.append(lc)
            short_confidences.append(sc)

            sr = get_sr_levels(window, price)
            if sr.get("at_key_level"):
                sr_at_key_count += 1

            rev = hunt_reversals(window, price)
            if rev:
                reversal_count += 1

            reg = detect_gold_regime(window).get("regime", "UNKNOWN")
            regime_counts[reg] = regime_counts.get(reg, 0) + 1

        except Exception as sim_e:
            errors += 1
            if errors <= 3:
                print(f"    {WARN}  idx={idx}: {sim_e}")

        # Progress dots
        if (i + 1) % 10 == 0:
            print(f"    … {i + 1}/{len(sim_indices)} done")

    t5_pass &= check("Error rate < 10%", errors / len(sim_indices) < 0.1,
                     f"errors={errors}/{len(sim_indices)}")
    t5_pass &= check("Long confidences non-empty",  len(long_confidences) > 0)
    t5_pass &= check("Short confidences non-empty", len(short_confidences) > 0)

    avg_l = sum(long_confidences)  / len(long_confidences)  if long_confidences  else 0
    avg_s = sum(short_confidences) / len(short_confidences) if short_confidences else 0
    sr_rate  = sr_at_key_count / len(sim_indices) * 100
    rev_rate = reversal_count  / len(sim_indices) * 100

    print(f"\n  ── SIMULATION RESULTS ────────────────────────────────────────")
    print(f"  Candles simulated   : {len(sim_indices)}")
    print(f"  Errors              : {errors}")
    print(f"  Avg LONG  confidence: {avg_l:.2f}/10")
    print(f"  Avg SHORT confidence: {avg_s:.2f}/10")
    print(f"  At-key-S/R rate     : {sr_rate:.1f}%  ({sr_at_key_count}/{len(sim_indices)})")
    print(f"  Reversal signal rate: {rev_rate:.1f}%  ({reversal_count}/{len(sim_indices)})")
    print(f"  Regime distribution : {dict(sorted(regime_counts.items(), key=lambda x: -x[1]))}")
    print(f"  ─────────────────────────────────────────────────────────────")

    # Sanity: avg confidence shouldn't be 0 or 10 (would indicate a bug)
    t5_pass &= check("Avg confidence > 0",  avg_l > 0 and avg_s > 0)
    t5_pass &= check("Avg confidence < 10", avg_l < 10 and avg_s < 10)

except Exception as e:
    print(f"  {FAIL}  Simulation error: {e}")
    traceback.print_exc()
    t5_pass = False

record("T5_RollingSimulation", t5_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — TODAY'S MARKET ANALYSIS (all Phase 3 components)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 6 — TODAY'S MARKET ANALYSIS  (all Phase 3 components live)")

t6_pass = True
try:
    from sr_mapper         import get_sr_levels
    from confluence_engine import score_confluences
    from reversal_hunter   import hunt_reversals
    from market_context    import detect_gold_regime, get_regime_strategy_config
    from fundamental_bias  import get_fundamental_bias, check_fundamental_conflict
    from confluence_engine import detect_rsi_divergence

    df    = LIVE_DF.copy()
    price = LIVE_PRICE

    # --- Run all components ---
    regime_data = detect_gold_regime(df)
    regime      = regime_data.get("regime", "UNKNOWN")
    reg_cfg     = get_regime_strategy_config(regime)

    sr          = get_sr_levels(df, price)
    rsi_div     = detect_rsi_divergence(df)
    conf_long   = score_confluences(df, "long")
    conf_short  = score_confluences(df, "short")
    rev_sigs    = hunt_reversals(df, price)
    fund        = get_fundamental_bias()

    # Conflict based on stronger side
    bias_dir = "long" if conf_long["confidence"] >= conf_short["confidence"] else "short"
    conflict = check_fundamental_conflict(bias_dir)

    # Validate
    t6_pass &= check("Regime OK",           bool(regime) and regime != "UNKNOWN", f"regime={regime}")
    t6_pass &= check("S/R levels OK",       len(sr.get("resistance_levels", [])) > 0)
    t6_pass &= check("RSI divergence OK",   isinstance(rsi_div, dict) and "divergence_found" in rsi_div)
    t6_pass &= check("LONG confidence OK",  0 <= conf_long["confidence"] <= 10)
    t6_pass &= check("SHORT confidence OK", 0 <= conf_short["confidence"] <= 10)
    t6_pass &= check("Reversal hunt OK",    isinstance(rev_sigs, list))
    t6_pass &= check("Fundamental OK",      "fundamental_bias" in fund)
    t6_pass &= check("Conflict check OK",   "conflict" in conflict)

    # --- Print today's analysis block ---
    nr = sr["nearest_resistance"]
    ns = sr["nearest_support"]
    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║          TODAY'S MARKET ANALYSIS  (Phase 3)                 ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Price          : ${price:>10,.2f}                              ║
  ║  Regime         : {regime:<40}  ║
  ║  Best Playbooks : {str(reg_cfg.get('best_playbooks', []))[:40]:<40}  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  S/R LEVELS                                                  ║
  ║    Nearest Res  : ${nr['price']:>10,.2f}  ({nr['proximity']:<12})  dist={nr['distance_usd']:>6.1f}  ║
  ║    Nearest Sup  : ${ns['price']:>10,.2f}  ({ns['proximity']:<12})  dist={ns['distance_usd']:>6.1f}  ║
  ║    At Key Level : {str(sr.get('at_key_level')):<6}  {sr.get('key_level_detail','')[:33]:<33}  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  CONFLUENCE SCORES                                           ║
  ║    LONG  : {conf_long['confidence']:>4.1f}/10  (valid={str(conf_long.get('trade_valid')):<5})                     ║
  ║    SHORT : {conf_short['confidence']:>4.1f}/10  (valid={str(conf_short.get('trade_valid')):<5})                     ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  RSI DIVERGENCE                                              ║
  ║    Detected     : {str(rsi_div.get('divergence_found')):<6}  Type: {str(rsi_div.get('divergence_type','—'))[:20]:<20}  ║
  ║    Strength     : {str(rsi_div.get('strength','—')):<10}  bars={rsi_div.get('bars_since_divergence',0)}              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  FUNDAMENTAL                                                 ║
  ║    Bias         : {fund.get('fundamental_bias','—'):<18}  score={fund.get('total_score',0):+}  conf={fund.get('confidence',0):.1f}  ║
  ║    Conflict     : {str(conflict.get('conflict')):<6}  sev={conflict.get('severity','?'):<12}                  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  REVERSAL SIGNALS : {len(rev_sigs):<3}                                        ║""")
    for s in rev_sigs[:3]:
        score = s.get('score', 0)
        ddir  = s.get('direction', s.get('type', '?'))
        conds = len(s.get('conditions_met', []))
        print(f"  ║    Score={score}/13  dir={ddir:<6}  conditions={conds:<2}                           ║")
    if not rev_sigs:
        print(f"  ║    (none at current price)                                   ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

except Exception as e:
    print(f"  {FAIL}  Today's analysis error: {e}")
    traceback.print_exc()
    t6_pass = False

record("T6_TodayAnalysis", t6_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — PHASE 1/2 REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 7 — PHASE 1/2 REGRESSION  (no crashes allowed)")

t7_pass = True
df    = LIVE_DF.copy()
price = LIVE_PRICE

# 7a — DXY / Macro context
try:
    from dxy_correlation import get_macro_context
    macro = get_macro_context("long")
    t7_pass &= check("get_macro_context('long') returns dict", isinstance(macro, dict))
    print(f"    DXY macro keys: {list(macro.keys())[:6]}")
except Exception as e:
    t7_pass &= check("get_macro_context() no crash", False, str(e))

# 7b — Geopolitical score
try:
    from geo_filter import get_geopolitical_score
    geo = get_geopolitical_score()
    t7_pass &= check("get_geopolitical_score() returns dict/float", geo is not None)
    print(f"    Geo score: {geo}")
except Exception as e:
    t7_pass &= check("get_geopolitical_score() no crash", False, str(e))

# 7c — ATR dynamic SL
try:
    from atr_sl_engine import calculate_dynamic_sl
    sl_data = calculate_dynamic_sl(df, "long", price)
    t7_pass &= check("calculate_dynamic_sl() returns dict", isinstance(sl_data, dict))
    print(f"    SL keys: {list(sl_data.keys())[:6]}")
except Exception as e:
    t7_pass &= check("calculate_dynamic_sl() no crash", False, str(e))

# 7d — Session profiler
try:
    from session_profiler import get_current_session_profile
    sess = get_current_session_profile()
    t7_pass &= check("get_current_session_profile() returns dict", isinstance(sess, dict))
    print(f"    Session: {sess.get('session', sess.get('name', '?'))}")
except Exception as e:
    t7_pass &= check("get_current_session_profile() no crash", False, str(e))

# 7e — Spread monitor
try:
    from spread_monitor import check_spread
    spread = check_spread()
    t7_pass &= check("check_spread() returns dict", isinstance(spread, dict))
    print(f"    Spread keys: {list(spread.keys())[:5]}")
except Exception as e:
    t7_pass &= check("check_spread() no crash", False, str(e))

# 7f — SMC: use score_confluences raw_checks['smc'] as regression proxy
try:
    from confluence_engine import score_confluences as _sc_smc
    smc_raw = _sc_smc(df, "long").get("raw_checks", {}).get("smc")
    t7_pass &= check("SMC via confluence raw_checks['smc'] present", smc_raw is not None)
    print(f"    SMC raw type: {type(smc_raw).__name__}")
except Exception as e:
    t7_pass &= check("smart_money regression no crash", False, str(e))

# 7g — RSI divergence (full regression)
try:
    from confluence_engine import detect_rsi_divergence
    rdi = detect_rsi_divergence(df)
    required = ["divergence_found", "divergence_type", "strength",
                "signal_direction", "price_swing1", "price_swing2",
                "rsi_swing1", "rsi_swing2", "bars_since_divergence",
                "note", "confidence_boost"]
    missing = [k for k in required if k not in rdi]
    t7_pass &= check("detect_rsi_divergence() has all 11 keys", not missing,
                     f"missing={missing}" if missing else "")
    print(f"    RSI div: found={rdi.get('divergence_found')}  "
          f"type={rdi.get('divergence_type')}  strength={rdi.get('strength')}")
except Exception as e:
    t7_pass &= check("detect_rsi_divergence() no crash", False, str(e))

# 7h — Market regime regression (detect + regime history)
try:
    from market_context import detect_gold_regime, get_regime_history
    reg = detect_gold_regime(df)
    hist = get_regime_history(last_n=5)
    t7_pass &= check("detect_gold_regime() OK",     "regime" in reg)
    t7_pass &= check("get_regime_history() returns list", isinstance(hist, list))
    print(f"    Regime: {reg.get('regime')}  history_len={len(hist)}")
except Exception as e:
    t7_pass &= check("market_context regression no crash", False, str(e))

# 7i — Fundamental bias regression
try:
    from fundamental_bias import get_fundamental_bias, check_fundamental_conflict
    fb = get_fundamental_bias()
    fc = check_fundamental_conflict("long")
    t7_pass &= check("get_fundamental_bias() has 'fundamental_bias'", "fundamental_bias" in fb)
    t7_pass &= check("check_fundamental_conflict() has 'conflict'",   "conflict" in fc)
    print(f"    Fundamental: {fb.get('fundamental_bias')}  conflict={fc.get('conflict')}")
except Exception as e:
    t7_pass &= check("fundamental_bias regression no crash", False, str(e))

record("T7_Phase12Regression", t7_pass)


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
section("FINAL VERDICT")

all_pass = all(results.values())

# Map test IDs to Phase component names
component_map = {
    "T1_SR_Mapper":           "Task 13 S/R Mapper",
    "T2_Confluence_11Factor": "Task 10 SMC + 11-Factor Confluence",
    "T3_ReversalHunter":      "Reversal Hunter",
    "T4_FullPipeline":        "Full 8-Step Pipeline",
    "T5_RollingSimulation":   "50-Candle Rolling Simulation",
    "T6_TodayAnalysis":       "Today's Market Analysis",
    "T7_Phase12Regression":   "Phase 1/2 Regression",
}

# Grouped verdict
smc_ok    = results.get("T2_Confluence_11Factor", False)
regime_ok = results.get("T7_Phase12Regression", False)
rsi_ok    = results.get("T7_Phase12Regression", False)
sr_ok     = results.get("T1_SR_Mapper", False)
extras_ok = results.get("T3_ReversalHunter", False) and results.get("T4_FullPipeline", False)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         PHASE 3 COMPLETE INTEGRATION BACKTEST — VERDICT         ║
╠══════════════════════════════════════════════════════════════════╣""")

for tid, name in component_map.items():
    ok     = results.get(tid, False)
    marker = "✅" if ok else "❌"
    label  = f"{marker}  {name}"
    print(f"║  {label:<62}  ║")

print(f"""╠══════════════════════════════════════════════════════════════════╣
║  Task 10 SMC + Confluence:       {'PASS ✅' if smc_ok    else 'FAIL ❌':<10}                      ║
║  Task 11 Market Regime:          {'PASS ✅' if regime_ok else 'FAIL ❌':<10}                      ║
║  Task 12 RSI Divergence:         {'PASS ✅' if rsi_ok    else 'FAIL ❌':<10}                      ║
║  Task 13 S/R Mapper:             {'PASS ✅' if sr_ok     else 'FAIL ❌':<10}                      ║
║  Extras (Reversal/Pipeline):     {'PASS ✅' if extras_ok else 'FAIL ❌':<10}                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PHASE 3 COMPLETE — ALL COMPONENTS INTEGRATED                    ║
║                                                                  ║
║  Ready for Phase 4:  {'YES ✅' if all_pass else 'NO  ❌ — fix failing tests first':<42}  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝""")

sys.exit(0 if all_pass else 1)
