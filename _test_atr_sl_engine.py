"""
_test_atr_sl_engine.py — Backtest verification for atr_sl_engine.py
─────────────────────────────────────────────────────────────────────
Phase 2 Task 8 test suite — DO NOT modify any production code.
Run: python _test_atr_sl_engine.py
"""

from __future__ import annotations
import sys
import math
import traceback
import pandas as pd
import numpy as np

# ── ANSI colours (ASCII-safe fallback for Windows cp1252) ──────────────────
PASS  = "[PASS]"
FAIL  = "[FAIL]"
INFO  = "[INFO]"
HEAD  = "=" * 64

TOTAL_PASS = 0
TOTAL_FAIL = 0
ISSUES: list[str] = []

# ── helpers ────────────────────────────────────────────────────────────────
def _ok(label: str) -> None:
    global TOTAL_PASS
    TOTAL_PASS += 1
    print(f"  {PASS}  {label}")

def _fail(label: str, detail: str = "") -> None:
    global TOTAL_FAIL
    TOTAL_FAIL += 1
    msg = f"  {FAIL}  {label}" + (f"  ({detail})" if detail else "")
    print(msg)
    ISSUES.append(label + (f" — {detail}" if detail else ""))

def _check(cond: bool, label: str, detail: str = "") -> bool:
    if cond:
        _ok(label)
    else:
        _fail(label, detail)
    return cond

# ── Load real data ─────────────────────────────────────────────────────────
def _load_df(tail: int | None = None) -> pd.DataFrame:
    df = pd.read_csv("data/historical_xauusd.csv")
    df.columns = [c.lower() for c in df.columns]
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    if tail:
        df = df.tail(tail).reset_index(drop=True)
    return df

# ── Import engine ──────────────────────────────────────────────────────────
try:
    from atr_sl_engine import calculate_dynamic_sl
    _ENGINE_OK = True
except Exception as e:
    print(f"{FAIL} Cannot import atr_sl_engine: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Unit test calculate_dynamic_sl()
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 1 — Unit test: calculate_dynamic_sl() key presence & sanity")
print(HEAD)

REQUIRED_KEYS = [
    "sl_price", "tp1_price", "tp2_price", "sl_distance", "atr_value",
    "atr_percentile", "volatility_state", "session_multiplier",
    "regime_multiplier", "vol_multiplier", "geo_buffer",
    "final_multiplier", "sl_breakdown", "quality", "rr_at_tp1", "rr_at_tp2",
]

try:
    df100 = _load_df(100)
    ENTRY = 3300.00
    r = calculate_dynamic_sl(
        df100, "long", entry=ENTRY,
        session="London", regime="RANGING",
        geo_multiplier=0.0, strategy_name="EMA Trend Continuation",
    )

    print(f"\n  {INFO} Returned values:")
    for k, v in r.items():
        print(f"       {k:24s} = {v}")

    print()
    for key in REQUIRED_KEYS:
        _check(key in r, f"key '{key}' present", "MISSING" if key not in r else "")

    _check(r["sl_price"]  < ENTRY,          "sl_price < entry (long)")
    _check(r["tp1_price"] > ENTRY,          "tp1_price > entry (long)")
    _check(r["tp2_price"] > r["tp1_price"], "tp2_price > tp1_price (long)")
    _check(
        r["sl_distance"] >= r["atr_value"] * 0.5,
        "sl_distance >= 0.5x ATR",
        f"sl_dist={r['sl_distance']:.4f}  atr={r['atr_value']:.4f}",
    )
    _check(
        r["sl_distance"] <= r["atr_value"] * 3.0,
        "sl_distance <= 3.0x ATR",
        f"sl_dist={r['sl_distance']:.4f}  atr={r['atr_value']:.4f}",
    )
    _check(
        r["rr_at_tp2"] >= 2.5,
        f"rr_at_tp2 >= 2.5  (actual={r['rr_at_tp2']})",
    )
except Exception:
    _fail("TEST 1 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — Session multiplier test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 2 — Session multiplier verification")
print(HEAD)

SESSION_EXPECTED: list[tuple[str, float]] = [
    ("London",    1.5),
    ("Asian",     2.0),
    ("Off-Hours", 2.5),
    ("NewYork",   1.5),
    ("Overlap",   1.5),
]

try:
    df100 = _load_df(100)
    for sess, exp in SESSION_EXPECTED:
        res = calculate_dynamic_sl(df100, "long", entry=3300.0, session=sess,
                                   regime="RANGING", geo_multiplier=0.0)
        got = res["session_multiplier"]
        _check(
            math.isclose(got, exp, rel_tol=1e-6),
            f"session={sess:10s}  expected={exp}  got={got}",
        )
except Exception:
    _fail("TEST 2 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Regime multiplier test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 3 — Regime multiplier verification")
print(HEAD)

REGIME_EXPECTED: list[tuple[str, float]] = [
    ("TRENDING_STRONG",    0.9),
    ("TRENDING_WEAK",      1.0),
    ("RANGING",            1.2),
    ("VOLATILE_EXPANDING", 1.5),
    ("SQUEEZE_BUILDING",   0.8),
]

try:
    df100 = _load_df(100)
    for regime, exp in REGIME_EXPECTED:
        res = calculate_dynamic_sl(df100, "long", entry=3300.0, session="London",
                                   regime=regime, geo_multiplier=0.0)
        got = res["regime_multiplier"]
        _check(
            math.isclose(got, exp, rel_tol=1e-6),
            f"regime={regime:22s}  expected={exp}  got={got}",
        )
except Exception:
    _fail("TEST 3 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Hard cap test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 4 — Hard cap enforcement (max 3.0x ATR, min 0.5x ATR)")
print(HEAD)

try:
    df_full = _load_df()

    # Scenario A: force high-vol environment via synthetic df
    n = 200
    np.random.seed(7)
    close_a = 3300.0 + np.cumsum(np.random.randn(n) * 25)   # very noisy
    high_a  = close_a + np.abs(np.random.randn(n) * 15)
    low_a   = close_a - np.abs(np.random.randn(n) * 15)
    df_a    = pd.DataFrame({"high": high_a, "low": low_a, "close": close_a,
                            "open": close_a, "volume": np.ones(n) * 1000})
    hl = df_a["high"] - df_a["low"];  hc = (df_a["high"] - df_a["close"].shift()).abs()
    lc = (df_a["low"] - df_a["close"].shift()).abs()
    df_a["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    res_a = calculate_dynamic_sl(df_a, "long", entry=3300.0,
                                  session="Asian", regime="VOLATILE_EXPANDING",
                                  geo_multiplier=1.5)
    atr_a = res_a["atr_value"]
    cap_a = round(atr_a * 3.0, 4)
    print(f"\n  Scenario A (Asian+VOLATILE+geo1.5):  sl_dist={res_a['sl_distance']:.4f}  "
          f"cap={cap_a:.4f}  atr={atr_a:.4f}  vol={res_a['volatility_state']}")
    _check(
        res_a["sl_distance"] <= cap_a + 0.01,
        f"Scenario A: sl_distance capped at 3.0x ATR (sl={res_a['sl_distance']:.4f} cap={cap_a:.4f})",
    )

    # Scenario B: force low-vol environment + squeeze
    np.random.seed(11)
    close_b = 3300.0 + np.cumsum(np.random.randn(n) * 0.5)  # very quiet
    high_b  = close_b + np.abs(np.random.randn(n) * 0.3)
    low_b   = close_b - np.abs(np.random.randn(n) * 0.3)
    df_b    = pd.DataFrame({"high": high_b, "low": low_b, "close": close_b,
                            "open": close_b, "volume": np.ones(n) * 1000})
    hl = df_b["high"] - df_b["low"]; hc = (df_b["high"] - df_b["close"].shift()).abs()
    lc = (df_b["low"] - df_b["close"].shift()).abs()
    df_b["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    res_b = calculate_dynamic_sl(df_b, "long", entry=3300.0,
                                  session="London", regime="SQUEEZE_BUILDING",
                                  geo_multiplier=0.0)
    atr_b   = res_b["atr_value"]
    floor_b = round(atr_b * 0.5, 4)
    print(f"  Scenario B (London+SQUEEZE+no geo):  sl_dist={res_b['sl_distance']:.4f}  "
          f"floor={floor_b:.4f}  atr={atr_b:.4f}  vol={res_b['volatility_state']}")
    _check(
        res_b["sl_distance"] >= floor_b - 0.01,
        f"Scenario B: sl_distance floored at 0.5x ATR (sl={res_b['sl_distance']:.4f} floor={floor_b:.4f})",
    )
except Exception:
    _fail("TEST 4 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Direction SHORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 5 — Direction SHORT sanity check")
print(HEAD)

try:
    df100 = _load_df(100)
    rs = calculate_dynamic_sl(df100, "short", entry=3300.0,
                               session="London", regime="RANGING",
                               geo_multiplier=0.0)
    print(f"\n  sl_price={rs['sl_price']}  tp1={rs['tp1_price']}  tp2={rs['tp2_price']}")
    _check(rs["sl_price"]  > 3300.0,          "sl_price > entry (short)")
    _check(rs["tp1_price"] < 3300.0,          "tp1_price < entry (short)")
    _check(rs["tp2_price"] < rs["tp1_price"], "tp2_price < tp1_price (short)")
except Exception:
    _fail("TEST 5 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Geo buffer test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 6 — Geo buffer additive accuracy")
print(HEAD)

try:
    df100 = _load_df(100)
    r0   = calculate_dynamic_sl(df100, "long", entry=3300.0, session="London",
                                 regime="RANGING", geo_multiplier=0.0)
    r1   = calculate_dynamic_sl(df100, "long", entry=3300.0, session="London",
                                 regime="RANGING", geo_multiplier=1.0)
    r15  = calculate_dynamic_sl(df100, "long", entry=3300.0, session="London",
                                 regime="RANGING", geo_multiplier=1.5)
    atr  = r0["atr_value"]

    print(f"\n  ATR={atr:.4f}")
    print(f"  geo=0.0 → geo_buffer={r0['geo_buffer']:.4f}  sl_dist={r0['sl_distance']:.4f}")
    print(f"  geo=1.0 → geo_buffer={r1['geo_buffer']:.4f}  sl_dist={r1['sl_distance']:.4f}")
    print(f"  geo=1.5 → geo_buffer={r15['geo_buffer']:.4f}  sl_dist={r15['sl_distance']:.4f}")

    _check(math.isclose(r0["geo_buffer"], 0.0, abs_tol=1e-6),
           "geo=0.0 → geo_buffer is 0")

    exp1 = 1.0 * atr
    _check(math.isclose(r1["geo_buffer"], exp1, rel_tol=1e-4),
           f"geo=1.0 → geo_buffer={r1['geo_buffer']:.4f} == 1.0×ATR={exp1:.4f}")

    exp15 = 1.5 * atr
    _check(math.isclose(r15["geo_buffer"], exp15, rel_tol=1e-4),
           f"geo=1.5 → geo_buffer={r15['geo_buffer']:.4f} == 1.5×ATR={exp15:.4f}")

    # Verify additive contribution to sl_distance (before cap)
    # base dist for geo=0 is base_sl; for geo=1 should be base_sl + 1×atr
    # after capping, may be equal if both hit cap; check uncapped scenario
    base_dist = r0["sl_distance"]
    expected_dist1 = min(base_dist + 1.0 * atr, 3.0 * atr)
    _check(
        math.isclose(r1["sl_distance"], expected_dist1, rel_tol=1e-4),
        f"geo=1.0 sl_dist={r1['sl_distance']:.4f} == expected={expected_dist1:.4f} (after cap)",
    )
except Exception:
    _fail("TEST 6 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Strategy minimum floor
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 7 — Strategy minimum floor enforcement")
print(HEAD)

STRAT_FLOORS: list[tuple[str, float]] = [
    ("News Spike Fade",       1.5),
    ("London Breakout",       1.0),
    ("Unknown Strategy",      0.8),
]

try:
    # Use low-vol synthetic df + squeeze to try to push below floor
    np.random.seed(22)
    n = 200
    close_s = 3300.0 + np.cumsum(np.random.randn(n) * 0.4)
    high_s  = close_s + np.abs(np.random.randn(n) * 0.25)
    low_s   = close_s - np.abs(np.random.randn(n) * 0.25)
    df_s    = pd.DataFrame({"high": high_s, "low": low_s, "close": close_s,
                            "open": close_s, "volume": np.ones(n) * 1000})
    hl = df_s["high"] - df_s["low"]; hc = (df_s["high"] - df_s["close"].shift()).abs()
    lc = (df_s["low"] - df_s["close"].shift()).abs()
    df_s["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    for strat, floor_mult in STRAT_FLOORS:
        rs = calculate_dynamic_sl(df_s, "long", entry=3300.0,
                                   session="London", regime="SQUEEZE_BUILDING",
                                   geo_multiplier=0.0, strategy_name=strat)
        atr_s       = rs["atr_value"]
        floor_dist  = floor_mult * atr_s
        # Raw multiplier before floor: 1.5 × 0.8 × 0.9 = 1.08 for low vol
        # After hard min cap (0.5x), floor must also hold
        effective_floor = max(floor_dist, 0.5 * atr_s)
        print(f"\n  strategy='{strat}': floor={floor_mult}x  "
              f"atr={atr_s:.4f}  sl_dist={rs['sl_distance']:.4f}  "
              f"final_mult={rs['final_multiplier']:.3f}")
        _check(
            rs["sl_distance"] >= effective_floor - 0.01,
            f"'{strat}': sl_dist={rs['sl_distance']:.4f} >= floor={effective_floor:.4f}",
        )
except Exception:
    _fail("TEST 7 crashed", traceback.format_exc().splitlines()[-1])

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 8 — Historical backtest simulation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 8 — Historical backtest simulation (every 10th candle)")
print(HEAD)

try:
    df_hist = _load_df()
    total_rows  = len(df_hist)
    indices     = list(range(60, total_rows, 10))   # need 60+ rows for ATR window

    sl_dists:        list[float] = []
    sl_mults:        list[float] = []
    vol_states:      dict[str, list[float]] = {"high_volatility": [], "low_volatility": [], "normal_volatility": []}
    capped_max_cnt  = 0
    capped_min_cnt  = 0
    static_dists:   list[float] = []

    # Estimate session from hour (if datetime column exists)
    has_dt = "datetime" in df_hist.columns or "date" in df_hist.columns
    dt_col = "datetime" if "datetime" in df_hist.columns else ("date" if "date" in df_hist.columns else None)

    SESSION_FROM_HOUR = {
        range(0, 3):   "Asian",
        range(3, 8):   "London",
        range(8, 13):  "Overlap",
        range(13, 17): "NewYork",
        range(17, 24): "Off-Hours",
    }
    session_sl: dict[str, list[float]] = {"Asian": [], "London": [], "Overlap": [], "NewYork": [], "Off-Hours": []}

    def _session_from_hour(h: int) -> str:
        for rng, s in SESSION_FROM_HOUR.items():
            if h in rng:
                return s
        return "London"

    for idx in indices:
        slice_df = df_hist.iloc[:idx + 1].copy()
        row      = df_hist.iloc[idx]
        entry_p  = float(row["close"])
        atr_val  = float(slice_df["atr"].dropna().iloc[-1]) if not slice_df["atr"].dropna().empty else entry_p * 0.005

        # Estimate session
        session = "London"
        if dt_col and pd.notna(row[dt_col]):
            try:
                hour = pd.to_datetime(str(row[dt_col])).hour
                session = _session_from_hour(hour)
            except Exception:
                pass

        for direction in ("long", "short"):
            try:
                res = calculate_dynamic_sl(slice_df, direction, entry=entry_p,
                                           session=session, regime="RANGING",
                                           geo_multiplier=0.0)
                sd   = res["sl_distance"]
                atr  = res["atr_value"]
                mult = res["final_multiplier"]
                vstate = res["volatility_state"]

                sl_dists.append(sd)
                sl_mults.append(mult)
                if vstate in vol_states:
                    vol_states[vstate].append(sd)
                session_sl.setdefault(session, []).append(sd)

                if math.isclose(mult, 3.0, rel_tol=0.01) or sd >= atr * 2.99:
                    capped_max_cnt += 1
                if math.isclose(mult, 0.5, rel_tol=0.01) or sd <= atr * 0.51:
                    capped_min_cnt += 1

                static_dists.append(atr * 1.5)
            except Exception:
                pass

    total_obs    = len(sl_dists)
    avg_dyn      = float(np.mean(sl_dists)) if sl_dists else 0.0
    avg_static   = float(np.mean(static_dists)) if static_dists else 0.0
    pct_max_cap  = 100.0 * capped_max_cnt / max(total_obs, 1)
    pct_min_cap  = 100.0 * capped_min_cnt / max(total_obs, 1)
    diff_pct     = 100.0 * (avg_dyn - avg_static) / max(avg_static, 1e-9)
    direction_lbl= "wider" if avg_dyn >= avg_static else "tighter"

    print(f"\n  Total observations   : {total_obs}")
    print(f"  Dynamic SL avg       : ${avg_dyn:.2f}")
    print(f"  Static  SL avg       : ${avg_static:.2f}  (1.5x ATR always)")
    print(f"  Dynamic is {direction_lbl} by {abs(diff_pct):.1f}% on average")
    print(f"  % at max cap (3.0x)  : {pct_max_cap:.1f}%")
    print(f"  % at min floor (0.5x): {pct_min_cap:.1f}%")

    print(f"\n  Avg SL by volatility state:")
    for vs, vals in vol_states.items():
        if vals:
            print(f"    {vs:22s}: ${float(np.mean(vals)):.2f}  (n={len(vals)})")

    print(f"\n  Avg SL by estimated session:")
    for sess in ("Asian", "London", "Overlap", "NewYork", "Off-Hours"):
        vals = session_sl.get(sess, [])
        if vals:
            asian_avg  = float(np.mean(session_sl.get("Asian", [1]) or [1]))
            london_avg = float(np.mean(session_sl.get("London", [1]) or [1]))
            print(f"    {sess:12s}: ${float(np.mean(vals)):.2f}  (n={len(vals)})")

    if session_sl.get("Asian") and session_sl.get("London"):
        asian_avg  = float(np.mean(session_sl["Asian"]))
        london_avg = float(np.mean(session_sl["London"]))
        print(f"\n  Dynamic adapts: Asian SL ${asian_avg:.2f} vs London SL ${london_avg:.2f}")
        _check(asian_avg > london_avg,
               f"Asian SL (${asian_avg:.2f}) > London SL (${london_avg:.2f}) — session adapts correctly")

    _check(total_obs > 0, f"Simulation ran with {total_obs} observations")
    _check(pct_max_cap < 30.0, f"< 30% of SLs at max cap (actual={pct_max_cap:.1f}%)")
    _check(pct_min_cap < 30.0, f"< 30% of SLs at min floor (actual={pct_min_cap:.1f}%)")

except Exception:
    _fail("TEST 8 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 9 — strategy_playbooks integration test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 9 — strategy_playbooks integration (london_breakout_gold)")
print(HEAD)

try:
    from strategy_playbooks import format_playbook_signal
    df100 = _load_df(100)
    atr_val = float(df100["atr"].dropna().iloc[-1])
    entry_p = float(df100["close"].iloc[-1])

    # Minimal playbook dict for london_breakout_gold
    pb = {
        "name":       "london_breakout_gold",
        "direction":  "long",
        "entry":      entry_p,
        "stop_loss":  entry_p - atr_val * 1.5,
        "take_profit":entry_p + atr_val * 3.0,
        "confidence_score": 7.0,
        "reason":     "Breakout above Asian high",
    }

    result = format_playbook_signal(pb, df100)

    # Returns tuple (entry_price, sl_price, tp_price); sl_breakdown stored as playbook side-effect
    _check(isinstance(result, tuple) and len(result) == 3,
           f"format_playbook_signal returns 3-tuple: {result}")

    ret_entry, sl_price, tp_price = result
    sl_dist = abs(ret_entry - float(sl_price)) if sl_price else None

    print(f"\n  Returned tuple: entry={ret_entry}  sl={sl_price}  tp={tp_price}")
    print(f"  playbook side-effect keys: {[k for k in pb if k.startswith('_')]}")

    _check("_sl_breakdown" in pb, "playbook has '_sl_breakdown' side-effect key")
    _check(sl_price is not None and sl_price != 0.0, "sl_price is non-zero")

    breakdown = pb.get("_sl_breakdown", "")
    _check(
        isinstance(breakdown, str) and len(breakdown) > 5,
        f"sl_breakdown is descriptive string: '{breakdown[:80]}'",
    )

    if sl_dist is not None and sl_dist > 0:
        _check(
            sl_dist >= atr_val * 0.5,
            f"sl_distance={sl_dist:.4f} >= 0.5x ATR={atr_val * 0.5:.4f}",
        )
        _check(
            sl_dist <= atr_val * 3.0 + 0.01,
            f"sl_distance={sl_dist:.4f} <= 3.0x ATR={atr_val * 3.0:.4f}",
        )
    else:
        _fail("Could not determine sl_distance from signal")
except Exception:
    _fail("TEST 9 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 10 — Entry checklist CHECK F
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("TEST 10 — entry_checklist CHECK F (dynamic ATR advisory)")
print(HEAD)

try:
    from entry_checklist import sl_quality_check
    df100 = _load_df(100)
    atr_val = float(df100["atr"].dropna().iloc[-1])
    entry_p = 3300.0

    print(f"\n  ATR (from real data): ${atr_val:.4f}")

    # ── Scenario A: SL much tighter than recommendation (only $2 away) ──────
    tight_sl = entry_p - 2.0   # $2 SL, ATR is ~$15
    res_a = sl_quality_check(df100, entry_p, tight_sl, "long")

    print(f"\n  Scenario A — tight SL (entry={entry_p}, sl={tight_sl}):")
    print(f"    checks: {list(res_a.get('checks', {}).keys())}")
    print(f"    warnings: {res_a.get('warnings', [])}")
    print(f"    passed: {res_a.get('passed', '?')}")

    _check(
        "dynamic_atr" in res_a.get("checks", {}),
        "CHECK F: dynamic_atr key exists in checks",
    )
    # Advisory only — signal must NOT be hard-failed
    _check(
        res_a.get("passed") is not False or res_a.get("hard_fail") is not True,
        "CHECK F: tight SL is advisory only — signal not hard-failed",
    )
    # Warning should mention recommended sl
    warnings_tight = res_a.get("warnings", [])
    has_sl_warning = any("SL tighter" in w or "recommend" in w.lower() or "$" in w
                         for w in warnings_tight)
    _check(has_sl_warning, "CHECK F: tight SL triggers width warning")

    # ── Scenario B: appropriate SL — within 70-150% of dynamic recommendation ─
    # First get the recommendation, then place SL exactly at it
    _rec_res = calculate_dynamic_sl(df100, "long", entry=entry_p)
    _rec_dist = _rec_res["sl_distance"]
    ok_sl = entry_p - _rec_dist   # exactly at recommendation = definitely within band
    res_b = sl_quality_check(df100, entry_p, ok_sl, "long")
    warnings_ok = res_b.get("warnings", [])
    sl_warning_ok = [w for w in warnings_ok if "SL tighter" in w or "SL wider" in w]

    print(f"\n  Scenario B — ok SL (entry={entry_p}, sl={ok_sl:.2f}):")
    print(f"    warnings: {warnings_ok}")
    _check(
        len(sl_warning_ok) == 0,
        f"CHECK F: appropriate SL triggers no width warning (got {sl_warning_ok})",
    )

    # ── Scenario C: SL much wider than recommendation ────────────────────────
    wide_sl = entry_p - max(atr_val * 5.0, 30.0)    # 5x ATR — clearly too wide
    res_c   = sl_quality_check(df100, entry_p, wide_sl, "long")
    warnings_wide = res_c.get("warnings", [])
    has_wide_warning = any("SL wider" in w or "wider" in w.lower() for w in warnings_wide)
    print(f"\n  Scenario C — wide SL (entry={entry_p}, sl={wide_sl:.2f}):")
    print(f"    warnings: {warnings_wide}")
    _check(has_wide_warning, "CHECK F: wide SL triggers 'wider than ATR' warning")

except Exception:
    _fail("TEST 10 crashed", traceback.format_exc().splitlines()[-1])
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{HEAD}")
print("FINAL VERDICT")
print(HEAD)
print(f"\n  Total PASS : {TOTAL_PASS}")
print(f"  Total FAIL : {TOTAL_FAIL}")
print(f"  Total Checks: {TOTAL_PASS + TOTAL_FAIL}")

if ISSUES:
    print("\n  Issues found:")
    for issue in ISSUES:
        print(f"    - {issue}")

print()
if TOTAL_FAIL == 0:
    print("  Phase 2 Task 8 -- READY FOR LIVE USE")
else:
    print(f"  Phase 2 Task 8 -- {TOTAL_FAIL} ISSUE(S) FOUND — review before live use")
print()
