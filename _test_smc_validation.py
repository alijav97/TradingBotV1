"""
SMC Validation Test — verifies smart_money.py functions with real XAUUSD data.
Run: python _test_smc_validation.py
Expected: all PASS
"""

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join("data", "historical_xauusd.csv")
PASS = "PASS"; FAIL = "FAIL"
results = []

def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    results.append((tag, name, detail))
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── Build DataFrame ──────────────────────────────────────────────────────────
section("Loading data")
try:
    raw = pd.read_csv(CSV_PATH)
    raw.columns = [c.lower().strip() for c in raw.columns]
    # Support common column name variants
    rename_map = {
        "time": "timestamp", "date": "timestamp",
        "vol":  "volume", "tick_volume": "volume",
    }
    raw.rename(columns=rename_map, inplace=True)
    for col in ("open", "high", "low", "close"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df = raw.tail(500).copy().reset_index(drop=True)

    # Add ATR (14)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # Add EMA20 / EMA50
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Add RSI(14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Volume fallback
    if "volume" not in df.columns:
        df["volume"] = 1000.0

    check("CSV loaded", True, f"{len(df)} rows")
    current_price = float(df["close"].iloc[-1])
    check("current_price > 0", current_price > 0, f"${current_price:,.2f}")
except Exception as e:
    check("CSV loaded", False, str(e))
    print("\nFATAL: Cannot load data — aborting.")
    sys.exit(1)

# ── Import SmartMoneyAnalyzer ─────────────────────────────────────────────────
section("Import SmartMoneyAnalyzer")
try:
    from smart_money import SmartMoneyAnalyzer
    sma = SmartMoneyAnalyzer()
    check("SmartMoneyAnalyzer imported", True)
except Exception as e:
    check("SmartMoneyAnalyzer imported", False, str(e))
    print("\nFATAL: Cannot import SmartMoneyAnalyzer — aborting.")
    sys.exit(1)

# ── Test 1: find_order_blocks ─────────────────────────────────────────────────
section("Test 1 — find_order_blocks")
try:
    obs = sma.find_order_blocks(df)
    check("Returns a list", isinstance(obs, list))
    check("At least 1 order block found", len(obs) >= 1, f"count={len(obs)}")
    if obs:
        ob = obs[0]
        required_keys = {"ob_high", "ob_low", "ob_level", "untested", "bar_index"}
        check("Required keys present", required_keys.issubset(ob.keys()),
              f"keys={set(ob.keys())}")
        check("ob_high > ob_low", ob["ob_high"] > ob["ob_low"],
              f"high={ob['ob_high']}, low={ob['ob_low']}")
        check("ob_level within range",
              ob["ob_low"] <= ob["ob_level"] <= ob["ob_high"],
              f"level={ob['ob_level']}")
except Exception as e:
    check("find_order_blocks", False, str(e))

# ── Test 2: find_fair_value_gaps ──────────────────────────────────────────────
section("Test 2 — find_fair_value_gaps")
try:
    fvgs = sma.find_fair_value_gaps(df)
    check("Returns a list", isinstance(fvgs, list))
    check("At least 1 FVG found", len(fvgs) >= 1, f"count={len(fvgs)}")
    if fvgs:
        fv = fvgs[0]
        required_keys = {"fvg_top", "fvg_bottom", "fvg_midpoint", "filled", "bar_index"}
        check("Required keys present", required_keys.issubset(fv.keys()),
              f"keys={set(fv.keys())}")
        check("fvg_top > fvg_bottom", fv["fvg_top"] > fv["fvg_bottom"],
              f"top={fv['fvg_top']}, bottom={fv['fvg_bottom']}")
except Exception as e:
    check("find_fair_value_gaps", False, str(e))

# ── Test 3: find_liquidity_levels ─────────────────────────────────────────────
section("Test 3 — find_liquidity_levels")
try:
    liq = sma.find_liquidity_levels(df)
    check("Returns a dict", isinstance(liq, dict))
    check("buy_side_liquidity key present", "buy_side_liquidity" in liq)
    check("sell_side_liquidity key present", "sell_side_liquidity" in liq)
    bsl_list = liq.get("buy_side_liquidity") or []
    ssl_list = liq.get("sell_side_liquidity") or []
    check("buy_side_liquidity is list", isinstance(bsl_list, list), f"count={len(bsl_list)}")
    check("sell_side_liquidity is list", isinstance(ssl_list, list), f"count={len(ssl_list)}")
    if bsl_list:
        check("buy_side levels > current_price", all(v > current_price for v in bsl_list[:3]),
              f"top3={bsl_list[:3]}")
    else:
        check("buy_side_liquidity found (informational)", False, "0 levels — may be normal")
    if ssl_list:
        check("sell_side levels < current_price", all(v < current_price for v in ssl_list[:3]),
              f"top3={ssl_list[:3]}")
    else:
        check("sell_side_liquidity found (informational)", False, "0 levels — may be normal")
except Exception as e:
    check("find_liquidity_levels", False, str(e))

# ── Test 4: detect_market_structure ───────────────────────────────────────────
section("Test 4 — detect_market_structure")
try:
    struct = sma.detect_market_structure(df)
    check("Returns a dict", isinstance(struct, dict))
    required_keys = {"structure", "bias", "last_bos", "last_choch"}
    check("Required keys present", required_keys.issubset(struct.keys()),
          f"keys={set(struct.keys())}")
    valid_structures = {"trending_up", "trending_down", "ranging"}
    check("Structure value valid", struct.get("structure") in valid_structures,
          f"structure={struct.get('structure')}")
    valid_biases = {"bullish", "bearish", "neutral"}
    check("Bias value valid", struct.get("bias") in valid_biases,
          f"bias={struct.get('bias')}")
except Exception as e:
    check("detect_market_structure", False, str(e))

# ── Test 5: smc_score (long + short) ──────────────────────────────────────────
section("Test 5 — smc_score")
for _dir in ("long", "short"):
    try:
        result = sma.smc_score(df, _dir)
        check(f"smc_score({_dir}) returns dict", isinstance(result, dict))
        check(f"smc_score({_dir}) has 'score'", "score" in result,
              f"keys={list(result.keys())[:6]}")
        score_val = result.get("score", -1)
        check(f"smc_score({_dir}) score in 0–10",
              isinstance(score_val, (int, float)) and 0 <= score_val <= 10,
              f"score={score_val}")
    except Exception as e:
        check(f"smc_score({_dir})", False, str(e))

# ── Test 6: find_breaker_blocks (NEW) ─────────────────────────────────────────
section("Test 6 — find_breaker_blocks (NEW)")
try:
    bb_long  = sma.find_breaker_blocks(df, "long")
    bb_short = sma.find_breaker_blocks(df, "short")
    check("Returns list (long)",  isinstance(bb_long,  list))
    check("Returns list (short)", isinstance(bb_short, list))
    all_bb = bb_long + bb_short
    check("Method callable (no crash)", True, f"long={len(bb_long)}, short={len(bb_short)}")
    if all_bb:
        bb = all_bb[0]
        required_keys = {"breaker_type", "breaker_level", "breaker_high", "breaker_low", "active", "bar_index"}
        check("Required keys present", required_keys.issubset(bb.keys()),
              f"keys={set(bb.keys())}")
        check("breaker_high > breaker_low",
              bb["breaker_high"] > bb["breaker_low"],
              f"high={bb['breaker_high']}, low={bb['breaker_low']}")
    else:
        check("Breaker blocks found (informational)", False,
              "0 found — may be expected on clean trend data")
except Exception as e:
    check("find_breaker_blocks", False, str(e))

# ── Test 7: find_premium_discount_zones (NEW) ─────────────────────────────────
section("Test 7 — find_premium_discount_zones (NEW)")
try:
    zones = sma.find_premium_discount_zones(df)
    check("Returns a dict", isinstance(zones, dict))
    required_keys = {"highest_high", "lowest_low", "equilibrium", "current_price",
                     "current_zone", "zone_bias", "fib_levels"}
    check("Required keys present", required_keys.issubset(zones.keys()),
          f"keys={set(zones.keys())}")
    check("highest_high > lowest_low",
          zones["highest_high"] > zones["lowest_low"],
          f"H={zones['highest_high']:.2f}, L={zones['lowest_low']:.2f}")
    check("equilibrium between H/L",
          zones["lowest_low"] < zones["equilibrium"] < zones["highest_high"],
          f"eq={zones['equilibrium']:.2f}")
    check("current_price matches df close",
          abs(zones["current_price"] - current_price) < 1.0,
          f"zone_price={zones['current_price']:.2f}, df_price={current_price:.2f}")
    valid_zones = {"premium", "discount", "equilibrium"}
    check("current_zone valid", zones["current_zone"] in valid_zones,
          f"zone={zones['current_zone']}")
    valid_biases = {"long", "short", "neutral"}
    check("zone_bias valid", zones["zone_bias"] in valid_biases,
          f"bias={zones['zone_bias']}")
    fib = zones.get("fib_levels", {})
    check("fib_levels has 5 levels", len(fib) == 5,
          f"levels={list(fib.keys())}")
except Exception as e:
    check("find_premium_discount_zones", False, str(e))

# ── Test 8: get_smc_context (NEW master function) ─────────────────────────────
section("Test 8 — get_smc_context (NEW)")
for _dir in ("long", "short"):
    try:
        ctx = sma.get_smc_context(df, _dir)
        check(f"get_smc_context({_dir}) returns dict", isinstance(ctx, dict))
        required_keys = {
            "smc_score", "order_blocks", "fair_value_gaps", "breaker_blocks",
            "liquidity", "structure", "premium_discount",
            "zone_aligned", "entry_quality", "entry_quality_label",
            "entry_reasons", "avoid_reasons", "confidence_adjustment",
        }
        check(f"Required keys present ({_dir})", required_keys.issubset(ctx.keys()),
              f"missing={required_keys - ctx.keys()}")
        check(f"entry_quality in A/B/C/D ({_dir})",
              ctx.get("entry_quality") in ("A", "B", "C", "D"),
              f"grade={ctx.get('entry_quality')}")
        check(f"confidence_adjustment numeric ({_dir})",
              isinstance(ctx.get("confidence_adjustment"), (int, float)),
              f"adj={ctx.get('confidence_adjustment')}")
        check(f"order_blocks is list ({_dir})", isinstance(ctx.get("order_blocks"), list))
        check(f"fair_value_gaps is list ({_dir})", isinstance(ctx.get("fair_value_gaps"), list))
        check(f"entry_reasons is list ({_dir})", isinstance(ctx.get("entry_reasons"), list))
        check(f"avoid_reasons is list ({_dir})", isinstance(ctx.get("avoid_reasons"), list))
        print(f"       Grade={ctx['entry_quality']} | adj={ctx['confidence_adjustment']:+.1f} | "
              f"reasons={ctx['entry_reasons']}")
    except Exception as e:
        check(f"get_smc_context({_dir})", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total  = len(results)
print(f"\n  {passed}/{total} PASS  |  {failed} FAIL\n")
if failed:
    print("  FAILED checks:")
    for tag, name, detail in results:
        if tag == FAIL:
            print(f"    ✗ {name}" + (f"  — {detail}" if detail else ""))
else:
    print("  All checks PASSED ✓")
print()
sys.exit(0 if failed == 0 else 1)
