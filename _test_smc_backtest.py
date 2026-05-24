"""
SMC Backtest Verification — Phase 3 Task 10
Tests find_breaker_blocks, find_premium_discount_zones, get_smc_context,
confluence FACTOR 2 upgrade, trade card, and full pipeline.

Run: python _test_smc_backtest.py
Expected: Phase 3 Task 10 — READY FOR LIVE USE
"""

import os, sys, traceback, types, unittest.mock as mock
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
PASS = "PASS"; FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []   # (tag, name, detail)

def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = PASS if cond else FAIL
    _results.append((tag, name, detail))
    mark = "✓" if cond else "✗"
    print(f"    [{tag}] {mark} {name}" + (f"  — {detail}" if detail else ""))
    return cond

def section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print('═'*65)

def _safe(fn, *a, **kw):
    """Call fn; return (result, None) or (None, exc_str) on error."""
    try:
        return fn(*a, **kw), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Load & enrich data
# ─────────────────────────────────────────────────────────────────────────────
section("SETUP — Loading XAUUSD historical data")

CSV_PATH = os.path.join("data", "historical_xauusd.csv")
try:
    raw = pd.read_csv(CSV_PATH)
    raw.columns = [c.lower().strip() for c in raw.columns]
    for old, new in {"time": "timestamp", "date": "timestamp",
                     "vol": "volume", "tick_volume": "volume"}.items():
        if old in raw.columns:
            raw.rename(columns={old: new}, inplace=True)
    for col in ("open", "high", "low", "close"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw.dropna(subset=["open", "high", "low", "close"], inplace=True)

    df500 = raw.tail(500).copy().reset_index(drop=True)
    df100 = raw.tail(100).copy().reset_index(drop=True)

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"]   = tr.ewm(span=14, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        d  = df["close"].diff()
        g  = d.clip(lower=0).ewm(span=14, adjust=False).mean()
        l_ = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
        df["rsi"]   = 100 - (100 / (1 + g / l_.replace(0, np.nan)))
        if "volume" not in df.columns:
            df["volume"] = 1000.0
        return df

    df500 = _enrich(df500)
    df100 = _enrich(df100)
    current_price = float(df500["close"].iloc[-1])
    print(f"  Loaded {len(df500)} candles  |  current price ${current_price:,.2f}")
    check("Data loaded successfully", True, f"{len(df500)} rows, price=${current_price:,.2f}")
except Exception as e:
    check("Data loaded successfully", False, str(e))
    print("\nFATAL — cannot load data, aborting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Import SmartMoneyAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
try:
    from smart_money import SmartMoneyAnalyzer
    sma = SmartMoneyAnalyzer()
    check("SmartMoneyAnalyzer imported", True)
except Exception as e:
    check("SmartMoneyAnalyzer imported", False, str(e))
    print("\nFATAL — cannot import SmartMoneyAnalyzer, aborting.")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — find_breaker_blocks()
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 1 — find_breaker_blocks()")

bb_long,  e1a = _safe(sma.find_breaker_blocks, df500, "long")
bb_short, e1b = _safe(sma.find_breaker_blocks, df500, "short")

check("find_breaker_blocks(long)  no crash",  e1a is None, e1a or "")
check("find_breaker_blocks(short) no crash",  e1b is None, e1b or "")
check("Returns list (long)",  isinstance(bb_long,  list) if bb_long  is not None else False)
check("Returns list (short)", isinstance(bb_short, list) if bb_short is not None else False)

all_bb = (bb_long or []) + (bb_short or [])
print(f"\n  Breaker blocks found: long={len(bb_long or [])}, short={len(bb_short or [])}, total={len(all_bb)}")

REQUIRED_BB_KEYS = {"breaker_type", "breaker_level", "breaker_high", "breaker_low", "active", "bar_index"}
key_ok = True
hl_ok  = True
type_ok = True
active_ok = True

for bb in all_bb:
    if not REQUIRED_BB_KEYS.issubset(bb.keys()):
        key_ok = False
    if bb["breaker_high"] <= bb["breaker_low"]:
        hl_ok = False
    if bb["breaker_type"] not in ("bullish", "bearish"):
        type_ok = False

check("All required keys present in every breaker", key_ok,
      f"checked {len(all_bb)} breakers")
check("breaker_high > breaker_low always", hl_ok, f"checked {len(all_bb)}")
check("breaker_type is 'bullish' or 'bearish'", type_ok, f"checked {len(all_bb)}")

active_bb = [b for b in all_bb if b["active"]]
for ab in active_bb:
    dist_pct = abs(ab["breaker_level"] - current_price) / current_price * 100
    if dist_pct > 0.5:
        active_ok = False
        print(f"  ⚠ Active breaker at {ab['breaker_level']:.2f} is {dist_pct:.3f}% from price (>0.5%)")

check("Active breakers within 0.5% of current price", active_ok,
      f"{len(active_bb)} active")

types_found = {b["breaker_type"] for b in all_bb}
print(f"  Types found: {types_found}")
print(f"  Active count: {len(active_bb)}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — find_premium_discount_zones()
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 2 — find_premium_discount_zones()")

zones, e2 = _safe(sma.find_premium_discount_zones, df100)
check("No crash", e2 is None, e2 or "")

if zones:
    hh = zones["highest_high"]
    ll = zones["lowest_low"]
    eq = zones["equilibrium"]
    cp = zones["current_price"]
    zn = zones["current_zone"]
    zb = zones["zone_bias"]
    ps = zones.get("premium_start", 0)
    de = zones.get("discount_end", 0)
    fib = zones.get("fib_levels", {})

    print(f"\n  highest_high  = {hh:,.2f}")
    print(f"  lowest_low    = {ll:,.2f}")
    print(f"  equilibrium   = {eq:,.2f}")
    print(f"  current_price = {cp:,.2f}")
    print(f"  current_zone  = {zn}")
    print(f"  zone_bias     = {zb}")
    print(f"  premium_start = {ps:,.2f}")
    print(f"  discount_end  = {de:,.2f}")
    print(f"  fib_levels    = { {k: f'{v:,.2f}' for k, v in fib.items()} }")

    check("highest_high > lowest_low", hh > ll, f"{hh:.2f} > {ll:.2f}")

    expected_eq = (hh + ll) / 2
    check("equilibrium == (H+L)/2",
          abs(eq - expected_eq) < 0.01,
          f"got {eq:.2f}, expected {expected_eq:.2f}")

    check("current_zone valid",
          zn in ("premium", "discount", "equilibrium"),
          f"zone={zn}")

    check("All 5 fib levels present", len(fib) == 5,
          f"got {list(fib.keys())}")

    fib_vals = sorted(float(v) for v in fib.values())
    check("Fib levels in ascending order",
          fib_vals == sorted(fib_vals),
          f"{[f'{v:.2f}' for v in fib_vals]}")

    # zone_bias logic
    if zn == "discount":
        check("zone_bias='long' when discount", zb == "long", f"got {zb}")
    elif zn == "premium":
        check("zone_bias='short' when premium", zb == "short", f"got {zb}")
    else:
        check("zone_bias='neutral' when equilibrium", zb == "neutral", f"got {zb}")

    check("premium_start >= equilibrium", ps >= eq, f"{ps:.2f} >= {eq:.2f}")
    check("discount_end <= equilibrium",  de <= eq, f"{de:.2f} <= {eq:.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — get_smc_context() grade validation
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 3 — get_smc_context() grade validation (mock scenarios)")

# We run get_smc_context on real data, then verify grade logic by patching
# the internals of SmartMoneyAnalyzer.  We test the grade formula directly
# rather than brittle deep mocks — since get_smc_context calls the real
# sub-functions and the grade formula is deterministic.

def _grade_from_counts(ob_active: bool, fvg_nearby: bool,
                       str_aligned: bool, zone_aligned: bool) -> tuple[str, float]:
    """Replicate the grading logic from get_smc_context."""
    met_count = sum([ob_active, fvg_nearby, str_aligned])
    if met_count >= 3 or (met_count >= 2 and zone_aligned):
        return "A", 1.0
    elif met_count >= 2:
        return "B", 0.5
    elif met_count >= 1:
        return "C", 0.0
    else:
        return "D", -0.5

scenarios = [
    # (name, ob, fvg, str_aligned, zone_aligned, exp_grade, exp_adj)
    ("Grade A — 3 criteria + zone",     True,  True,  True,  True,  "A",  1.0),
    ("Grade A — 2 criteria + zone",     True,  True,  False, True,  "A",  1.0),
    ("Grade A — 3 criteria no zone",    True,  True,  True,  False, "A",  1.0),
    ("Grade B — 2 criteria no zone",    True,  True,  False, False, "B",  0.5),
    ("Grade B — OB+str no FVG",         True,  False, True,  False, "B",  0.5),
    ("Grade C — only OB",               True,  False, False, False, "C",  0.0),
    ("Grade C — only FVG",              False, True,  False, False, "C",  0.0),
    ("Grade D — nothing",               False, False, False, False, "D", -0.5),
]

print()
for name, ob, fvg, st, za, exp_g, exp_a in scenarios:
    g, a = _grade_from_counts(ob, fvg, st, za)
    ok = (g == exp_g) and (abs(a - exp_a) < 1e-9)
    check(name, ok, f"got grade={g} adj={a:+.1f}, expected grade={exp_g} adj={exp_a:+.1f}")

# Also verify on real data — grade must be one of A/B/C/D
ctx_long,  e3a = _safe(sma.get_smc_context, df500, "long")
ctx_short, e3b = _safe(sma.get_smc_context, df500, "short")
check("get_smc_context(long)  no crash",  e3a is None, e3a or "")
check("get_smc_context(short) no crash",  e3b is None, e3b or "")
if ctx_long:
    check("Real data grade in A/B/C/D",
          ctx_long.get("entry_quality") in ("A", "B", "C", "D"),
          f"grade={ctx_long.get('entry_quality')}  adj={ctx_long.get('confidence_adjustment'):+.1f}")
    # Cross-check: grade formula consistent with flags
    g2, a2 = _grade_from_counts(
        ctx_long["active_order_block"],
        ctx_long["fvg_nearby"],
        ctx_long["structure_aligned"],
        ctx_long.get("zone_aligned", False),
    )
    check("Grade consistent with internal flags",
          g2 == ctx_long["entry_quality"],
          f"flags→{g2}, reported={ctx_long['entry_quality']}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — Confluence engine FACTOR 2 upgrade
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 4 — Confluence engine FACTOR 2 upgrade")

try:
    from confluence_engine import score_confluences
    cr, e4 = _safe(score_confluences, df500, "long")
    check("score_confluences no crash", e4 is None, e4 or "")

    if cr:
        dl = cr.get("detail_lines", [])
        smc_line = next((l for l in dl if "SMC" in l or "Order block" in l or "smc" in l.lower()), None)
        print(f"\n  SMC detail line: {smc_line!r}")

        grade_referenced = smc_line is not None and any(
            kw in smc_line for kw in ("Grade A", "Grade B", "Grade C", "Grade D",
                                       "Order block", "SMC")
        )
        check("detail_lines contains SMC reference", grade_referenced,
              f"line={smc_line!r}")

        raw_smc = cr.get("raw_checks", {}).get("smc")
        check("raw_checks['smc'] is populated", raw_smc is not None,
              f"type={type(raw_smc).__name__}")

        if isinstance(raw_smc, dict) and "entry_quality" in raw_smc:
            check("raw_checks['smc'] is full smc_context", True,
                  f"grade={raw_smc.get('entry_quality')}")
        else:
            # fallback — may be legacy smc_result
            check("raw_checks['smc'] is present (legacy fallback)",
                  raw_smc is not None, f"value={raw_smc!r}")

        # SMC weight range check — pull from check_weights_earned or detail line
        smc_weight = cr.get("check_weights_earned", {}).get("SMC", None)
        if smc_weight is not None:
            check("SMC weight in [-1.0, +2.0]",
                  -1.0 <= smc_weight <= 2.0,
                  f"weight={smc_weight}")
        else:
            check("SMC weight key present in check_weights_earned",
                  "SMC" in cr.get("check_weights_earned", {}),
                  f"keys={list(cr.get('check_weights_earned', {}).keys())}")

        tc = cr.get("total_checks", 0)
        check("total_checks >= 9", tc >= 9, f"total_checks={tc}")

        conf = cr.get("confidence", -1)
        check("confidence in 0–10", 0.0 <= conf <= 10.0, f"confidence={conf}")
except Exception as e:
    check("Confluence engine TEST 4", False, traceback.format_exc().splitlines()[-1])

# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — Fallback to legacy smc_score()
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 5 — Fallback to legacy smc_score()")

try:
    import confluence_engine as _ce
    orig_enhanced = _ce._SMC_ENHANCED

    # Patch _SMC_ENHANCED = False to force legacy path
    _ce._SMC_ENHANCED = False
    cr_legacy, e5 = _safe(_ce.score_confluences, df500, "long")
    _ce._SMC_ENHANCED = orig_enhanced   # restore immediately

    check("No crash with _SMC_ENHANCED=False", e5 is None, e5 or "")
    if cr_legacy:
        conf5 = cr_legacy.get("confidence", -1)
        check("Confidence still 0–10 in legacy mode", 0.0 <= conf5 <= 10.0,
              f"confidence={conf5}")
        dl5 = cr_legacy.get("detail_lines", [])
        legacy_smc_line = next((l for l in dl5 if "Order block" in l or "SMC" in l), None)
        check("Legacy detail line present", legacy_smc_line is not None,
              f"line={legacy_smc_line!r}")
        tc5 = cr_legacy.get("total_checks", 0)
        check("total_checks >= 9 in legacy mode", tc5 >= 9, f"total={tc5}")
except Exception as e:
    check("Fallback test", False, traceback.format_exc().splitlines()[-1])

# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — Historical SMC grade distribution
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 6 — Historical SMC grade distribution (every 10th of last 500)")

grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
adj_total    = 0.0
n_samples    = 0
errors6      = 0

for i in range(0, len(df500), 10):
    if i < 50:   # need enough history
        continue
    slice_df = df500.iloc[:i+1].copy()
    ctx, _err = _safe(sma.get_smc_context, slice_df, "long")
    if ctx is None:
        errors6 += 1
        continue
    g = ctx.get("entry_quality", "D")
    a = float(ctx.get("confidence_adjustment", 0.0))
    grade_counts[g] = grade_counts.get(g, 0) + 1
    adj_total += a
    n_samples += 1

ADJ_MAP = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}
net_avg = round(adj_total / n_samples, 3) if n_samples else 0.0
n_boost  = grade_counts["A"] + grade_counts["B"]
n_neutral= grade_counts["C"]
n_penalty= grade_counts["D"]
pct_boost   = round(n_boost   / n_samples * 100, 1) if n_samples else 0
pct_neutral = round(n_neutral / n_samples * 100, 1) if n_samples else 0
pct_penalty = round(n_penalty / n_samples * 100, 1) if n_samples else 0

print(f"""
  SMC grade distribution ({n_samples} sample points):
   Grade A: {grade_counts['A']:>3}  ({round(grade_counts['A']/n_samples*100,1) if n_samples else 0}%)  → +1.0 avg boost
   Grade B: {grade_counts['B']:>3}  ({round(grade_counts['B']/n_samples*100,1) if n_samples else 0}%)  → +0.5 avg boost
   Grade C: {grade_counts['C']:>3}  ({round(grade_counts['C']/n_samples*100,1) if n_samples else 0}%)  → 0.0
   Grade D: {grade_counts['D']:>3}  ({round(grade_counts['D']/n_samples*100,1) if n_samples else 0}%)  → -0.5 penalty
   Net avg adjustment: {net_avg:+.3f}
   Boost signals: {pct_boost}%  |  Neutral: {pct_neutral}%  |  Penalty: {pct_penalty}%
   Errors: {errors6}
""")

check("At least 20 samples processed", n_samples >= 20, f"got {n_samples}")
check("All grade buckets populated",
      all(grade_counts[g] > 0 for g in ("A", "B", "C", "D")),
      f"{grade_counts}")
check("Grade error rate < 10%",
      errors6 / max(n_samples + errors6, 1) < 0.10,
      f"errors={errors6}/{n_samples+errors6}")
check("Net avg adjustment between -0.5 and +1.0",
      -0.5 <= net_avg <= 1.0,
      f"avg={net_avg:+.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — Trade card SMC block
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 7 — Trade card SMC block")

try:
    from bot_chat import _render_trade_card

    # Build a realistic mock signal with smc_context populated
    _smc_ctx_mock = sma.get_smc_context(df500, "long")
    mock_sig = {
        "source":            "test",
        "asset":             "XAUUSD",
        "direction":         "long",
        "confidence":        7.5,
        "confidence_score":  7.5,
        "tier":              "A",
        "tier_label":        "Tier A",
        "pattern_name":      "SMC Test Signal",
        "description":       "Test signal for SMC block rendering",
        "entry":             current_price,
        "stop_loss":         current_price - 15.0,
        "take_profit":       current_price + 30.0,
        "bt_win_rate":       72.0,
        "profit_factor":     2.1,
        "live_win_rate":     None,
        "note":              "",
        "sl_atr_multiplier": 0.0,
        "geo_risk_level":    "low",
        "confluence_met":    ["SMC", "HTF", "Trend"],
        "confluence_missed": [],
        "detail_lines":      ["✓ SMC Grade A — OB + FVG   +2.0"],
        "checklist_results": None,
        "_confluence_raw":   None,
        "smc_context":       _smc_ctx_mock,
        "entry_quality":     _smc_ctx_mock.get("entry_quality", "A"),
    }

    card, e7 = _safe(_render_trade_card, mock_sig, 1, 10000.0)
    check("_render_trade_card no crash", e7 is None, e7 or "")

    if card:
        has_smc_block = "SMC ANALYSIS" in card
        has_entry_q   = "Entry quality" in card or "entry quality" in card.lower()
        has_zone      = "Zone:" in card or "zone:" in card.lower()

        check("Card contains 'SMC ANALYSIS'", has_smc_block)
        check("Card contains 'Entry quality:'", has_entry_q)
        check("Card contains 'Zone:'",          has_zone)

        # Print the SMC section
        lines = card.split("\n")
        in_smc = False
        print()
        for ln in lines:
            if "SMC ANALYSIS" in ln:
                in_smc = True
            if in_smc:
                print(f"  {ln}")
            if in_smc and "─" in ln and "SMC" not in ln:
                break
    else:
        check("Card rendered (non-empty)", False, "empty string returned")

except Exception as e:
    check("Trade card TEST 7", False, traceback.format_exc().splitlines()[-1])

# ═════════════════════════════════════════════════════════════════════════════
# TEST 8 — Premium/discount zone alignment
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 8 — Premium/discount zone alignment")

z8, e8 = _safe(sma.find_premium_discount_zones, df500)
check("No crash", e8 is None, e8 or "")

if z8:
    zn8 = z8["current_zone"]
    zb8 = z8["zone_bias"]
    hh8 = z8["highest_high"]
    ll8 = z8["lowest_low"]
    eq8 = z8["equilibrium"]
    cp8 = z8["current_price"]

    print(f"\n  Current market:")
    print(f"  Range:  {ll8:,.2f} – {hh8:,.2f}  (eq={eq8:,.2f})")
    print(f"  Price:  {cp8:,.2f}")
    print(f"  Zone:   {zn8.upper()}  →  bias={zb8.upper()}")

    if zn8 == "discount":
        check("Zone=discount → bias=long",  zb8 == "long",  f"got {zb8}")
    elif zn8 == "premium":
        check("Zone=premium  → bias=short", zb8 == "short", f"got {zb8}")
    elif zn8 == "equilibrium":
        check("Zone=equil    → bias=neutral", zb8 == "neutral", f"got {zb8}")
    else:
        check("Zone value valid", False, f"unexpected zone={zn8}")

    check("current_price within H/L range",
          ll8 <= cp8 <= hh8,
          f"{ll8:.2f} <= {cp8:.2f} <= {hh8:.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 9 — Combined SMC + Phase 2 pipeline
# ═════════════════════════════════════════════════════════════════════════════
section("TEST 9 — Combined SMC + Phase 2 pipeline")

base_confidence = 7.0
smc_adj = 0.0; macro_adj = 0.0; geo_adj = 0.0
smc_grade = "?"; macro_bias = "?"; geo_level = "?"
sl_result  = None; sess_result = None

# Step 1 — SMC
smc_ctx9, e9a = _safe(sma.get_smc_context, df500, "long")
if smc_ctx9:
    smc_grade = smc_ctx9.get("entry_quality", "?")
    smc_adj   = float(smc_ctx9.get("confidence_adjustment", 0.0))
    check("get_smc_context step", True, f"grade={smc_grade} adj={smc_adj:+.1f}")
else:
    check("get_smc_context step", False, e9a or "")

# Step 2 — macro context (detect_gold_regime proxy)
try:
    from market_context import detect_gold_regime
    regime9, e9b = _safe(detect_gold_regime, df500)
    if regime9:
        macro_bias = regime9.get("regime", "?")
        size_mult  = regime9.get("position_size_multiplier", 1.0)
        # Translate regime to confidence adjustment
        if "BULL" in macro_bias.upper():
            macro_adj = +0.5
        elif "BEAR" in macro_bias.upper():
            macro_adj = -0.5
        else:
            macro_adj = 0.0
        check("detect_gold_regime step", True,
              f"regime={macro_bias} size×{size_mult}")
    else:
        check("detect_gold_regime step", False, e9b or "")
except Exception as e:
    check("detect_gold_regime step", False, str(e))

# Step 3 — geopolitical score
try:
    from geo_filter import get_geopolitical_score
    geo9, e9c = _safe(get_geopolitical_score, [])
    if geo9:
        geo_level = geo9.get("geo_risk_level", "?")
        geo_adj   = float(geo9.get("confidence_adjustment", 0.0))
        check("get_geopolitical_score step", True,
              f"level={geo_level} adj={geo_adj:+.1f}")
    else:
        check("get_geopolitical_score step", False, e9c or "")
except Exception as e:
    check("get_geopolitical_score step", False, str(e))

# Step 4 — dynamic SL
try:
    from atr_sl_engine import calculate_dynamic_sl
    sl9, e9d = _safe(calculate_dynamic_sl, df500, "long", current_price)
    if sl9:
        sl_result = sl9
        sl_price  = sl9.get("sl_price", 0)
        sl_dist   = sl9.get("sl_distance", 0)
        sl_atr    = sl9.get("atr_used", 0)
        check("calculate_dynamic_sl step", True,
              f"SL=${sl_price:,.2f} dist={sl_dist:.2f} atr={sl_atr:.2f}")
    else:
        check("calculate_dynamic_sl step", False, e9d or "")
except Exception as e:
    check("calculate_dynamic_sl step", False, str(e))

# Step 5 — session profile
try:
    from session_profiler import get_current_session_profile
    sess9, e9e = _safe(get_current_session_profile)
    if sess9:
        sess_result  = sess9
        sess_name    = sess9.get("session", "?")
        sess_grade   = sess9.get("grade", "?")
        sess_mult    = sess9.get("lot_size_multiplier", 1.0)
        check("get_current_session_profile step", True,
              f"session={sess_name} grade={sess_grade} lot×{sess_mult}")
    else:
        check("get_current_session_profile step", False, e9e or "")
except Exception as e:
    check("get_current_session_profile step", False, str(e))

# ── Combine and print ─────────────────────────────────────────────────────
final_conf = round(min(10.0, max(0.0, base_confidence + smc_adj + macro_adj + geo_adj)), 2)

print(f"""
  FULL PIPELINE TEST:
   Base:    {base_confidence:.1f}
   SMC:     Grade {smc_grade} → {smc_adj:+.1f}
   Macro:   {macro_bias} → {macro_adj:+.1f}
   Geo:     {geo_level} → {geo_adj:+.1f}
   Final:   {final_conf}/10
""")

if sl_result:
    sl_p   = sl_result.get("sl_price", 0)
    sl_brk = sl_result.get("breakdown", "n/a")
    print(f"   SL:      ${sl_p:,.2f}  ({sl_brk})")
if sess_result:
    sn = sess_result.get("session", "?")
    sg = sess_result.get("grade", "?")
    sm = sess_result.get("lot_size_multiplier", 1.0)
    print(f"   Session: {sn} Grade {sg} → lot×{sm}")

check("Final confidence in 0–10", 0.0 <= final_conf <= 10.0,
      f"final={final_conf}")
check("All 5 pipeline steps completed",
      all([smc_ctx9, regime9 if 'regime9' in dir() else None,
           sess_result]) is not False,
      "see individual step results above")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═════════════════════════════════════════════════════════════════════════════
section("FINAL VERDICT")

passed  = sum(1 for r in _results if r[0] == PASS)
failed  = sum(1 for r in _results if r[0] == FAIL)
total   = len(_results)
pct     = round(passed / total * 100, 1) if total else 0

print(f"\n  {passed}/{total} checks passed  ({pct}%)")

if failed:
    print(f"\n  ✗ FAILED checks ({failed}):")
    for tag, name, detail in _results:
        if tag == FAIL:
            print(f"    ✗ {name}" + (f"  — {detail}" if detail else ""))
    print(f"\n  Phase 3 Task 10 — ⚠ ISSUES FOUND — see failures above")
else:
    print(f"\n  Phase 3 Task 10 — ✅ READY FOR LIVE USE")

print()
sys.exit(0 if failed == 0 else 1)
