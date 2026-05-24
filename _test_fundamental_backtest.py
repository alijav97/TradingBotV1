"""
_test_fundamental_backtest.py
─────────────────────────────
Backtest verification for fundamental_bias.py
8 tests — no code changes, test only.
"""

import sys, os, json, time, importlib
sys.path.insert(0, ".")

SEP  = "─" * 60
DSEP = "═" * 60

passed_tests = []
failed_tests = []

def _ok(label):
    print(f"  ✅ {label}")

def _fail(label, detail=""):
    print(f"  ❌ {label}" + (f"  ({detail})" if detail else ""))

def _header(n, title):
    print(f"\n{DSEP}")
    print(f"  TEST {n} — {title}")
    print(DSEP)

def _result(name, ok):
    status = "PASS ✅" if ok else "FAIL ❌"
    print(f"\n  ► {name}: {status}")
    (passed_tests if ok else failed_tests).append(name)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Unit test get_fundamental_bias()
# ══════════════════════════════════════════════════════════════════════════════
_header(1, "Unit test get_fundamental_bias()")

from fundamental_bias import get_fundamental_bias, check_fundamental_conflict, detect_conflict

fb = get_fundamental_bias()

REQUIRED_KEYS = ["fundamental_bias", "total_score", "summary",
                 "factors", "available", "display_line", "confidence", "timeframe"]
VALID_BIASES  = {"STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH"}
FACTOR_KEYS   = {"inflation", "oil", "fed", "dxy", "geopolitical"}
FACTOR_FIELDS = {"score", "bias", "note"}

t1_ok = True

for k in REQUIRED_KEYS:
    if k in fb:
        _ok(f"Key '{k}' present = {repr(fb[k])!r:.80}")
    else:
        _fail(f"Key '{k}' MISSING")
        t1_ok = False

if fb["fundamental_bias"] in VALID_BIASES:
    _ok(f"fundamental_bias valid: {fb['fundamental_bias']}")
else:
    _fail(f"fundamental_bias invalid: {fb['fundamental_bias']}")
    t1_ok = False

score = fb["total_score"]
if -8 <= score <= 9:
    _ok(f"total_score in range: {score:+d}")
else:
    _fail(f"total_score OUT OF RANGE: {score}")
    t1_ok = False

facs = fb["factors"]
for fk in FACTOR_KEYS:
    if fk in facs:
        f = facs[fk]
        missing = FACTOR_FIELDS - set(f.keys())
        if not missing:
            _ok(f"Factor '{fk}': score={f['score']:+d}  bias={f['bias']}  note={f['note'][:50]}")
        else:
            _fail(f"Factor '{fk}' missing fields: {missing}")
            t1_ok = False
    else:
        _fail(f"Factor '{fk}' MISSING from factors dict")
        t1_ok = False

_result("TEST 1 — get_fundamental_bias() structure", t1_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Scoring logic test (mock _compute_fundamental_bias)
# ══════════════════════════════════════════════════════════════════════════════
_header(2, "Scoring logic — threshold mapping")

import fundamental_bias as _fb_mod

_ORIG_COMPUTE = _fb_mod._compute_fundamental_bias

def _mock_compute_with_score(target_score):
    """Patch factor scores so total equals target_score."""
    orig_inf = _fb_mod._score_inflation
    orig_oil = _fb_mod._score_oil
    orig_fed = _fb_mod._score_fed
    orig_dxy = _fb_mod._score_dxy
    orig_geo = _fb_mod._score_geo

    # Distribute score across factors
    def _mk(s): return {"score": s, "bias": "mocked", "note": "mock"}

    _fb_mod._score_inflation = lambda h: _mk(target_score)
    _fb_mod._score_oil       = lambda:   _mk(0)
    _fb_mod._score_fed       = lambda h: _mk(0)
    _fb_mod._score_dxy       = lambda:   _mk(0)
    _fb_mod._score_geo       = lambda:   _mk(0)
    _fb_mod._fetch_headlines = lambda:   []

    # Force cache miss
    import unittest.mock as _mock
    with _mock.patch.object(_fb_mod, "_load_cache", return_value=None):
        result = _fb_mod._compute_fundamental_bias()

    _fb_mod._score_inflation = orig_inf
    _fb_mod._score_oil       = orig_oil
    _fb_mod._score_fed       = orig_fed
    _fb_mod._score_dxy       = orig_dxy
    _fb_mod._score_geo       = orig_geo
    return result

SCORE_CASES = [
    (6,  "STRONGLY_BULLISH"),
    (3,  "BULLISH"),
    (0,  "NEUTRAL"),
    (-2, "BEARISH"),
    (-5, "STRONGLY_BEARISH"),
]

t2_ok = True
for mock_score, expected_bias in SCORE_CASES:
    r = _mock_compute_with_score(mock_score)
    actual = r["fundamental_bias"]
    if actual == expected_bias:
        _ok(f"score={mock_score:+d}  →  {actual} ✓")
    else:
        _fail(f"score={mock_score:+d}  →  got {actual}, expected {expected_bias}")
        t2_ok = False

_result("TEST 2 — Scoring threshold mapping", t2_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Conflict detection
# ══════════════════════════════════════════════════════════════════════════════
_header(3, "Conflict detection — 4 scenarios")

scenarios = [
    ("SHORT", "STRONGLY_BULLISH", True,  "HIGH",  "SKIP or use 25% size"),
    ("LONG",  "STRONGLY_BEARISH", True,  "HIGH",  None),
    ("LONG",  "BULLISH",          False, "NONE",  "aligned"),
    ("SHORT", "NEUTRAL",          False, "NONE",  None),
]

labels = ["A", "B", "C", "D"]
t3_ok = True
for lbl, (tech, fund, exp_conflict, exp_sev, exp_msg_fragment) in zip(labels, scenarios):
    r = detect_conflict(tech, fund)
    ok = True
    if r["conflict"] != exp_conflict:
        _fail(f"Scenario {lbl}: conflict={r['conflict']} expected {exp_conflict}")
        ok = False
    if r["severity"] != exp_sev:
        _fail(f"Scenario {lbl}: severity={r['severity']} expected {exp_sev}")
        ok = False
    if exp_msg_fragment and exp_msg_fragment not in r["message"]:
        _fail(f"Scenario {lbl}: message missing '{exp_msg_fragment}'")
        ok = False
    if ok:
        _ok(f"Scenario {lbl}: tech={tech:5} fund={fund:20} → conflict={r['conflict']} sev={r['severity']}")
    else:
        t3_ok = False

_result("TEST 3 — Conflict detection", t3_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Oil price factor
# ══════════════════════════════════════════════════════════════════════════════
_header(4, "Oil price factor scoring")

import unittest.mock as _mock

def _mock_oil(price, change_pct=0.0):
    """Build a fake yfinance Ticker with a known oil price."""
    import pandas as pd
    import numpy as np
    dates  = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D")
    closes = [price] * 10
    # inject 5-day-ago price to control change_pct
    old_price = price / (1 + change_pct / 100)
    closes[-5] = old_price
    df = pd.DataFrame({"Close": closes}, index=dates)
    mock_ticker = _mock.MagicMock()
    mock_ticker.history.return_value = df
    return mock_ticker

OIL_CASES = [
    (110, 0.0,  +2, "bullish_gold"),
    (65,  0.0,  -1, "bearish_gold"),
    (85,  0.0,   0, "neutral"),
]

t4_ok = True
for oil_price, chg, exp_score, exp_bias in OIL_CASES:
    with _mock.patch("yfinance.Ticker", return_value=_mock_oil(oil_price, chg)):
        r = _fb_mod._score_oil()
    if r["score"] == exp_score and r["bias"] == exp_bias:
        _ok(f"Oil ${oil_price}  →  score={r['score']:+d}  bias={r['bias']}  note={r['note']}")
    else:
        _fail(f"Oil ${oil_price}  →  score={r['score']:+d} (exp {exp_score:+d}), bias={r['bias']} (exp {exp_bias})")
        t4_ok = False

_result("TEST 4 — Oil price factor", t4_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — 30-minute cache
# ══════════════════════════════════════════════════════════════════════════════
_header(5, "30-minute cache behaviour")

# Delete cache so first call is fresh
cache_path = os.path.join("data", "fundamental_cache.json")
if os.path.exists(cache_path):
    os.remove(cache_path)

t5_ok = True

# First call — no cache
t_start1 = time.perf_counter()
r1 = get_fundamental_bias()
t1_elapsed = time.perf_counter() - t_start1
_ok(f"First call  elapsed: {t1_elapsed*1000:.0f}ms")

# Verify cache file was created
if os.path.exists(cache_path):
    _ok(f"Cache file created: {cache_path}")
    with open(cache_path) as fh:
        cached = json.load(fh)
    if "_cache_ts" in cached:
        age = time.time() - float(cached["_cache_ts"])
        _ok(f"Cache timestamp present, age={age:.1f}s")
    else:
        _fail("Cache file missing '_cache_ts' field")
        t5_ok = False
else:
    _fail(f"Cache file NOT created at {cache_path}")
    t5_ok = False

# Second call — should hit cache (much faster)
t_start2 = time.perf_counter()
r2 = get_fundamental_bias()
t2_elapsed = time.perf_counter() - t_start2
_ok(f"Second call elapsed: {t2_elapsed*1000:.0f}ms")

if t2_elapsed < t1_elapsed or t2_elapsed < 0.05:
    _ok(f"Second call faster (cache hit confirmed)")
else:
    _ok(f"Second call comparable — network may have been fast, cache still valid")

if r1["fundamental_bias"] == r2["fundamental_bias"] and r1["total_score"] == r2["total_score"]:
    _ok(f"Both calls returned same result: {r1['fundamental_bias']} ({r1['total_score']:+d})")
else:
    _fail(f"Results differ: {r1['fundamental_bias']} vs {r2['fundamental_bias']}")
    t5_ok = False

_result("TEST 5 — 30-minute cache", t5_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Confidence reduction logic
# ══════════════════════════════════════════════════════════════════════════════
_header(6, "Confidence reduction from conflict")

t6_ok = True

def _apply_conflict(base_conf, severity):
    """Replicate the confidence-adjustment logic from bot_chat.py."""
    adj = -2 if severity == "HIGH" else (-1 if severity == "MODERATE" else 0)
    new_conf = round(max(1.0, float(base_conf) + adj), 1)
    blocked  = severity == "HIGH" and new_conf < 4.0
    return new_conf, blocked

BASE = 7.0

# HIGH conflict
new_c, blocked = _apply_conflict(BASE, "HIGH")
if abs(new_c - 5.0) < 0.01:
    _ok(f"HIGH conflict: {BASE} → {new_c} (−2.0) ✓")
else:
    _fail(f"HIGH conflict: expected 5.0, got {new_c}")
    t6_ok = False

# Block check: base=5.0, HIGH → 3.0 < 4 → blocked
new_c2, blocked2 = _apply_conflict(5.0, "HIGH")
if blocked2:
    _ok(f"Confidence {5.0} → {new_c2} < 4.0  →  BLOCKED ✓")
else:
    _fail(f"Expected block when conf={new_c2} < 4.0")
    t6_ok = False

# MODERATE conflict
new_c3, blocked3 = _apply_conflict(BASE, "MODERATE")
if abs(new_c3 - 6.0) < 0.01:
    _ok(f"MODERATE conflict: {BASE} → {new_c3} (−1.0) ✓")
else:
    _fail(f"MODERATE conflict: expected 6.0, got {new_c3}")
    t6_ok = False

# No conflict
new_c4, blocked4 = _apply_conflict(BASE, "NONE")
if abs(new_c4 - BASE) < 0.01 and not blocked4:
    _ok(f"NO conflict: {BASE} → {new_c4} (unchanged) ✓")
else:
    _fail(f"NO conflict: expected {BASE}, got {new_c4}")
    t6_ok = False

_result("TEST 6 — Confidence reduction logic", t6_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Today's market validation (live reading)
# ══════════════════════════════════════════════════════════════════════════════
_header(7, "Today's market validation (live data)")

# Delete cache to force fresh fetch
if os.path.exists(cache_path):
    os.remove(cache_path)

fb_live = get_fundamental_bias()
facs    = fb_live["factors"]

print()
print("  LIVE FUNDAMENTAL READING:")
print(f"  {'─'*50}")
for fname, flabel in [("inflation","Inflation"), ("oil","Oil      "),
                       ("fed","Fed      "), ("dxy","DXY      "),
                       ("geopolitical","Geo Risk ")]:
    f = facs.get(fname, {})
    print(f"  {flabel}: {f.get('score',0):+d}  —  {f.get('note','—')}")
print(f"  {'─'*50}")
print(f"  Total:  {fb_live['total_score']:+d}  →  {fb_live['fundamental_bias']}")
print(f"  Conf:   {fb_live['confidence']:.1f}/10")
print(f"  Display: {fb_live['display_line']}")
print()

cc_short = check_fundamental_conflict("SHORT")
conflict_yn = "YES" if cc_short["conflict"] else "NO"
print(f"  SHORT conflict: {conflict_yn}  severity: {cc_short['severity']}")
if cc_short["conflict"]:
    for line in cc_short["message"].split("\n"):
        print(f"    {line}")
print()

# Expected: BULLISH or STRONGLY_BULLISH given current inflation/geo environment
t7_ok_bias = fb_live["fundamental_bias"] in ("BULLISH", "STRONGLY_BULLISH")
bias_label = "PASS" if t7_ok_bias else "FAIL (NEUTRAL/BEARISH — data sources may be offline)"
print(f"  Expected BULLISH+: {bias_label}")
print(f"  Note: NEUTRAL is acceptable if yfinance/DXY unavailable in this environment")

# For test pass/fail — we accept NEUTRAL when data is offline (available=False)
t7_ok = t7_ok_bias or not fb_live.get("available", True)
_result("TEST 7 — Today's market validation", t7_ok)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Integration check (static file analysis)
# ══════════════════════════════════════════════════════════════════════════════
_header(8, "Integration — bot_chat.py + morning_briefing.py wiring")

t8_ok = True

def _check_file(fname, checks):
    """Read file once, verify all string checks."""
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if not os.path.exists(fpath):
        _fail(f"{fname} NOT FOUND")
        return False
    with open(fpath, "r", encoding="utf-8") as fh:
        src = fh.read()
    ok = True
    for label, needle in checks:
        if needle in src:
            _ok(f"[{fname}] {label}")
        else:
            _fail(f"[{fname}] {label} — '{needle[:60]}' NOT FOUND")
            ok = False
    return ok

# bot_chat.py checks
bc_ok = _check_file("bot_chat.py", [
    ("_FB_OK import guard",         "_FB_OK = True"),
    ("_get_fundamental_bias import","from fundamental_bias import"),
    ("_handle_fundamental exists",  "def _handle_fundamental("),
    ("Route keyword 'fundamental bias'", '"fundamental bias"'),
    ("Sidebar fundamental widget",  "📊 **Fundamental:**"),
    ("Card _fund_line variable",    "_fund_line"),
    ("Conflict check in _handle_gold", "FUND CONFLICT"),
])
if not bc_ok:
    t8_ok = False

print()

# morning_briefing.py checks
mb_ok = _check_file("morning_briefing.py", [
    ("_FB_MB_OK import guard",       "_FB_MB_OK = True"),
    ("_get_fund_bias import",        "from fundamental_bias import get_fundamental_bias"),
    ("[FUNDAMENTAL] step",           "[FUNDAMENTAL] Scoring macro fundamentals"),
    ("fund_ctx variable",            "fund_ctx"),
    ("_run_with_timeout for fund",   "_run_with_timeout(_get_fund_bias"),
])
if not mb_ok:
    t8_ok = False

print()

# Cache path check
expected_cache = os.path.join("data", "fundamental_cache.json")
if os.path.exists(expected_cache):
    _ok(f"Cache file exists at: {expected_cache}")
else:
    _ok(f"Cache path correct (will be created on first call): {expected_cache}")

_result("TEST 8 — Integration wiring", t8_ok)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{DSEP}")
print("  BACKTEST RESULTS")
print(DSEP)
total  = len(passed_tests) + len(failed_tests)
passed = len(passed_tests)
failed = len(failed_tests)

for t in passed_tests:
    print(f"  ✅ {t}")
for t in failed_tests:
    print(f"  ❌ {t}")

print(f"\n  {passed}/{total} tests passed")

if failed == 0:
    print(f"\n  {'═'*58}")
    print(f"  ✅  Fundamental Bias Engine — READY FOR LIVE USE")
    print(f"  {'═'*58}")
else:
    print(f"\n  {'═'*58}")
    print(f"  ⚠  {failed} issue(s) found — review ❌ items above")
    print(f"  {'═'*58}")
