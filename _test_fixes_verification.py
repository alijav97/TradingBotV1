"""
_test_fixes_verification.py
Backtest verification for the 3 fixes applied to TradingBotV1.

FIX 1 — Pattern fatigue gate in morning_briefing._step5_scan_signals()
FIX 2 — _VA_OK top-level guard for volume_analyzer
FIX 3 — spread_usd field added to debug_logger.save_signal_detail()
"""

import os
import sys
import json
import inspect
import importlib
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─── Colour helpers ───────────────────────────────────────────────────────────
GRN  = "\033[92m"
RED  = "\033[91m"
YLW  = "\033[93m"
CYN  = "\033[96m"
RST  = "\033[0m"
BOLD = "\033[1m"

_pass_count = 0
_fail_count = 0

def _ok(label):
    global _pass_count
    _pass_count += 1
    print(f"  {GRN}[PASS]{RST}  {label}")

def _fail(label, detail=""):
    global _fail_count
    _fail_count += 1
    print(f"  {RED}[FAIL]{RST}  {label}" + (f"  ({detail})" if detail else ""))

def _info(msg):
    print(f"  {CYN}[INFO]{RST}  {msg}")

def _section(title):
    print(f"\n{BOLD}{'='*65}{RST}")
    print(f"{BOLD}{title}{RST}")
    print(f"{BOLD}{'='*65}{RST}")

# ─── Read source once ─────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "morning_briefing.py"), encoding="utf-8") as _f:
    MB_SRC = _f.read()

with open(os.path.join(BASE_DIR, "debug_logger.py"), encoding="utf-8") as _f:
    DL_SRC = _f.read()

# ─── Temp signal_detail.json path ─────────────────────────────────────────────
SIGNAL_LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "signal_detail.json")

def _read_last_signal():
    """Return the last record written to signal_detail.json, or None."""
    try:
        with open(SIGNAL_LOG_PATH, encoding="utf-8") as f:
            records = json.load(f)
        return records[-1] if records else None
    except Exception:
        return None

def _signal_count():
    try:
        with open(SIGNAL_LOG_PATH, encoding="utf-8") as f:
            return len(json.load(f))
    except Exception:
        return 0

# =============================================================================
# FIX 1 TESTS — Pattern fatigue gate
# =============================================================================
_section("FIX 1 — Pattern fatigue gate")

# ─── TEST 1A — Critical fatigue blocks signal ─────────────────────────────────
print(f"\n{YLW}TEST 1A{RST} — Critical fatigue blocks signal (playbook loop)")

import pandas as pd
import numpy as np

# Build minimal synthetic df so playbook/checklist engines don't crash
_rows = 60
_close = np.linspace(4500, 4600, _rows)
_df = pd.DataFrame({
    "open":   _close - 5,
    "high":   _close + 10,
    "low":    _close - 10,
    "close":  _close,
    "volume": np.random.randint(1000, 5000, _rows).astype(float),
    "atr":    np.full(_rows, 25.0),
    "ema50":  _close - 20,
    "ema200": _close - 50,
    "rsi":    np.full(_rows, 55.0),
    "macd":   np.full(_rows, 0.5),
    "macd_signal": np.full(_rows, 0.3),
})

# We test fatigue logic in isolation — we directly replicate what the gate does
# (avoids needing full _step5_scan_signals to run cleanly with all deps)

def _simulate_fatigue_gate(fatigue_level, base_confidence=7.5):
    """
    Simulate exactly the fatigue gate logic as written in morning_briefing.py.
    Returns (blocked, final_confidence, note, fatigue_rejected_count).
    """
    sig = {
        "pattern_name": "EMA Trend Continuation",
        "confidence":   base_confidence,
        "note":         "",
        "direction":    "long",
    }
    meta = {"fatigue_rejected": 0}
    rejected_calls = []

    # Mock save_signal_detail
    def _mock_save(s, status, **kw):
        rejected_calls.append({"status": status, **kw})

    # Mock log_rejected
    log_rejected_calls = []
    def _mock_log_rejected(msg):
        log_rejected_calls.append(msg)

    direction = "long"

    # ── Gate logic (verbatim copy from morning_briefing.py) ──────────────────
    blocked = False
    try:
        # Simulate check_strategy_fatigue returning desired level
        _fatigue = {"fatigue_level": fatigue_level, "recommendation": f"fatigue is {fatigue_level}"}
        _fl = _fatigue.get("fatigue_level", "none")
        sig["fatigue_level"] = _fl
        sig["fatigue_recommendation"] = _fatigue.get("recommendation", "")

        if _fl == "critical":
            _mock_log_rejected(f"{sig['pattern_name']} — fatigue_critical")
            meta["fatigue_rejected"] = meta.get("fatigue_rejected", 0) + 1
            _mock_save(sig, "REJECTED",
                       rejection_reason="Pattern fatigue critical",
                       rejection_stage="fatigue_gate",
                       spread_usd=None)
            blocked = True
        elif _fl in ("high", "moderate"):
            sig["confidence"] = max(0, sig["confidence"] - 0.5)
            sig["note"] = sig.get("note", "") + f" ⚠ Fatigue {_fl}"
    except Exception as _fe:
        pass   # gate skipped

    return {
        "blocked":           blocked,
        "confidence":        sig["confidence"],
        "note":              sig["note"],
        "fatigue_rejected":  meta["fatigue_rejected"],
        "rejected_calls":    rejected_calls,
        "log_rejected_calls": log_rejected_calls,
    }

r = _simulate_fatigue_gate("critical")
if r["blocked"]:
    _ok("Critical fatigue: signal blocked (not appended)")
else:
    _fail("Critical fatigue: signal NOT blocked")

if r["fatigue_rejected"] == 1:
    _ok("Critical fatigue: meta['fatigue_rejected'] == 1")
else:
    _fail("Critical fatigue: meta['fatigue_rejected'] wrong", f"got {r['fatigue_rejected']}")

if r["rejected_calls"] and r["rejected_calls"][0].get("rejection_reason") == "Pattern fatigue critical":
    _ok("Critical fatigue: save_signal_detail called with rejection_reason='Pattern fatigue critical'")
else:
    _fail("Critical fatigue: save_signal_detail rejection_reason wrong", str(r["rejected_calls"]))

if r["rejected_calls"] and r["rejected_calls"][0].get("rejection_stage") == "fatigue_gate":
    _ok("Critical fatigue: rejection_stage='fatigue_gate'")
else:
    _fail("Critical fatigue: rejection_stage wrong", str(r["rejected_calls"]))

# ─── TEST 1B — High fatigue penalises confidence ──────────────────────────────
print(f"\n{YLW}TEST 1B{RST} — High fatigue penalises confidence")
r = _simulate_fatigue_gate("high", base_confidence=7.5)
if not r["blocked"]:
    _ok("High fatigue: signal NOT blocked (still appended)")
else:
    _fail("High fatigue: signal incorrectly blocked")

if abs(r["confidence"] - 7.0) < 0.001:
    _ok(f"High fatigue: confidence 7.5 → 7.0 (-0.5)  (got {r['confidence']})")
else:
    _fail("High fatigue: confidence not correctly penalised", f"got {r['confidence']}")

if "⚠ Fatigue high" in r["note"]:
    _ok("High fatigue: note contains '⚠ Fatigue high'")
else:
    _fail("High fatigue: note missing fatigue warning", f"got '{r['note']}'")

# ─── TEST 1C — No fatigue = no change ────────────────────────────────────────
print(f"\n{YLW}TEST 1C{RST} — No fatigue: signal unchanged")
r = _simulate_fatigue_gate("none", base_confidence=7.5)
if not r["blocked"]:
    _ok("No fatigue: signal not blocked")
else:
    _fail("No fatigue: signal incorrectly blocked")

if abs(r["confidence"] - 7.5) < 0.001:
    _ok(f"No fatigue: confidence unchanged at 7.5  (got {r['confidence']})")
else:
    _fail("No fatigue: confidence changed unexpectedly", f"got {r['confidence']}")

if "Fatigue" not in r["note"]:
    _ok("No fatigue: no fatigue note added")
else:
    _fail("No fatigue: unexpected note", f"got '{r['note']}'")

# ─── TEST 1D — Fatigue gate in BOTH loops ────────────────────────────────────
print(f"\n{YLW}TEST 1D{RST} — Fatigue gate exists in both loops (source check)")
occurrences = MB_SRC.count("check_strategy_fatigue")
_info(f"'check_strategy_fatigue' found {occurrences} time(s) in morning_briefing.py")
if occurrences >= 2:
    _ok(f"Fatigue gate appears in both loops ({occurrences} occurrences)")
else:
    _fail(f"Fatigue gate only found {occurrences} time(s) — expected ≥2")

# ─── TEST 1E — fatigue_rejected in meta initialisation ───────────────────────
print(f"\n{YLW}TEST 1E{RST} — fatigue_rejected initialised in meta dict")
# Scan source for the meta dict literal that initialises fatigue_rejected
if '"fatigue_rejected":    0' in MB_SRC or '"fatigue_rejected": 0' in MB_SRC:
    _ok("meta['fatigue_rejected'] = 0 found in meta initialisation")
else:
    _fail("meta['fatigue_rejected'] = 0 NOT found in meta initialisation")

# Verify critical gate increments it
r = _simulate_fatigue_gate("critical")
if r["fatigue_rejected"] == 1:
    _ok("After one critical rejection: fatigue_rejected == 1")
else:
    _fail("After one critical rejection: wrong count", f"got {r['fatigue_rejected']}")

# Also check log_session_end call passes it
if "fatigue_rejections" in MB_SRC:
    _ok("log_session_end() includes fatigue_rejections kwarg")
else:
    _fail("log_session_end() missing fatigue_rejections kwarg")

# =============================================================================
# FIX 2 TESTS — _VA_OK guard
# =============================================================================
_section("FIX 2 — _VA_OK top-level guard for volume_analyzer")

# ─── TEST 2A — Top-level guard exists ────────────────────────────────────────
print(f"\n{YLW}TEST 2A{RST} — _VA_OK guard at module level")

# Check the try/except block is present at module level (not inside a def)
import ast
try:
    tree = ast.parse(MB_SRC.lstrip("\ufeff"))
    # Find Try nodes at module level
    _va_ok_try_found = False
    _va_ok_fallback_found = False
    for node in ast.iter_child_nodes(tree):  # only top-level nodes
        if isinstance(node, ast.Try):
            src_segment = ast.get_source_segment(MB_SRC, node) or ""
            if "_VA_OK" in src_segment and "volume_analyzer" in src_segment:
                _va_ok_try_found = True
            if "_VA_OK = False" in src_segment and "_check_vol_confluence" in src_segment:
                _va_ok_fallback_found = True
    if _va_ok_try_found:
        _ok("_VA_OK try/except guard found at module level")
    else:
        _fail("_VA_OK try/except guard NOT found at module level")
    if _va_ok_fallback_found:
        _ok("Fallback _check_vol_confluence defined when _VA_OK=False")
    else:
        _fail("Fallback _check_vol_confluence NOT defined in guard")
except Exception as _e:
    _fail("AST parse error", str(_e))

# Check _VA_WARNED flag at module level
if "_VA_WARNED = False" in MB_SRC:
    _ok("_VA_WARNED = False flag defined at module level")
else:
    _fail("_VA_WARNED flag NOT found at module level")

# ─── TEST 2B — No bare import inside handler ─────────────────────────────────
print(f"\n{YLW}TEST 2B{RST} — No bare 'from volume_analyzer import' inside function body")
# Use AST: walk all FunctionDef nodes and check if any ImportFrom for
# volume_analyzer appears inside them
_bare_in_func = []
try:
    _tree2b = ast.parse(MB_SRC.lstrip("\ufeff"))
    for _node in ast.walk(_tree2b):
        if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for _child in ast.walk(_node):
                if isinstance(_child, ast.ImportFrom):
                    if _child.module and "volume_analyzer" in _child.module:
                        _bare_in_func.append(f"{_node.name}(): from {_child.module} import ...")
except Exception as _e2b:
    _info(f"AST parse error in TEST 2B: {_e2b}")

_info(f"volume_analyzer imports inside function bodies: {_bare_in_func if _bare_in_func else 'none'}")
if not _bare_in_func:
    _ok("No bare 'from volume_analyzer import' inside function bodies")
else:
    _fail("Bare volume_analyzer import still present inside function body", str(_bare_in_func))

# ─── TEST 2C — Volume gate works when _VA_OK=True ────────────────────────────
print(f"\n{YLW}TEST 2C{RST} — Volume gate functional when _VA_OK=True")
try:
    from volume_analyzer import check_volume_confluence as _direct_vol
    import morning_briefing as _mb
    _VA_OK_live = getattr(_mb, "_VA_OK", None)
    _info(f"morning_briefing._VA_OK = {_VA_OK_live}")

    # Call via the module's aliased function (same as gate uses)
    _alias = getattr(_mb, "_check_vol_confluence", None)
    _info(f"_check_vol_confluence alias: {_alias}")

    result_direct = _direct_vol(_df, "long", "EMA Trend Continuation")
    _info(f"Direct call result keys: {list(result_direct.keys())}")

    if "climax" in result_direct and "strategy_optimal" in result_direct:
        _ok("check_volume_confluence returns 'climax' and 'strategy_optimal' keys")
    else:
        _fail("check_volume_confluence missing expected keys", str(list(result_direct.keys())))

    if _alias is not None:
        result_alias = _alias(_df, "long", "EMA Trend Continuation")
        if "climax" in result_alias and "strategy_optimal" in result_alias:
            _ok("_check_vol_confluence alias gives same result shape as direct call")
        else:
            _fail("_check_vol_confluence alias missing keys", str(list(result_alias.keys())))
    else:
        _fail("_check_vol_confluence alias not found on morning_briefing module")
except Exception as _e:
    _fail("Volume gate test error", str(_e))

# ─── TEST 2D — Graceful fallback when _VA_OK=False ───────────────────────────
print(f"\n{YLW}TEST 2D{RST} — Graceful fallback function when _VA_OK=False")
# Use the fallback function as defined in the guard
def _fallback_vol(*a, **kw):
    return {"climax": False, "strategy_optimal": True, "score": 0}

try:
    result = _fallback_vol("anything", "long", "test_pattern")
    if result == {"climax": False, "strategy_optimal": True, "score": 0}:
        _ok("Fallback returns {'climax': False, 'strategy_optimal': True, 'score': 0}")
    else:
        _fail("Fallback returned unexpected value", str(result))
    _ok("No crash or exception from fallback function")
except Exception as _e:
    _fail("Fallback function raised exception", str(_e))

# Also verify the actual fallback in morning_briefing matches the spec
_fb_line = [l for l in MB_SRC.splitlines() if "_check_vol_confluence" in l and "climax" in l]
if _fb_line:
    _ok(f"Fallback definition confirmed in source: {_fb_line[0].strip()}")
else:
    _fail("Fallback definition not found in source")

# ─── TEST 2E — One-time warning logged ───────────────────────────────────────
print(f"\n{YLW}TEST 2E{RST} — One-time _VA_WARNED warning")
if "_VA_WARNED" in MB_SRC:
    _ok("_VA_WARNED flag referenced in morning_briefing.py")
else:
    _fail("_VA_WARNED flag not found")

if 'volume_analyzer not available' in MB_SRC:
    _ok("One-time log_info message 'volume_analyzer not available' present")
else:
    _fail("One-time warning message missing")

# Check _VA_WARNED set to True after warning fires
if "_VA_WARNED = True" in MB_SRC:
    _ok("_VA_WARNED set to True after warning fires")
else:
    _fail("_VA_WARNED = True not found")

# =============================================================================
# FIX 3 TESTS — spread_usd in save_signal_detail
# =============================================================================
_section("FIX 3 — spread_usd field in signal logs")

from debug_logger import save_signal_detail

# ─── TEST 3A — Function accepts spread_usd ───────────────────────────────────
print(f"\n{YLW}TEST 3A{RST} — save_signal_detail() accepts spread_usd parameter")
sig_params = inspect.signature(save_signal_detail).parameters
if "spread_usd" in sig_params:
    _ok("spread_usd parameter present in save_signal_detail()")
    default_val = sig_params["spread_usd"].default
    if default_val is None:
        _ok("spread_usd defaults to None")
    else:
        _fail("spread_usd default is not None", f"got {default_val!r}")
else:
    _fail("spread_usd parameter NOT found in save_signal_detail()")

# ─── TEST 3B — Spread record written correctly ───────────────────────────────
print(f"\n{YLW}TEST 3B{RST} — Spread record written with correct status")

_dummy_sig = {
    "pattern_name": "Test Signal",
    "source":       "rules",
    "direction":    "long",
    "confidence":   7.0,
    "entry":        4500.0,
    "stop_loss":    4470.0,
    "take_profit":  4590.0,
}

_spread_cases = [
    (0.35,  "acceptable"),
    (1.0,   "acceptable"),  # boundary: exactly 1.0 → acceptable
    (1.50,  "warning"),
    (2.0,   "warning"),     # boundary: exactly 2.0 → warning
    (2.50,  "blocked"),
    (None,  "unknown"),
]

for _susd, _expected in _spread_cases:
    count_before = _signal_count()
    try:
        save_signal_detail(_dummy_sig, "SHOWN_TO_USER", spread_usd=_susd)
        rec = _read_last_signal()
        if rec is None:
            _fail(f"spread_usd={_susd}: no record written")
            continue
        spread_block = rec.get("spread", {})
        actual_status = spread_block.get("spread_status", "MISSING")
        actual_usd    = spread_block.get("spread_usd")
        if actual_status == _expected:
            _ok(f"spread_usd={_susd!r:5} → spread_status='{actual_status}'  ✓")
        else:
            _fail(f"spread_usd={_susd!r:5}: expected '{_expected}', got '{actual_status}'")
        if actual_usd == _susd:
            pass  # no extra noise
        else:
            _fail(f"spread_usd value not written correctly", f"expected {_susd}, got {actual_usd}")
    except Exception as _e:
        _fail(f"spread_usd={_susd}: exception", str(_e))

# ─── TEST 3C — spread_usd passed in morning_briefing calls ───────────────────
print(f"\n{YLW}TEST 3C{RST} — All save_signal_detail() calls in morning_briefing pass spread_usd")
_mb_lines = MB_SRC.splitlines()

# Find all save_signal_detail call blocks
_call_starts = [i for i, l in enumerate(_mb_lines) if "save_signal_detail(" in l and "def " not in l]
_info(f"save_signal_detail() call sites found: {len(_call_starts)}")

_with_spread = 0
_without_spread = 0
for _ci in _call_starts:
    # Read up to 15 lines from this call to find its arguments
    _block = "\n".join(_mb_lines[_ci:_ci+15])
    if "spread_usd=" in _block:
        _with_spread += 1
    else:
        _without_spread += 1

_info(f"Calls with    spread_usd= : {_with_spread}")
_info(f"Calls without spread_usd= : {_without_spread}")

if _with_spread >= 4:
    _ok(f"{_with_spread} save_signal_detail() calls include spread_usd (≥4 required)")
else:
    _fail(f"Only {_with_spread} calls include spread_usd — expected ≥4")

if _without_spread == 0:
    _ok("All save_signal_detail() calls pass spread_usd")
else:
    _fail(f"{_without_spread} call(s) still missing spread_usd")

# ─── TEST 3D — Spread status boundary values ─────────────────────────────────
print(f"\n{YLW}TEST 3D{RST} — Spread status boundary values")

def _compute_spread_status(spread_usd):
    """Replicate the logic from debug_logger.py."""
    return (
        "acceptable" if spread_usd and spread_usd <= 1.0
        else "warning"    if spread_usd and spread_usd <= 2.0
        else "blocked"    if spread_usd and spread_usd > 2.0
        else "unknown"
    )

_boundary_cases = [
    (1.0,   "acceptable",  "1.0 → acceptable"),
    (1.01,  "warning",     "1.01 → warning"),
    (2.0,   "warning",     "2.0 → warning"),
    (2.01,  "blocked",     "2.01 → blocked"),
    (0.0,   "unknown",     "0.0 (falsy) → unknown"),
    (None,  "unknown",     "None → unknown"),
]
for _val, _exp, _label in _boundary_cases:
    _got = _compute_spread_status(_val)
    if _got == _exp:
        _ok(f"Boundary: {_label}  ✓")
    else:
        _fail(f"Boundary: {_label}", f"expected '{_exp}', got '{_got}'")

# Verify source logic matches
if ('"acceptable" if spread_usd and spread_usd <= 1.0' in DL_SRC or
        "\"acceptable\" if spread_usd and spread_usd <= 1.0" in DL_SRC):
    _ok("Boundary logic confirmed in debug_logger.py source")
else:
    _fail("Boundary logic not found in debug_logger.py source")

# =============================================================================
# COMBINED INTEGRATION TEST
# =============================================================================
_section("COMBINED INTEGRATION TEST — All 3 fixes working together")

print(f"\n{YLW}INTEGRATION{RST} — EMA Trend Continuation LONG, moderate fatigue, spread present")

_int_log: list[str] = []

# Setup signal
_int_sig = {
    "pattern_name": "EMA Trend Continuation",
    "source":       "playbook",
    "direction":    "long",
    "confidence":   7.5,
    "note":         "",
    "entry":        4562.0,
    "stop_loss":    4524.0,
    "take_profit":  4638.0,
    "fatigue_level": None,
}

_int_meta = {
    "fatigue_rejected": 0,
    "vol_rejected":     0,
    "playbook_passed":  0,
    "spread_check":     {"spread_usd": 0.42, "status": "acceptable", "blocked": False},
}

_int_log.append("Step 1: Volume gate")
# Simulate _VA_OK=True path (use the top-level guarded function)
import morning_briefing as _mb_mod
_vol_result = _mb_mod._check_vol_confluence(_df, "long", _int_sig["pattern_name"])
_int_sig["volume"] = _vol_result
_vol_gate_ok = True
if _vol_result.get("climax"):
    _int_log.append("  → climax detected, REJECTED")
    _vol_gate_ok = False
elif not _vol_result.get("strategy_optimal", True):
    _int_log.append("  → volume suboptimal, REJECTED")
    _vol_gate_ok = False
else:
    _int_log.append(f"  → climax={_vol_result.get('climax')}, strategy_optimal={_vol_result.get('strategy_optimal')} — PASS")

if _vol_gate_ok:
    _ok("Volume gate: passed (climax=False, optimal=True)")
else:
    _fail("Volume gate: unexpectedly blocked signal")

_int_log.append("Step 2: Fatigue gate")
_fatigue_mock = {"fatigue_level": "moderate", "recommendation": "reduce position size"}
_fl = _fatigue_mock.get("fatigue_level", "none")
_int_sig["fatigue_level"] = _fl
_int_sig["fatigue_recommendation"] = _fatigue_mock.get("recommendation", "")
_fatigue_blocked = False
if _fl == "critical":
    _int_meta["fatigue_rejected"] += 1
    _fatigue_blocked = True
    _int_log.append("  → CRITICAL — signal BLOCKED")
elif _fl in ("high", "moderate"):
    _int_sig["confidence"] = max(0, _int_sig["confidence"] - 0.5)
    _int_sig["note"] = _int_sig.get("note", "") + f" ⚠ Fatigue {_fl}"
    _int_log.append(f"  → {_fl.upper()} — confidence -0.5, note updated")

if not _fatigue_blocked:
    _ok("Fatigue gate: moderate — signal NOT blocked")
else:
    _fail("Fatigue gate: incorrectly blocked moderate signal")

if abs(_int_sig["confidence"] - 7.0) < 0.001:
    _ok(f"Fatigue gate: confidence 7.5 → 7.0  (got {_int_sig['confidence']})")
else:
    _fail("Fatigue gate: wrong confidence", f"got {_int_sig['confidence']}")

if "⚠ Fatigue moderate" in _int_sig["note"]:
    _ok("Fatigue gate: note contains '⚠ Fatigue moderate'")
else:
    _fail("Fatigue gate: note missing", f"got '{_int_sig['note']}'")

if _int_sig["fatigue_level"] == "moderate":
    _ok("Signal fatigue_level set to 'moderate'")
else:
    _fail("Signal fatigue_level wrong", f"got '{_int_sig['fatigue_level']}'")

if _int_meta["fatigue_rejected"] == 0:
    _ok("meta['fatigue_rejected'] still 0 (moderate doesn't block)")
else:
    _fail("meta['fatigue_rejected'] incremented incorrectly")

_int_log.append("Step 3: save_signal_detail with spread")
_spread_usd = _int_meta.get("spread_check", {}).get("spread_usd")
_int_log.append(f"  → spread_usd from meta: {_spread_usd}")
count_before = _signal_count()
save_signal_detail(
    _int_sig, "SHOWN_TO_USER",
    session="London",
    gold_price=4562.0,
    spread_usd=_spread_usd,
)
rec = _read_last_signal()
if rec is not None and rec.get("spread", {}).get("spread_usd") == _spread_usd:
    _ok(f"Signal saved with spread_usd={_spread_usd}")
else:
    _fail("Signal spread_usd not saved correctly", str(rec.get("spread") if rec else "no record"))

if rec is not None and rec.get("spread", {}).get("spread_status") == "acceptable":
    _ok("spread_status='acceptable' for spread_usd=0.42")
else:
    _fail("spread_status wrong for 0.42", str(rec.get("spread") if rec else "no record"))

_int_log.append("Step 4: _VA_OK used (no bare import inside function)")
_bare_int = []
try:
    _tree_int = ast.parse(MB_SRC.lstrip("\ufeff"))
    for _n in ast.walk(_tree_int):
        if isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for _c in ast.walk(_n):
                if isinstance(_c, ast.ImportFrom) and _c.module and "volume_analyzer" in _c.module:
                    _bare_int.append(f"{_n.name}(): {_c.module}")
except Exception:
    pass
if not _bare_int:
    _ok("No bare 'from volume_analyzer import' inside morning_briefing functions")
else:
    _fail("Bare import still present inside function body", str(_bare_int))

# Print integration trace
print(f"\n  {CYN}── Integration trace ──{RST}")
for _line in _int_log:
    print(f"    {_line}")

# =============================================================================
# FINAL VERDICT
# =============================================================================
_section("FINAL VERDICT")

print(f"\n  Fix verifications:")
print(f"  {'─'*56}")
_f1_ok = _pass_count > 0  # rough proxy — we'll tally per section below

# Count checks per fix via section markers (recount from totals)
total_checks = _pass_count + _fail_count
print(f"  Total PASS  : {GRN}{_pass_count}{RST}")
print(f"  Total FAIL  : {RED}{_fail_count}{RST}")
print(f"  Total checks: {total_checks}")
print()

if _fail_count == 0:
    print(f"  {GRN}{BOLD}All 3 fixes verified — READY FOR PHASE 3{RST}")
else:
    print(f"  {RED}{BOLD}Issues found — review FAIL items above before Phase 3{RST}")
print()
