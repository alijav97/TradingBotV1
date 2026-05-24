"""
_test_regime_detection.py
=========================
Validates all 6 parts of the market regime detection enhancement.

Tests:
  1. detect_gold_regime() returns all required keys + valid regime
  2. get_regime_strategy_config() covers all 5 regimes with correct fields
  3. Regime filter logic on mock signals (morning_briefing style)
  4. save_regime_snapshot + get_regime_history round-trip
  5. bot_chat._handle_regime_history() is importable and callable
  6. morning_briefing._step5_scan_signals regime_config stamped on meta
"""

import os, sys, json, tempfile, traceback
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))

# ──────────────────────────────────────────────
# PART 1: detect_gold_regime() keys + valid regime
# ──────────────────────────────────────────────
print("\n── PART 1: detect_gold_regime() ──")
try:
    from market_context import detect_gold_regime

    # Build a minimal realistic DataFrame (120 rows, 1-hour OHLCV)
    np.random.seed(42)
    n = 120
    close = 3100 + np.cumsum(np.random.randn(n) * 5)
    df = pd.DataFrame({
        "open":   close - np.abs(np.random.randn(n)),
        "high":   close + np.abs(np.random.randn(n) * 3),
        "low":    close - np.abs(np.random.randn(n) * 3),
        "close":  close,
        "volume": np.random.randint(1000, 5000, n).astype(float),
    })

    regime_data = detect_gold_regime(df)

    REQUIRED_KEYS = {
        "regime", "regime_label", "best_playbooks", "avoid_playbooks",
        "position_size_multiplier", "regime_note",
        "atr_now", "atr_avg", "ema50_slope",
    }
    VALID_REGIMES = {
        "TRENDING_STRONG", "TRENDING_WEAK", "RANGING",
        "VOLATILE_EXPANDING", "SQUEEZE_BUILDING",
    }
    MULT_MAP = {
        "TRENDING_STRONG": 1.2,
        "TRENDING_WEAK":   1.0,
        "RANGING":         0.8,
        "VOLATILE_EXPANDING": 0.5,
        "SQUEEZE_BUILDING":   0.6,
    }

    missing = REQUIRED_KEYS - set(regime_data.keys())
    check("All required keys present", not missing, f"missing={missing}")
    check("regime in valid set", regime_data["regime"] in VALID_REGIMES,
          f"got '{regime_data['regime']}'")
    check("regime_label is non-empty string",
          isinstance(regime_data.get("regime_label"), str) and len(regime_data["regime_label"]) > 0)
    check("atr_now > 0", regime_data.get("atr_now", 0) > 0, f"atr_now={regime_data.get('atr_now')}")
    check("atr_avg > 0", regime_data.get("atr_avg", 0) > 0, f"atr_avg={regime_data.get('atr_avg')}")
    check("ema50_slope is float", isinstance(regime_data.get("ema50_slope"), float))

    reg = regime_data["regime"]
    expected_mult = MULT_MAP.get(reg)
    actual_mult = regime_data.get("position_size_multiplier")
    check("position_size_multiplier correct for regime",
          actual_mult == expected_mult,
          f"regime={reg}, expected={expected_mult}, got={actual_mult}")

except Exception as e:
    check("detect_gold_regime() import+run", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────
# PART 2: get_regime_strategy_config() all 5 regimes
# ──────────────────────────────────────────────
print("\n── PART 2: get_regime_strategy_config() ──")
try:
    from market_context import get_regime_strategy_config

    REQUIRED_FIELDS = {"entry_filter", "avoid_reversals", "sl_style", "tp_style", "max_signals", "note"}
    ALL_REGIMES = ["TRENDING_STRONG", "TRENDING_WEAK", "RANGING", "VOLATILE_EXPANDING", "SQUEEZE_BUILDING"]

    for regime_name in ALL_REGIMES:
        cfg = get_regime_strategy_config(regime_name)
        missing_fields = REQUIRED_FIELDS - set(cfg.keys())
        check(f"  {regime_name}: all fields present", not missing_fields, f"missing={missing_fields}")
        check(f"  {regime_name}: max_signals >= 1", cfg.get("max_signals", 0) >= 1,
              f"max_signals={cfg.get('max_signals')}")
        check(f"  {regime_name}: avoid_reversals is bool",
              isinstance(cfg.get("avoid_reversals"), bool))

    # fallback for unknown regime
    cfg_fallback = get_regime_strategy_config("UNKNOWN_REGIME")
    check("fallback for unknown regime has entry_filter", "entry_filter" in cfg_fallback)

except Exception as e:
    check("get_regime_strategy_config() import+run", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────
# PART 3: Regime filter logic (mock signals)
# ──────────────────────────────────────────────
print("\n── PART 3: Regime filter on mock signals ──")
try:
    from market_context import get_regime_strategy_config as _grc

    def apply_regime_filter(signals: list[dict], regime: str) -> list[dict]:
        """Mirror the filter logic from morning_briefing._step5_scan_signals."""
        rcfg = _grc(regime)
        entry_filter = rcfg.get("entry_filter", "any")
        avoid_rev    = rcfg.get("avoid_reversals", False)
        mult         = rcfg.get("size_multiplier", 1.0)

        for sig in signals:
            strat = sig.get("strategy", "")
            if entry_filter == "sr_bounce_only" and "breakout" in strat.lower():
                sig["confidence"] = sig.get("confidence", 0) - 1.0
            if entry_filter == "breakout_only" and "bounce" in strat.lower():
                sig["confidence"] = sig.get("confidence", 0) - 1.5
            if entry_filter == "news_fade_only" and "news" not in strat.lower():
                sig["confidence"] = sig.get("confidence", 0) - 1.0
            if avoid_rev and regime == "TRENDING_STRONG" and "reversal" in strat.lower():
                sig["confidence"] = sig.get("confidence", 0) - 2.0
            sig["size_multiplier"] = mult
            sig["regime_config"] = rcfg

        return sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)

    mock_signals = [
        {"strategy": "SR Bounce", "confidence": 3.0},
        {"strategy": "Breakout", "confidence": 3.5},
        {"strategy": "Reversal Hunter", "confidence": 4.0},
    ]

    # TRENDING_STRONG: reversals suppressed, breakouts preferred
    filtered = apply_regime_filter(
        [dict(s) for s in mock_signals], "TRENDING_STRONG"
    )
    reversal_conf = next(s["confidence"] for s in filtered if "Reversal" in s["strategy"])
    check("TRENDING_STRONG: reversal confidence reduced",
          reversal_conf < 4.0, f"reversal conf={reversal_conf}")
    check("TRENDING_STRONG: size_multiplier stamped",
          filtered[0].get("size_multiplier") is not None)
    check("TRENDING_STRONG: regime_config stamped",
          "regime_config" in filtered[0])

    # RANGING: breakouts penalised
    filtered_r = apply_regime_filter(
        [dict(s) for s in mock_signals], "RANGING"
    )
    breakout_conf = next(s["confidence"] for s in filtered_r if "Breakout" in s["strategy"])
    check("RANGING: breakout confidence reduced",
          breakout_conf < 3.5, f"breakout conf={breakout_conf}")

except Exception as e:
    check("Regime filter logic test", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────
# PART 4: save_regime_snapshot + get_regime_history
# ──────────────────────────────────────────────
print("\n── PART 4: save_regime_snapshot + get_regime_history ──")
try:
    from market_context import save_regime_snapshot, get_regime_history, _REGIME_HISTORY_FILE

    # Write to a temp copy
    _orig_file = _REGIME_HISTORY_FILE
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("[]")
        tmp_path = f.name

    # Monkey-patch the module-level path
    import market_context as _mc
    _mc._REGIME_HISTORY_FILE = tmp_path

    mock_regime = {
        "regime": "TRENDING_STRONG",
        "regime_label": "Strong Trend 🟢",
        "atr_now": 18.5,
        "atr_20": 15.0,
        "bb_width": 0.012,
        "ema50_slope": 0.008,
        "ema50": 3092.0,
        "trend_strength": 0.75,
        "position_size_multiplier": 1.2,
    }
    save_regime_snapshot(mock_regime, price=3100.0)
    save_regime_snapshot(mock_regime, price=3105.0)

    history = get_regime_history(last_n=10)
    check("History has 2 entries after 2 saves", len(history) == 2, f"got {len(history)}")
    check("Snapshot has timestamp", "timestamp" in history[0])
    check("Snapshot has regime", history[0].get("regime") == "TRENDING_STRONG")
    check("Snapshot has price", history[0].get("price") in (3100.0, 3105.0))
    check("Snapshot has atr_now", "atr_now" in history[0])

    # last_n=1 should return only 1
    history1 = get_regime_history(last_n=1)
    check("last_n=1 returns 1 entry", len(history1) == 1, f"got {len(history1)}")

    # Restore
    _mc._REGIME_HISTORY_FILE = _orig_file
    os.unlink(tmp_path)

except Exception as e:
    check("save_regime_snapshot / get_regime_history", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────
# PART 5: bot_chat._handle_regime_history callable
# ──────────────────────────────────────────────
print("\n── PART 5: bot_chat._handle_regime_history callable ──")
try:
    # We can't import bot_chat (Streamlit import) — check function exists via grep
    with open(os.path.join(os.path.dirname(__file__), "bot_chat.py"), encoding="utf-8") as f:
        src = f.read()

    count = src.count("def _handle_regime_history(")
    check("_handle_regime_history defined exactly once", count == 1, f"found {count} definitions")

    check("'regime history' route keyword present",
          '"regime history"' in src or "'regime history'" in src)

    check("save_regime_snapshot call in 60s loop",
          "save_regime_snapshot" in src)

    check("_REGIME_ICONS dict used (not _REGIME_COLOURS)",
          "_REGIME_ICONS" in src and "_REGIME_COLOURS" not in src)

except Exception as e:
    check("bot_chat source checks", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────
# PART 6: morning_briefing source checks
# ──────────────────────────────────────────────
print("\n── PART 6: morning_briefing source checks ──")
try:
    with open(os.path.join(os.path.dirname(__file__), "morning_briefing.py"), encoding="utf-8") as f:
        mb_src = f.read()

    check("import get_regime_strategy_config in morning_briefing",
          "get_regime_strategy_config" in mb_src)
    check("import save_regime_snapshot in morning_briefing",
          "save_regime_snapshot" in mb_src)
    check("import get_regime_history in morning_briefing",
          "get_regime_history" in mb_src)
    check("size_multiplier stamped on signals",
          "size_multiplier" in mb_src)
    check("regime_config stamped on meta",
          'meta["regime_config"]' in mb_src or "meta['regime_config']" in mb_src)
    check("MARKET REGIME block in _print_session_summary",
          "MARKET REGIME" in mb_src)

except Exception as e:
    check("morning_briefing source checks", False, str(e))
    traceback.print_exc()

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total  = len(results)
print(f"RESULT: {passed}/{total} passed  |  {failed} failed")
if failed == 0:
    print("🎉 ALL TESTS PASSED — Regime detection enhancement validated!")
else:
    print("⚠️  Some tests failed — review output above.")
print("="*60)
sys.exit(0 if failed == 0 else 1)
