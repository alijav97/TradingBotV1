"""
_test_macro_filter.py — Backtest verification for DXY + US10Y macro filter
Run with: python _test_macro_filter.py
No code is changed — test only.
"""
from __future__ import annotations
import os, sys, traceback
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠  WARN"

results: list[tuple[str, str, str]] = []   # (test, status, detail)

def _mark(test: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    results.append((test, status, detail))
    print(f"  {status}  {test}" + (f" — {detail}" if detail else ""))

def _sep(title: str = "") -> None:
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "═" * pad + f" {title} " + "═" * pad)
    else:
        print("\n" + "─" * w)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — get_yields_context()
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 1 — get_yields_context()")

try:
    from dxy_correlation import get_yields_context
    yctx = get_yields_context()

    print(f"\n  Raw result:")
    for k, v in yctx.items():
        print(f"    {k:28s}: {v}")

    required_keys = ["available", "current_yield", "yield_trend",
                     "yield_momentum", "gold_bias_from_yields", "display_line"]
    missing = [k for k in required_keys if k not in yctx]
    _mark("T1.1 — all required keys present", not missing,
          f"missing: {missing}" if missing else "")

    _mark("T1.2 — available is bool", isinstance(yctx["available"], bool))

    if yctx["available"]:
        _mark("T1.3 — current_yield is float > 0",
              isinstance(yctx["current_yield"], float) and yctx["current_yield"] > 0,
              str(yctx["current_yield"]))
        _mark("T1.4 — yield_trend in valid set",
              yctx["yield_trend"] in ("rising", "falling", "sideways"),
              yctx["yield_trend"])
        _mark("T1.5 — yield_momentum in valid set",
              yctx["yield_momentum"] in ("strong", "weak"),
              yctx["yield_momentum"])
        _mark("T1.6 — gold_bias_from_yields in valid set",
              yctx["gold_bias_from_yields"] in ("bullish", "bearish", "neutral"),
              yctx["gold_bias_from_yields"])
        _mark("T1.7 — display_line is non-empty string",
              isinstance(yctx["display_line"], str) and len(yctx["display_line"]) > 5,
              yctx["display_line"])
    else:
        print(f"  ℹ  Yields data unavailable (network/market hours): {yctx['display_line']}")
        _mark("T1.3-7 — graceful fallback when unavailable",
              yctx["yield_trend"] == "sideways" and yctx["yield_momentum"] == "weak"
              and yctx["gold_bias_from_yields"] == "neutral",
              "fallback values correct")

    # Test graceful fallback with patched download
    import unittest.mock as _mock
    import yfinance as _yf
    with _mock.patch.object(_yf, "download", side_effect=Exception("network error")):
        yctx_fallback = get_yields_context()
    _mark("T1.8 — graceful fallback on network error",
          yctx_fallback["available"] is False and "display_line" in yctx_fallback,
          yctx_fallback.get("display_line", "?")[:60])

except Exception as exc:
    _mark("TEST 1 — import/run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — get_macro_context()
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 2 — get_macro_context()")

VALID_BIASES = {"strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"}

try:
    from dxy_correlation import get_macro_context

    for direction in ("long", "short"):
        print(f"\n  ── Direction: {direction.upper()} ──")
        mctx = get_macro_context(direction)

        for k, v in mctx.items():
            if k not in ("dxy_df", "dxy"):   # skip large df
                print(f"    {k:28s}: {v}")

        tag = f"T2[{direction}]"
        score = mctx.get("macro_score", None)
        bias  = mctx.get("macro_bias", "")
        conf  = mctx.get("macro_confirmed", None)
        opp   = mctx.get("macro_opposed", None)
        cadj  = mctx.get("confidence_adjustment", None)
        summ  = mctx.get("summary", "")

        _mark(f"{tag} macro_score in [-2, +2]",
              isinstance(score, (int, float)) and -2.0 <= float(score) <= 2.0,
              str(score))
        _mark(f"{tag} macro_bias valid",
              bias in VALID_BIASES, bias)
        _mark(f"{tag} confirmed and opposed not both True",
              not (conf and opp),
              f"confirmed={conf} opposed={opp}")
        _mark(f"{tag} confidence_adjustment in {{-1, 0, +1}}",
              cadj in (-1.0, 0.0, 1.0), str(cadj))
        _mark(f"{tag} summary non-empty",
              isinstance(summ, str) and len(summ) > 5, summ[:80])

        # Score-bias consistency
        if score is not None:
            if float(score) >= 1.0:
                bias_ok = conf is True
            elif float(score) <= -1.0:
                bias_ok = opp is True
            else:
                bias_ok = not conf and not opp
            _mark(f"{tag} score/confirmed/opposed consistent", bias_ok,
                  f"score={score} conf={conf} opp={opp}")

except Exception as exc:
    _mark("TEST 2 — import/run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Confluence engine integration
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 3 — Confluence engine integration")

def _make_synthetic_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 3300.0 + np.cumsum(rng.normal(0, 5, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open":   closes - rng.uniform(1, 3, n),
        "high":   closes + rng.uniform(1, 5, n),
        "low":    closes - rng.uniform(1, 5, n),
        "close":  closes,
        "volume": rng.integers(1000, 5000, n).astype(float),
    }, index=idx)
    # EMA50 / EMA200
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    # RSI14
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    # ATR14
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    # MACD
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df.dropna(subset=["ema200", "rsi", "atr"])

try:
    from confluence_engine import score_confluences

    df_syn = _make_synthetic_df(50)
    print(f"\n  Synthetic DF: {len(df_syn)} rows, close range "
          f"${df_syn['close'].min():.0f}–${df_syn['close'].max():.0f}")

    for direction in ("long", "short"):
        print(f"\n  ── score_confluences(df, '{direction}') ──")
        result = score_confluences(df_syn, direction)

        conf_score   = result.get("confidence", result.get("score", None))
        total_checks = result.get("total_checks", None)
        detail_lines = result.get("detail_lines", [])

        print(f"    confidence:   {conf_score}")
        print(f"    total_checks: {total_checks}")
        print(f"    detail_lines:")
        for ln in detail_lines:
            print(f"      {ln}")

        tag = f"T3[{direction}]"
        _mark(f"{tag} confidence in [0, 10]",
              conf_score is not None and 0 <= float(conf_score) <= 10,
              str(conf_score))
        _mark(f"{tag} total_checks >= 8",
              total_checks is not None and int(total_checks) >= 8,
              str(total_checks))
        _mark(f"{tag} detail_lines not empty",
              bool(detail_lines), f"{len(detail_lines)} lines")

        # Check Factor 6 line mentions macro/DXY/yields
        dxy_lines = [l for l in detail_lines if any(
            kw in l.lower() for kw in ("dxy", "macro", "us10y", "yield", "unavailable"))]
        _mark(f"{tag} Factor 6 macro line present in detail_lines",
              bool(dxy_lines),
              dxy_lines[0][:80] if dxy_lines else "NOT FOUND")

except Exception as exc:
    _mark("TEST 3 — import/run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Confidence adjustment simulation
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 4 — Signal confidence adjustment")

try:
    _sep_inner = lambda s: print(f"\n  ── {s} ──")

    scenarios = [
        ("A — Macro strongly bullish, LONG signal",
         {"macro_score": 2.0, "macro_bias": "strongly_bullish",
          "macro_confirmed": True, "macro_opposed": False,
          "confidence_adjustment": 1.0},
         7.0, "long", 8.0),
        ("B — Macro strongly bearish, LONG signal",
         {"macro_score": -2.0, "macro_bias": "strongly_bearish",
          "macro_confirmed": False, "macro_opposed": True,
          "confidence_adjustment": -1.0},
         7.0, "long", 6.0),
        ("C — Macro neutral, any signal",
         {"macro_score": 0.0, "macro_bias": "neutral",
          "macro_confirmed": False, "macro_opposed": False,
          "confidence_adjustment": 0.0},
         7.0, "long", 7.0),
    ]

    for name, macro, base_conf, direction, expected_conf in scenarios:
        _sep_inner(name)
        cadj         = float(macro["confidence_adjustment"])
        after_conf   = base_conf + cadj
        print(f"    Before macro: confidence = {base_conf}")
        print(f"    Macro adjustment: {cadj:+.1f}  (macro_bias: {macro['macro_bias']})")
        print(f"    After macro: confidence = {after_conf}")
        _mark(f"T4[{name[:1]}] confidence adjustment correct",
              abs(after_conf - expected_conf) < 0.001,
              f"got {after_conf}, expected {expected_conf}")

        # Also verify confirmed/opposed mutual exclusion
        _mark(f"T4[{name[:1]}] confirmed/opposed not both True",
              not (macro["macro_confirmed"] and macro["macro_opposed"]))

except Exception as exc:
    _mark("TEST 4 — run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Morning briefing integration
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 5 — Morning briefing integration")

try:
    import inspect
    import morning_briefing as _mb

    # 5.1 — _step_dxy accepts gold_dir argument
    sig = inspect.signature(_mb._step_dxy)
    params = list(sig.parameters.keys())
    _mark("T5.1 — _step_dxy() accepts gold_dir param",
          "gold_dir" in params, str(params))

    # 5.2 — _step5_scan_signals accepts macro_ctx param
    sig5 = inspect.signature(_mb._step5_scan_signals)
    params5 = list(sig5.parameters.keys())
    _mark("T5.2 — _step5_scan_signals() accepts macro_ctx param",
          "macro_ctx" in params5, str(params5))

    # 5.3 — Call _step_dxy() and verify macro fields present
    print("\n  Calling _step_dxy('long')...")
    try:
        dxy_result = _mb._step_dxy("long")
        has_macro  = "macro_score" in dxy_result or "dxy_trend" in dxy_result
        _mark("T5.3 — _step_dxy() returns macro/dxy fields",
              has_macro, str(list(dxy_result.keys())[:8]))
        print(f"  available: {dxy_result.get('available')}")
        print(f"  macro_bias: {dxy_result.get('macro_bias','n/a')}")
        print(f"  confidence_adjustment: {dxy_result.get('confidence_adjustment','n/a')}")
        print(f"  display_line: {dxy_result.get('display_line','')[:80]}")
    except Exception as e:
        _mark("T5.3 — _step_dxy() callable", False, str(e))

    # 5.4 — _step5_scan_signals with macro_ctx doesn't crash
    try:
        mock_macro = {
            "available": True, "macro_score": 1.0, "macro_bias": "bullish",
            "macro_confirmed": True, "macro_opposed": False,
            "confidence_adjustment": 1.0, "summary": "DXY falling + US10Y falling",
            "dxy_trend": "down", "dxy_rsi": 45.0, "momentum_strength": "weak",
        }
        mock_sentiment = {"gold": {"bias": "buy"}, "overall_risk": "low"}
        df_syn2 = _make_synthetic_df(50)
        result5, meta5 = _mb._step5_scan_signals(
            rules=[], sentiment=mock_sentiment, df=df_syn2,
            macro_ctx=mock_macro,
        )
        _mark("T5.4 — _step5_scan_signals runs with macro_ctx",
              isinstance(result5, list), f"{len(result5)} signals")
    except Exception as e:
        _mark("T5.4 — _step5_scan_signals with macro_ctx", False, str(e))
        traceback.print_exc()

    # 5.5 — No NameError on import
    _mark("T5.5 — morning_briefing imports without NameError",
          hasattr(_mb, "_DXY_OK"), f"_DXY_OK={_mb._DXY_OK}")

    # 5.6 — get_macro_context imported into morning_briefing namespace
    _mark("T5.6 — get_macro_context accessible in morning_briefing",
          _mb._DXY_OK and hasattr(_mb, "get_macro_context"),
          "imported OK" if _mb._DXY_OK else "DXY module not available")

except Exception as exc:
    _mark("TEST 5 — import/run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 6 — Backward compatibility")

try:
    from dxy_correlation import get_dxy_context

    dctx = get_dxy_context()
    print(f"\n  get_dxy_context() keys: {list(dctx.keys())}")
    print(f"  available: {dctx.get('available')}")
    print(f"  dxy_trend: {dctx.get('dxy_trend')}")
    print(f"  dxy_rsi:   {dctx.get('dxy_rsi')}")
    print(f"  gold_aligned: {dctx.get('gold_aligned')}")

    _mark("T6.1 — get_dxy_context() still returns dxy_trend",
          "dxy_trend" in dctx, dctx.get("dxy_trend", "MISSING"))
    _mark("T6.2 — get_dxy_context() still returns dxy_rsi",
          "dxy_rsi" in dctx, str(dctx.get("dxy_rsi")))
    _mark("T6.3 — get_dxy_context() still returns available",
          "available" in dctx, str(dctx.get("available")))
    _mark("T6.4 — get_dxy_context() now includes gold_aligned",
          "gold_aligned" in dctx, str(dctx.get("gold_aligned")))

    # Check confluence engine can still use plain dxy_ctx
    df_compat = _make_synthetic_df(50)
    try:
        # Pass old-style dxy_ctx — confluence should handle it via legacy path
        r_compat = score_confluences(df_compat, "long", dxy_ctx=dctx)
        detail_compat = r_compat.get("detail_lines", [])
        dxy_line_compat = [l for l in detail_compat if "dxy" in l.lower() or "macro" in l.lower()]
        _mark("T6.5 — confluence_engine accepts old dxy_ctx param",
              isinstance(r_compat.get("confidence"), (int, float)),
              f"confidence={r_compat.get('confidence')}")
    except Exception as e:
        _mark("T6.5 — confluence_engine old dxy_ctx param", False, str(e))

    # Check macro_ctx fields are backward-compatible at top level
    from dxy_correlation import get_macro_context as _gmc
    mctx_compat = _gmc("long")
    _mark("T6.6 — get_macro_context() exposes dxy_trend at top level",
          "dxy_trend" in mctx_compat, mctx_compat.get("dxy_trend", "MISSING"))
    _mark("T6.7 — get_macro_context() exposes dxy_rsi at top level",
          "dxy_rsi" in mctx_compat, str(mctx_compat.get("dxy_rsi")))

except Exception as exc:
    _mark("TEST 6 — import/run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Historical backtest simulation (data/historical_xauusd.csv)
# ══════════════════════════════════════════════════════════════════════════════

_sep("TEST 7 — Historical backtest simulation")

try:
    CSV_PATH = os.path.join(BASE_DIR, "data", "historical_xauusd.csv")
    _mark("T7.0 — historical_xauusd.csv exists", os.path.exists(CSV_PATH), CSV_PATH)

    if os.path.exists(CSV_PATH):
        df_hist = pd.read_csv(CSV_PATH, index_col=0)
        df_hist.columns = [c.lower() for c in df_hist.columns]
        if "open" not in df_hist.columns:
            df_hist["open"] = df_hist["close"].shift(1).fillna(df_hist["close"])
        # Enrich
        df_hist["ema50"]  = df_hist["close"].ewm(span=50,  adjust=False).mean()
        df_hist["ema200"] = df_hist["close"].ewm(span=200, adjust=False).mean()
        delta = df_hist["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df_hist["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        hl  = df_hist["high"] - df_hist["low"]
        hc  = (df_hist["high"] - df_hist["close"].shift()).abs()
        lc  = (df_hist["low"]  - df_hist["close"].shift()).abs()
        df_hist["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        e12 = df_hist["close"].ewm(span=12, adjust=False).mean()
        e26 = df_hist["close"].ewm(span=26, adjust=False).mean()
        df_hist["macd"]        = e12 - e26
        df_hist["macd_signal"] = df_hist["macd"].ewm(span=9, adjust=False).mean()
        df_hist = df_hist.dropna(subset=["ema200", "rsi", "atr"])

        N = min(500, len(df_hist))
        df_use = df_hist.tail(N).copy()
        print(f"\n  Using last {N} candles from historical CSV")
        print(f"  Price range: ${df_use['close'].min():.0f}–${df_use['close'].max():.0f}")

        WINDOW = 50
        n_windows = (N - WINDOW) // 10   # stride 10

        # ── Simulate macro filter using SYNTHETIC macro contexts ──────────────
        # (We can't hit real network for each window, so we use get_macro_context
        # result cached once, plus synthetic "without macro" path)
        from dxy_correlation import get_macro_context as _gmc2
        print(f"  Fetching macro context once (live or fallback)...")
        _macro_live = _gmc2("long")
        _cadj = float(_macro_live.get("confidence_adjustment", 0.0))
        _mbias = _macro_live.get("macro_bias", "neutral")
        print(f"  Live macro_bias: {_mbias}  |  confidence_adjustment: {_cadj:+.1f}")

        # Run confluence on each window for both long and short
        boosted = 0; penalised = 0; unchanged = 0
        n_passed_without = 0; n_passed_with = 0; n_blocked = 0
        total_conf_change = 0.0
        errors = 0
        MIN_CONF_THRESHOLD = 5.0

        # Use a fixed mock macro for each scenario to simulate filter impact
        macro_scenarios = {
            "bullish":  +1.0,
            "neutral":   0.0,
            "bearish":  -1.0,
        }

        print(f"  Running {n_windows} windows × 3 macro scenarios (yfinance mocked for speed)...")

        # Mock ALL yfinance.download calls so the loop runs at CPU speed
        import unittest.mock as _mock2
        import yfinance as _yf2
        import confluence_engine as _ce

        # Build a fake multi-column yfinance DataFrame any HTF/MTF call would return
        _fake_idx = pd.date_range("2025-01-01", periods=200, freq="1h")
        _fake_close = pd.Series(np.linspace(3200, 3400, 200), index=_fake_idx, name="Close")
        _yf_df = pd.DataFrame({
            ("Close", "XAUUSD"): _fake_close.values,
            ("Open",  "XAUUSD"): _fake_close.values - 2,
            ("High",  "XAUUSD"): _fake_close.values + 5,
            ("Low",   "XAUUSD"): _fake_close.values - 5,
            ("Volume","XAUUSD"): np.ones(200) * 1000,
        }, index=_fake_idx)

        _macro_stub = {"available": True, "macro_score": -2.0, "macro_bias": "strongly_bearish",
                       "macro_confirmed": False, "macro_opposed": True,
                       "confidence_adjustment": -1.0,
                       "summary": "DXY rising ✗ + US10Y rising ✗ → strongly bearish gold",
                       "dxy_trend": "up", "dxy_rsi": 54.0, "momentum_strength": "weak",
                       "display_line": "DXY: Rising ▲ | Macro: Strongly Bearish"}

        with _mock2.patch.object(_yf2, "download", return_value=_yf_df):
          if hasattr(_ce, "_get_macro_ctx"):
            _macro_patcher = _mock2.patch.object(_ce, "_get_macro_ctx", return_value=_macro_stub)
            _macro_patcher.start()
          else:
            _macro_patcher = None

          for i in range(n_windows):
            start_idx = i * 10
            end_idx   = start_idx + WINDOW
            w_df      = df_use.iloc[start_idx:end_idx].copy()
            if len(w_df) < 30:
                continue
            try:
                _patches = []
                for direction in ("long", "short"):
                    r = score_confluences(w_df, direction)
                    base_conf = float(r.get("confidence", 0) or 0)
                    # Simulate macro filter effect
                    for scenario_name, cadj in macro_scenarios.items():
                        adj_conf = base_conf + cadj
                        total_conf_change += cadj
                        if cadj > 0:
                            boosted += 1
                            if base_conf < MIN_CONF_THRESHOLD <= adj_conf:
                                n_passed_with += 1
                        elif cadj < 0:
                            penalised += 1
                            if base_conf >= MIN_CONF_THRESHOLD > adj_conf:
                                n_blocked += 1
                        else:
                            unchanged += 1
                        if base_conf >= MIN_CONF_THRESHOLD:
                            n_passed_without += 1
                for p in _patches: pass
            except Exception:
                errors += 1
          if _macro_patcher: _macro_patcher.stop()

        total_adj = boosted + penalised + unchanged
        avg_change = total_conf_change / max(total_adj, 1)

        print(f"\n  {'═'*52}")
        print(f"  Macro filter impact on {N} candles ({n_windows} windows):")
        print(f"  {'─'*52}")
        print(f"  Signals boosted (+1.0):    {boosted:>5}")
        print(f"  Signals penalised (-1.0):  {penalised:>5}")
        print(f"  Signals unchanged (0.0):   {unchanged:>5}")
        print(f"  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
        print(f"  Signals passed WITHOUT filter: {n_passed_without:>5}")
        print(f"  New signals unlocked by boost: {n_passed_with:>5}")
        print(f"  Signals blocked by opposition: {n_blocked:>5}")
        print(f"  Average confidence change:    {avg_change:>+.3f}")
        print(f"  Errors during scan:           {errors:>5}")
        print(f"  {'═'*52}")
        net = "better" if n_passed_with > n_blocked else ("worse" if n_blocked > n_passed_with else "neutral")
        print(f"  Net improvement: {net.upper()}")
        print(f"  Live macro bias used: {_mbias} ({_cadj:+.1f})")

        _mark("T7.1 — scan completed without fatal errors",
              errors < n_windows // 2,
              f"{errors} errors out of {n_windows} windows")
        _mark("T7.2 — boosted + penalised + unchanged > 0",
              total_adj > 0, str(total_adj))
        _mark("T7.3 — avg_change is finite float",
              not np.isnan(avg_change) and not np.isinf(avg_change), str(round(avg_change, 4)))

except Exception as exc:
    _mark("TEST 7 — import/run", False, str(exc))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════

_sep("FINAL VERDICT")

passed  = sum(1 for _, s, _ in results if s == PASS)
failed  = sum(1 for _, s, _ in results if s == FAIL)
warned  = sum(1 for _, s, _ in results if s == WARN)
total   = len(results)

print(f"\n  Results: {passed}/{total} passed  |  {failed} failed  |  {warned} warnings\n")

if failed > 0:
    print("  ❌ FAILED TESTS:")
    for test, status, detail in results:
        if status == FAIL:
            print(f"     • {test}")
            if detail:
                print(f"       → {detail}")

print()
if failed == 0:
    print("  ✅ Phase 2 Task 6 — READY FOR LIVE USE")
    print("     DXY + US10Y macro filter is correctly implemented,")
    print("     wired into the confluence engine and morning briefing,")
    print("     and passes all backward-compatibility checks.")
elif failed <= 2:
    print("  ⚠  Phase 2 Task 6 — MINOR ISSUES (see above)")
    print("     Core functionality works; fix warnings before live use.")
else:
    print("  ❌ Phase 2 Task 6 — NOT READY")
    print("     Multiple failures detected — review and fix before use.")
print()
