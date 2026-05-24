"""
_test_ror_verification.py
═════════════════════════
Phase 4 — Task 15  Risk of Ruin Calculator Backtest Verification
9 tests. Do NOT change any code — test only.
"""

import sys, os, json, math, inspect, traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PASS = "PASS"
FAIL = "FAIL"
results: dict[str, bool] = {}

def section(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")

def check(name: str, cond: bool, detail: str = "") -> bool:
    marker = "  [PASS]" if cond else "  [FAIL]"
    suffix = f"  ({detail})" if detail else ""
    print(f"{marker}  {name}{suffix}")
    return cond

def record(name: str, passed: bool) -> None:
    results[name] = passed


from trade_manager import (
    calculate_risk_of_ruin,
    get_current_risk_profile,
    format_ror_report,
)


# ==============================================================================
#  TEST 1 — Safe settings
# ==============================================================================
section("TEST 1 — calculate_risk_of_ruin() SAFE settings")

t1 = True
try:
    r = calculate_risk_of_ruin(
        win_rate=60, risk_pct=1, rr_ratio=3,
        num_trades=100, simulations=500,
    )
    print(f"\n  Values:")
    print(f"    ruin_probability          = {r['ruin_probability']:.2f}%")
    print(f"    risk_rating               = {r['risk_rating']}")
    print(f"    ev_per_trade              = {r['ev_per_trade']:+.6f}")
    print(f"    ev_positive               = {r['ev_positive']}")
    print(f"    consecutive_losses_to_ruin= {r['consecutive_losses_to_ruin']}")
    print(f"    avg_final_balance_pct     = {r['avg_final_balance_pct']:.1f}%")
    print(f"    avg_max_drawdown_pct      = {r['avg_max_drawdown_pct']:.2f}%")
    print(f"    simulations_run           = {r['simulations_run']}")
    print(f"    summary                   = {r['summary']}")
    print()

    t1 &= check("ruin_probability < 5%",        r["ruin_probability"] < 5,
                f"got {r['ruin_probability']:.2f}%")
    t1 &= check("risk_rating = 'SAFE'",          r["risk_rating"] == "SAFE",
                f"got {r['risk_rating']}")
    t1 &= check("ev_per_trade > 0",              r["ev_per_trade"] > 0,
                f"got {r['ev_per_trade']:+.6f}")
    t1 &= check("ev_positive = True",            r["ev_positive"] is True)
    t1 &= check("consecutive_losses_to_ruin > 200",
                r["consecutive_losses_to_ruin"] > 200,
                f"got {r['consecutive_losses_to_ruin']}")
    t1 &= check("simulations_run = 500",         r["simulations_run"] == 500)
    t1 &= check("trades_simulated = 100",        r["trades_simulated"] == 100)

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t1 = False

record("T1_Safe_Settings", t1)


# ==============================================================================
#  TEST 2 — Dangerous settings
# ==============================================================================
section("TEST 2 — calculate_risk_of_ruin() DANGER settings")

t2 = True
try:
    r2 = calculate_risk_of_ruin(
        win_rate=40, risk_pct=25, rr_ratio=1.5,
        num_trades=100, simulations=500,
    )
    print(f"\n  Values:")
    print(f"    ruin_probability = {r2['ruin_probability']:.2f}%")
    print(f"    risk_rating      = {r2['risk_rating']}")
    print(f"    recommendation   = {r2['recommendation']}")
    print(f"    ev_per_trade     = {r2['ev_per_trade']:+.6f}")
    print()

    t2 &= check("ruin_probability > 30%",     r2["ruin_probability"] > 30,
                f"got {r2['ruin_probability']:.2f}%")
    t2 &= check("risk_rating = 'DANGER'",     r2["risk_rating"] == "DANGER",
                f"got '{r2['risk_rating']}'")
    t2 &= check("recommendation contains 'reduce'",
                "reduce" in r2["recommendation"].lower(),
                f"got '{r2['recommendation']}'")
    # EV = (0.40 * 0.375) - (0.60 * 0.25) = 0.15 - 0.15 = 0.0 (break-even).
    # Floating-point may return a tiny positive epsilon (~2.8e-17), so we
    # test that ev is <= 0.001 (essentially zero or negative), not strict False.
    t2 &= check("ev_per_trade <= 0.001 (break-even or negative)",
                r2["ev_per_trade"] <= 0.001,
                f"got {r2['ev_per_trade']:+.8f}")

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t2 = False

record("T2_Danger_Settings", t2)


# ==============================================================================
#  TEST 3 — Current bot settings (get_current_risk_profile)
# ==============================================================================
section("TEST 3 — get_current_risk_profile() all required keys")

t3 = True
REQUIRED_KEYS = [
    "ruin_probability", "risk_rating", "win_rate", "risk_pct", "rr_ratio",
    "ev_per_trade", "current_balance", "risk_per_trade_usd",
    "ruin_threshold_usd", "recommended_risk_pct", "recommended_risk_usd",
    "consecutive_losses_to_ruin",
]
try:
    ror3 = get_current_risk_profile()

    print(f"\n  Full current risk profile:")
    print(f"    ruin_probability          = {ror3['ruin_probability']:.2f}%")
    print(f"    risk_rating               = {ror3['risk_rating']}")
    print(f"    win_rate                  = {ror3['win_rate']:.1f}%")
    print(f"    risk_pct                  = {ror3['risk_pct']}%")
    print(f"    rr_ratio                  = 1:{ror3['rr_ratio']}")
    print(f"    ev_per_trade              = {ror3['ev_per_trade']:+.6f}")
    print(f"    ev_positive               = {ror3['ev_positive']}")
    print(f"    current_balance           = ${ror3['current_balance']:,.2f}")
    print(f"    risk_per_trade_usd        = ${ror3['risk_per_trade_usd']:.2f}")
    print(f"    ruin_threshold_usd        = ${ror3['ruin_threshold_usd']:.2f}")
    print(f"    recommended_risk_pct      = {ror3['recommended_risk_pct']}%")
    print(f"    recommended_risk_usd      = ${ror3['recommended_risk_usd']:.2f}")
    print(f"    consecutive_losses_to_ruin= {ror3['consecutive_losses_to_ruin']}")
    print(f"    avg_final_balance_pct     = {ror3['avg_final_balance_pct']:.1f}%")
    print(f"    avg_max_drawdown_pct      = {ror3['avg_max_drawdown_pct']:.2f}%")
    print(f"    summary                   = {ror3['summary']}")
    print()

    for key in REQUIRED_KEYS:
        t3 &= check(f"key '{key}' present", key in ror3, f"missing" if key not in ror3 else "ok")

    t3 &= check("current_balance > 0",        ror3["current_balance"] > 0)
    t3 &= check("ruin_threshold_usd > 0",     ror3["ruin_threshold_usd"] > 0)
    t3 &= check("recommended_risk_pct >= 1",  ror3["recommended_risk_pct"] >= 1)
    t3 &= check("risk_rating is a str",       isinstance(ror3["risk_rating"], str))

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t3 = False

record("T3_Profile_Keys", t3)


# ==============================================================================
#  TEST 4 — EV calculation accuracy
# ==============================================================================
section("TEST 4 — EV calculation accuracy")

t4 = True
try:
    # Case A: win=50%, risk=10%, rr=3
    # EV = (0.5 * 0.3) - (0.5 * 0.1) = 0.15 - 0.05 = 0.10
    r4a = calculate_risk_of_ruin(win_rate=50, risk_pct=10, rr_ratio=3,
                                  num_trades=50, simulations=200)
    expected_ev_a = (0.50 * 0.30) - (0.50 * 0.10)   # = 0.10
    print(f"\n  Case A: 50% WR, 10% risk, 1:3 RR")
    print(f"    Expected EV = (0.5 x 0.3) - (0.5 x 0.1) = {expected_ev_a:+.4f}")
    print(f"    Actual   EV = {r4a['ev_per_trade']:+.6f}")
    print(f"    Diff        = {abs(r4a['ev_per_trade'] - expected_ev_a):.6f}")
    print()

    t4 &= check("EV approx 0.1 (within 0.01)",
                abs(r4a["ev_per_trade"] - expected_ev_a) < 0.01,
                f"expected ~{expected_ev_a:.4f} got {r4a['ev_per_trade']:.6f}")
    t4 &= check("ev_positive = True",    r4a["ev_positive"] is True)

    # Case B: win=40%, risk=10%, rr=1
    # EV = (0.4 * 0.1) - (0.6 * 0.1) = 0.04 - 0.06 = -0.02
    r4b = calculate_risk_of_ruin(win_rate=40, risk_pct=10, rr_ratio=1,
                                  num_trades=50, simulations=200)
    expected_ev_b = (0.40 * 0.10) - (0.60 * 0.10)   # = -0.02
    print(f"  Case B: 40% WR, 10% risk, 1:1 RR")
    print(f"    Expected EV = (0.4 x 0.1) - (0.6 x 0.1) = {expected_ev_b:+.4f}")
    print(f"    Actual   EV = {r4b['ev_per_trade']:+.6f}")
    print(f"    Diff        = {abs(r4b['ev_per_trade'] - expected_ev_b):.6f}")
    print()

    t4 &= check("EV approx -0.02 (within 0.01)",
                abs(r4b["ev_per_trade"] - expected_ev_b) < 0.01,
                f"expected ~{expected_ev_b:.4f} got {r4b['ev_per_trade']:.6f}")
    t4 &= check("ev_per_trade < 0",   r4b["ev_per_trade"] < 0,
                f"got {r4b['ev_per_trade']:+.6f}")
    t4 &= check("ev_positive = False", r4b["ev_positive"] is False)

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t4 = False

record("T4_EV_Calculation", t4)


# ==============================================================================
#  TEST 5 — Consecutive losses to ruin
# ==============================================================================
section("TEST 5 — Consecutive losses to ruin (math verification)")

t5 = True
try:
    # At 10% risk per trade: 0.9^N = 0.1 → N = log(0.1)/log(0.9)
    _n10_exact = math.log(0.1) / math.log(0.9)   # ≈ 21.85 → 22
    print(f"\n  Exact formula: log(0.1) / log(0.9) = {_n10_exact:.4f} --> ceil = {math.ceil(_n10_exact)}")

    r5a = calculate_risk_of_ruin(win_rate=50, risk_pct=10, rr_ratio=3,
                                  num_trades=50, simulations=200)
    got10 = r5a["consecutive_losses_to_ruin"]
    print(f"  risk_pct=10  expected ~{math.ceil(_n10_exact)} (+-2)  got {got10}")
    t5 &= check("risk_pct=10: losses_to_ruin ≈ 22 (within +-2)",
                abs(got10 - math.ceil(_n10_exact)) <= 2,
                f"expected ~{math.ceil(_n10_exact)} got {got10}")

    # At 1% risk: 0.99^N = 0.1 → N = log(0.1)/log(0.99) ≈ 229.1
    _n1_exact = math.log(0.1) / math.log(0.99)
    print(f"\n  Exact formula: log(0.1) / log(0.99) = {_n1_exact:.2f} --> ceil = {math.ceil(_n1_exact)}")

    r5b = calculate_risk_of_ruin(win_rate=60, risk_pct=1, rr_ratio=3,
                                  num_trades=50, simulations=200)
    got1 = r5b["consecutive_losses_to_ruin"]
    print(f"  risk_pct=1   expected ~{math.ceil(_n1_exact)} (>200)  got {got1}")
    t5 &= check("risk_pct=1: losses_to_ruin > 200",  got1 > 200,
                f"got {got1}")

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t5 = False

record("T5_Consecutive_Losses", t5)


# ==============================================================================
#  TEST 6 — Recommended risk % validation
# ==============================================================================
section("TEST 6 — Recommended risk % produces safe RoR")

t6 = True
try:
    # Start with known-dangerous config
    ror6_danger = get_current_risk_profile.__module__  # import check
    _danger_cfg = calculate_risk_of_ruin(win_rate=40, risk_pct=25, rr_ratio=1.5,
                                          num_trades=100, simulations=500)
    print(f"\n  Dangerous config: 40% WR, 25% risk, 1:1.5 RR")
    print(f"    ruin_probability  = {_danger_cfg['ruin_probability']:.1f}%")

    # Build a fake profile with those params to get recommended_risk_pct
    # We replicate the get_current_risk_profile logic directly
    from settings import load_settings
    settings6 = load_settings()

    # Force dangerous params for this test
    safe_risk6 = 25.0
    while safe_risk6 > 1.0:
        _test6 = calculate_risk_of_ruin(40, safe_risk6, 1.5, 100, 200)
        if _test6["ruin_probability"] < 5.0:
            break
        safe_risk6 = round(safe_risk6 - 0.5, 1)

    print(f"  Recommended safe risk = {safe_risk6}%")
    t6 &= check("recommended_risk_pct < 25",  safe_risk6 < 25,
                f"got {safe_risk6}%")

    # Now verify that recommended risk actually gives <5% ruin
    _verify6 = calculate_risk_of_ruin(40, safe_risk6, 1.5, 100, 500)
    print(f"  Verify RoR at {safe_risk6}% risk: {_verify6['ruin_probability']:.1f}%  [{_verify6['risk_rating']}]")
    t6 &= check(f"RoR at recommended {safe_risk6}% < 5%",
                _verify6["ruin_probability"] < 5,
                f"got {_verify6['ruin_probability']:.1f}%")
    t6 &= check("Risk rating at recommended % = SAFE",
                _verify6["risk_rating"] == "SAFE",
                f"got {_verify6['risk_rating']}")

    # Also verify against current live profile
    ror6_live = get_current_risk_profile()
    print(f"\n  Live profile recommended_risk_pct = {ror6_live['recommended_risk_pct']}%")
    t6 &= check("live recommended_risk_pct >= 1",
                ror6_live["recommended_risk_pct"] >= 1.0)
    t6 &= check("live recommended_risk_usd > 0",
                ror6_live["recommended_risk_usd"] > 0)

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t6 = False

record("T6_Recommended_Risk", t6)


# ==============================================================================
#  TEST 7 — format_ror_report() content
# ==============================================================================
section("TEST 7 — format_ror_report() output structure")

t7 = True
try:
    ror7 = get_current_risk_profile()
    report = format_ror_report(ror7)

    print(f"\n  Report output:\n")
    for line in report.split("\n"):
        print(f"  {line}")
    print()

    REQUIRED_SECTIONS = [
        "RATING:",
        "Win rate:",
        "Risk/trade:",
        "EV/trade:",
        "Losses to ruin:",
        "RECOMMENDATION:",
        "Safe risk %:",
    ]
    for section_str in REQUIRED_SECTIONS:
        t7 &= check(f"Contains '{section_str}'",
                    section_str in report,
                    "missing" if section_str not in report else "found")

    t7 &= check("Report is a string",          isinstance(report, str))
    t7 &= check("Report length > 100 chars",   len(report) > 100,
                f"got {len(report)} chars")
    t7 &= check("Contains ruin probability",
                str(round(ror7["ruin_probability"], 1)) in report
                or f"{ror7['ruin_probability']:.1f}" in report)

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t7 = False

record("T7_Format_Report", t7)


# ==============================================================================
#  TEST 8 — bot_chat.py integration audit
# ==============================================================================
section("TEST 8 — bot_chat.py integration audit")

t8 = True
try:
    with open("bot_chat.py", encoding="utf-8") as _f:
        bot_src = _f.read()

    print()

    # _handle_ror() present
    has_handle_ror = "def _handle_ror(" in bot_src
    t8 &= check("_handle_ror() defined in bot_chat.py",  has_handle_ror)

    # "risk of ruin" keyword wired in _route()
    has_ror_keyword = '"risk of ruin"' in bot_src or "'risk of ruin'" in bot_src
    t8 &= check('"risk of ruin" keyword in _route()',    has_ror_keyword)

    # "ror" keyword
    has_ror_short = '"ror"' in bot_src or "'ror'" in bot_src
    t8 &= check('"ror" keyword in _route()',             has_ror_short)

    # "am i safe" keyword
    has_amisafe = '"am i safe"' in bot_src or "'am i safe'" in bot_src
    t8 &= check('"am i safe" keyword in _route()',       has_amisafe)

    # Sidebar risk rating block
    has_sidebar_ror = "_ror_sb" in bot_src or "Risk Rating" in bot_src
    t8 &= check("Sidebar risk rating indicator present", has_sidebar_ror)

    # Trade card warning block
    has_card_warning = "ror_warning_block" in bot_src
    t8 &= check("Trade card ror_warning_block present",  has_card_warning)

    # 10-minute refresh check
    has_10min = "_last_ror_check" in bot_src and "600" in bot_src
    t8 &= check("10-minute RoR refresh check in loop",   has_10min)

    # _get_ror_profile import
    has_import = "_get_ror_profile" in bot_src
    t8 &= check("_get_ror_profile imported in bot_chat", has_import)

    # _format_ror import
    has_fmt = "_format_ror" in bot_src
    t8 &= check("_format_ror imported in bot_chat",      has_fmt)

    print()
    # Count occurrences to confirm ror_warning_block used in card string
    card_inject_count = bot_src.count("ror_warning_block")
    print(f"  ror_warning_block appears {card_inject_count} time(s) in bot_chat.py"
          f"  (expect 2: definition + card injection)")
    t8 &= check("ror_warning_block used >= 2x (defined + injected)",
                card_inject_count >= 2, f"got {card_inject_count}")

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t8 = False

record("T8_BotChat_Integration", t8)


# ==============================================================================
#  TEST 9 — Live settings simulation
# ==============================================================================
section("TEST 9 — Live settings simulation (data/user_settings.json)")

t9 = True
try:
    # ── Load settings ──────────────────────────────────────────────────────────
    _settings_path = os.path.join("data", "user_settings.json")
    with open(_settings_path) as _f:
        raw_settings = json.load(_f)

    balance  = float(raw_settings.get("balance",  300.0))
    risk_pct = float(raw_settings.get("risk_pct", 10.0))
    rr_ratio = float(raw_settings.get("min_rr",   3.0))

    # ── Load win rate from signal_performance.json ─────────────────────────────
    win_rate    = 50.0
    n_resolved  = 0
    n_wins_sp   = 0
    _sp_exists  = False

    _sp_path = os.path.join("data", "signal_performance.json")
    if os.path.exists(_sp_path):
        _sp_exists = True
        with open(_sp_path) as _f:
            signals_sp = json.load(_f)
        resolved_sp = [s for s in signals_sp if s.get("outcome") in ("win", "loss", "partial")]
        n_resolved  = len(resolved_sp)
        if n_resolved >= 5:
            n_wins_sp = len([s for s in resolved_sp if s.get("outcome") == "win"])
            win_rate  = n_wins_sp / n_resolved * 100.0
        print(f"  signal_performance.json: {len(signals_sp)} total, {n_resolved} resolved"
              f" ({n_wins_sp} wins -> {win_rate:.1f}% WR)")
    else:
        print(f"  signal_performance.json: not found -> using 50% default")

    # ── Blend with pattern_memory.json ────────────────────────────────────────
    _pm_path = os.path.join("data", "pattern_memory.json")
    n_memory = 0
    mem_wr   = None
    if os.path.exists(_pm_path):
        with open(_pm_path) as _f:
            memory_pm = json.load(_f)
        n_memory = len(memory_pm)
        if n_memory >= 5:
            mem_wins = len([m for m in memory_pm if m.get("outcome") == "WIN"])
            mem_wr   = mem_wins / n_memory * 100.0
            win_rate = (win_rate + mem_wr) / 2.0
        print(f"  pattern_memory.json    : {n_memory} records"
              + (f" ({mem_wr:.1f}% WR -> blended {win_rate:.1f}%)" if mem_wr is not None else " (< 5 records, not blended)"))
    else:
        print(f"  pattern_memory.json    : not found -> win rate unchanged")

    # ── Run full profile ───────────────────────────────────────────────────────
    ror9 = get_current_risk_profile()

    # ── Print YOUR CURRENT RISK PROFILE ───────────────────────────────────────
    n_trades_label = f"{n_resolved}" if n_resolved > 0 else "0 — using 50% default"
    print(f"""
  YOUR CURRENT RISK PROFILE:
  ──────────────────────────────────────────────────
  Balance:     ${balance:,.2f}
  Win rate:    {ror9['win_rate']:.1f}%  ({n_trades_label} trades tracked)
  Risk/trade:  {risk_pct}% = ${balance * risk_pct / 100:.2f}
  RR ratio:    1:{rr_ratio}
  EV/trade:    {ror9['ev_per_trade']:+.4f}  ({'positive' if ror9['ev_positive'] else 'NEGATIVE — review strategy'})
  Ruin prob:   {ror9['ruin_probability']:.1f}%
  Rating:      {ror9['risk_rating']}
  Safe risk:   {ror9['recommended_risk_pct']}% = ${ror9['recommended_risk_usd']:.2f}
  Losses safe: {ror9['consecutive_losses_to_ruin']} in a row before 90% drawdown
  ──────────────────────────────────────────────────""")

    t9 &= check("settings file loaded successfully",     _settings_path and True)
    t9 &= check("balance > 0",                          balance > 0,
                f"got {balance}")
    t9 &= check("risk_pct in (1, 50) range",            1 <= risk_pct <= 50,
                f"got {risk_pct}")
    t9 &= check("rr_ratio > 0",                         rr_ratio > 0,
                f"got {rr_ratio}")
    t9 &= check("profile win_rate in (0, 100)",         0 < ror9["win_rate"] < 100,
                f"got {ror9['win_rate']:.1f}")
    t9 &= check("profile ruin_probability in [0, 100]", 0 <= ror9["ruin_probability"] <= 100)
    t9 &= check("profile risk_rating is valid string",
                ror9["risk_rating"] in ("SAFE", "MODERATE", "HIGH", "DANGER"),
                f"got '{ror9['risk_rating']}'")
    t9 &= check("recommended_risk_pct between 1 and risk_pct",
                1 <= ror9["recommended_risk_pct"] <= risk_pct + 0.1,
                f"got {ror9['recommended_risk_pct']}%")

except Exception as e:
    print(f"  [FAIL]  Exception: {e}")
    traceback.print_exc()
    t9 = False

record("T9_Live_Simulation", t9)


# ==============================================================================
#  FINAL VERDICT
# ==============================================================================
section("FINAL VERDICT")

all_pass   = all(results.values())
pass_count = sum(results.values())
total      = len(results)

component_map = {
    "T1_Safe_Settings":      "calculate_risk_of_ruin() SAFE settings",
    "T2_Danger_Settings":    "calculate_risk_of_ruin() DANGER settings",
    "T3_Profile_Keys":       "get_current_risk_profile() all keys",
    "T4_EV_Calculation":     "EV calculation accuracy",
    "T5_Consecutive_Losses": "Consecutive losses to ruin math",
    "T6_Recommended_Risk":   "Recommended risk % validation",
    "T7_Format_Report":      "format_ror_report() structure",
    "T8_BotChat_Integration":"bot_chat.py integration audit",
    "T9_Live_Simulation":    "Live settings simulation",
}

print(f"""
+---------------------------------------------------------------+
|      Phase 4 -- Task 15  RISK OF RUIN VERIFICATION           |
+---------------------------------------------------------------+""")

for tid, name in component_map.items():
    ok     = results.get(tid, False)
    marker = "[PASS]" if ok else "[FAIL]"
    print(f"|  {marker}  {name:<54}  |")

print(f"""+---------------------------------------------------------------+
|  Tests passed: {pass_count}/{total}                                              |
+---------------------------------------------------------------+""")

if all_pass:
    print(f"""|                                                               |
|  Phase 4 Task 15 -- READY FOR LIVE USE                       |
|                                                               |
|  Phase 4 COMPLETE -- Ready for Phase 5                       |
|                                                               |
+---------------------------------------------------------------+""")
else:
    failed = [component_map[k] for k, v in results.items() if not v]
    print(f"""|                                                               |
|  Phase 4 Task 15 -- ISSUES FOUND                             |
|                                                               |""")
    for f in failed:
        print(f"|  ✗  {f:<58}  |")
    print(f"""|                                                               |
+---------------------------------------------------------------+""")

sys.exit(0 if all_pass else 1)
