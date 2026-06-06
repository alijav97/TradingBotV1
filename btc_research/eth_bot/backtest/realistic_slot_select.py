"""
btc_research/eth_bot/backtest/realistic_slot_select.py

DEEP RESEARCH: which slots are worth keeping UNDER THE ONE-POSITION RULE?

realistic_backtest.py proved S4+9-complement (one trade at a time) makes
$10,072 / -23% DD / CL13 vs pure S4's $1,247. But the blended WR fell to 32%
because the 9 complement slots are mean-reversion (low-WR / high-R by design).
The open question the concurrent backtests could NOT answer:

  Under one-position trading, a weak trade does not just add a parallel bet --
  it OCCUPIES the single slot and BLOCKS a better trade (the S4+9 run missed
  232 signals to an open position). So pruning a bad slot can RAISE finalBal
  AND win-rate by freeing the slot for stronger setups. (This is the opposite
  of the concurrent model, where trimming always hurt via lost diversification.)

This script attacks that directly:

  1. PER-SLOT CONTRIBUTION under the one-position sim (full S4+9 set):
       for every slot -- taken count, WR, TotR, AvgR. Shows who drags the WR.
  2. GREEDY FORWARD SELECTION: start with the 8 S4 slots, add the complement
       slot that most improves finalBal, repeat until nothing helps.
  3. GREEDY BACKWARD ELIMINATION: start with full S4+9, drop the slot whose
       removal most improves finalBal, repeat.
  4. Reports WR / MaxDD / MaxCL at every step so the trade-offs are visible.

Risk model is IDENTICAL to realistic_backtest.py (Config D 2/3/5%, Oct x0.5,
monthly -10% CB, equity throttle x0.20 @ >=15% below peak), one trade at a time.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.realistic_slot_select
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR = Path(__file__).parent
CSV  = _DIR / "data" / "backtest_trades.csv"

# -- Fixed sim parameters (must match realistic_backtest.py) --------------------
STARTING_BALANCE = 500.0
START_YEAR       = 2023
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.020, 0.030, 0.050  # Config D
DD_THROTTLE_TRIGGER = -0.15
DD_THROTTLE_FACTOR  = 0.20

# -- Slot sets -----------------------------------------------------------------
S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]
COMPLEMENT_9 = [
    ("engulfing",16,False), ("engulfing", 1,False), ("pin_bar",  2,False),
    ("engulfing", 8,False), ("engulfing", 2,False), ("engulfing",5,False),
    ("pin_bar",  19,False), ("pin_bar",  13,False), ("pin_bar", 11,False),
]


def _bar(c: str = "-", w: int = 100) -> str:
    return c * w


def _slot_label(slot) -> str:
    strat, hour, btc = slot
    return f"{strat}[H{hour:02d}]{'*' if btc else ''}"


def _risk_pct(adx: float) -> float:
    if adx >= 40:
        return RISK_STRONG
    if adx <= 25:
        return RISK_EARLY
    return RISK_TRANS


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _simulate_onepos(cand: pd.DataFrame, tag_slots: bool = False) -> dict:
    """One-position-at-a-time compound sim. cand sorted by entry_time.

    If tag_slots, also returns per-(strategy,hour) taken aggregates.
    """
    balance = STARTING_BALANCE
    peak = balance
    last_exit = pd.Timestamp.min
    bals, outcomes, rs = [], [], []
    per_slot: dict[tuple, dict] = {}

    cur_month = None
    month_start_bal = balance
    month_halted = False
    n_taken = n_missed = 0

    for _, t in cand.iterrows():
        et = t["entry_time"]
        xt = t["exit_time"]
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False

        if et < last_exit:           # one-trade rule: position busy -> miss
            n_missed += 1
            continue
        if month_halted:
            continue

        rp = _risk_pct(float(t["adx"]))
        if et.month == 10:
            rp *= OCT_RISK_FACTOR
        if (balance - peak) / peak <= DD_THROTTLE_TRIGGER:
            rp *= DD_THROTTLE_FACTOR

        r_val = float(t["r_achieved"])
        balance += balance * rp * r_val
        peak = max(peak, balance)
        last_exit = xt
        n_taken += 1
        bals.append(balance)
        outcomes.append(t["outcome"])
        rs.append(r_val)

        if tag_slots:
            key = (t["strategy"], int(t["hour_utc"]))
            d = per_slot.setdefault(key, {"n": 0, "w": 0, "totr": 0.0})
            d["n"] += 1
            d["w"] += 1 if t["outcome"] == "win" else 0
            d["totr"] += r_val

        if (balance - month_start_bal) / month_start_bal <= CB_THRESHOLD:
            month_halted = True

    if not bals:
        return {"final": STARTING_BALANCE, "cagr": 0.0, "maxdd": 0.0,
                "maxcl": 0, "wr": 0.0, "totr": 0.0, "n_taken": 0,
                "n_missed": n_missed, "per_slot": per_slot}

    bser = pd.Series(bals)
    max_dd = ((bser - bser.cummax()) / bser.cummax() * 100).min()
    max_cl = cur = 0
    for o in outcomes:
        cur = cur + 1 if o == "loss" else 0
        max_cl = max(max_cl, cur)

    days = (cand["entry_time"].max() - cand["entry_time"].min()).days or 1
    n_years = days / 365.25
    final_bal = bals[-1]
    cagr = ((final_bal / STARTING_BALANCE) ** (1 / n_years) - 1) * 100
    wins = sum(1 for o in outcomes if o == "win")

    return {
        "final": final_bal, "cagr": cagr, "maxdd": max_dd, "maxcl": max_cl,
        "wr": wins / n_taken * 100 if n_taken else 0.0,
        "totr": sum(rs), "n_taken": n_taken, "n_missed": n_missed,
        "per_slot": per_slot,
    }


def _run(df: pd.DataFrame, slots) -> dict:
    return _simulate_onepos(_select(df, slots))


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for _col in ("entry_time", "exit_time"):
        if df[_col].dt.tz is not None:
            df[_col] = df[_col].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)

    print()
    print(_bar("="))
    print(f"  ETH BOT -- ONE-POSITION SLOT SELECTION  ({START_YEAR}+)")
    print(f"  Goal: prune slots that BLOCK better trades under the one-trade rule.")
    print(f"  Config D 2/3/5% | Oct x0.5 | monthly CB -10% | throttle x0.2 @ -15% peak")
    print(_bar("="))

    # -- 1. Per-slot contribution under full S4+9 one-position sim --------------
    full = _simulate_onepos(_select(df, S4_SLOTS + COMPLEMENT_9), tag_slots=True)
    ps = full["per_slot"]
    print()
    print(f"  PER-SLOT CONTRIBUTION (inside full S4+9 one-position sim: "
          f"{full['n_taken']} taken, {full['n_missed']} missed)")
    print(_bar())
    print(f"  {'slot':<18} {'set':>4} {'taken':>6} {'WR%':>6} {'TotR':>8} {'AvgR':>7}")
    print(_bar())
    rows = []
    for slot in S4_SLOTS + COMPLEMENT_9:
        strat, hour, _ = slot
        d = ps.get((strat, hour), {"n": 0, "w": 0, "totr": 0.0})
        setname = "S4" if slot in S4_SLOTS else "comp"
        avgr = d["totr"] / d["n"] if d["n"] else 0.0
        wr = d["w"] / d["n"] * 100 if d["n"] else 0.0
        rows.append((slot, setname, d["n"], wr, d["totr"], avgr))
    # show S4 first, then complement sorted by TotR ascending (worst first)
    s4r = [r for r in rows if r[1] == "S4"]
    cor = sorted([r for r in rows if r[1] == "comp"], key=lambda r: r[4])
    for slot, setname, n, wr, totr, avgr in s4r + cor:
        flag = "  <== drag" if (setname == "comp" and totr <= 0) else ""
        print(f"  {_slot_label(slot):<18} {setname:>4} {n:>6} {wr:>5.1f}% "
              f"{totr:>+8.1f} {avgr:>+7.2f}{flag}")
    print(_bar())
    print("  NOTE: 'taken' here is AFTER one-trade blocking -- a slot with few")
    print("  taken is often getting blocked by, or blocking, other slots.")

    # -- baselines -------------------------------------------------------------
    base_s4   = _run(df, S4_SLOTS)
    base_full = _run(df, S4_SLOTS + COMPLEMENT_9)

    def _line(label, r):
        print(f"  {label:<40} {r['final']:>11,.0f} {r['wr']:>5.1f}% "
              f"{r['totr']:>+7.1f} {r['maxdd']:>+7.1f}% {r['maxcl']:>5} "
              f"{r['n_taken']:>5} {r['n_missed']:>6}")

    print()
    print(_bar("="))
    print("  BASELINES")
    print(_bar())
    print(f"  {'config':<40} {'finalBal':>11} {'WR%':>6} {'TotR':>7} "
          f"{'MaxDD%':>8} {'MaxCL':>5} {'take':>5} {'miss':>6}")
    print(_bar())
    _line("PURE S4 (8 slots)", base_s4)
    _line("S4 + ALL 9 COMPLEMENT", base_full)
    print(_bar())

    # -- 2. GREEDY FORWARD SELECTION ------------------------------------------
    print()
    print(_bar("="))
    print("  GREEDY FORWARD: start S4, add the complement slot that most raises finalBal")
    print(_bar())
    print(f"  {'step (added slot)':<40} {'finalBal':>11} {'WR%':>6} {'TotR':>7} "
          f"{'MaxDD%':>8} {'MaxCL':>5} {'take':>5} {'miss':>6}")
    print(_bar())
    chosen = list(S4_SLOTS)
    remaining = list(COMPLEMENT_9)
    best = base_s4
    _line("[S4 base]", best)
    fwd_set = list(S4_SLOTS)
    while remaining:
        scored = [(c, _run(df, chosen + [c])) for c in remaining]
        c_best, r_best = max(scored, key=lambda x: x[1]["final"])
        if r_best["final"] <= best["final"]:
            print("  (no remaining slot improves finalBal -- stop)")
            break
        chosen.append(c_best)
        remaining.remove(c_best)
        best = r_best
        fwd_set = list(chosen)
        _line(f"+ {_slot_label(c_best)}", r_best)
    print(_bar())
    print(f"  FORWARD-OPTIMAL SET ({len(fwd_set)} slots): "
          f"{', '.join(_slot_label(s) for s in fwd_set if s not in S4_SLOTS) or '(none added)'}")

    # -- 3. GREEDY BACKWARD ELIMINATION ---------------------------------------
    print()
    print(_bar("="))
    print("  GREEDY BACKWARD: start S4+9, drop the slot whose removal most raises finalBal")
    print(_bar())
    print(f"  {'step (dropped slot)':<40} {'finalBal':>11} {'WR%':>6} {'TotR':>7} "
          f"{'MaxDD%':>8} {'MaxCL':>5} {'take':>5} {'miss':>6}")
    print(_bar())
    cur_set = list(S4_SLOTS + COMPLEMENT_9)
    best_b = base_full
    _line("[S4+9 full]", best_b)
    bwd_set = list(cur_set)
    while True:
        # only consider dropping complement slots (keep the S4 core intact)
        droppable = [s for s in cur_set if s not in S4_SLOTS]
        if not droppable:
            break
        scored = [(s, _run(df, [x for x in cur_set if x != s])) for s in droppable]
        s_drop, r_drop = max(scored, key=lambda x: x[1]["final"])
        if r_drop["final"] <= best_b["final"]:
            print("  (no further drop improves finalBal -- stop)")
            break
        cur_set = [x for x in cur_set if x != s_drop]
        best_b = r_drop
        bwd_set = list(cur_set)
        _line(f"- {_slot_label(s_drop)}", r_drop)
    print(_bar())
    kept = [s for s in bwd_set if s not in S4_SLOTS]
    print(f"  BACKWARD-OPTIMAL SET ({len(bwd_set)} slots): kept complement = "
          f"{', '.join(_slot_label(s) for s in kept) or '(none)'}")

    print()
    print(_bar("="))
    print("  READ: compare FORWARD-OPTIMAL vs BACKWARD-OPTIMAL vs S4+9 full.")
    print("  Want the set with the best finalBal that keeps MaxCL<16 and MaxDD>-42%.")
    print("  If forward/backward converge on the same slots -> that's robust signal.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
