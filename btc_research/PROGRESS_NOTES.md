# BTC Research — Progress Notes

## What This Folder Is
Pure research / backtesting for a BTC/USD strategy.
**Completely isolated from `v2/` (WTI live bot).** Never mix the two.

---

## Key Findings (2-Year H1 Backtest on BTCUSD, Pepperstone MT5)

### Session Scanner Results — all 24 UTC hours tested
| Hour (UTC) | WR     | Notes                  |
|-----------|--------|------------------------|
| 02:00     | 57.1%  | Best single hour       |
| 03:00     | 52.6%  | Second best            |
| 00:00     | 31.7%  | Avoid                  |
| 01:00     | 26.7%  | Avoid                  |
| 13–17     | 38.2%  | WTI assumption — wrong for BTC |

### Per-Strategy Optimal Sessions
| Strategy            | Best Session          | Key Metric              |
|--------------------|-----------------------|-------------------------|
| Volatility Breakout | Asia Night 00–04 UTC  | +0.70R avg, 16.6% MaxDD |
| Swing Level Break   | Asia Night 00–04 UTC  | 51.0% WR, 19.9% MaxDD   |
| Morning Range Break | US Open 13–17 UTC     | 40.6% WR, 23.1% MaxDD   |
| EMA Trend Follow    | EU Session 08–12 UTC  | 40.1% WR, 23.5% MaxDD   |

### Combined 3-Strategy — Best Session: US Late 21–24 UTC ✅ (Current Setting)
- **223 trades | WR = 43.0% | AvgR = +0.47R | PnL = +$23,733 | MaxDD = 16.1%**
- Beats every individual strategy at their own best session
- UAE time equivalent: 01:00 AM – 04:00 AM (bot runs overnight)

---

## Current Optimised Settings (`btc_research/settings.py`)

```
KZ_START_UTC     = 21       # Kill-zone open
KZ_END_UTC       = 24       # Kill-zone close
STARTING_BALANCE = 500      # USD
RISK_PCT         = 0.02     # 2% per trade (reduced from 3% — MaxDD too high at 3%)
TP1_RR           = 2.0      # Partial close + SL to breakeven
TP2_RR           = 5.0      # Full close
MAX_HOLD_BARS    = 96       # 4 days max hold
MIN_CONFLUENCE_SCORE = 3.0  # Tune up/down for WR vs. trade count
LOOKBACK_YEARS   = 2
```

### Why 2% not 3%?
At 43–46% WR, 3% risk pushed MaxDD above 45%. Dropping to 2% keeps MaxDD ~16% —
much more survivable through losing streaks.

---

## Strategy Logic (`btc_research/strategy/confluence.py`)

Four factors scored and summed:
1. **BTC Momentum** — trend direction on H1/H4
2. **Gold Factor** — inverse correlation (BTC up = Gold down, and vice versa)
3. **Nasdaq Factor** — risk-on/off alignment
4. **Time Factor** — inside kill-zone window bonus

Signal fires when `total_score >= MIN_CONFLUENCE_SCORE (3.0)`.

### Entry / Exit Rules
- One trade at a time
- Entry on current bar close when score threshold met inside kill-zone
- TP1 hit → close 50%, SL moves to breakeven (BE)
- TP2 hit → close remaining 50%
- SL hit after TP1 → "SL_AFTER_TP1" (partial profit already banked)
- MAX_HOLD_BARS → force-close after 96 H1 bars regardless

---

## File Map

```
btc_research/
├── settings.py                  ← All config (kill-zone, risk, TP/SL)
├── run_backtest.py              ← Main entry: run full 2-yr backtest + print report
├── backtest/
│   ├── engine.py                ← Bar-by-bar simulation loop
│   ├── report.py                ← Summary stats + print formatting
│   ├── session_scanner.py       ← Tested all 24 hours → found 21-24 UTC best
│   ├── optimizer.py             ← Parameter grid search
│   └── strategy_comparison.py  ← Per-strategy session breakdown
├── strategy/
│   └── confluence.py            ← score_bar() — core signal logic
├── strategies/
│   ├── volatility_breakout.py
│   ├── swing_level.py
│   ├── morning_range.py
│   └── ema_trend.py
├── factors/
│   ├── btc_momentum.py
│   ├── gold_factor.py
│   ├── nasdaq_factor.py
│   └── time_factor.py
└── data/
    └── fetcher.py               ← Pulls H1 BTCUSD + XAUUSD + NAS100 from MT5
```

---

## How to Re-run Backtest

```powershell
# From project root (C:\Temp\TradingBotV1 on VPS)
.\venv\Scripts\python.exe -m btc_research.run_backtest
```

Requires MT5 to be connected (data fetch). Results cached in `btc_research/data/cache/`.

---

## Next Steps (Planned for Future Session)
- [ ] Forward-test on paper account once live infrastructure is ready
- [ ] Consider live integration into a `v3/` module (keep isolated from WTI `v2/`)
- [ ] Investigate 02:00–03:00 UTC individual-hour performance (57% WR) vs. combined session
- [ ] Review MIN_CONFLUENCE_SCORE sensitivity (3.0 → 3.5 / 4.0 effect on WR vs. trade count)
