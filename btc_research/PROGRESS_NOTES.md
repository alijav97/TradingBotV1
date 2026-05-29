# BTC Research — Progress Notes

## What This Folder Is
Pure research / backtesting for a BTC/USD strategy.
**Completely isolated from `v2/` (WTI live bot).** Never mix the two.

---

## File Map (Full)

```
btc_research/
├── settings.py                  ← All config (kill-zone, risk, TP/SL) — completely isolated
├── run_backtest.py              ← Main entry: run full 2-yr backtest + print report
│
├── analysis_ema_filter.py       ← Compared Versions A/B/C/D (EMA200 + flipped risk)
├── analysis_quarterly.py        ← Version D quarterly backtest
├── analysis_weak_months.py      ← Weak month filters (E1/E2/E3)
├── analysis_high_wr.py          ← High WR month deep-dive
│
├── strategies/                  ← 7 BTC-specific strategy variants
│   ├── combined.py              ← Version D (EMA200 + flipped risk) ← FINAL
│   ├── volatility_breakout.py
│   ├── swing_level.py
│   ├── morning_range.py
│   ├── ema_trend.py
│   └── rsi_reversion.py
│
├── backtest/
│   ├── engine.py                ← Bar-by-bar simulation loop
│   ├── session_scanner.py       ← Found 21–24 UTC as optimal session
│   ├── strategy_comparison.py   ← Per-strategy session breakdown
│   └── optimizer.py             ← Parameter grid search
│
├── factors/                     ← Correlation factors
│   ├── btc_momentum.py
│   ├── gold_factor.py           ← XAUUSD inverse correlation
│   ├── nasdaq_factor.py         ← NAS100 risk-on/off
│   └── time_factor.py
│
├── strategy/
│   └── confluence.py            ← score_bar() — core signal logic
│
└── data/cache/                  ← Cached H1 bars (BTCUSD, XAUUSD, NAS100)
```

---

## Strategy Version History

### Version A — Baseline
- 3-strategy combined (Volatility Breakout + Swing Level + Morning Range)
- No EMA filter, standard risk sizing

### Version B — Session Optimised
- Added session scanner → found **US Late 21–24 UTC** as best session
- 223 trades | WR = 43.0% | AvgR = +0.47R | PnL = +$23,733 | MaxDD = 16.1%

### Version C — Risk Tuning
- Reduced RISK_PCT from 3% → 2%
- At 43–46% WR, 3% risk pushed MaxDD above 45%. At 2% → MaxDD ~16%

### Version D — EMA200 + Flipped Risk ← FINAL ✅
- Added **EMA200 filter**: only take longs above EMA200, shorts below
- **Flipped risk**: size positions larger on high-confluence setups, smaller on marginal ones
- Lives in `strategies/combined.py`
- This is the version to use for live deployment

---

## Session Scanner Results — All 24 UTC Hours Tested (2-Year Data)

| Hour (UTC) | WR     | Notes                        |
|-----------|--------|------------------------------|
| 02:00     | 57.1%  | Best single hour             |
| 03:00     | 52.6%  | Second best                  |
| 00:00     | 31.7%  | Avoid                        |
| 01:00     | 26.7%  | Avoid                        |
| 13–17     | 38.2%  | WTI assumption — wrong for BTC |

### Per-Strategy Optimal Sessions
| Strategy            | Best Session          | Key Metric              |
|--------------------|-----------------------|-------------------------|
| Volatility Breakout | Asia Night 00–04 UTC  | +0.70R avg, 16.6% MaxDD |
| Swing Level Break   | Asia Night 00–04 UTC  | 51.0% WR, 19.9% MaxDD   |
| Morning Range Break | US Open 13–17 UTC     | 40.6% WR, 23.1% MaxDD   |
| EMA Trend Follow    | EU Session 08–12 UTC  | 40.1% WR, 23.5% MaxDD   |

### Combined 3-Strategy — Best: US Late 21–24 UTC
- 223 trades | WR = 43.0% | AvgR = +0.47R | PnL = +$23,733 | MaxDD = 16.1%
- Beats every individual strategy at their own best session
- UAE equivalent: 01:00 AM – 04:00 AM (bot runs overnight)

---

## Quarterly & Monthly Analysis

### `analysis_quarterly.py` — Version D quarterly breakdown
- Ran Version D through each quarter to check consistency (no one good quarter masking bad ones)

### `analysis_weak_months.py` — Weak month filters E1/E2/E3
- Identified months with consistently poor BTC performance
- Three filter variants tested (E1 = remove weakest, E2 = remove worst 2, E3 = remove worst 3)
- Helps avoid trading during historically bad BTC months

### `analysis_high_wr.py` — High WR month deep-dive
- Deep-dive into months that consistently produced 50%+ WR
- Identifies if there are seasonal patterns worth leaning into

---

## Current Optimised Settings (`btc_research/settings.py`)

```python
KZ_START_UTC         = 21       # Kill-zone open (US Late session)
KZ_END_UTC           = 24       # Kill-zone close
STARTING_BALANCE     = 500      # USD — backtest starting capital
RISK_PCT             = 0.02     # 2% per trade
TP1_RR               = 2.0      # Partial close + SL to breakeven
TP2_RR               = 5.0      # Full close
MAX_HOLD_BARS        = 96       # 4 days max hold
MIN_CONFLUENCE_SCORE = 3.0      # Tune: higher = fewer trades, higher WR
LOOKBACK_YEARS       = 2
```

---

## Core Strategy Logic (`strategy/confluence.py`)

`score_bar()` scores four factors and sums them:
1. **BTC Momentum** — H1/H4 trend direction
2. **Gold Factor** — inverse correlation (BTC up ↔ Gold down)
3. **Nasdaq Factor** — risk-on/off alignment
4. **Time Factor** — inside kill-zone window bonus

Signal fires when `total_score >= MIN_CONFLUENCE_SCORE`.

### Entry / Exit Rules
- One trade at a time (never double up)
- Entry on bar close when score threshold met inside kill-zone
- **TP1 hit** → close 50%, SL moves to breakeven
- **TP2 hit** → close remaining 50%
- **SL hit after TP1** → "SL_AFTER_TP1" (partial profit already banked)
- **MAX_HOLD** → force-close after 96 H1 bars regardless

---

## How to Re-run Backtest

```powershell
# From project root (C:\Temp\TradingBotV1 on VPS)
.\venv\Scripts\python.exe -m btc_research.run_backtest
```

Requires MT5 connected (data fetch). Results cached in `btc_research/data/cache/`.

---

## Next Steps (Planned for Future Session)
- [ ] Review E1/E2/E3 weak month filter results — decide which to apply in Version D
- [ ] Sensitivity test: MIN_CONFLUENCE_SCORE 3.0 → 3.5 → 4.0 (WR vs. trade count)
- [ ] Forward-test Version D on paper account once live infrastructure is ready
- [ ] Live integration into `v3/` module (keep fully isolated from WTI `v2/`)
- [ ] Investigate 02:00–03:00 UTC individually (57% WR) — worth a separate day-session bot?
