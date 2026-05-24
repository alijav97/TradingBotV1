"""
Part 5 — Historical validation test for the $4,460→$4,560 bounce.

Simulates the bot state just before the bounce by slicing the CSV at the
local low (close < 4490) then computing RSI / ATR / EMA indicators
identically to _load_df() in bot_chat.py.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, ".")

CSV = "data/historical_xauusd.csv"
if not os.path.exists(CSV):
    print("CSV not found:", CSV)
    sys.exit(1)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the same indicators as _load_df() in bot_chat.py."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df = df.dropna(subset=["ema200", "rsi", "atr"])
    return df.reset_index(drop=True)


# ── Load and add indicators to full dataset ───────────────────────────────────
raw = pd.read_csv(CSV)
print(f"Loaded {len(raw)} raw rows")

df_full = _add_indicators(raw)
print(f"After indicator warmup: {len(df_full)} rows")
print(f"Price range: ${df_full['close'].min():,.2f}  –  ${df_full['close'].max():,.2f}")

# ── Find the dip: last bar where close is in the $4,440–$4,490 range ─────────
dip_rows = df_full[df_full["close"].between(4440, 4490)]

if dip_rows.empty:
    # Fallback — just use the bar with the lowest recent close
    recent = df_full.tail(500)
    dip_idx = recent["close"].idxmin()
    print(f"No dip in $4,440–$4,490 range — using recent low at index {dip_idx}")
else:
    dip_idx = dip_rows.index[-1]   # last bar in the dip zone

dip_row = df_full.loc[dip_idx]
print(f"\nDip bar index : {dip_idx}")
if "datetime" in df_full.columns:
    print(f"Dip bar time  : {df_full.loc[dip_idx, 'datetime']}")
print(f"Dip bar close : ${dip_row['close']:,.2f}")
print(f"Dip bar RSI   : {dip_row['rsi']:.1f}")
print(f"Dip bar ATR   : ${dip_row['atr']:,.2f}")

# ── Slice up to and including the dip bar ─────────────────────────────────────
df_cut = df_full.iloc[: dip_idx + 1].copy()
print(f"\nRunning hunt_reversals() on {len(df_cut)} bars (up to the dip)…")

from reversal_hunter import hunt_reversals

sigs = hunt_reversals(df_cut)

print()
print("=" * 55)
if sigs:
    s = sigs[0]
    print("Missed trade test: CAUGHT")
    print("=" * 55)
    print(f"Pattern    : {s['pattern_name']}")
    print(f"Direction  : {s['direction'].upper()}")
    print(f"Score      : {s['score']}/11  ({s['reversal_strength']})")
    print(f"Confidence : {s['confidence']}/10")
    print(f"Entry      : ${s['entry']:,.2f}")
    print(f"Stop Loss  : ${s['stop_loss']:,.2f}")
    print(f"Take Profit: ${s['take_profit']:,.2f}")
    print(f"SL Dist    : ${s['sl_distance']:,.2f}")
    print(f"Note       : {s['note']}")
    print("\nConditions met:")
    for c in s["conditions_met"]:
        print(f"  ✔ {c}")
    print(f"\nKey reason : {s['key_reason']}")
else:
    print("Missed trade test: MISSED")
    print("=" * 55)
    print("No reversal signal generated for this window.")
    print(f"Last bar RSI : {df_cut['rsi'].iloc[-1]:.1f}")
    print(f"RSI 3 bars ago: {df_cut['rsi'].iloc[-3]:.1f}")
    move = df_cut['close'].iloc[-1] - df_cut['close'].iloc[-4]
    atr  = df_cut['atr'].iloc[-1]
    print(f"3-bar move   : ${move:.2f}  (threshold: ${atr*2.5:.2f})")
    print(f"vs EMA200    : close={df_cut['close'].iloc[-1]:.2f}  ema200={df_cut['ema200'].iloc[-1]:.2f}")
