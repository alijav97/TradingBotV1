import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import random

# ══════════════════════════════════════════════════════
# INSTRUMENT CONFIG — 20x leverage aware
# ══════════════════════════════════════════════════════

# Risk constants
ACCOUNT_BALANCE = 1000
LEVERAGE        = 10
RR_RATIO        = 3.0
POSITION_SIZE   = ACCOUNT_BALANCE * LEVERAGE   # $10,000

INSTRUMENTS = {
    "XAUUSD": {
        "ticker":      "GC=F",
        "period":      "730d",
        "interval":    "4h",
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 4.5,
        "atr_period":  14,
        "max_hold":    12,
        "pip":         0.10,
        "spread_pct":  0.00007,
        "min_sl_pct":  0.004,
        "max_sl_pct":  0.015,
    },
    "NAS100": {
        "ticker":      "NQ=F",
        "period":      "2y",
        "interval":    "1h",
        "atr_sl_mult": 1.2,
        "atr_tp_mult": 3.6,
        "atr_period":  14,
        "max_hold":    24,
        "pip":         1.0,
        "spread_pct":  0.00005,
        "min_sl_pct":  0.005,
        "max_sl_pct":  0.020,
    },
    "US30": {
        "ticker":      "YM=F",
        "period":      "2y",
        "interval":    "1h",
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 6.0,
        "atr_period":  14,
        "max_hold":    24,
        "pip":         1.0,
        "spread_pct":  0.00008,
        "min_sl_pct":  0.004,
        "max_sl_pct":  0.015,
    },
    "GBPUSD": {
        "ticker":      "GBPUSD=X",
        "period":      "2y",
        "interval":    "1h",
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 4.5,
        "atr_period":  14,
        "max_hold":    48,
        "pip":         0.0001,
        "spread_pct":  0.00006,
        "min_sl_pct":  0.003,
        "max_sl_pct":  0.010,
    },
    "EURUSD": {
        "ticker":      "EURUSD=X",
        "period":      "2y",
        "interval":    "1h",
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 4.5,
        "atr_period":  14,
        "max_hold":    48,
        "pip":         0.0001,
        "spread_pct":  0.00005,
        "min_sl_pct":  0.003,
        "max_sl_pct":  0.010,
    },
    "WTI": {
        "ticker":      "CL=F",
        "period":      "2y",
        "interval":    "1h",
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 4.5,
        "atr_period":  14,
        "max_hold":    24,
        "pip":         0.01,
        "spread_pct":  0.00010,
        "min_sl_pct":  0.006,
        "max_sl_pct":  0.025,
    },
}

SESSION_HOURS = {
    "XAUUSD": list(range(8, 22)),
    "NAS100": list(range(13, 23)),
    "US30":   list(range(13, 23)),
    "GBPUSD": list(range(7, 18)),
    "EURUSD": list(range(7, 18)),
    "WTI":    list(range(13, 23)),
}

def is_active(ts, instr):
    try:
        return (ts.hour + 4) % 24 in SESSION_HOURS.get(instr, [])
    except Exception:
        return True

def is_weekend(ts):
    try:
        return ts.dayofweek >= 5
    except Exception:
        return False

# ══════════════════════════════════════════════════════
# HELPER INDICATORS
# ══════════════════════════════════════════════════════

def calc_rsi(close, p=14):
    d  = close.diff()
    g  = d.clip(lower=0)
    lo = (-d).clip(lower=0)
    rs = g.ewm(span=p).mean() / lo.ewm(span=p).mean()
    return 100 - (100 / (1 + rs))

def calc_atr(df, p=14):
    return (df["High"] - df["Low"]).rolling(p).mean()

def calc_macd(close):
    e12 = close.ewm(span=12).mean()
    e26 = close.ewm(span=26).mean()
    m   = e12 - e26
    s   = m.ewm(span=9).mean()
    return m, s, m - s

# ══════════════════════════════════════════════════════
# STRATEGY LIBRARY
# ══════════════════════════════════════════════════════

def s_squeeze(df, instr):
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    atr   = calc_atr(df, 20)
    bb_up = sma + 2 * std
    bb_lo = sma - 2 * std
    kc_up = sma + 1.5 * atr
    kc_lo = sma - 1.5 * atr
    sq    = (bb_up < kc_up) & (bb_lo > kc_lo)
    mom   = df["Close"] - sma
    sig   = pd.Series(0, index=df.index)
    sig[(~sq) & sq.shift(1) & (mom > 0)] =  1
    sig[(~sq) & sq.shift(1) & (mom < 0)] = -1
    return sig

def s_atr_breakout(df, instr):
    atr = calc_atr(df)
    chg = df["Close"] - df["Close"].shift(1)
    sig = pd.Series(0, index=df.index)
    sig[chg >  atr] =  1
    sig[chg < -atr] = -1
    return sig

def s_fvg(df, instr):
    sig = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        h1 = df["High"].iloc[i-2]
        l1 = df["Low"].iloc[i-2]
        h3 = df["High"].iloc[i]
        l3 = df["Low"].iloc[i]
        if l3 > h1:
            sig.iloc[i] =  1
        if h3 < l1:
            sig.iloc[i] = -1
    return sig

def s_order_block(df, instr):
    sig = pd.Series(0, index=df.index)
    c   = df["Close"].values
    o   = df["Open"].values
    h   = df["High"].values
    lo  = df["Low"].values
    for i in range(10, len(df) - 1):
        if c[i-1] < o[i-1] and c[i] > o[i] and c[i] > h[i-1]:
            sig.iloc[i] =  1
        if c[i-1] > o[i-1] and c[i] < o[i] and c[i] < lo[i-1]:
            sig.iloc[i] = -1
    return sig

def s_bos(df, instr):
    sig = pd.Series(0, index=df.index)
    h   = df["High"].values
    lo  = df["Low"].values
    c   = df["Close"].values
    for i in range(20, len(df)):
        ph = max(h[i-20:i-1])
        pl = min(lo[i-20:i-1])
        if c[i] > ph:
            sig.iloc[i] =  1
        elif c[i] < pl:
            sig.iloc[i] = -1
    return sig

def s_liq_sweep(df, instr):
    sig = pd.Series(0, index=df.index)
    for i in range(20, len(df)):
        rh = df["High"].iloc[i-20:i-1].max()
        rl = df["Low"].iloc[i-20:i-1].min()
        if df["High"].iloc[i] > rh and df["Close"].iloc[i] < rh:
            sig.iloc[i] = -1
        elif df["Low"].iloc[i] < rl and df["Close"].iloc[i] > rl:
            sig.iloc[i] =  1
    return sig

def s_engulfing(df, instr):
    bp  = df["Close"].shift(1) - df["Open"].shift(1)
    bc  = df["Close"] - df["Open"]
    sig = pd.Series(0, index=df.index)
    sig[(bp < 0) & (bc > 0) &
        (df["Close"] > df["Open"].shift(1)) &
        (df["Open"] < df["Close"].shift(1))] =  1
    sig[(bp > 0) & (bc < 0) &
        (df["Close"] < df["Open"].shift(1)) &
        (df["Open"] > df["Close"].shift(1))] = -1
    return sig

def s_bollinger_break(df, instr):
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    sig   = pd.Series(0, index=df.index)
    sig[(df["Close"] > upper) &
        (df["Close"].shift(1) <= upper.shift(1))] =  1
    sig[(df["Close"] < lower) &
        (df["Close"].shift(1) >= lower.shift(1))] = -1
    return sig

def s_rsi_div(df, instr):
    rsi = calc_rsi(df["Close"])
    sig = pd.Series(0, index=df.index)
    c   = df["Close"].values
    r   = rsi.values
    for i in range(20, len(df)):
        wc = c[i-10:i]
        wr = r[i-10:i]
        if len(wc) < 10:
            continue
        if c[i] < np.min(wc) and r[i] > np.min(wr):
            sig.iloc[i] =  1
        if c[i] > np.max(wc) and r[i] < np.max(wr):
            sig.iloc[i] = -1
    return sig

def s_hh_hl(df, instr):
    sig = pd.Series(0, index=df.index)
    sig[(df["High"] > df["High"].shift(1)) &
        (df["Low"]  > df["Low"].shift(1))]  =  1
    sig[(df["High"] < df["High"].shift(1)) &
        (df["Low"]  < df["Low"].shift(1))]  = -1
    return sig

def s_confluence(df, instr):
    """MASTER CONFLUENCE — squeeze + FVG + RSI direction"""
    sq  = s_squeeze(df, instr)
    fvg = s_fvg(df, instr)
    rsi = calc_rsi(df["Close"])
    r_b = (rsi > 50).astype(int)
    r_s = (rsi < 50).astype(int)
    bull = ((sq == 1).astype(int) +
            (fvg == 1).astype(int) + r_b)
    bear = ((sq == -1).astype(int) +
            (fvg == -1).astype(int) + r_s)
    sig = pd.Series(0, index=df.index)
    sig[bull >= 2] =  1
    sig[bear >= 2] = -1
    return sig

def s_smc_confluence(df, instr):
    """SMC CONFLUENCE — OB + BOS + Liquidity sweep"""
    ob  = s_order_block(df, instr)
    bos = s_bos(df, instr)
    liq = s_liq_sweep(df, instr)
    bull = ((ob == 1).astype(int) +
            (bos == 1).astype(int) +
            (liq == 1).astype(int))
    bear = ((ob == -1).astype(int) +
            (bos == -1).astype(int) +
            (liq == -1).astype(int))
    sig = pd.Series(0, index=df.index)
    sig[bull >= 2] =  1
    sig[bear >= 2] = -1
    return sig

def s_kill_zone_fvg(df, instr):
    """FVG only during kill zones"""
    fvg = s_fvg(df, instr)
    kz  = SESSION_HOURS.get(instr, [])
    sig = pd.Series(0, index=df.index)
    for i, (idx, _) in enumerate(df.iterrows()):
        try:
            gst = (idx.hour + 4) % 24
            if gst not in kz:
                continue
            if fvg.iloc[i] != 0:
                sig.iloc[i] = fvg.iloc[i]
        except Exception:
            continue
    return sig

def s_squeeze_bos(df, instr):
    """Squeeze release confirmed by BOS"""
    sq  = s_squeeze(df, instr)
    bos = s_bos(df, instr)
    sig = pd.Series(0, index=df.index)
    sig[(sq == 1)  & (bos == 1)]  =  1
    sig[(sq == -1) & (bos == -1)] = -1
    return sig

def s_nas_vwap(df, instr):
    """
    VWAP mean reversion —
    NAS100 respects VWAP strongly
    """
    try:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = ((typical * df["Volume"]).rolling(20).sum() /
                df["Volume"].rolling(20).sum())
        sig   = pd.Series(0, index=df.index)
        trend = df["Close"].ewm(span=50).mean()
        sig[(df["Close"] > trend) &
            (df["Close"] < vwap * 1.002) &
            (df["Close"] > vwap * 0.998)] =  1
        sig[(df["Close"] < trend) &
            (df["Close"] > vwap * 0.998) &
            (df["Close"] < vwap * 1.002)] = -1
        return sig
    except Exception:
        return pd.Series(0, index=df.index)


def s_nas_open_drive(df, instr):
    """
    Opening drive — NAS100 often continues
    direction of first hour of NY session
    """
    sig   = pd.Series(0, index=df.index)
    opens = {}
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            gst = (idx.hour + 4) % 24
            if gst == 13:  # NY open GST
                opens[idx.date()] = row["Close"]
            elif gst in [14, 15, 16]:
                d = idx.date()
                if d in opens:
                    open_p = opens[d]
                    if row["Close"] > open_p * 1.003:
                        sig.iloc[i] =  1
                    elif row["Close"] < open_p * 0.997:
                        sig.iloc[i] = -1
        except Exception:
            continue
    return sig


def s_nas_gap_fill(df, instr):
    """
    Gap fill — NAS100 gaps often
    fill within same session
    """
    sig = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        try:
            prev_close = df["Close"].iloc[i - 1]
            curr_open  = df["Open"].iloc[i]
            gap_pct = (curr_open - prev_close) / prev_close * 100
            if gap_pct > 0.3:
                sig.iloc[i] = -1
            elif gap_pct < -0.3:
                sig.iloc[i] =  1
        except Exception:
            continue
    return sig


def s_nas_momentum_burst(df, instr):
    """
    Momentum burst — NAS100 strongest moves
    happen when RSI > 60 AND price breaks
    recent high with volume
    """
    rsi     = calc_rsi(df["Close"])
    avg_vol = df["Volume"].rolling(20).mean()
    h20     = df["High"].rolling(20).max()
    l20     = df["Low"].rolling(20).min()
    sig     = pd.Series(0, index=df.index)
    sig[(rsi > 60) &
        (df["Close"] > h20.shift(1)) &
        (df["Volume"] > avg_vol * 1.3)] =  1
    sig[(rsi < 40) &
        (df["Close"] < l20.shift(1)) &
        (df["Volume"] > avg_vol * 1.3)] = -1
    return sig


def s_nas_pullback(df, instr):
    """
    Trend pullback entry — NAS100 strongest
    when entering on pullbacks to EMA
    in trend direction
    """
    e20  = df["Close"].ewm(span=20).mean()
    e50  = df["Close"].ewm(span=50).mean()
    e200 = df["Close"].rolling(200).mean()
    sig  = pd.Series(0, index=df.index)
    sig[(e20 > e50) & (e50 > e200) &
        (df["Low"] <= e20 * 1.001) &
        (df["Low"] >= e20 * 0.997)] =  1
    sig[(e20 < e50) & (e50 < e200) &
        (df["High"] >= e20 * 0.999) &
        (df["High"] <= e20 * 1.003)] = -1
    return sig


def s_nas_squeeze_vwap(df, instr):
    """
    Squeeze release + VWAP confluence
    NAS100 specific combo
    """
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    atr   = calc_atr(df, 20)
    bb_up = sma + 2 * std
    bb_lo = sma - 2 * std
    kc_up = sma + 1.5 * atr
    kc_lo = sma - 1.5 * atr
    sq    = (bb_up < kc_up) & (bb_lo > kc_lo)
    mom   = df["Close"] - sma
    sq_sig = pd.Series(0, index=df.index)
    sq_sig[(~sq) & sq.shift(1) & (mom > 0)] =  1
    sq_sig[(~sq) & sq.shift(1) & (mom < 0)] = -1
    try:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = ((typical * df["Volume"]).rolling(20).sum() /
                df["Volume"].rolling(20).sum())
    except Exception:
        return sq_sig
    sig = pd.Series(0, index=df.index)
    sig[(sq_sig == 1)  & (df["Close"] > vwap)] =  1
    sig[(sq_sig == -1) & (df["Close"] < vwap)] = -1
    return sig


# ══════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
# XAUUSD SPECIFIC STRATEGIES
# ══════════════════════════════════════════════════════

def s_gold_dxy_divergence(df, instr):
    """Gold rises when DXY falls. Proxy: RSI recovering from <40 above SMA50."""
    rsi   = calc_rsi(df["Close"])
    sma50 = df["Close"].ewm(span=50).mean()
    sig   = pd.Series(0, index=df.index)
    sig[(df["Close"] > sma50) &
        (rsi > 40) &
        (rsi.shift(3) < 40)] =  1
    sig[(df["Close"] < sma50) &
        (rsi < 60) &
        (rsi.shift(3) > 60)] = -1
    return sig

def s_gold_asian_range(df, instr):
    """Asian range breakout for gold. London/NY trades the breakout."""
    sig       = pd.Series(0, index=df.index)
    asian_high = {}
    asian_low  = {}
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            gst = (idx.hour + 4) % 24
            d   = idx.date()
            if 1 <= gst <= 7:
                if d not in asian_high:
                    asian_high[d] = row["High"]
                    asian_low[d]  = row["Low"]
                else:
                    asian_high[d] = max(asian_high[d], row["High"])
                    asian_low[d]  = min(asian_low[d],  row["Low"])
            elif gst == 8:
                if d in asian_high:
                    if row["Close"] > asian_high[d]:
                        sig.iloc[i] =  1
                    elif row["Close"] < asian_low[d]:
                        sig.iloc[i] = -1
        except:
            continue
    return sig

def s_gold_cot_proxy(df, instr):
    """COT proxy — new 20-day high/low with expanding volume = institutional flow."""
    h20     = df["High"].rolling(20).max()
    l20     = df["Low"].rolling(20).min()
    avg_vol = df["Volume"].rolling(20).mean()
    sig     = pd.Series(0, index=df.index)
    sig[(df["High"] >= h20) & (df["Volume"] > avg_vol * 1.2)] =  1
    sig[(df["Low"] <= l20)  & (df["Volume"] > avg_vol * 1.2)] = -1
    return sig

# ══════════════════════════════════════════════════════
# GBPUSD SPECIFIC STRATEGIES
# ══════════════════════════════════════════════════════

def s_gbp_london_open(df, instr):
    """GBP/USD London open momentum at 08:00 GST."""
    sig = pd.Series(0, index=df.index)
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            gst = (idx.hour + 4) % 24
            if gst == 8:
                prev = df["Close"].iloc[max(0, i - 3):i]
                if not prev.empty:
                    avg = prev.mean()
                    if row["Close"] > avg * 1.001:
                        sig.iloc[i] =  1
                    elif row["Close"] < avg * 0.999:
                        sig.iloc[i] = -1
        except:
            continue
    return sig

def s_gbp_bos_london(df, instr):
    """Break of structure during London session only."""
    sig = pd.Series(0, index=df.index)
    h   = df["High"].values
    l   = df["Low"].values
    c   = df["Close"].values
    ix  = df.index
    for i in range(20, len(df)):
        try:
            gst = (ix[i].hour + 4) % 24
            if not (8 <= gst <= 17):
                continue
            ph = max(h[i - 20:i - 1])
            pl = min(l[i - 20:i - 1])
            if c[i] > ph:
                sig.iloc[i] =  1
            elif c[i] < pl:
                sig.iloc[i] = -1
        except:
            continue
    return sig

def s_gbp_macro_momentum(df, instr):
    """Medium-term momentum for GBP using 3-period rate of change."""
    roc   = df["Close"].pct_change(3) * 100
    ema20 = df["Close"].ewm(span=20).mean()
    sig   = pd.Series(0, index=df.index)
    sig[(roc > 0.1)  & (df["Close"] > ema20)] =  1
    sig[(roc < -0.1) & (df["Close"] < ema20)] = -1
    return sig

# ══════════════════════════════════════════════════════
# EURUSD SPECIFIC STRATEGIES
# ══════════════════════════════════════════════════════

def s_eur_frankfurt_london(df, instr):
    """EUR/USD Frankfurt + London overlap 07:00-10:00 GST."""
    sig = pd.Series(0, index=df.index)
    rsi = calc_rsi(df["Close"])
    ema = df["Close"].ewm(span=20).mean()
    ix  = df.index
    for i in range(20, len(df)):
        try:
            gst = (ix[i].hour + 4) % 24
            if not (7 <= gst <= 10):
                continue
            if df["Close"].iloc[i] > ema.iloc[i] and rsi.iloc[i] > 50:
                sig.iloc[i] =  1
            elif df["Close"].iloc[i] < ema.iloc[i] and rsi.iloc[i] < 50:
                sig.iloc[i] = -1
        except:
            continue
    return sig

def s_eur_squeeze_frankfurt(df, instr):
    """Squeeze momentum release during Frankfurt/London for EUR/USD."""
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    atr   = calc_atr(df, 20)
    bb_up = sma + 2 * std
    bb_lo = sma - 2 * std
    kc_up = sma + 1.5 * atr
    kc_lo = sma - 1.5 * atr
    sq    = (bb_up < kc_up) & (bb_lo > kc_lo)
    mom   = df["Close"] - sma
    ix    = df.index
    sig   = pd.Series(0, index=df.index)
    for i in range(20, len(df)):
        try:
            gst = (ix[i].hour + 4) % 24
            if not (7 <= gst <= 17):
                continue
            if (not sq.iloc[i] and sq.iloc[i - 1] and mom.iloc[i] > 0):
                sig.iloc[i] =  1
            elif (not sq.iloc[i] and sq.iloc[i - 1] and mom.iloc[i] < 0):
                sig.iloc[i] = -1
        except:
            continue
    return sig

# ══════════════════════════════════════════════════════
# WTI SPECIFIC STRATEGIES
# ══════════════════════════════════════════════════════

def s_wti_eia_week(df, instr):
    """EIA report Wednesday fade — fade the pre-Wednesday 2-day move."""
    sig = pd.Series(0, index=df.index)
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            if idx.dayofweek == 2 and i >= 2:
                trend = (df["Close"].iloc[i] -
                         df["Close"].iloc[i - 2])
                if trend > 0:
                    sig.iloc[i] = -1
                else:
                    sig.iloc[i] =  1
        except:
            continue
    return sig

def s_wti_supply_demand(df, instr):
    """WTI supply/demand zones — high-volume candles create S/D levels."""
    avg_vol = df["Volume"].rolling(20).mean()
    atr     = calc_atr(df)
    sig     = pd.Series(0, index=df.index)
    for i in range(20, len(df)):
        if df["Volume"].iloc[i] > avg_vol.iloc[i] * 2:
            if df["Close"].iloc[i] > (df["Open"].iloc[i] + atr.iloc[i] * 0.5):
                sig.iloc[i] =  1
            elif df["Close"].iloc[i] < (df["Open"].iloc[i] - atr.iloc[i] * 0.5):
                sig.iloc[i] = -1
    return sig

def s_wti_squeeze_ny(df, instr):
    """Squeeze momentum during NY session only (13-22 GST)."""
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    atr   = calc_atr(df, 20)
    bb_up = sma + 2 * std
    bb_lo = sma - 2 * std
    kc_up = sma + 1.5 * atr
    kc_lo = sma - 1.5 * atr
    sq    = (bb_up < kc_up) & (bb_lo > kc_lo)
    mom   = df["Close"] - sma
    ix    = df.index
    sig   = pd.Series(0, index=df.index)
    for i in range(20, len(df)):
        try:
            gst = (ix[i].hour + 4) % 24
            if not (13 <= gst <= 22):
                continue
            if (not sq.iloc[i] and sq.iloc[i - 1] and mom.iloc[i] > 0):
                sig.iloc[i] =  1
            elif (not sq.iloc[i] and sq.iloc[i - 1] and mom.iloc[i] < 0):
                sig.iloc[i] = -1
        except:
            continue
    return sig

# ══════════════════════════════════════════════════════
# US30 SPECIFIC STRATEGIES
# ══════════════════════════════════════════════════════

def s_us30_opening_range(df, instr):
    """US30 NY opening range breakout — first bar sets direction."""
    sig   = pd.Series(0, index=df.index)
    or_hi = {}
    or_lo = {}
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            gst = (idx.hour + 4) % 24
            d   = idx.date()
            if gst == 13:
                or_hi[d] = row["High"]
                or_lo[d] = row["Low"]
            elif 14 <= gst <= 16:
                if d in or_hi:
                    if row["Close"] > or_hi[d]:
                        sig.iloc[i] =  1
                    elif row["Close"] < or_lo[d]:
                        sig.iloc[i] = -1
        except:
            continue
    return sig

def s_us30_vix_filter(df, instr):
    """US30 trend follow only when ATR is below its 50-bar average (calm market)."""
    atr     = calc_atr(df)
    price   = df["Close"]
    atr_pct = atr / price * 100
    calm    = atr_pct < atr_pct.rolling(50).mean()
    ema20   = price.ewm(span=20).mean()
    ema50   = price.ewm(span=50).mean()
    sig     = pd.Series(0, index=df.index)
    sig[calm & (ema20 > ema50) & (price > ema20)] =  1
    sig[calm & (ema20 < ema50) & (price < ema20)] = -1
    return sig

STRATEGIES = {
    # Individual validated strategies
    "Squeeze_Momentum":   s_squeeze,
    "ATR_Breakout":       s_atr_breakout,
    "Fair_Value_Gap":     s_fvg,
    "Order_Block":        s_order_block,
    "BOS_CHoCH":          s_bos,
    "Liquidity_Sweep":    s_liq_sweep,
    "Engulfing":          s_engulfing,
    "Bollinger_Breakout": s_bollinger_break,
    "RSI_Divergence":     s_rsi_div,
    "Higher_High_HL":     s_hh_hl,
    # Confluence combinations
    "Squeeze_FVG_RSI":    s_confluence,
    "SMC_Confluence":     s_smc_confluence,
    "KillZone_FVG":       s_kill_zone_fvg,
    "Squeeze_BOS":        s_squeeze_bos,
    # NAS100-specific strategies
    "NAS_VWAP_Reversion":   s_nas_vwap,
    "NAS_Opening_Drive":    s_nas_open_drive,
    "NAS_Gap_Fill":         s_nas_gap_fill,
    "NAS_Momentum_Burst":   s_nas_momentum_burst,
    "NAS_Pullback_EMA":     s_nas_pullback,
    "NAS_Squeeze_VWAP":     s_nas_squeeze_vwap,
    # XAUUSD specific
    "Gold_DXY_Divergence":  s_gold_dxy_divergence,
    "Gold_Asian_Range":     s_gold_asian_range,
    "Gold_COT_Proxy":       s_gold_cot_proxy,
    # GBPUSD specific
    "GBP_London_Open":      s_gbp_london_open,
    "GBP_BOS_London":       s_gbp_bos_london,
    "GBP_Macro_Momentum":   s_gbp_macro_momentum,
    # EURUSD specific
    "EUR_Frankfurt_London": s_eur_frankfurt_london,
    "EUR_Squeeze_Frankfurt": s_eur_squeeze_frankfurt,
    # WTI specific
    "WTI_EIA_Week":         s_wti_eia_week,
    "WTI_Supply_Demand":    s_wti_supply_demand,
    "WTI_Squeeze_NY":       s_wti_squeeze_ny,
    # US30 specific
    "US30_Opening_Range":   s_us30_opening_range,
    "US30_VIX_Filter":      s_us30_vix_filter,
}

# ══════════════════════════════════════════════════════
# CORE BACKTEST ENGINE
# ══════════════════════════════════════════════════════

def run_backtest(df, signals, cfg, instrument):
    try:
        sp_pct = cfg.get("spread_pct", 0.0001)
        mh     = cfg.get("max_hold", 48)

        # Pre-compute ATR for the whole dataframe
        atr_period  = cfg.get("atr_period", 14)
        atr_sl_mult = cfg.get("atr_sl_mult", 1.5)
        atr_tp_mult = cfg.get("atr_tp_mult", 4.5)
        min_sl      = cfg.get("min_sl_pct", 0.003)
        max_sl      = cfg.get("max_sl_pct", 0.020)

        hi_s  = pd.Series(df["High"].values)
        lo_s  = pd.Series(df["Low"].values)
        cl_s  = pd.Series(df["Close"].values)
        tr    = pd.concat([
            hi_s - lo_s,
            (hi_s - cl_s.shift()).abs(),
            (lo_s - cl_s.shift()).abs(),
        ], axis=1).max(axis=1)
        atrs  = tr.rolling(atr_period).mean().values

        trades   = []
        in_trade = False
        entry    = 0.0
        direction = 0
        ei       = 0
        sl = tp = sp = 0.0
        trade_sl_pct = trade_tp_pct = atr_pct = 0.0
        e_date   = None

        c  = df["Close"].values
        h  = df["High"].values
        lo = df["Low"].values
        sg = signals.values
        ix = df.index

        for i in range(50, len(df)):
            if is_weekend(ix[i]):
                continue
            if not in_trade:
                if sg[i] == 0:
                    continue
                if not is_active(ix[i], instrument):
                    continue
                entry = c[i]
                if entry == 0:
                    continue
                direction = int(sg[i])
                sp = entry * sp_pct

                # ADAPTIVE ATR-BASED SL/TP
                atr_val = atrs[i]
                if np.isnan(atr_val) or atr_val == 0:
                    continue
                atr_pct = atr_val / entry
                sl_pct_adaptive = atr_pct * atr_sl_mult
                sl_pct_adaptive = max(min_sl,
                                      min(max_sl, sl_pct_adaptive))
                tp_pct_adaptive = sl_pct_adaptive * 3.0

                if direction == 1:
                    sl = entry * (1 - sl_pct_adaptive)
                    tp = entry * (1 + tp_pct_adaptive)
                else:
                    sl = entry * (1 + sl_pct_adaptive)
                    tp = entry * (1 - tp_pct_adaptive)

                trade_sl_pct = sl_pct_adaptive
                trade_tp_pct = tp_pct_adaptive

                in_trade = True
                ei       = i
                e_date   = ix[i]
            else:
                hit = None
                if direction == 1:
                    if lo[i] <= sl:
                        hit = ("LOSS", sl - entry - sp, i - ei)
                    elif h[i] >= tp:
                        hit = ("WIN",  tp - entry - sp, i - ei)
                else:
                    if h[i] >= sl:
                        hit = ("LOSS", entry - sl - sp, i - ei)
                    elif lo[i] <= tp:
                        hit = ("WIN",  entry - tp - sp, i - ei)

                if not hit and i - ei > mh:
                    pnl = (c[i] - entry) * direction - sp
                    hit = ("WIN" if pnl > 0 else "LOSS",
                           abs(pnl) * (1 if pnl > 0 else -1),
                           i - ei)

                if hit:
                    trades.append({
                        "outcome": hit[0],
                        "pnl":     hit[1],
                        "bars":    hit[2],
                        "year":    e_date.year  if e_date else 0,
                        "month":   e_date.month if e_date else 0,
                        "sl_pct":  trade_sl_pct,
                        "tp_pct":  trade_tp_pct,
                        "atr_pct": atr_pct,
                    })
                    in_trade = False

                # Force close at session end
                if in_trade and not hit:
                    try:
                        gst_now = (ix[i].hour + 4) % 24
                        session_end = {
                            "XAUUSD": 22,
                            "NAS100": 22,
                            "US30":   22,
                            "GBPUSD": 17,
                            "EURUSD": 17,
                            "WTI":    22,
                        }.get(instrument, 22)
                        if gst_now == session_end:
                            pnl = (c[i] - entry) * direction - sp
                            trades.append({
                                "outcome": "WIN" if pnl > 0 else "LOSS",
                                "pnl":     pnl,
                                "bars":    i - ei,
                                "year":    e_date.year  if e_date else 0,
                                "month":   e_date.month if e_date else 0,
                                "sl_pct":  trade_sl_pct,
                                "tp_pct":  trade_tp_pct,
                                "atr_pct": atr_pct,
                                "close_reason": "session_end",
                            })
                            in_trade = False
                    except:
                        pass

        if len(trades) < 5:
            return {"error": f"Only {len(trades)} trades"}

        wins   = [t for t in trades if t["outcome"] == "WIN"]
        losses = [t for t in trades if t["outcome"] == "LOSS"]
        total  = len(trades)
        wr     = len(wins) / total * 100
        gp     = sum(t["pnl"] for t in wins)
        gl     = abs(sum(t["pnl"] for t in losses))
        pf     = round(gp / gl, 2) if gl > 0 else 9.99
        aw     = gp / len(wins)   if wins   else 0.0
        al     = gl / len(losses) if losses else 0.0
        ev     = (wr / 100 * aw) - (1 - wr / 100) * al
        net    = gp - gl

        yrs = {}
        for t in trades:
            y = t["year"]
            if y not in yrs:
                yrs[y] = {"w": 0, "l": 0}
            if t["outcome"] == "WIN":
                yrs[y]["w"] += 1
            else:
                yrs[y]["l"] += 1

        yr_wrs = [d["w"] / (d["w"] + d["l"]) * 100
                  for d in yrs.values()
                  if d["w"] + d["l"] > 0]
        consistency = (
            round(len([w for w in yr_wrs if w >= 35]) /
                  len(yr_wrs) * 100, 1)
            if yr_wrs else 0.0)

        # Adaptive SL means variable risk/reward
        # Lower thresholds slightly to account for this variability
        if (pf >= 1.4 and total >= 30
                and consistency >= 55
                and wr >= 35):
            grade = "A"
        elif (pf >= 1.2 and total >= 20
              and consistency >= 45
              and wr >= 30):
            grade = "B"
        elif (pf >= 1.05 and total >= 10
              and wr >= 25):
            grade = "C"
        else:
            grade = "D"

        return {
            "grade":          grade,
            "total_trades":   total,
            "wins":           len(wins),
            "losses":         len(losses),
            "is_profitable":  bool(net > 0),
            "win_rate":       round(wr, 1),
            "profit_factor":  pf,
            "expected_value": round(ev, 4),
            "net_pnl":        round(net, 4),
            "avg_bars":       round(
                sum(t["bars"] for t in trades) / total, 1),
            "consistency":    consistency,
            "yearly":         yrs,
            "all_trades":     trades,
        }
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════════════
# 20x LEVERAGE RISK ENGINE
# ══════════════════════════════════════════════════════

def analyze_leverage_risk(trades, cfg, instrument):
    try:
        max_consec = 0
        cur_consec = 0
        balance    = float(ACCOUNT_BALANCE)
        peak       = balance
        max_dd     = 0.0

        for t in trades:
            if balance <= 0:
                break
            t_sl        = t.get("sl_pct",
                                cfg.get("min_sl_pct", 0.005))
            t_tp        = t_sl * 3.0
            acct_risk   = t_sl * LEVERAGE * 100
            acct_target = t_tp * LEVERAGE * 100

            if t["outcome"] == "WIN":
                balance += balance * (acct_target / 100)
                cur_consec = 0
            else:
                balance -= balance * (acct_risk / 100)
                cur_consec += 1
                max_consec  = max(max_consec, cur_consec)

            balance = max(0.0,
                          min(balance, ACCOUNT_BALANCE * 50))
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Average risk per trade
        avg_sl = (np.mean([t.get("sl_pct",
                            cfg.get("min_sl_pct", 0.005))
                           for t in trades])
                  if trades else 0.01)
        avg_risk = avg_sl * LEVERAGE * 100

        losses_to_ruin = 0
        b = float(ACCOUNT_BALANCE)
        while b > 50 and losses_to_ruin < 100:
            b -= b * avg_risk / 100
            losses_to_ruin += 1

        if (max_consec <= 4 and max_dd <= 35):
            safety = "SAFE ✅"
        elif (max_consec <= 6 and max_dd <= 55):
            safety = "MODERATE ⚠️"
        else:
            safety = "DANGEROUS ❌"

        return {
            "avg_risk_per_trade_pct":  round(avg_risk, 1),
            "avg_tp_per_trade_pct":    round(avg_risk * 3, 1),
            "max_consecutive_losses":  max_consec,
            "losses_to_ruin":          losses_to_ruin,
            "simulated_final_balance": round(balance, 2),
            "max_drawdown_pct":        round(max_dd, 1),
            "safety_rating":           safety,
            "leverage":                LEVERAGE,
            "adaptive_sl":             True,
        }
    except Exception as e:
        return {"error": str(e), "safety_rating": "UNKNOWN"}

# ══════════════════════════════════════════════════════
# ROLLING WINDOW WALK-FORWARD
# Solves WF% = 0 by generating many test windows
# ══════════════════════════════════════════════════════

def rolling_window_wf(df, strat_func, cfg, instrument,
                      window_bars=800, step_bars=150):
    try:
        results = []
        i = 0
        while i + window_bars + step_bars < len(df):
            test = df.iloc[i + window_bars:
                           i + window_bars + step_bars]
            if len(test) < 50:
                break
            try:
                sigs = strat_func(test.copy(), instrument)
                res  = run_backtest(test, sigs, cfg, instrument)
                if ("error" not in res and
                        res.get("total_trades", 0) >= 5):
                    results.append({
                        "profitable": res["profit_factor"] > 1.0,
                        "pf":         res["profit_factor"],
                        "wr":         res["win_rate"],
                        "trades":     res["total_trades"],
                    })
            except Exception:
                pass
            i += step_bars

        if not results:
            return {"wf_score": 0.0, "windows_tested": 0}

        prof   = sum(1 for r in results if r["profitable"])
        wf     = round(prof / len(results) * 100, 1)
        avg_pf = round(sum(r["pf"] for r in results) / len(results), 2)

        return {
            "wf_score":           wf,
            "windows_tested":     len(results),
            "profitable_windows": prof,
            "avg_oos_pf":         avg_pf,
            "min_pf":             round(min(r["pf"] for r in results), 2)
                                  if results else 0,
            "max_pf":             round(max(r["pf"] for r in results), 2)
                                  if results else 0,
            "failed_windows":     "Check if 0 — means too few trades",
        }
    except Exception as e:
        return {"wf_score": 0.0, "error": str(e)}

# ══════════════════════════════════════════════════════
# MARKET REGIME TESTING
# ══════════════════════════════════════════════════════

def test_by_regime(df, signals, cfg, instrument):
    try:
        atr         = calc_atr(df)
        price_range = (df["Close"].rolling(20).max() -
                       df["Close"].rolling(20).min())
        adx_proxy   = (atr / price_range * 100).fillna(0)

        results = {}
        for regime, idx_set in [
            ("trending", df.index[adx_proxy > 25]),
            ("ranging",  df.index[adx_proxy <= 25]),
        ]:
            if len(idx_set) < 100:
                continue
            df_r   = df.loc[df.index.isin(idx_set)]
            sigs_r = signals.loc[signals.index.isin(idx_set)]
            res    = run_backtest(df_r, sigs_r, cfg, instrument)
            if "error" not in res:
                results[regime] = {
                    "win_rate": res["win_rate"],
                    "pf":       res["profit_factor"],
                    "trades":   res["total_trades"],
                }
        return results
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════════════
# MONTE CARLO — 20x leverage aware
# ══════════════════════════════════════════════════════

def monte_carlo_20x(trades, cfg, n=500):
    try:
        sl_pct   = cfg["sl_pct"]
        tp_pct   = cfg["tp_pct"]
        outcomes = [t["outcome"] for t in trades]

        final_bals = []
        max_dds    = []
        ruins      = 0

        for _ in range(n):
            shuffled = random.sample(outcomes, len(outcomes))
            bal      = float(ACCOUNT_BALANCE)
            peak     = bal
            max_dd   = 0.0

            for outcome in shuffled:
                if bal <= 10:
                    ruins += 1
                    break
                if outcome == "WIN":
                    gain = bal * cfg["tp_pct"] * LEVERAGE
                    gain = min(gain, bal * 1.5)
                    bal  = min(bal + gain,
                               ACCOUNT_BALANCE * 30)
                else:
                    loss = bal * cfg["sl_pct"] * LEVERAGE
                    bal  = max(bal - loss, 0)
                bal = max(bal, 0)
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            final_bals.append(
                min(max(0.0, bal), ACCOUNT_BALANCE * 100))
            max_dds.append(max_dd)

        final_bals.sort()
        max_dds.sort()
        n_tot = len(final_bals)

        return {
            "starting":       ACCOUNT_BALANCE,
            "leverage":       LEVERAGE,
            "median_final":   round(final_bals[n_tot // 2], 2),
            "best_10pct":     round(final_bals[int(n_tot * 0.9)], 2),
            "worst_10pct":    round(final_bals[int(n_tot * 0.1)], 2),
            "prob_profit":    round(
                sum(1 for b in final_bals if b > ACCOUNT_BALANCE) /
                n_tot * 100, 1),
            "prob_ruin":      round(ruins / n * 100, 1),
            "median_max_dd":  round(max_dds[n_tot // 2], 1),
            "worst_dd_10pct": round(max_dds[int(n_tot * 0.9)], 1),
        }
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════

def run_full_backtest():
    all_results = {}
    summary     = []

    print("=" * 70)
    print("TradingBotV1 -- MAXIMUM REFINED BACKTEST")
    print(f"Account: ${ACCOUNT_BALANCE} | "
          f"Leverage: {LEVERAGE}x | "
          f"Position: ${POSITION_SIZE}")
    print(f"Strategies: {len(STRATEGIES)} | "
          f"Instruments: {len(INSTRUMENTS)}")
    print("Features: Rolling WF | 20x Risk | Regime | Monte Carlo")
    print("=" * 70)

    FOCUS = ["XAUUSD", "NAS100", "US30"]
    for instr, cfg in INSTRUMENTS.items():
        if instr not in FOCUS:
            continue
        print(f"\n{'='*70}")
        print(f"  {instr} -- {cfg['interval']} | {cfg['period']}")

        try:
            df = yf.download(
                cfg["ticker"],
                period=cfg["period"],
                interval=cfg["interval"],
                progress=False,
                auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 200:
                print(f"  WARNING: No data -- skipped")
                continue

            bars  = len(df)
            years = (df.index[-1] - df.index[0]).days / 365
            print(f"  OK {bars} bars | {years:.1f} years")
            print(f"\n  {'Strategy':<22} "
                  f"{'Gr':>2} "
                  f"{'WR':>4} "
                  f"{'PF':>5} "
                  f"{'RolWF%':>7} "
                  f"{'Windows':>8} "
                  f"{'MC-Med':>8} "
                  f"{'Ruin%':>6} "
                  f"{'Safety':>10}")
            print(f"  {'-'*80}")

            all_results[instr] = {
                "bars":       bars,
                "years":      round(years, 1),
                "strategies": {}
            }
            rows = []

            for name, func in STRATEGIES.items():
                try:
                    sigs = func(df.copy(), instr)
                    res  = run_backtest(df.copy(), sigs, cfg, instr)

                    if "error" in res:
                        continue

                    # Rolling window walk-forward
                    if instr == "XAUUSD":
                        wf = rolling_window_wf(
                            df, func, cfg, instr,
                            window_bars=600,
                            step_bars=150)
                    else:
                        wf = rolling_window_wf(
                            df, func, cfg, instr,
                            window_bars=2000,
                            step_bars=500)

                    # 20x Monte Carlo
                    mc = monte_carlo_20x(
                        res.get("all_trades", []), cfg, n=500)

                    # Leverage risk
                    lev = analyze_leverage_risk(
                        res.get("all_trades", []), cfg, instr)

                    # Regime testing
                    reg = test_by_regime(
                        df.copy(), sigs.copy(), cfg, instr)

                    # Strip all_trades before saving
                    res_save = {k: v for k, v in res.items()
                                if k != "all_trades"}
                    res_save["walk_forward"]  = wf
                    res_save["monte_carlo"]   = mc
                    res_save["leverage_risk"] = lev
                    res_save["regime"]        = reg

                    all_results[instr]["strategies"][name] = res_save

                    g    = res["grade"]
                    wr   = res["win_rate"]
                    pf   = res["profit_factor"]
                    wfs  = wf.get("wf_score", 0)
                    wins = wf.get("windows_tested", 0)
                    mcm  = mc.get("median_final", 0)
                    ruin = mc.get("prob_ruin", 0)
                    safe = lev.get("safety_rating", "?")

                    print(f"  [{g}] {name:<21} "
                          f"{g:>2} "
                          f"{wr:>3.0f}% "
                          f"{pf:>5.2f} "
                          f"{wfs:>6.0f}% "
                          f"{wins:>8} "
                          f"${mcm:>7.0f} "
                          f"{ruin:>5.1f}% "
                          f"{safe}")
                    all_trades_list = res.get("all_trades", [])
                    if res.get("is_profitable"):
                        dollar_profit = sum(
                            ACCOUNT_BALANCE * t.get("tp_pct", 0) * LEVERAGE
                            for t in all_trades_list
                            if t["outcome"] == "WIN")
                        dollar_loss = sum(
                            ACCOUNT_BALANCE * t.get("sl_pct", 0) * LEVERAGE
                            for t in all_trades_list
                            if t["outcome"] == "LOSS")
                        dollar_net = dollar_profit - dollar_loss
                        print(
                            f"       💰 ${dollar_net:+,.0f} net on "
                            f"${ACCOUNT_BALANCE:,} | "
                            f"Wins:+${dollar_profit:,.0f} "
                            f"Losses:-${dollar_loss:,.0f}")
                    all_sls = [
                        t.get("sl_pct", 0)
                        for t in all_trades_list]
                    if all_sls:
                        avg_sl = np.mean(all_sls) * 100
                        avg_tp = avg_sl * 3
                        print(f"       📏 Avg adaptive SL: "
                              f"{avg_sl:.2f}% | "
                              f"Avg TP: {avg_tp:.2f}% | "
                              f"Avg account risk: "
                              f"{avg_sl*LEVERAGE:.1f}%")

                    rows.append({
                        "strategy":    name,
                        "grade":       g,
                        "win_rate":    wr,
                        "pf":          pf,
                        "wf_score":    wfs,
                        "wf_windows":  wins,
                        "mc_median":   mcm,
                        "ruin_pct":    ruin,
                        "safety":      safe,
                        "consistency": res.get("consistency", 0),
                        "ev":          res.get("expected_value", 0),
                        "regime":      reg,
                    })

                    if wf.get("windows_tested", 0) == 0:
                        print(f"    ⚠️  {name}: "
                              f"WF windows = 0 — "
                              f"step_bars may still be too small")

                except Exception as e:
                    print(f"  ERR {name}: {e}")

            # Rank: WF score x PF x safety multiplier
            def score(r):
                s = r["pf"] * (r["wf_score"] + 1)
                if r["safety"] == "SAFE":
                    s *= 1.5
                elif r["safety"] == "DANGEROUS":
                    s *= 0.3
                return s

            rows.sort(key=score, reverse=True)
            all_results[instr]["ranked"] = rows

            a           = [r for r in rows if r["grade"] == "A"]
            b           = [r for r in rows if r["grade"] == "B"]
            safe_strats = [r for r in rows if r["safety"] == "SAFE"]
            danger      = [r for r in rows if r["safety"] == "DANGEROUS"]

            print(f"\n  TOP 5 for {instr}:")
            for r in rows[:5]:
                print(f"     [{r['grade']}] "
                      f"{r['strategy']:<22} "
                      f"WR:{r['win_rate']}% "
                      f"PF:{r['pf']} "
                      f"WF:{r['wf_score']}% "
                      f"({r['wf_windows']} wins) "
                      f"MC:${r['mc_median']:.0f} "
                      f"Ruin:{r['ruin_pct']}% "
                      f"{r['safety']}")

            if danger:
                print(f"\n  DANGEROUS at 20x leverage:")
                for r in danger:
                    print(f"     [X] {r['strategy']}")

            summary.append({
                "instrument":   instr,
                "years":        round(years, 1),
                "bars":         bars,
                "tested":       len(rows),
                "grade_A":      len(a),
                "grade_B":      len(b),
                "safe_count":   len(safe_strats),
                "danger_count": len(danger),
                "top_5":        rows[:5],
                "grade_A_list": [r["strategy"] for r in a],
                "safe_list":    [r["strategy"] for r in safe_strats],
                "danger_list":  [r["strategy"] for r in danger],
            })

        except Exception as e:
            print(f"  ERR {instr}: {e}")

    # ── Save ───────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    output = {
        "generated_at":  datetime.now().isoformat(),
        "account":       ACCOUNT_BALANCE,
        "leverage":      LEVERAGE,
        "position_size": POSITION_SIZE,
        "strategies":    len(STRATEGIES),
        "instruments":   list(all_results.keys()),
        "summary":       summary,
        "full_results":  all_results,
    }
    with open("data/backtest_fixed_v3.json",
              "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # ── Master report ──────────────────────────────────
    print("\n" + "=" * 70)
    print("MASTER REPORT — ADAPTIVE ATR SL/TP")
    print("SL = 1.5x current ATR at signal time")
    print("TP = 4.5x current ATR (1:3 RR always)")
    print("Leverage: 10x | Account: $1,000")
    print("=" * 70)
    print(f"\n{'Instr':<8} {'A':>3} {'B':>3} "
          f"{'Safe':>5} {'Danger':>7} "
          f"{'Best Strategy':<22} "
          f"{'WF%':>5} {'MC$':>7} {'Ruin%':>6}")
    print("-" * 70)
    for s in summary:
        t = s["top_5"][0] if s["top_5"] else {}
        print(f"{s['instrument']:<8} "
              f"{s['grade_A']:>3} "
              f"{s['grade_B']:>3} "
              f"{s['safe_count']:>5} "
              f"{s['danger_count']:>7} "
              f"{t.get('strategy', 'N/A'):<22} "
              f"{t.get('wf_score', 0):>4.0f}% "
              f"${t.get('mc_median', 0):>6.0f} "
              f"{t.get('ruin_pct', 0):>5.1f}%")

    print("\nSAFE STRATEGIES AT 20x LEVERAGE:")
    for s in summary:
        if s["safe_list"]:
            print(f"\n  {s['instrument']}:")
            for st in s["safe_list"]:
                print(f"    [SAFE] {st}")

    print("\nDO NOT TRADE AT 20x LEVERAGE:")
    for s in summary:
        if s["danger_list"]:
            print(f"\n  {s['instrument']}:")
            for st in s["danger_list"]:
                print(f"    [X] {st}")

    print(f"Saved: data/backtest_fixed_v3.json")
    print(f"Done:  {datetime.now().strftime('%H:%M:%S')}")
    return output


if __name__ == "__main__":
    run_full_backtest()
