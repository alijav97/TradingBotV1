#!/usr/bin/env python3
"""
XAUUSD Comprehensive Strategy Backtest
Tests 47 strategies across 30min, 1hr, 4hr, 1day timeframes.
Ranks by win rate, profit factor, and Sharpe ratio composite score.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

TICKER = 'GC=F'           # Gold Futures (COMEX)
SL_ATR_MULT = 1.5         # Stop loss: 1.5x ATR(14)
TP_ATR_MULT = 3.0         # Take profit: 3.0x ATR(14)  → 1:2 risk/reward
MAX_HOLD_BARS = 20        # Max bars to hold a trade
MIN_TRADES = 12           # Minimum trades required to include strategy

TIMEFRAMES = {
    '30m': {'interval': '30m', 'period': '60d',  'resample': None},
    '1h':  {'interval': '1h',  'period': '2y',   'resample': None},
    '4h':  {'interval': '1h',  'period': '2y',   'resample': '4h'},
    '1d':  {'interval': '1d',  'period': '5y',   'resample': None},
}

# ============================================================
# SYNTHETIC DATA GENERATOR (fallback when network is unavailable)
# Calibrated to gold's real statistical properties:
#   Annual return ~8%, Annual vol ~16%, fat tails, regime-switching
# ============================================================

def generate_synthetic_xauusd(timeframe_key, seed=42):
    """
    Regime-switching GBM calibrated to historical XAUUSD behavior.
    Used as fallback when yfinance is unavailable.
    """
    np.random.seed(seed)
    tf_cfg = {
        '30m': {'n': 2880,  'start': '2024-01-01', 'freq': '30min', 'h_scale': 0.5},
        '1h':  {'n': 11500, 'start': '2022-01-01', 'freq': 'h',     'h_scale': 1.0},
        '4h':  {'n': 7000,  'start': '2019-01-01', 'freq': '4h',    'h_scale': 4.0},
        '1d':  {'n': 1300,  'start': '2019-01-01', 'freq': 'B',     'h_scale': 23.0},
    }
    cfg = tf_cfg[timeframe_key]
    n = cfg['n']
    sc = cfg['h_scale']

    # Per-bar annualised params (gold: 8% drift, 16% vol, ~5796 trading hours/year)
    base_hours = 5796
    mu0 = (0.08 / base_hours) * sc
    s0  = (0.16 / np.sqrt(base_hours)) * np.sqrt(sc)

    # Four market regimes: bull, bear, range, spike
    reg_mu    = [mu0*3, -mu0*2,  0.0,      mu0]
    reg_sig   = [s0*0.8, s0*0.9, s0*0.6,  s0*2.5]
    reg_ac    = [0.12,   0.12,  -0.08,    0.0]
    # Transition probs (rows=from, cols=to: bull, bear, range, spike)
    P = np.array([
        [1-0.005,  0.001,  0.003,  0.001],
        [0.003,  1-0.007,  0.003,  0.001],
        [0.004,    0.002, 1-0.007, 0.001],
        [0.010,    0.005,  0.005, 1-0.020],
    ])

    regime = 2          # start in range
    closes = [2000.0]   # typical gold price ~$2000
    last_z = 0.0

    for _ in range(n - 1):
        regime = np.random.choice(4, p=P[regime])
        z = np.random.randn()
        if np.random.random() < 0.025:          # fat-tail event
            z = np.sign(z) * (abs(z) * 2.2 + 1.8)
        z_corr = z + reg_ac[regime] * last_z
        ret = reg_mu[regime] + reg_sig[regime] * z_corr
        closes.append(max(closes[-1] * (1 + ret), 100.0))
        last_z = z_corr

    closes = np.array(closes)
    opens  = np.concatenate([[closes[0]], closes[:-1]])
    hl_rng = reg_sig[0] * 2.5 * closes
    rnd_hi = np.abs(np.random.randn(n)) * 0.6
    rnd_lo = np.abs(np.random.randn(n)) * 0.6
    highs  = np.maximum(opens, closes) + hl_rng * rnd_hi
    lows   = np.minimum(opens, closes) - hl_rng * rnd_lo
    vols   = np.random.lognormal(10, 0.5, n).astype(int)

    idx = pd.date_range(start=cfg['start'], periods=n, freq=cfg['freq'])
    df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows,
                       'Close': closes, 'Volume': vols}, index=idx)
    return df.round(2)


# ============================================================
# DATA FETCHING
# ============================================================

_DATA_SOURCE = 'LIVE'   # updated to 'SYNTHETIC' if live fails

def fetch_data(timeframe_key):
    global _DATA_SOURCE
    cfg = TIMEFRAMES[timeframe_key]
    print(f"  Downloading XAUUSD {timeframe_key} data...")
    df = None
    try:
        raw = yf.download(TICKER, interval=cfg['interval'], period=cfg['period'],
                          auto_adjust=True, progress=False)
        if raw.empty:
            raw = yf.download('XAUUSD=X', interval=cfg['interval'], period=cfg['period'],
                              auto_adjust=True, progress=False)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            if cfg['resample']:
                df = df.resample(cfg['resample']).agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min',
                     'Close': 'last', 'Volume': 'sum'}).dropna()
    except Exception:
        pass

    if df is None or len(df) < 100:
        _DATA_SOURCE = 'SYNTHETIC'
        df = generate_synthetic_xauusd(timeframe_key)
        print(f"  [SYNTHETIC data – real statistical profile of gold]  {len(df)} bars")
    else:
        print(f"  Got {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")

    return df

# ============================================================
# INDICATOR LIBRARY
# ============================================================

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _atr(df, p=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def _rsi(s, p=14):
    d = s.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1/p, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def _macd(s, fast=12, slow=26, sig=9):
    ml = _ema(s, fast) - _ema(s, slow)
    sl = _ema(ml, sig)
    return ml, sl, ml - sl

def _bbands(s, p=20, k=2.0):
    mid = _sma(s, p)
    std = s.rolling(p).std()
    return mid + k*std, mid, mid - k*std

def _keltner(df, p=20, m=2.0):
    mid = _ema(df['Close'], p)
    a = _atr(df, p)
    return mid + m*a, mid, mid - m*a

def _stoch(df, k=14, d=3):
    lo = df['Low'].rolling(k).min()
    hi = df['High'].rolling(k).max()
    pct_k = 100 * (df['Close'] - lo) / (hi - lo)
    return pct_k, pct_k.rolling(d).mean()

def _cci(df, p=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - _sma(tp, p)) / (0.015 * mad)

def _williams_r(df, p=14):
    hi = df['High'].rolling(p).max()
    lo = df['Low'].rolling(p).min()
    return -100 * (hi - df['Close']) / (hi - lo)

def _adx_dmi(df, p=14):
    hi, lo, cl = df['High'], df['Low'], df['Close']
    pdm = hi.diff().clip(lower=0)
    ndm = (-lo.diff()).clip(lower=0)
    pdm = pdm.where(pdm > ndm, 0)
    ndm = ndm.where(ndm > pdm, 0)
    tr = pd.concat([hi-lo, (hi-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()], axis=1).max(axis=1)
    atr_p = tr.ewm(span=p, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / atr_p
    ndi = 100 * ndm.ewm(span=p, adjust=False).mean() / atr_p
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(span=p, adjust=False).mean()
    return adx, pdi, ndi

def _parabolic_sar(df, af0=0.02, af_max=0.2):
    n = len(df)
    sar = np.zeros(n)
    trend = np.ones(n)
    ep = np.zeros(n)
    af = np.zeros(n)
    h = df['High'].values
    l = df['Low'].values
    sar[0] = l[0]; ep[0] = h[0]; af[0] = af0
    for i in range(1, n):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[max(i-2,0)])
            if l[i] < sar[i]:
                trend[i] = -1; sar[i] = ep[i-1]; ep[i] = l[i]; af[i] = af0
            else:
                trend[i] = 1
                if h[i] > ep[i-1]: ep[i] = h[i]; af[i] = min(af[i-1]+af0, af_max)
                else: ep[i] = ep[i-1]; af[i] = af[i-1]
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[max(i-2,0)])
            if h[i] > sar[i]:
                trend[i] = 1; sar[i] = ep[i-1]; ep[i] = h[i]; af[i] = af0
            else:
                trend[i] = -1
                if l[i] < ep[i-1]: ep[i] = l[i]; af[i] = min(af[i-1]+af0, af_max)
                else: ep[i] = ep[i-1]; af[i] = af[i-1]
    return pd.Series(trend, index=df.index)

def _supertrend(df, p=10, mult=3.0):
    a = _atr(df, p).values
    hl2 = ((df['High'] + df['Low']) / 2).values
    cl = df['Close'].values
    n = len(df)
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    fub = ub.copy(); flb = lb.copy()
    direction = np.ones(n)
    for i in range(1, n):
        fub[i] = ub[i] if fub[i-1] < ub[i] or cl[i-1] > fub[i-1] else fub[i-1]
        flb[i] = lb[i] if flb[i-1] > lb[i] or cl[i-1] < flb[i-1] else flb[i-1]
        if direction[i-1] == -1:
            direction[i] = 1 if cl[i] > fub[i-1] else -1
        else:
            direction[i] = -1 if cl[i] < flb[i-1] else 1
    return pd.Series(direction, index=df.index)

def _aroon(df, p=25):
    n = len(df)
    h = df['High'].values; l = df['Low'].values
    au = np.zeros(n); ad = np.zeros(n)
    for i in range(p, n):
        au[i] = (p - np.argmax(h[i-p:i+1][::-1])) / p * 100
        ad[i] = (p - np.argmin(l[i-p:i+1][::-1])) / p * 100
    return pd.Series(au, index=df.index), pd.Series(ad, index=df.index)

def _ichimoku(df):
    tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    kijun  = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    sa = ((tenkan + kijun) / 2).shift(26)
    sb = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, sa, sb

def _hma(s, p):
    return _ema(2 * _ema(s, p//2) - _ema(s, p), int(np.sqrt(p)))

def _dema(s, p):
    e = _ema(s, p)
    return 2*e - _ema(e, p)

def _trix(s, p=15):
    e1 = _ema(s, p); e2 = _ema(e1, p); e3 = _ema(e2, p)
    return e3.pct_change() * 100

def _obv(df):
    return (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()

def _vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

# ============================================================
# PRECOMPUTE ALL INDICATORS
# ============================================================

def compute_indicators(df):
    df = df.copy()
    c = df['Close']

    df['ATR14'] = _atr(df, 14)

    for p in [9, 20, 21, 50, 100, 200]:
        df[f'EMA{p}'] = _ema(c, p)
    for p in [20, 50, 200]:
        df[f'SMA{p}'] = _sma(c, p)

    df['HMA20'] = _hma(c, 20);  df['HMA50'] = _hma(c, 50)
    df['DEMA20'] = _dema(c, 20); df['DEMA50'] = _dema(c, 50)

    df['MACD'], df['MACD_sig'], df['MACD_hist'] = _macd(c)
    df['RSI14'] = _rsi(c, 14)

    df['BB_up'], df['BB_mid'], df['BB_lo'] = _bbands(c)
    df['KC_up'], df['KC_mid'], df['KC_lo'] = _keltner(df)

    df['Stoch_K'], df['Stoch_D'] = _stoch(df)
    df['CCI20'] = _cci(df)
    df['WR14'] = _williams_r(df)
    df['ROC14'] = c.pct_change(14) * 100
    df['MOM14'] = c - c.shift(14)

    df['ADX'], df['DI_plus'], df['DI_minus'] = _adx_dmi(df)
    df['SAR_dir'] = _parabolic_sar(df)
    df['ST_dir'] = _supertrend(df)
    df['Aroon_up'], df['Aroon_dn'] = _aroon(df)

    df['Tenkan'], df['Kijun'], df['Senkou_A'], df['Senkou_B'] = _ichimoku(df)

    df['TRIX'] = _trix(c); df['TRIX_sig'] = _sma(df['TRIX'], 9)

    df['DC20_hi'] = df['High'].rolling(20).max()
    df['DC20_lo'] = df['Low'].rolling(20).min()
    df['DC50_hi'] = df['High'].rolling(50).max()
    df['DC50_lo'] = df['Low'].rolling(50).min()

    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2*df['Pivot'] - df['Low'].shift(1)
    df['S1'] = 2*df['Pivot'] - df['High'].shift(1)

    df['OBV'] = _obv(df); df['OBV_EMA'] = _ema(df['OBV'], 20)
    df['VWAP'] = _vwap(df)

    return df.dropna(subset=['ATR14', 'EMA50'])

# ============================================================
# HELPER: crossover signal
# ============================================================

def _cross(fast, slow):
    above = (fast > slow).astype(int)
    d = above.diff()
    sig = pd.Series(0, index=fast.index)
    sig[d == 1] = 1; sig[d == -1] = -1
    return sig

# ============================================================
# 47 STRATEGY SIGNAL GENERATORS
# ============================================================

# --- TREND FOLLOWING (EMA/SMA) ---

def sig_ema_9_21(df):        return _cross(df['EMA9'],  df['EMA21'])
def sig_ema_20_50(df):       return _cross(df['EMA20'], df['EMA50'])
def sig_ema_50_200(df):      return _cross(df['EMA50'], df['EMA200'])
def sig_sma_20_50(df):       return _cross(df['SMA20'], df['SMA50'])
def sig_hma_cross(df):       return _cross(df['HMA20'], df['HMA50'])
def sig_dema_cross(df):      return _cross(df['DEMA20'],df['DEMA50'])

# --- MACD ---

def sig_macd_signal(df):     return _cross(df['MACD'], df['MACD_sig'])
def sig_macd_zero(df):       return _cross(df['MACD'], pd.Series(0.0, index=df.index))

def sig_macd_histogram(df):
    h = df['MACD_hist']
    s = pd.Series(0, index=df.index)
    s[(h > 0) & (h.shift(1) <= 0)] = 1
    s[(h < 0) & (h.shift(1) >= 0)] = -1
    return s

# --- RSI ---

def sig_rsi_mean_rev(df):
    r = df['RSI14']
    s = pd.Series(0, index=df.index)
    s[(r < 30) & (r.shift(1) >= 30)] = 1
    s[(r > 70) & (r.shift(1) <= 70)] = -1
    return s

def sig_rsi_50_cross(df):    return _cross(df['RSI14'], pd.Series(50.0, index=df.index))

def sig_rsi_40_60(df):
    r = df['RSI14']
    s = pd.Series(0, index=df.index)
    s[(r > 60) & (r.shift(1) <= 60)] = 1
    s[(r < 40) & (r.shift(1) >= 40)] = -1
    return s

def sig_rsi_divergence(df, lback=12):
    sig = pd.Series(0, index=df.index)
    c = df['Close'].values; r = df['RSI14'].values
    for i in range(lback, len(df)):
        wc = c[i-lback:i+1]; wr = r[i-lback:i+1]
        if np.any(np.isnan(wr)): continue
        if c[i] == wc.min():
            pi = np.argmin(wc[:-1])
            if r[i] > wr[pi]: sig.iloc[i] = 1
        if c[i] == wc.max():
            pi = np.argmax(wc[:-1])
            if r[i] < wr[pi]: sig.iloc[i] = -1
    return sig

# --- OSCILLATORS ---

def sig_stochastic(df):
    k, d = df['Stoch_K'], df['Stoch_D']
    s = pd.Series(0, index=df.index)
    s[(k > d) & (k.shift(1) <= d.shift(1)) & (k < 40)] = 1
    s[(k < d) & (k.shift(1) >= d.shift(1)) & (k > 60)] = -1
    return s

def sig_cci_breakout(df):
    cc = df['CCI20']
    s = pd.Series(0, index=df.index)
    s[(cc > 100) & (cc.shift(1) <= 100)] = 1
    s[(cc < -100) & (cc.shift(1) >= -100)] = -1
    return s

def sig_cci_zero(df):        return _cross(df['CCI20'], pd.Series(0.0, index=df.index))

def sig_williams_r(df):
    wr = df['WR14']
    s = pd.Series(0, index=df.index)
    s[(wr > -20) & (wr.shift(1) <= -20)] = -1  # leaving overbought
    s[(wr < -80) & (wr.shift(1) >= -80)] = 1   # entering oversold
    return s

def sig_roc_zero(df):        return _cross(df['ROC14'], pd.Series(0.0, index=df.index))
def sig_momentum_zero(df):   return _cross(df['MOM14'], pd.Series(0.0, index=df.index))

# --- VOLATILITY / CHANNEL ---

def sig_bb_breakout(df):
    c = df['Close']
    s = pd.Series(0, index=df.index)
    s[(c > df['BB_up']) & (c.shift(1) <= df['BB_up'].shift(1))] = 1
    s[(c < df['BB_lo']) & (c.shift(1) >= df['BB_lo'].shift(1))] = -1
    return s

def sig_bb_mean_rev(df):
    s = pd.Series(0, index=df.index)
    s[(df['Low'] <= df['BB_lo']) & (df['Low'].shift(1) > df['BB_lo'].shift(1))] = 1
    s[(df['High'] >= df['BB_up']) & (df['High'].shift(1) < df['BB_up'].shift(1))] = -1
    return s

def sig_keltner_break(df):
    c = df['Close']
    s = pd.Series(0, index=df.index)
    s[c > df['KC_up']] = 1
    s[c < df['KC_lo']] = -1
    return s

def sig_donchian_20(df):
    c = df['Close']
    s = pd.Series(0, index=df.index)
    s[(c > df['DC20_hi'].shift(1))] = 1
    s[(c < df['DC20_lo'].shift(1))] = -1
    return s

def sig_donchian_50(df):
    c = df['Close']
    s = pd.Series(0, index=df.index)
    s[(c > df['DC50_hi'].shift(1))] = 1
    s[(c < df['DC50_lo'].shift(1))] = -1
    return s

def sig_atr_breakout(df):
    pc = df['Close'].shift(1); a = df['ATR14'].shift(1)
    s = pd.Series(0, index=df.index)
    s[df['Close'] > pc + a] = 1
    s[df['Close'] < pc - a] = -1
    return s

# --- TREND STRENGTH ---

def sig_adx_di(df):
    s = pd.Series(0, index=df.index)
    adx = df['ADX']; di_p = df['DI_plus']; di_n = df['DI_minus']
    s[(adx > 25) & (di_p > di_n) & (di_p.shift(1) <= di_n.shift(1))] = 1
    s[(adx > 25) & (di_n > di_p) & (di_n.shift(1) <= di_p.shift(1))] = -1
    return s

def sig_parabolic_sar(df):
    sar = df['SAR_dir']
    s = pd.Series(0, index=df.index)
    s[(sar == 1) & (sar.shift(1) == -1)] = 1
    s[(sar == -1) & (sar.shift(1) == 1)] = -1
    return s

def sig_supertrend(df):
    st = df['ST_dir']
    s = pd.Series(0, index=df.index)
    s[(st == 1) & (st.shift(1) == -1)] = 1
    s[(st == -1) & (st.shift(1) == 1)] = -1
    return s

def sig_aroon(df):
    au = df['Aroon_up']; ad = df['Aroon_dn']
    s = pd.Series(0, index=df.index)
    s[(au > ad) & (au.shift(1) <= ad.shift(1))] = 1
    s[(ad > au) & (ad.shift(1) <= au.shift(1))] = -1
    return s

# --- COMPLEX ---

def sig_ichimoku(df):
    s = pd.Series(0, index=df.index)
    c = df['Close']; tk = df['Tenkan']; kj = df['Kijun']
    sa = df['Senkou_A']; sb = df['Senkou_B']
    cloud_top = sa.combine(sb, max)
    cloud_bot = sa.combine(sb, min)
    tk_cross_up = (tk > kj) & (tk.shift(1) <= kj.shift(1))
    tk_cross_dn = (tk < kj) & (tk.shift(1) >= kj.shift(1))
    s[tk_cross_up & (c > cloud_top)] = 1
    s[tk_cross_dn & (c < cloud_bot)] = -1
    return s

def sig_squeeze_momentum(df):
    in_squeeze = (df['BB_up'] < df['KC_up']) & (df['BB_lo'] > df['KC_lo'])
    released = ~in_squeeze & in_squeeze.shift(1).fillna(False)
    s = pd.Series(0, index=df.index)
    s[released & (df['MACD_hist'] > 0)] = 1
    s[released & (df['MACD_hist'] < 0)] = -1
    return s

def sig_trix(df):            return _cross(df['TRIX'], df['TRIX_sig'])
def sig_obv_ema(df):         return _cross(df['OBV'], df['OBV_EMA'])
def sig_vwap_cross(df):      return _cross(df['Close'], df['VWAP'])

# --- PRICE ACTION ---

def sig_engulfing(df):
    po = df['Open'].shift(1); pc = df['Close'].shift(1)
    co = df['Open']; cc = df['Close']
    p_body = (pc - po).abs(); c_body = (cc - co).abs()
    s = pd.Series(0, index=df.index)
    bull = (pc < po) & (cc > co) & (c_body > p_body) & (co <= pc) & (cc >= po)
    bear = (pc > po) & (cc < co) & (c_body > p_body) & (co >= pc) & (cc <= po)
    s[bull] = 1; s[bear] = -1
    return s

def sig_inside_bar(df):
    s = pd.Series(0, index=df.index)
    is_inside = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    mother_hi = df['High'].shift(1); mother_lo = df['Low'].shift(1)
    break_up = is_inside.shift(1).fillna(False) & (df['Close'] > mother_hi.shift(1))
    break_dn = is_inside.shift(1).fillna(False) & (df['Close'] < mother_lo.shift(1))
    s[break_up] = 1; s[break_dn] = -1
    return s

def sig_pin_bar(df):
    s = pd.Series(0, index=df.index)
    body = (df['Close'] - df['Open']).abs()
    rng = df['High'] - df['Low']
    lo_wick = df[['Open','Close']].min(axis=1) - df['Low']
    hi_wick = df['High'] - df[['Open','Close']].max(axis=1)
    valid = rng > 0
    hammer = valid & (lo_wick >= 2*body) & (lo_wick >= 0.6*rng)
    star   = valid & (hi_wick >= 2*body) & (hi_wick >= 0.6*rng)
    s[hammer] = 1; s[star] = -1
    return s

def sig_outside_bar(df):
    s = pd.Series(0, index=df.index)
    outside = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))
    bull_close = df['Close'] > df['Open']
    s[outside & bull_close] = 1
    s[outside & ~bull_close] = -1
    return s

def sig_three_candle(df):
    s = pd.Series(0, index=df.index)
    c = df['Close']; o = df['Open']
    bull = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) & \
           (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    bear = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) & \
           (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    s[bull] = 1; s[bear] = -1
    return s

# --- SUPPORT / RESISTANCE ---

def sig_pivot_points(df):
    c = df['Close']
    s = pd.Series(0, index=df.index)
    s[(c > df['R1']) & (c.shift(1) <= df['R1'].shift(1))] = 1
    s[(c < df['S1']) & (c.shift(1) >= df['S1'].shift(1))] = -1
    return s

# --- MULTI-INDICATOR COMBOS ---

def sig_ma_ribbon(df):
    s = pd.Series(0, index=df.index)
    c = df['Close']; e20 = df['EMA20']; e50 = df['EMA50']
    s[(c > e20) & (c.shift(1) <= e20.shift(1)) & (e20 > e50)] = 1
    s[(c < e20) & (c.shift(1) >= e20.shift(1)) & (e20 < e50)] = -1
    return s

def sig_golden_zone(df):
    s = pd.Series(0, index=df.index)
    r = df['RSI14']; e50 = df['EMA50']; e200 = df['EMA200']
    c = df['Close']
    up_trend = e50 > e200
    s[up_trend & (r >= 38) & (r <= 50) & (c > e50*0.995) & (c > c.shift(1))] = 1
    s[~up_trend & (r >= 50) & (r <= 62) & (c < e50*1.005) & (c < c.shift(1))] = -1
    return s

def sig_rsi_bb_combo(df):
    s = pd.Series(0, index=df.index)
    r = df['RSI14']; c = df['Close']
    s[(r < 35) & (c <= df['BB_lo'])] = 1
    s[(r > 65) & (c >= df['BB_up'])] = -1
    return s

def sig_macd_rsi_combo(df):
    macd_up = (df['MACD'] > df['MACD_sig']) & (df['MACD'].shift(1) <= df['MACD_sig'].shift(1))
    macd_dn = (df['MACD'] < df['MACD_sig']) & (df['MACD'].shift(1) >= df['MACD_sig'].shift(1))
    s = pd.Series(0, index=df.index)
    s[macd_up & (df['RSI14'] > 40)] = 1
    s[macd_dn & (df['RSI14'] < 60)] = -1
    return s

def sig_adx_macd_combo(df):
    s = pd.Series(0, index=df.index)
    trending = df['ADX'] > 25
    macd_up = (df['MACD'] > df['MACD_sig']) & (df['MACD'].shift(1) <= df['MACD_sig'].shift(1))
    macd_dn = (df['MACD'] < df['MACD_sig']) & (df['MACD'].shift(1) >= df['MACD_sig'].shift(1))
    s[trending & macd_up] = 1
    s[trending & macd_dn] = -1
    return s

def sig_ema_triple_align(df):
    e9 = df['EMA9']; e20 = df['EMA20']; e50 = df['EMA50']
    bull_now = (e9 > e20) & (e20 > e50)
    bull_prev = (e9.shift(1) > e20.shift(1)) & (e20.shift(1) > e50.shift(1))
    bear_now = (e9 < e20) & (e20 < e50)
    bear_prev = (e9.shift(1) < e20.shift(1)) & (e20.shift(1) < e50.shift(1))
    s = pd.Series(0, index=df.index)
    s[bull_now & ~bull_prev] = 1
    s[bear_now & ~bear_prev] = -1
    return s

# ============================================================
# STRATEGY REGISTRY
# ============================================================

STRATEGIES = {
    # Trend Following
    'EMA_9_21_Cross':        (sig_ema_9_21,          'Trend Following'),
    'EMA_20_50_Cross':       (sig_ema_20_50,          'Trend Following'),
    'EMA_50_200_GoldenX':    (sig_ema_50_200,         'Trend Following'),
    'SMA_20_50_Cross':       (sig_sma_20_50,          'Trend Following'),
    'HMA_20_50_Cross':       (sig_hma_cross,          'Trend Following'),
    'DEMA_20_50_Cross':      (sig_dema_cross,         'Trend Following'),
    # MACD
    'MACD_Signal_Cross':     (sig_macd_signal,        'MACD'),
    'MACD_Zero_Cross':       (sig_macd_zero,          'MACD'),
    'MACD_Histogram_Rev':    (sig_macd_histogram,     'MACD'),
    # RSI
    'RSI_30_70_MeanRev':     (sig_rsi_mean_rev,       'RSI'),
    'RSI_50_Cross':          (sig_rsi_50_cross,       'RSI'),
    'RSI_40_60_Momentum':    (sig_rsi_40_60,          'RSI'),
    'RSI_Divergence':        (sig_rsi_divergence,     'RSI'),
    # Oscillators
    'Stochastic_Cross':      (sig_stochastic,         'Oscillator'),
    'CCI_100_Break':         (sig_cci_breakout,       'Oscillator'),
    'CCI_Zero_Cross':        (sig_cci_zero,           'Oscillator'),
    'Williams_R':            (sig_williams_r,         'Oscillator'),
    'ROC_Zero_Cross':        (sig_roc_zero,           'Momentum'),
    'Momentum_Zero_Cross':   (sig_momentum_zero,      'Momentum'),
    # Volatility/Channel
    'BB_Breakout':           (sig_bb_breakout,        'Volatility'),
    'BB_Mean_Reversion':     (sig_bb_mean_rev,        'Volatility'),
    'Keltner_Breakout':      (sig_keltner_break,      'Volatility'),
    'Donchian_20':           (sig_donchian_20,        'Breakout'),
    'Donchian_50':           (sig_donchian_50,        'Breakout'),
    'ATR_Breakout':          (sig_atr_breakout,       'Breakout'),
    # Trend Strength
    'ADX_DI_Cross':          (sig_adx_di,             'Trend Strength'),
    'Parabolic_SAR':         (sig_parabolic_sar,      'Trend Strength'),
    'Supertrend':            (sig_supertrend,         'Trend Strength'),
    'Aroon_Cross':           (sig_aroon,              'Trend Strength'),
    # Complex
    'Ichimoku_TK_Cross':     (sig_ichimoku,           'Complex'),
    'Squeeze_Momentum':      (sig_squeeze_momentum,   'Complex'),
    'TRIX_Signal_Cross':     (sig_trix,               'Complex'),
    'OBV_EMA_Cross':         (sig_obv_ema,            'Volume'),
    'VWAP_Cross':            (sig_vwap_cross,         'Volume'),
    # Price Action
    'Engulfing_Pattern':     (sig_engulfing,          'Price Action'),
    'Inside_Bar_Breakout':   (sig_inside_bar,         'Price Action'),
    'Pin_Bar':               (sig_pin_bar,            'Price Action'),
    'Outside_Bar':           (sig_outside_bar,        'Price Action'),
    'Three_Candle_Combo':    (sig_three_candle,       'Price Action'),
    # S/R
    'Pivot_Points':          (sig_pivot_points,       'Support/Resistance'),
    # Multi-indicator
    'MA_Ribbon_Pullback':    (sig_ma_ribbon,          'Multi-Indicator'),
    'Golden_Zone_Pullback':  (sig_golden_zone,        'Multi-Indicator'),
    'RSI_BB_Combo':          (sig_rsi_bb_combo,       'Multi-Indicator'),
    'MACD_RSI_Combo':        (sig_macd_rsi_combo,     'Multi-Indicator'),
    'ADX_MACD_Combo':        (sig_adx_macd_combo,     'Multi-Indicator'),
    'EMA_Triple_Alignment':  (sig_ema_triple_align,   'Multi-Indicator'),
}

# ============================================================
# BACKTESTING ENGINE
# ============================================================

def backtest(df, signals, sl_m=SL_ATR_MULT, tp_m=TP_ATR_MULT, max_hold=MAX_HOLD_BARS):
    atr_v  = df['ATR14'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values
    sigs   = signals.fillna(0).values

    trades = []
    in_trade = False
    entry_px = sl = tp = 0.0
    entry_i = direction = 0

    for i in range(len(df)):
        if in_trade:
            if direction == 1:
                if lows[i] <= sl:
                    trades.append((sl - entry_px, 0)); in_trade = False
                elif highs[i] >= tp:
                    trades.append((tp - entry_px, 1)); in_trade = False
                elif (i - entry_i) >= max_hold:
                    p = closes[i] - entry_px
                    trades.append((p, int(p > 0))); in_trade = False
            else:
                if highs[i] >= sl:
                    trades.append((entry_px - sl, 0)); in_trade = False
                elif lows[i] <= tp:
                    trades.append((entry_px - tp, 1)); in_trade = False
                elif (i - entry_i) >= max_hold:
                    p = entry_px - closes[i]
                    trades.append((p, int(p > 0))); in_trade = False

        if not in_trade and i > 0:
            s = sigs[i]
            if s != 0 and not np.isnan(s):
                av = atr_v[i]
                if np.isnan(av) or av <= 0:
                    continue
                in_trade = True
                direction = int(s)
                entry_px = closes[i]
                entry_i = i
                if direction == 1:
                    sl = entry_px - sl_m * av
                    tp = entry_px + tp_m * av
                else:
                    sl = entry_px + sl_m * av
                    tp = entry_px - tp_m * av

    if len(trades) < MIN_TRADES:
        return None

    pnls  = np.array([t[0] for t in trades])
    wins  = np.array([t[1] for t in trades])

    win_rate = wins.mean() * 100
    total_r  = pnls.sum()

    pos = pnls[pnls > 0]; neg = pnls[pnls < 0]
    pf = abs(pos.sum()) / abs(neg.sum()) if neg.size > 0 and neg.sum() != 0 else 9.99

    cum = np.cumsum(pnls)
    roll_max = np.maximum.accumulate(cum)
    max_dd = abs((cum - roll_max).min())

    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0
    avg_price = closes.mean()
    ret_pct = (total_r / avg_price) * 100

    return {
        'total_trades': len(trades),
        'wins':         int(wins.sum()),
        'losses':       int((1-wins).sum()),
        'win_rate':     round(win_rate, 2),
        'profit_factor':round(min(pf, 9.99), 3),
        'total_return': round(total_r, 2),
        'return_pct':   round(ret_pct, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe':       round(sharpe, 3),
        'avg_win':      round(pos.mean(), 2) if pos.size > 0 else 0,
        'avg_loss':     round(neg.mean(), 2) if neg.size > 0 else 0,
    }

# ============================================================
# MAIN
# ============================================================

def run():
    print("=" * 72)
    print("  XAUUSD COMPREHENSIVE STRATEGY BACKTEST")
    print(f"  SL={SL_ATR_MULT}×ATR  TP={TP_ATR_MULT}×ATR  MaxHold={MAX_HOLD_BARS} bars  RR=1:{TP_ATR_MULT/SL_ATR_MULT:.0f}")
    print(f"  {len(STRATEGIES)} strategies × 4 timeframes  (30m / 1h / 4h / 1d)")
    print("=" * 72)

    all_results = []

    for tf_name in TIMEFRAMES:
        print(f"\n{'─'*60}")
        print(f"  TIMEFRAME: {tf_name.upper()}")
        print(f"{'─'*60}")

        df_raw = fetch_data(tf_name)
        if df_raw is None or len(df_raw) < 150:
            print(f"  Skipping {tf_name}: insufficient data"); continue

        df = compute_indicators(df_raw)
        if len(df) < 100:
            print(f"  Skipping {tf_name}: too few rows after indicator warmup"); continue

        tf_res = []
        for name, (fn, cat) in STRATEGIES.items():
            try:
                signals = fn(df)
                result  = backtest(df, signals)
                if result:
                    result['strategy']  = name
                    result['category']  = cat
                    result['timeframe'] = tf_name
                    tf_res.append(result)
            except Exception:
                pass  # skip silently

        if not tf_res:
            print(f"  No valid strategies for {tf_name}"); continue

        tf_res.sort(key=lambda x: (x['win_rate'], x['profit_factor']), reverse=True)

        print(f"\n  {'#':<3} {'Strategy':<28} {'Cat':<18} {'Trd':>5} {'Win%':>7} {'PF':>6} {'Ret%':>7} {'Sharpe':>7}")
        print(f"  {'─'*82}")
        for rank, r in enumerate(tf_res[:12], 1):
            print(f"  {rank:<3} {r['strategy']:<28} {r['category']:<18} "
                  f"{r['total_trades']:>5} {r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} "
                  f"{r['return_pct']:>6.1f}% {r['sharpe']:>7.2f}")

        all_results.extend(tf_res)

    if not all_results:
        print("\nNo results generated – check network access to yfinance."); return

    # ── COMPOSITE SCORING ──────────────────────────────────────────────────
    rdf = pd.DataFrame(all_results)

    def _norm(col, cap=None):
        s = rdf[col].clip(upper=cap) if cap else rdf[col]
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    rdf['WR_n']  = _norm('win_rate')
    rdf['PF_n']  = _norm('profit_factor', cap=8)
    rdf['SH_n']  = _norm('sharpe', cap=None)
    dd_inv       = 1 - _norm('max_drawdown')
    rdf['DD_n']  = dd_inv

    rdf['score'] = (0.35 * rdf['WR_n'] + 0.30 * rdf['PF_n'] +
                    0.25 * rdf['SH_n'] + 0.10 * rdf['DD_n'])

    rdf = rdf.sort_values('score', ascending=False).reset_index(drop=True)

    # ── TOP-20 GLOBAL RANKINGS ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GLOBAL RANKINGS  (all strategies × all timeframes)")
    print("=" * 72)
    print(f"\n  {'#':<4} {'Strategy':<28} {'TF':<5} {'Cat':<18} {'Trd':>5} "
          f"{'Win%':>7} {'PF':>6} {'Ret%':>7} {'Sharpe':>7} {'Score':>7}")
    print(f"  {'─'*95}")
    for i, row in rdf.head(20).iterrows():
        print(f"  {i+1:<4} {row['strategy']:<28} {row['timeframe']:<5} {row['category']:<18} "
              f"{row['total_trades']:>5} {row['win_rate']:>6.1f}% {row['profit_factor']:>6.2f} "
              f"{row['return_pct']:>6.1f}% {row['sharpe']:>7.2f} {row['score']:>7.3f}")

    # ── CHAMPION ───────────────────────────────────────────────────────────
    best = rdf.iloc[0]
    print(f"\n{'═'*72}")
    print(f"  🏆  CHAMPION STRATEGY")
    print(f"{'═'*72}")
    print(f"  Strategy     : {best['strategy']}")
    print(f"  Category     : {best['category']}")
    print(f"  Timeframe    : {best['timeframe']}")
    print(f"  Total Trades : {best['total_trades']}")
    print(f"  Win Rate     : {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.3f}")
    print(f"  Total Return : {best['return_pct']:.1f}%")
    print(f"  Sharpe Ratio : {best['sharpe']:.3f}")
    print(f"  Max Drawdown : {best['max_drawdown']:.1f} USD/unit")
    print(f"  Comp Score   : {best['score']:.4f}")

    # ── BEST PER TIMEFRAME ─────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  BEST STRATEGY PER TIMEFRAME")
    print(f"{'═'*72}")
    summary_tf = {}
    for tf in TIMEFRAMES:
        sub = rdf[rdf['timeframe'] == tf]
        if sub.empty: continue
        b = sub.iloc[0]
        print(f"\n  {tf.upper():>4}  ►  {b['strategy']}")
        print(f"         Win Rate: {b['win_rate']:.1f}%  |  PF: {b['profit_factor']:.2f}  "
              f"|  Return: {b['return_pct']:.1f}%  |  Sharpe: {b['sharpe']:.2f}  "
              f"|  Trades: {b['total_trades']}")
        summary_tf[tf] = b[['strategy','category','win_rate','profit_factor',
                              'return_pct','sharpe','total_trades','score']].to_dict()

    # ── BEST PER CATEGORY ──────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  BEST STRATEGY PER CATEGORY")
    print(f"{'═'*72}")
    for cat in sorted(rdf['category'].unique()):
        sub = rdf[rdf['category'] == cat]
        if sub.empty: continue
        b = sub.iloc[0]
        print(f"  {cat:<22} ►  {b['strategy']} [{b['timeframe']}]  "
              f"WR={b['win_rate']:.1f}%  PF={b['profit_factor']:.2f}")

    # ── SAVE RESULTS ──────────────────────────────────────────────────────
    os.makedirs('data', exist_ok=True)

    csv_path  = 'data/xauusd_strategy_backtest_results.csv'
    json_path = 'data/xauusd_strategy_backtest_summary.json'

    rdf.drop(columns=['WR_n','PF_n','SH_n','DD_n'], errors='ignore').to_csv(csv_path, index=False)

    summary = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'ticker': TICKER,
            'sl_atr_mult': SL_ATR_MULT,
            'tp_atr_mult': TP_ATR_MULT,
            'max_hold_bars': MAX_HOLD_BARS,
            'risk_reward': f"1:{TP_ATR_MULT/SL_ATR_MULT:.1f}",
        },
        'champion': best[['strategy','category','timeframe','total_trades',
                           'win_rate','profit_factor','return_pct',
                           'sharpe','max_drawdown','score']].to_dict(),
        'best_per_timeframe': summary_tf,
        'top_20': rdf.head(20)[['strategy','timeframe','category','total_trades',
                                  'win_rate','profit_factor','return_pct',
                                  'sharpe','score']].to_dict(orient='records'),
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Full results  → {csv_path}")
    print(f"  Summary JSON  → {json_path}")
    print(f"\n  Total combos tested : {len(rdf)}")
    print(f"  Data source         : {_DATA_SOURCE}")
    if _DATA_SOURCE == 'SYNTHETIC':
        print(f"  NOTE: Synthetic data mirrors gold's real stats (8% ann.return,")
        print(f"        16% vol, regime-switching, fat tails). Re-run with network")
        print(f"        access to yfinance for live XAUUSD historical data.")
    print(f"{'═'*72}\n")

    return rdf


if __name__ == '__main__':
    run()
