"""
analysis/indicators.py — 14 technical indicators for TradingBotV2.
Ported from V1 indicators.py. Every function takes a OHLCV DataFrame
and returns a dict with at minimum a "bias" key.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

GST = timezone(timedelta(hours=4))


# ── 1. Williams Alligator ─────────────────────────────────────────────────────

def get_alligator(df: pd.DataFrame) -> dict:
    def smma(series: pd.Series, period: int) -> pd.Series:
        sma = series.rolling(period).mean()
        out = sma.copy()
        for i in range(period, len(series)):
            out.iloc[i] = (out.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
        return out

    jaw   = smma(df["close"], 13).shift(8)
    teeth = smma(df["close"],  8).shift(5)
    lips  = smma(df["close"],  5).shift(3)

    jaw_v, teeth_v, lips_v = float(jaw.iloc[-1]), float(teeth.iloc[-1]), float(lips.iloc[-1])
    spread     = abs(jaw_v - lips_v)
    avg_spread = float(abs(jaw - lips).mean())

    if   spread < avg_spread * 0.3:                      state, bias = "SLEEPING", "neutral"
    elif lips_v > teeth_v > jaw_v:                       state, bias = "EATING_BULLISH", "bullish"
    elif lips_v < teeth_v < jaw_v:                       state, bias = "EATING_BEARISH", "bearish"
    elif spread > avg_spread * 0.7:                      state, bias = "WAKING", ("bullish" if lips_v > jaw_v else "bearish")
    else:                                                state, bias = "RESTING", "neutral"

    return {"jaw": round(jaw_v,2), "teeth": round(teeth_v,2), "lips": round(lips_v,2),
            "state": state, "bias": bias, "sleeping": state == "SLEEPING"}


# ── 2. ADX ────────────────────────────────────────────────────────────────────

def get_adx(df: pd.DataFrame, period: int = 14) -> dict:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    dmp = (high - high.shift()).clip(lower=0)
    dmm = (low.shift() - low).clip(lower=0)
    dmp = dmp.where(dmp > dmm, 0)
    dmm = dmm.where(dmm > dmp, 0)
    atr14   = tr.rolling(period).mean()
    di_plus = 100 * dmp.rolling(period).mean() / atr14
    di_minus= 100 * dmm.rolling(period).mean() / atr14
    dx      = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx     = dx.rolling(period).mean()

    a, dip, dim = float(adx.iloc[-1]), float(di_plus.iloc[-1]), float(di_minus.iloc[-1])
    strength = "VERY_STRONG" if a > 40 else "STRONG" if a > 25 else "MODERATE" if a > 20 else "WEAK"
    return {"adx": round(a,1), "di_plus": round(dip,1), "di_minus": round(dim,1),
            "strength": strength, "trending": a > 20,
            "bias": "bullish" if dip > dim else "bearish"}


# ── 3. MACD ───────────────────────────────────────────────────────────────────

def get_macd(df: pd.DataFrame) -> dict:
    ema12  = df["close"].ewm(span=12).mean()
    ema26  = df["close"].ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist   = macd - signal

    m, s, h, hp = float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2])
    bc = macd.iloc[-2] < signal.iloc[-2] and m > s
    sc = macd.iloc[-2] > signal.iloc[-2] and m < s

    if   m > s and h > hp: bias = "strongly_bullish"
    elif m > s:            bias = "bullish"
    elif m < s and h < hp: bias = "strongly_bearish"
    else:                  bias = "bearish"

    return {"macd": round(m,3), "signal": round(s,3), "histogram": round(h,3),
            "bullish_cross": bc, "bearish_cross": sc, "histogram_growing": h > hp, "bias": bias}


# ── 4. Stochastic RSI ─────────────────────────────────────────────────────────

def get_stoch_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min() + 1e-10)
    k, d  = stoch.rolling(3).mean() * 100, stoch.rolling(3).mean().rolling(3).mean() * 100

    kv, dv = float(k.iloc[-1]), float(d.iloc[-1])
    ov, ob = kv < 20, kv > 80
    bx     = k.iloc[-2] < d.iloc[-2] and kv > dv
    sx     = k.iloc[-2] > d.iloc[-2] and kv < dv

    if   ov and bx: bias = "strongly_bullish"
    elif ov:        bias = "bullish"
    elif ob and sx: bias = "strongly_bearish"
    elif ob:        bias = "bearish"
    else:           bias = "neutral"

    return {"k": round(kv,1), "d": round(dv,1), "oversold": ov, "overbought": ob,
            "bullish_cross": bx, "bearish_cross": sx, "bias": bias}


# ── 5. Ichimoku ───────────────────────────────────────────────────────────────

def get_ichimoku(df: pd.DataFrame) -> dict:
    def midpoint(n: int) -> pd.Series:
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2

    tenkan = midpoint(9)
    kijun  = midpoint(26)
    spanA  = ((tenkan + kijun) / 2).shift(26)
    spanB  = midpoint(52).shift(26)

    price = float(df["close"].iloc[-1])
    tv, kv, sAv, sBv = float(tenkan.iloc[-1]), float(kijun.iloc[-1]), float(spanA.iloc[-1]), float(spanB.iloc[-1])
    top, bot = max(sAv, sBv), min(sAv, sBv)

    above, below = price > top, price < bot

    if   above and tv > kv and sAv > sBv: bias = "strongly_bullish"
    elif above:                            bias = "bullish"
    elif below and tv < kv and sAv < sBv: bias = "strongly_bearish"
    elif below:                            bias = "bearish"
    else:                                  bias = "neutral"

    return {"tenkan": round(tv,2), "kijun": round(kv,2), "span_a": round(sAv,2),
            "span_b": round(sBv,2), "cloud_top": round(top,2), "cloud_bot": round(bot,2),
            "above_cloud": above, "below_cloud": below, "in_cloud": bot <= price <= top, "bias": bias}


# ── 6. VWAP ───────────────────────────────────────────────────────────────────

def get_vwap(df: pd.DataFrame) -> dict:
    typical  = (df["high"] + df["low"] + df["close"]) / 3
    vwap     = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    vv, price = float(vwap.iloc[-1]), float(df["close"].iloc[-1])
    above    = price > vv
    dist_pct = round(abs(price - vv) / vv * 100, 3) if vv else 0.0

    bias = "bullish" if above else "bearish"

    return {"vwap": round(vv,2), "price": round(price,2), "above": above,
            "distance": round(price - vv, 2), "dist_pct": dist_pct, "bias": bias}


# ── 7. Bollinger / Keltner Squeeze ────────────────────────────────────────────

def get_squeeze(df: pd.DataFrame) -> dict:
    p = 20
    basis = df["close"].rolling(p).mean()
    std   = df["close"].rolling(p).std()
    bb_up, bb_dn = basis + 2.0 * std, basis - 2.0 * std

    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"]  - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(p).mean()
    kc_up = basis + 1.5 * atr
    kc_dn = basis - 1.5 * atr

    sq_on  = bb_up.iloc[-1] < kc_up.iloc[-1] and bb_dn.iloc[-1] > kc_dn.iloc[-1]
    sq_off = bb_up.iloc[-2] < kc_up.iloc[-2] and bb_up.iloc[-1] >= kc_up.iloc[-1]
    mom    = float(df["close"].iloc[-1] - basis.iloc[-1])

    bias = "squeeze_building" if sq_on else ("bullish" if mom > 0 else "bearish")
    return {"squeeze_on": sq_on, "squeeze_off": sq_off, "momentum": round(mom,2),
            "bb_width": round(float(bb_up.iloc[-1] - bb_dn.iloc[-1]),2),
            "kc_width": round(float(kc_up.iloc[-1] - kc_dn.iloc[-1]),2), "bias": bias}


# ── 8. Supertrend ─────────────────────────────────────────────────────────────

def get_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict:
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"]  - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (df["high"] + df["low"]) / 2
    up  = hl2 - mult * atr
    dn  = hl2 + mult * atr

    st    = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > dn.iloc[i - 1]:
            trend.iloc[i], st.iloc[i] = 1, float(up.iloc[i])
        elif df["close"].iloc[i] < up.iloc[i - 1]:
            trend.iloc[i], st.iloc[i] = -1, float(dn.iloc[i])
        else:
            t_prev = int(trend.iloc[i - 1]) if not pd.isna(trend.iloc[i - 1]) else 1
            trend.iloc[i] = t_prev
            s_prev = float(st.iloc[i - 1]) if not pd.isna(st.iloc[i - 1]) else float(up.iloc[i])
            st.iloc[i] = max(float(up.iloc[i]), s_prev) if t_prev == 1 else min(float(dn.iloc[i]), s_prev)

    tn = int(trend.iloc[-1]) if not pd.isna(trend.iloc[-1]) else 1
    tp = int(trend.iloc[-2]) if not pd.isna(trend.iloc[-2]) else tn
    sn = float(st.iloc[-1])  if not pd.isna(st.iloc[-1])  else 0.0

    return {"supertrend": round(sn,2), "trend": "bullish" if tn == 1 else "bearish",
            "just_flipped": tn != tp, "sl_level": round(sn,2),
            "bias": "bullish" if tn == 1 else "bearish"}


# ── 9. KAMA ───────────────────────────────────────────────────────────────────

def get_kama(df: pd.DataFrame, period: int = 10) -> dict:
    close  = df["close"]
    change = (close - close.shift(period)).abs()
    volat  = close.diff().abs().rolling(period).sum()
    er     = change / volat.replace(0, 1e-10)
    sc     = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2

    kama = close.copy().astype(float)
    for i in range(1, len(close)):
        kama.iloc[i] = float(kama.iloc[i-1]) + float(sc.iloc[i]) * (float(close.iloc[i]) - float(kama.iloc[i-1]))

    kv, kp, price = float(kama.iloc[-1]), float(kama.iloc[-5]), float(close.iloc[-1])
    slope, above = kv - kp, price > kv

    if   slope > 0.5 and above:     bias = "strongly_bullish"
    elif above:                      bias = "bullish"
    elif slope < -0.5 and not above: bias = "strongly_bearish"
    else:                            bias = "bearish"

    return {"kama": round(kv,2), "slope": round(slope,2), "above": above,
            "bias": bias, "trending": abs(slope) > 0.3}


# ── 10. ICT Kill Zones ────────────────────────────────────────────────────────

def get_ict_killzones() -> dict:
    now = datetime.now(GST)
    t   = now.hour + now.minute / 60

    kz = {
        "Asian KZ":             (4.0,  6.0),
        "London KZ":            (12.0, 13.0),
        "NY AM KZ":             (17.5, 19.0),
        "NY PM KZ":             (20.0, 21.0),
        "Silver Bullet London": (14.0, 15.0),
        "Silver Bullet NY AM":  (18.0, 19.0),
    }
    active = [n for n, (s, e) in kz.items() if s <= t < e]
    next_kz, next_mins = None, 999.0
    for n, (s, _) in kz.items():
        if s > t and (s - t) * 60 < next_mins:
            next_mins = (s - t) * 60
            next_kz   = (n, int(next_mins))

    high_quality = any("London" in z or "NY AM" in z for z in active)
    return {"in_killzone": bool(active), "active_zones": active, "high_quality": high_quality,
            "next_killzone": next_kz,
            "bias": "high_probability" if high_quality else "active" if active else "normal"}


# ── 11. Wyckoff Phase ─────────────────────────────────────────────────────────

def get_wyckoff_phase(df: pd.DataFrame) -> dict:
    c, v = df["close"], df["volume"]
    st = float(c.iloc[-5:].mean() - c.iloc[-10:-5].mean())
    mt = float(c.iloc[-10:].mean() - c.iloc[-20:-10].mean())
    vr = float(v.iloc[-10:].mean()) / max(float(v.iloc[-20:-10].mean()), 1)
    rr = float((df["high"].iloc[-10:] - df["low"].iloc[-10:]).mean()) / \
         max(float((df["high"].iloc[-20:-10] - df["low"].iloc[-20:-10]).mean()), 1)

    if   st < 0 and vr > 1.2 and rr > 1.1: phase, bias = "MARKDOWN",     "bearish"
    elif st > 0 and vr > 1.2:               phase, bias = "MARKUP",       "bullish"
    elif abs(st) < 2 and vr < 0.8:          phase, bias = ("ACCUMULATION" if mt < 0 else "DISTRIBUTION"), ("bullish" if mt < 0 else "bearish")
    elif rr < 0.7:                          phase, bias = "SPRING_TEST",  "bullish"
    else:                                   phase, bias = "UNKNOWN",      "neutral"

    return {"phase": phase, "bias": bias, "vol_ratio": round(vr,2), "note": f"{phase} phase"}


# ── 12. Real Rate Model ───────────────────────────────────────────────────────

def get_real_rate_model() -> dict:
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX").history(period="5d")
        cl  = yf.Ticker("CL=F").history(period="5d")
        yield_10y = float(tnx["Close"].iloc[-1]) if not tnx.empty else 4.5
        oil       = float(cl["Close"].iloc[-1])  if not cl.empty  else 80.0
        infl_est  = 3.0 if oil > 90 else 2.5 if oil > 70 else 2.0
        real_rate = yield_10y - infl_est

        if   real_rate < 0: bias, note = "strongly_bullish", f"Real rate {real_rate:.1f}% negative → gold bullish"
        elif real_rate < 1: bias, note = "bullish",           f"Real rate {real_rate:.1f}% low → mild support"
        elif real_rate < 2: bias, note = "neutral",           f"Real rate {real_rate:.1f}% neutral"
        else:               bias, note = "bearish",           f"Real rate {real_rate:.1f}% high → headwind"

        return {"yield_10y": round(yield_10y,2), "inflation_est": round(infl_est,1),
                "real_rate": round(real_rate,2), "oil_price": round(oil,1),
                "bias": bias, "note": note, "available": True}
    except Exception:
        return {"available": False, "bias": "neutral", "real_rate": 0, "note": "unavailable"}


# ── 13. Market Cipher B Replica ───────────────────────────────────────────────

def get_market_cipher(df: pd.DataFrame) -> dict:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa  = hlc3.ewm(span=10).mean()
    d    = (hlc3 - esa).abs().ewm(span=10).mean()
    ci   = (hlc3 - esa) / (0.015 * d)
    wt1  = ci.ewm(span=21).mean()
    wt2  = wt1.rolling(4).mean()

    wu = wt1.iloc[-2] < wt2.iloc[-2] and wt1.iloc[-1] > wt2.iloc[-1]
    wd = wt1.iloc[-2] > wt2.iloc[-2] and wt1.iloc[-1] < wt2.iloc[-1]
    mf = (hlc3 - hlc3.ewm(span=3).mean()).ewm(span=3).mean()

    w1, w2v, mfv = float(wt1.iloc[-1]), float(wt2.iloc[-1]), float(mf.iloc[-1])
    ov, ob = w1 < -53, w1 > 53

    if   wu and ov:                   bias = "strongly_bullish"
    elif wu or (w1 > w2v and mfv > 0): bias = "bullish"
    elif wd and ob:                    bias = "strongly_bearish"
    elif wd or (w1 < w2v and mfv < 0): bias = "bearish"
    else:                              bias = "neutral"

    return {"wt1": round(w1,1), "wt2": round(w2v,1), "money_flow": round(mfv,3),
            "oversold": ov, "overbought": ob,
            "bullish_cross": wu, "bearish_cross": wd, "bias": bias}


# ── 14. OBV ───────────────────────────────────────────────────────────────────

def get_obv(df: pd.DataFrame) -> dict:
    sign      = (~df["close"].diff().le(0)) * 2 - 1
    obv       = (df["volume"] * sign).cumsum()
    obv_ema   = obv.ewm(span=20).mean()
    ov, oev   = float(obv.iloc[-1]), float(obv_ema.iloc[-1])
    obv_trend = ov - float(obv.iloc[-10])
    px_trend  = float(df["close"].iloc[-1]) - float(df["close"].iloc[-10])

    if   px_trend < 0 and obv_trend > 0: bias, div = "bullish", "bullish_divergence"
    elif px_trend > 0 and obv_trend < 0: bias, div = "bearish", "bearish_divergence"
    elif ov > oev:                       bias, div = "bullish", None
    else:                                bias, div = "bearish", None

    return {"obv": round(ov,0), "obv_ema": round(oev,0), "obv_trend": round(obv_trend,0),
            "divergence": div, "bias": bias}


# ── Master function ───────────────────────────────────────────────────────────

def get_all_indicators(df: pd.DataFrame) -> dict:
    """Run all 14 indicators. Each result has a 'bias' key."""
    results: dict = {}
    tasks = [
        ("alligator",     lambda: get_alligator(df)),
        ("adx",           lambda: get_adx(df)),
        ("macd",          lambda: get_macd(df)),
        ("stoch_rsi",     lambda: get_stoch_rsi(df)),
        ("ichimoku",      lambda: get_ichimoku(df)),
        ("vwap",          lambda: get_vwap(df)),
        ("squeeze",       lambda: get_squeeze(df)),
        ("supertrend",    lambda: get_supertrend(df)),
        ("kama",          lambda: get_kama(df)),
        ("killzones",     get_ict_killzones),
        ("wyckoff",       lambda: get_wyckoff_phase(df)),
        ("real_rate",     get_real_rate_model),
        ("market_cipher", lambda: get_market_cipher(df)),
        ("obv",           lambda: get_obv(df)),
    ]
    for name, func in tasks:
        try:
            results[name] = func()
        except Exception as exc:
            logger.debug("Indicator %s failed: %s", name, exc)
            results[name] = {"bias": "neutral", "error": str(exc)}
    return results
