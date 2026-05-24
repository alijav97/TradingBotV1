"""
indicators.py — 14 technical indicator calculations for TradingBotV1
All functions take df (OHLCV DataFrame) and return a dict with a "bias" key.
"""
from __future__ import annotations

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 1: Williams Alligator
# ─────────────────────────────────────────────────────────────────────────────
def get_alligator(df: pd.DataFrame) -> dict:
    def smma(series: pd.Series, period: int) -> pd.Series:
        sma = series.rolling(period).mean()
        smma_vals = sma.copy()
        for i in range(period, len(series)):
            smma_vals.iloc[i] = (smma_vals.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
        return smma_vals

    jaw   = smma(df["close"], 13).shift(8)
    teeth = smma(df["close"],  8).shift(5)
    lips  = smma(df["close"],  5).shift(3)

    jaw_now   = float(jaw.iloc[-1])
    teeth_now = float(teeth.iloc[-1])
    lips_now  = float(lips.iloc[-1])

    spread     = abs(jaw_now - lips_now)
    avg_spread = float(abs(jaw - lips).mean())

    if spread < avg_spread * 0.3:
        state = "SLEEPING"
        bias  = "neutral"
    elif lips_now > teeth_now > jaw_now:
        state = "EATING_BULLISH"
        bias  = "bullish"
    elif lips_now < teeth_now < jaw_now:
        state = "EATING_BEARISH"
        bias  = "bearish"
    elif spread > avg_spread * 0.7:
        state = "WAKING"
        bias  = "bullish" if lips_now > jaw_now else "bearish"
    else:
        state = "RESTING"
        bias  = "neutral"

    return {
        "jaw":     round(jaw_now, 2),
        "teeth":   round(teeth_now, 2),
        "lips":    round(lips_now, 2),
        "state":   state,
        "bias":    bias,
        "sleeping": state == "SLEEPING",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 2: ADX (Average Directional Index)
# ─────────────────────────────────────────────────────────────────────────────
def get_adx(df: pd.DataFrame, period: int = 14) -> dict:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr14    = tr.rolling(period).mean()
    di_plus  = 100 * dm_plus.rolling(period).mean()  / atr14
    di_minus = 100 * dm_minus.rolling(period).mean() / atr14
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx      = dx.rolling(period).mean()

    adx_val = float(adx.iloc[-1])
    dip_val = float(di_plus.iloc[-1])
    dim_val = float(di_minus.iloc[-1])

    if   adx_val > 40: strength = "VERY_STRONG"
    elif adx_val > 25: strength = "STRONG"
    elif adx_val > 20: strength = "MODERATE"
    else:              strength = "WEAK"

    return {
        "adx":      round(adx_val, 1),
        "di_plus":  round(dip_val, 1),
        "di_minus": round(dim_val, 1),
        "strength": strength,
        "trending": adx_val > 20,
        "bias":     "bullish" if dip_val > dim_val else "bearish",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 3: MACD
# ─────────────────────────────────────────────────────────────────────────────
def get_macd(df: pd.DataFrame) -> dict:
    ema12  = df["close"].ewm(span=12).mean()
    ema26  = df["close"].ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist   = macd - signal

    macd_now   = float(macd.iloc[-1])
    signal_now = float(signal.iloc[-1])
    hist_now   = float(hist.iloc[-1])
    hist_prev  = float(hist.iloc[-2])

    bullish_cross = macd.iloc[-2] < signal.iloc[-2] and macd_now > signal_now
    bearish_cross = macd.iloc[-2] > signal.iloc[-2] and macd_now < signal_now
    hist_growing  = hist_now > hist_prev

    if   macd_now > signal_now and hist_growing:     bias = "strongly_bullish"
    elif macd_now > signal_now:                       bias = "bullish"
    elif macd_now < signal_now and not hist_growing:  bias = "strongly_bearish"
    else:                                             bias = "bearish"

    return {
        "macd":              round(macd_now,   3),
        "signal":            round(signal_now, 3),
        "histogram":         round(hist_now,   3),
        "bullish_cross":     bullish_cross,
        "bearish_cross":     bearish_cross,
        "histogram_growing": hist_growing,
        "bias":              bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 4: Stochastic RSI
# ─────────────────────────────────────────────────────────────────────────────
def get_stoch_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-10)
    rsi   = 100 - (100 / (1 + rs))

    rsi_min   = rsi.rolling(period).min()
    rsi_max   = rsi.rolling(period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    k = stoch_rsi.rolling(3).mean() * 100
    d = k.rolling(3).mean()

    k_now = float(k.iloc[-1])
    d_now = float(d.iloc[-1])

    oversold   = k_now < 20
    overbought = k_now > 80
    bullish_x  = k.iloc[-2] < d.iloc[-2] and k_now > d_now
    bearish_x  = k.iloc[-2] > d.iloc[-2] and k_now < d_now

    if   oversold  and bullish_x: bias = "strongly_bullish"
    elif oversold:                 bias = "bullish"
    elif overbought and bearish_x: bias = "strongly_bearish"
    elif overbought:               bias = "bearish"
    else:                          bias = "neutral"

    return {
        "k":             round(k_now, 1),
        "d":             round(d_now, 1),
        "oversold":      oversold,
        "overbought":    overbought,
        "bullish_cross": bullish_x,
        "bearish_cross": bearish_x,
        "bias":          bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 5: Ichimoku Cloud
# ─────────────────────────────────────────────────────────────────────────────
def get_ichimoku(df: pd.DataFrame) -> dict:
    h9  = df["high"].rolling(9).max()
    l9  = df["low"].rolling(9).min()
    h26 = df["high"].rolling(26).max()
    l26 = df["low"].rolling(26).min()
    h52 = df["high"].rolling(52).max()
    l52 = df["low"].rolling(52).min()

    tenkan = (h9  + l9)  / 2
    kijun  = (h26 + l26) / 2
    spanA  = ((tenkan + kijun) / 2).shift(26)
    spanB  = ((h52 + l52) / 2).shift(26)

    price   = float(df["close"].iloc[-1])
    ten_now = float(tenkan.iloc[-1])
    kij_now = float(kijun.iloc[-1])
    spA_now = float(spanA.iloc[-1])
    spB_now = float(spanB.iloc[-1])

    cloud_top = max(spA_now, spB_now)
    cloud_bot = min(spA_now, spB_now)

    above_cloud = price > cloud_top
    below_cloud = price < cloud_bot
    in_cloud    = cloud_bot <= price <= cloud_top
    bullish_tk  = ten_now > kij_now
    cloud_bull  = spA_now > spB_now

    if   above_cloud and bullish_tk and cloud_bull:      bias = "strongly_bullish"
    elif above_cloud:                                     bias = "bullish"
    elif below_cloud and not bullish_tk and not cloud_bull: bias = "strongly_bearish"
    elif below_cloud:                                     bias = "bearish"
    else:                                                 bias = "neutral"

    return {
        "tenkan":      round(ten_now, 2),
        "kijun":       round(kij_now, 2),
        "span_a":      round(spA_now, 2),
        "span_b":      round(spB_now, 2),
        "cloud_top":   round(cloud_top, 2),
        "cloud_bot":   round(cloud_bot, 2),
        "above_cloud": above_cloud,
        "below_cloud": below_cloud,
        "in_cloud":    in_cloud,
        "bias":        bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 6: VWAP
# ─────────────────────────────────────────────────────────────────────────────
def get_vwap(df: pd.DataFrame) -> dict:
    typical  = (df["high"] + df["low"] + df["close"]) / 3
    vwap     = (typical * df["volume"]).cumsum() / df["volume"].cumsum()

    vwap_now = float(vwap.iloc[-1])
    price    = float(df["close"].iloc[-1])
    above    = price > vwap_now
    dist     = round(price - vwap_now, 2)
    dist_pct = round(abs(dist) / vwap_now * 100, 3) if vwap_now else 0.0

    if   above and dist_pct > 0.5:  bias = "bearish"   # too far above → mean-revert down
    elif above:                      bias = "bullish"
    elif not above and dist_pct > 0.5: bias = "bullish"  # too far below → mean-revert up
    else:                            bias = "bearish"

    return {
        "vwap":     round(vwap_now, 2),
        "price":    round(price, 2),
        "above":    above,
        "distance": dist,
        "dist_pct": dist_pct,
        "bias":     bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 7: Bollinger + Keltner Squeeze
# ─────────────────────────────────────────────────────────────────────────────
def get_squeeze(df: pd.DataFrame) -> dict:
    period   = 20
    mult_bb  = 2.0
    mult_kc  = 1.5

    basis  = df["close"].rolling(period).mean()
    std    = df["close"].rolling(period).std()
    bb_up  = basis + mult_bb * std
    bb_dn  = basis - mult_bb * std

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    kc_up = basis + mult_kc * atr
    kc_dn = basis - mult_kc * atr

    squeeze_on  = (bb_up.iloc[-1] < kc_up.iloc[-1] and
                   bb_dn.iloc[-1] > kc_dn.iloc[-1])
    squeeze_off = (bb_up.iloc[-2] < kc_up.iloc[-2] and
                   bb_up.iloc[-1] >= kc_up.iloc[-1])

    momentum = float(df["close"].iloc[-1] - basis.iloc[-1])

    if squeeze_on:
        bias = "squeeze_building"
    else:
        bias = "bullish" if momentum > 0 else "bearish"

    return {
        "squeeze_on":  squeeze_on,
        "squeeze_off": squeeze_off,
        "momentum":    round(momentum, 2),
        "bb_width":    round(float(bb_up.iloc[-1] - bb_dn.iloc[-1]), 2),
        "kc_width":    round(float(kc_up.iloc[-1] - kc_dn.iloc[-1]), 2),
        "bias":        bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 8: Supertrend
# ─────────────────────────────────────────────────────────────────────────────
def get_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (df["high"] + df["low"]) / 2
    up  = hl2 - mult * atr
    dn  = hl2 + mult * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    trend      = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > dn.iloc[i - 1]:
            trend.iloc[i]      = 1
            supertrend.iloc[i] = float(up.iloc[i])
        elif df["close"].iloc[i] < up.iloc[i - 1]:
            trend.iloc[i]      = -1
            supertrend.iloc[i] = float(dn.iloc[i])
        else:
            trend.iloc[i] = int(trend.iloc[i - 1]) if not pd.isna(trend.iloc[i - 1]) else 1
            prev_st = supertrend.iloc[i - 1]
            if trend.iloc[i] == 1:
                supertrend.iloc[i] = max(float(up.iloc[i]), float(prev_st) if not pd.isna(prev_st) else float(up.iloc[i]))
            else:
                supertrend.iloc[i] = min(float(dn.iloc[i]), float(prev_st) if not pd.isna(prev_st) else float(dn.iloc[i]))

    trend_now    = int(trend.iloc[-1]) if not pd.isna(trend.iloc[-1]) else 1
    trend_prev   = int(trend.iloc[-2]) if not pd.isna(trend.iloc[-2]) else trend_now
    st_now       = float(supertrend.iloc[-1]) if not pd.isna(supertrend.iloc[-1]) else 0.0
    just_flipped = trend_now != trend_prev

    return {
        "supertrend":   round(st_now, 2),
        "trend":        "bullish" if trend_now == 1 else "bearish",
        "just_flipped": just_flipped,
        "sl_level":     round(st_now, 2),
        "bias":         "bullish" if trend_now == 1 else "bearish",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 9: Adaptive EMA (KAMA — Kaufman Adaptive Moving Average)
# ─────────────────────────────────────────────────────────────────────────────
def get_kama(df: pd.DataFrame, period: int = 10) -> dict:
    close  = df["close"]
    change = (close - close.shift(period)).abs()
    volat  = close.diff().abs().rolling(period).sum()
    er     = change / volat.replace(0, 1e-10)
    fast   = 2 / (2  + 1)
    slow   = 2 / (30 + 1)
    sc     = (er * (fast - slow) + slow) ** 2

    kama = close.copy().astype(float)
    for i in range(1, len(close)):
        kama.iloc[i] = float(kama.iloc[i - 1]) + float(sc.iloc[i]) * (float(close.iloc[i]) - float(kama.iloc[i - 1]))

    kama_now  = float(kama.iloc[-1])
    kama_prev = float(kama.iloc[-5])
    price     = float(close.iloc[-1])
    slope     = kama_now - kama_prev
    above     = price > kama_now

    if   slope > 0.5 and above:        bias = "strongly_bullish"
    elif above:                         bias = "bullish"
    elif slope < -0.5 and not above:    bias = "strongly_bearish"
    else:                               bias = "bearish"

    return {
        "kama":     round(kama_now, 2),
        "slope":    round(slope, 2),
        "above":    above,
        "bias":     bias,
        "trending": abs(slope) > 0.3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 10: ICT Kill Zones
# ─────────────────────────────────────────────────────────────────────────────
def get_ict_killzones() -> dict:
    from datetime import datetime, timezone, timedelta
    GST          = timezone(timedelta(hours=4))
    now          = datetime.now(GST)
    time_decimal = now.hour + now.minute / 60

    killzones = {
        "Asian KZ":              (4.0,  6.0),
        "London KZ":             (12.0, 13.0),
        "NY AM KZ":              (17.5, 19.0),
        "NY PM KZ":              (20.0, 21.0),
        "Silver Bullet London":  (14.0, 15.0),
        "Silver Bullet NY AM":   (18.0, 19.0),
    }

    active_kz = [
        name for name, (start, end) in killzones.items()
        if start <= time_decimal < end
    ]

    next_kz = None
    next_mins = 999.0
    for name, (start, _) in killzones.items():
        if start > time_decimal:
            mins = (start - time_decimal) * 60
            if mins < next_mins:
                next_mins = mins
                next_kz   = (name, int(mins))

    in_killzone  = len(active_kz) > 0
    high_quality = any("London" in kz or "NY AM" in kz for kz in active_kz)

    return {
        "in_killzone":   in_killzone,
        "active_zones":  active_kz,
        "high_quality":  high_quality,
        "next_killzone": next_kz,
        "bias":          "high_probability" if high_quality else "active" if in_killzone else "normal",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 11: Wyckoff Phase Detection
# ─────────────────────────────────────────────────────────────────────────────
def get_wyckoff_phase(df: pd.DataFrame) -> dict:
    close  = df["close"]
    volume = df["volume"]

    short_trend = float(close.iloc[-5:].mean()    - close.iloc[-10:-5].mean())
    med_trend   = float(close.iloc[-10:].mean()   - close.iloc[-20:-10].mean())

    recent_vol = float(volume.iloc[-10:].mean())
    prev_vol   = float(volume.iloc[-20:-10].mean())
    vol_ratio  = recent_vol / prev_vol if prev_vol > 0 else 1.0

    recent_range = float((df["high"].iloc[-10:]   - df["low"].iloc[-10:]).mean())
    prev_range   = float((df["high"].iloc[-20:-10] - df["low"].iloc[-20:-10]).mean())
    range_ratio  = recent_range / prev_range if prev_range > 0 else 1.0

    if short_trend < 0 and vol_ratio > 1.2 and range_ratio > 1.1:
        phase = "MARKDOWN"
        bias  = "bearish"
        note  = "Distribution complete — markup ending"
    elif short_trend > 0 and vol_ratio > 1.2:
        phase = "MARKUP"
        bias  = "bullish"
        note  = "Accumulation complete — markup starting"
    elif abs(short_trend) < 2 and vol_ratio < 0.8:
        phase = "ACCUMULATION" if med_trend < 0 else "DISTRIBUTION"
        bias  = "bullish" if phase == "ACCUMULATION" else "bearish"
        note  = f"{phase} phase — sideways with low volume"
    elif range_ratio < 0.7:
        phase = "SPRING_TEST"
        bias  = "bullish"
        note  = "Tight range — spring/test likely"
    else:
        phase = "UNKNOWN"
        bias  = "neutral"
        note  = "No clear Wyckoff phase"

    return {
        "phase":     phase,
        "bias":      bias,
        "vol_ratio": round(vol_ratio, 2),
        "note":      note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 12: Real Interest Rate Model
# ─────────────────────────────────────────────────────────────────────────────
def get_real_rate_model() -> dict:
    try:
        import yfinance as yf
        tnx       = yf.Ticker("^TNX").history(period="5d")
        yield_10y = float(tnx["Close"].iloc[-1]) if not tnx.empty else 4.5
        cl        = yf.Ticker("CL=F").history(period="5d")
        oil       = float(cl["Close"].iloc[-1])  if not cl.empty  else 80.0

        inflation_est = 3.0 if oil > 90 else 2.5 if oil > 70 else 2.0
        real_rate     = yield_10y - inflation_est

        if   real_rate < 0: bias = "strongly_bullish"; note = f"Real rate {real_rate:.1f}% negative → gold bullish"
        elif real_rate < 1: bias = "bullish";           note = f"Real rate {real_rate:.1f}% low → mild gold support"
        elif real_rate < 2: bias = "neutral";           note = f"Real rate {real_rate:.1f}% neutral"
        else:               bias = "bearish";           note = f"Real rate {real_rate:.1f}% high → gold headwind"

        return {
            "yield_10y":     round(yield_10y, 2),
            "inflation_est": round(inflation_est, 1),
            "real_rate":     round(real_rate, 2),
            "oil_price":     round(oil, 1),
            "bias":          bias,
            "note":          note,
            "available":     True,
        }
    except Exception:
        return {"available": False, "bias": "neutral", "real_rate": 0, "note": "Unavailable"}


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 13: Market Cipher B Replica (Wave Trend Oscillator)
# ─────────────────────────────────────────────────────────────────────────────
def get_market_cipher(df: pd.DataFrame) -> dict:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa  = hlc3.ewm(span=10).mean()
    d    = (hlc3 - esa).abs().ewm(span=10).mean()
    ci   = (hlc3 - esa) / (0.015 * d)
    wt1  = ci.ewm(span=21).mean()
    wt2  = wt1.rolling(4).mean()

    wt_cross_up   = wt1.iloc[-2] < wt2.iloc[-2] and wt1.iloc[-1] > wt2.iloc[-1]
    wt_cross_down = wt1.iloc[-2] > wt2.iloc[-2] and wt1.iloc[-1] < wt2.iloc[-1]

    raw_mf = hlc3 - hlc3.ewm(span=3).mean()
    mf     = raw_mf.ewm(span=3).mean()

    wt1_now = float(wt1.iloc[-1])
    mf_now  = float(mf.iloc[-1])
    oversold   = wt1_now < -53
    overbought = wt1_now >  53

    if   wt_cross_up   and oversold:                              bias = "strongly_bullish"
    elif wt_cross_up   or (wt1_now > float(wt2.iloc[-1]) and mf_now > 0): bias = "bullish"
    elif wt_cross_down and overbought:                            bias = "strongly_bearish"
    elif wt_cross_down or (wt1_now < float(wt2.iloc[-1]) and mf_now < 0): bias = "bearish"
    else:                                                          bias = "neutral"

    return {
        "wt1":           round(wt1_now, 1),
        "wt2":           round(float(wt2.iloc[-1]), 1),
        "money_flow":    round(mf_now, 3),
        "oversold":      oversold,
        "overbought":    overbought,
        "bullish_cross": wt_cross_up,
        "bearish_cross": wt_cross_down,
        "bias":          bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 14: OBV (On Balance Volume)
# ─────────────────────────────────────────────────────────────────────────────
def get_obv(df: pd.DataFrame) -> dict:
    close = df["close"]
    vol   = df["volume"]
    sign  = (~close.diff().le(0)) * 2 - 1   # +1 if up, -1 if down/flat
    obv   = (vol * sign).cumsum()
    obv_ema = obv.ewm(span=20).mean()

    obv_now     = float(obv.iloc[-1])
    obv_ema_now = float(obv_ema.iloc[-1])
    obv_trend   = float(obv.iloc[-1] - obv.iloc[-10])
    price_trend = float(close.iloc[-1] - close.iloc[-10])

    if   price_trend < 0 and obv_trend > 0:
        bias       = "bullish"
        divergence = "bullish_divergence"
    elif price_trend > 0 and obv_trend < 0:
        bias       = "bearish"
        divergence = "bearish_divergence"
    elif obv_now > obv_ema_now:
        bias       = "bullish"
        divergence = None
    else:
        bias       = "bearish"
        divergence = None

    return {
        "obv":        round(obv_now, 0),
        "obv_ema":    round(obv_ema_now, 0),
        "obv_trend":  round(obv_trend, 0),
        "divergence": divergence,
        "bias":       bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
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
        except Exception as e:
            results[name] = {"bias": "neutral", "error": str(e)}
    return results
