"""
instrument_data.py — OHLCV + market context fetcher for TradingBotV1
Wraps yfinance for XAUUSD/NAS100/US30/GBPUSD/EURUSD/WTI with safe fallbacks.
"""
from __future__ import annotations
import datetime

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── Ticker mapping ────────────────────────────────────────────────────────────
YF_TICKERS: dict = {
    "XAUUSD": "GC=F",
    "NAS100": "NQ=F",
    "US30":   "YM=F",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "WTI":    "CL=F",
}

VIX_TICKER  = "^VIX"
DXY_TICKER  = "DX-Y.NYB"
US10Y_TICKER = "^TNX"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_last(ticker_obj, fallback: float = 0.0) -> float:
    """Return the most-recent closing price from a yfinance Ticker."""
    try:
        hist = ticker_obj.history(period="2d", interval="1h")
        if hist.empty:
            return fallback
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return fallback


def _safe_hist(ticker_sym: str, period: str = "5d",
               interval: str = "1h") -> list[dict]:
    """Return list of OHLCV dicts or [] on failure."""
    if not _YF_OK:
        return []
    try:
        data = yf.download(ticker_sym, period=period,
                           interval=interval, progress=False,
                           auto_adjust=True)
        if data.empty:
            return []
        rows = []
        for ts, row in data.iterrows():
            rows.append({
                "time":   str(ts),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
        return rows
    except Exception:
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def get_ohlcv(instrument: str, period: str = "5d",
              interval: str = "1h") -> list[dict]:
    """
    Return OHLCV history for an instrument.
    period: yfinance period string e.g. "5d", "1mo"
    interval: "1h", "15m", "1d"
    """
    ticker_sym = YF_TICKERS.get(instrument, "")
    if not ticker_sym:
        return []
    return _safe_hist(ticker_sym, period=period, interval=interval)


def get_market_context(instrument: str) -> dict:
    """
    Return DXY, VIX, US10Y context relevant to the instrument.
    VIX / US10Y only returned for index / commodity instruments.
    """
    ctx: dict = {}
    if not _YF_OK:
        return ctx
    try:
        dxy_val = _safe_last(yf.Ticker(DXY_TICKER))
        if dxy_val:
            ctx["dxy"] = round(dxy_val, 3)

        # VIX for risk-sensitive instruments
        if instrument in ("NAS100", "US30", "WTI", "XAUUSD"):
            vix_val = _safe_last(yf.Ticker(VIX_TICKER))
            if vix_val:
                ctx["vix"] = round(vix_val, 2)

        # US10Y for rate-sensitive instruments
        if instrument in ("XAUUSD", "GBPUSD", "EURUSD"):
            us10y = _safe_last(yf.Ticker(US10Y_TICKER))
            if us10y:
                ctx["us10y"] = round(us10y, 3)
    except Exception:
        pass
    return ctx


def get_instrument_summary(instrument: str) -> dict:
    """
    Return a rich price + context summary dict for the instrument.

    Keys returned (where data available):
      price, open, high, low, change_pct, change_abs,
      week_high, week_low, source,
      dxy, vix, us10y, fetched_at
    """
    summary: dict = {
        "instrument": instrument,
        "source":     "unavailable",
        "fetched_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }

    ticker_sym = YF_TICKERS.get(instrument, "")
    if not ticker_sym or not _YF_OK:
        return summary

    try:
        hist = yf.download(ticker_sym, period="5d", interval="1h",
                           progress=False, auto_adjust=True)
        if hist.empty:
            return summary

        closes  = hist["Close"].dropna()
        highs   = hist["High"].dropna()
        lows    = hist["Low"].dropna()

        price = float(closes.iloc[-1])
        prev  = float(closes.iloc[-2]) if len(closes) >= 2 else price

        summary["price"]      = round(price, 5)
        summary["open"]       = round(float(hist["Open"].dropna().iloc[0]), 5)
        summary["high"]       = round(float(highs.iloc[-1]), 5)
        summary["low"]        = round(float(lows.iloc[-1]), 5)
        summary["week_high"]  = round(float(highs.max()), 5)
        summary["week_low"]   = round(float(lows.min()), 5)
        summary["change_abs"] = round(price - prev, 5)
        summary["change_pct"] = round((price - prev) / prev * 100, 3) if prev else 0.0
        summary["source"]     = "yfinance"
    except Exception as e:
        summary["error"] = str(e)

    # Attach market context
    try:
        ctx = get_market_context(instrument)
        summary.update(ctx)
    except Exception:
        pass

    return summary
