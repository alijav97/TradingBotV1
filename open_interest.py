"""
open_interest.py — Volume / open-interest proxy analyzer for TradingBotV1
Tracks volume trends as an institutional flow signal.
Rising price + rising volume = real buying (strong trend).
Rising price + falling volume = weak move, fade it.
"""
from __future__ import annotations
from datetime import datetime

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

YF_TICKERS: dict = {
    "XAUUSD": "GC=F",
    "NAS100": "NQ=F",
    "US30":   "YM=F",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "WTI":    "CL=F",
}


class OpenInterestAnalyzer:

    def get_volume_analysis(self, instrument: str) -> dict:
        """
        Analyse volume trends as a proxy for open interest.

        Signal matrix
        ─────────────────────────────────────────────────────
        ↑ price + ↑ volume  →  STRONG TREND   (institutional buying)
        ↑ price + ↓ volume  →  WEAK MOVE      (shorts covering, fade it)
        ↓ price + ↑ volume  →  STRONG DOWNTREND (institutional selling)
        ↓ price + ↓ volume  →  TREND EXHAUSTION (reversal likely)
        """
        if not _YF_OK:
            return {
                "signal":     "ERROR",
                "bias":       "NEUTRAL",
                "error":      "yfinance not available",
                "instrument": instrument,
            }

        try:
            ticker = YF_TICKERS.get(instrument, "GC=F")
            tk     = yf.Ticker(ticker)
            hist   = tk.history(period="10d", interval="1d")

            if len(hist) < 5:
                return {
                    "instrument": instrument,
                    "signal":     "INSUFFICIENT_DATA",
                    "bias":       "NEUTRAL",
                    "reason":     "Not enough history (need 5+ days)",
                }

            closes  = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()

            # Average of all-but-last-3 days vs last-3-day average
            baseline   = volumes[:-3]
            avg_vol    = sum(baseline) / len(baseline) if baseline else 1
            recent_vol = sum(volumes[-3:]) / 3
            vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1.0

            price_chg = float(closes[-1]) - float(closes[-5])
            price_dir = "UP"     if price_chg > 0 else "DOWN"
            vol_dir   = ("RISING"  if vol_ratio > 1.10 else
                         "FALLING" if vol_ratio < 0.90 else
                         "FLAT")

            # Classify signal
            if price_dir == "UP" and vol_dir == "RISING":
                signal     = "STRONG TREND"
                bias       = "BULLISH"
                reason     = ("Rising price + rising volume = "
                              "real institutional buying")
                confidence = 75

            elif price_dir == "UP" and vol_dir in ("FALLING", "FLAT"):
                signal     = "WEAK MOVE"
                bias       = "CAUTION"
                reason     = ("Rising price + falling volume = "
                              "shorts covering, not real buying. "
                              "Fade this move.")
                confidence = 60

            elif price_dir == "DOWN" and vol_dir == "RISING":
                signal     = "STRONG DOWNTREND"
                bias       = "BEARISH"
                reason     = ("Falling price + rising volume = "
                              "real institutional selling")
                confidence = 75

            elif price_dir == "DOWN" and vol_dir in ("FALLING", "FLAT"):
                signal     = "TREND EXHAUSTION"
                bias       = "REVERSAL WATCH"
                reason     = ("Falling price + falling volume = "
                              "sellers exhausted, reversal likely soon")
                confidence = 65

            else:
                signal     = "NEUTRAL"
                bias       = "NEUTRAL"
                reason     = "No clear volume signal"
                confidence = 40

            return {
                "instrument":      instrument,
                "signal":          signal,
                "bias":            bias,
                "reason":          reason,
                "confidence":      confidence,
                "price_direction": price_dir,
                "volume_direction": vol_dir,
                "volume_ratio":    round(vol_ratio, 2),
                "price_change_5d": round(price_chg, 4),
                "avg_volume":      round(float(avg_vol), 0),
                "recent_volume":   round(float(recent_vol), 0),
                "fetched_at":      datetime.utcnow().isoformat(
                                       timespec="seconds"),
            }

        except Exception as e:
            return {
                "instrument": instrument,
                "signal":     "ERROR",
                "bias":       "NEUTRAL",
                "error":      str(e),
            }
