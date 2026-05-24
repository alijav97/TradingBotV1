"""
sector_rotation.py — Sector ETF flow tracker for TradingBotV1
Tracks money flow between market sectors and generates
instrument-specific trading bias.
"""
from __future__ import annotations
from datetime import datetime

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── Sector ETF tickers ────────────────────────────────────────────────────────
SECTOR_ETFS: dict = {
    "Technology":   "XLK",
    "Energy":       "XLE",
    "Financials":   "XLF",
    "Healthcare":   "XLV",
    "Industrials":  "XLI",
    "Materials":    "XLB",
    "Utilities":    "XLU",
    "ConsumerDisc": "XLY",
    "ConsumerStap": "XLP",
    "RealEstate":   "XLRE",
}

# ── Which sectors drive which instruments ─────────────────────────────────────
INSTRUMENT_SECTORS: dict = {
    "NAS100": ["Technology", "ConsumerDisc"],
    "US30":   ["Financials", "Industrials", "Healthcare"],
    "WTI":    ["Energy"],
    "XAUUSD": ["Materials"],
    "GBPUSD": ["Financials"],
    "EURUSD": ["Financials", "Industrials"],
}

# ── Risk-off sectors (money here = markets fearful) ───────────────────────────
RISK_OFF_SECTORS: list = [
    "Utilities", "ConsumerStap", "Healthcare", "RealEstate"
]


class SectorRotation:

    def get_sector_flows(self) -> dict:
        """
        Fetch 5-day performance of all sector ETFs.
        Returns ranked list from strongest to weakest.
        """
        if not _YF_OK:
            return {"error": "yfinance not available"}

        flows: dict = {}
        try:
            for sector, ticker in SECTOR_ETFS.items():
                try:
                    tk   = yf.Ticker(ticker)
                    hist = tk.history(period="5d", interval="1d")
                    if len(hist) >= 2:
                        start = float(hist["Close"].iloc[0])
                        end   = float(hist["Close"].iloc[-1])
                        chg   = round(((end - start) / start) * 100, 2) if start else 0.0
                        flows[sector] = {
                            "ticker":    ticker,
                            "change_5d": chg,
                            "direction": "INFLOW" if chg > 0 else "OUTFLOW",
                            "strength":  (
                                "STRONG"   if abs(chg) > 2 else
                                "MODERATE" if abs(chg) > 0.5 else
                                "WEAK"
                            ),
                        }
                    else:
                        flows[sector] = {
                            "ticker":    ticker,
                            "change_5d": 0.0,
                            "direction": "UNKNOWN",
                            "strength":  "UNKNOWN",
                        }
                except Exception:
                    flows[sector] = {
                        "ticker":    ticker,
                        "change_5d": 0.0,
                        "direction": "UNKNOWN",
                        "strength":  "UNKNOWN",
                    }
        except Exception as e:
            return {"error": str(e)}

        return flows

    def get_risk_appetite(self, flows: dict) -> dict:
        """
        Determine if market is risk-on or risk-off based on
        which sectors are getting inflows.
        """
        try:
            if "error" in flows:
                return {"regime": "NEUTRAL", "error": flows["error"]}

            risk_off_flow = sum(
                flows.get(s, {}).get("change_5d", 0)
                for s in RISK_OFF_SECTORS
            )
            risk_on_flow = sum(
                flows.get(s, {}).get("change_5d", 0)
                for s in ["Technology", "ConsumerDisc", "Energy"]
            )

            diff = risk_on_flow - risk_off_flow

            if diff > 3:
                regime  = "RISK_ON"
                meaning = "Money flowing into growth sectors — markets bullish"
            elif diff < -3:
                regime  = "RISK_OFF"
                meaning = "Money fleeing to safety sectors — markets fearful"
            else:
                regime  = "NEUTRAL"
                meaning = "Mixed sector flows — no clear bias"

            return {
                "regime":        regime,
                "meaning":       meaning,
                "risk_on_flow":  round(risk_on_flow, 2),
                "risk_off_flow": round(risk_off_flow, 2),
                "diff":          round(diff, 2),
            }
        except Exception as e:
            return {"regime": "NEUTRAL", "error": str(e)}

    def get_instrument_bias(self, instrument: str,
                            flows: dict | None = None) -> dict:
        """
        Generate trading bias for a specific instrument
        based on its relevant sector flows.
        """
        try:
            if flows is None:
                flows = self.get_sector_flows()

            if "error" in flows:
                return {
                    "bias":       "NEUTRAL",
                    "reason":     "Sector data unavailable — markets may be closed",
                    "confidence": 40,
                }

            sectors = INSTRUMENT_SECTORS.get(instrument, [])
            if not sectors:
                return {
                    "bias":       "NEUTRAL",
                    "reason":     "No sector mapping for this instrument",
                    "confidence": 40,
                }

            total_flow     = 0.0
            sector_details = []
            for s in sectors:
                chg         = flows.get(s, {}).get("change_5d", 0)
                total_flow += chg
                sector_details.append(f"{s}: {chg:+.1f}%")

            avg_flow = total_flow / len(sectors)

            if avg_flow > 1.5:
                bias, confidence = "BULLISH",      70
            elif avg_flow > 0.3:
                bias, confidence = "MILD BULLISH", 55
            elif avg_flow < -1.5:
                bias, confidence = "BEARISH",      70
            elif avg_flow < -0.3:
                bias, confidence = "MILD BEARISH", 55
            else:
                bias, confidence = "NEUTRAL",      40

            risk = self.get_risk_appetite(flows)

            # Risk-off override for risk-sensitive indices
            if (risk["regime"] == "RISK_OFF"
                    and bias == "BULLISH"
                    and instrument in ("NAS100", "US30")):
                bias       = "CAUTION — risk-off override"
                confidence = 35

            return {
                "instrument":      instrument,
                "bias":            bias,
                "confidence":      confidence,
                "avg_sector_flow": round(avg_flow, 2),
                "sectors_tracked": sector_details,
                "risk_regime":     risk["regime"],
                "risk_meaning":    risk["meaning"],
            }
        except Exception as e:
            return {
                "bias":       "NEUTRAL",
                "error":      str(e),
                "confidence": 40,
            }

    def get_full_report(self, instrument: str) -> dict:
        """Full sector rotation report for display."""
        try:
            flows = self.get_sector_flows()

            if "error" in flows:
                return {
                    "instrument":      instrument,
                    "error":           flows["error"],
                    "instrument_bias": {
                        "bias":   "NEUTRAL",
                        "reason": "Sector data unavailable — markets may be closed",
                    },
                    "risk_appetite":   {"regime": "NEUTRAL"},
                    "sector_ranking":  [],
                    "top_3_inflow":    [],
                    "top_3_outflow":   [],
                }

            bias = self.get_instrument_bias(instrument, flows)
            risk = self.get_risk_appetite(flows)

            ranked = sorted(
                [
                    (s, d.get("change_5d", 0))
                    for s, d in flows.items()
                    if isinstance(d, dict) and "change_5d" in d
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            return {
                "instrument":      instrument,
                "instrument_bias": bias,
                "risk_appetite":   risk,
                "sector_ranking":  ranked,
                "top_3_inflow":    ranked[:3],
                "top_3_outflow":   ranked[-3:],
                "all_flows":       flows,
                "fetched_at":      datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"instrument": instrument, "error": str(e)}
