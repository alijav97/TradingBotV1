"""
macro_scorer.py — Economic health scorer for forex pairs in TradingBotV1
Scores USD, GBP, EUR using interest rates, GDP, unemployment, CPI,
retail sales and trade balance. Generates macro bias for GBPUSD / EURUSD.
"""
from __future__ import annotations


class MacroScorer:
    """Scores currency economic health and generates forex pair bias."""

    # ── Hardcoded latest macro data (update monthly) ─────────────────────────
    # Source: centralbanknews.info, tradingeconomics.com
    LATEST_DATA: dict = {
        "USD": {
            "interest_rate": 5.50,
            "cpi":           3.5,
            "unemployment":  3.9,
            "gdp_growth":    2.8,
            "trade_balance": -63.3,
            "retail_sales":  0.6,
        },
        "GBP": {
            "interest_rate": 5.25,
            "cpi":           3.2,
            "unemployment":  4.2,
            "gdp_growth":    0.1,
            "trade_balance": -2.1,
            "retail_sales":  0.2,
        },
        "EUR": {
            "interest_rate": 4.00,
            "cpi":           2.4,
            "unemployment":  6.1,
            "gdp_growth":    0.4,
            "trade_balance": 1.2,
            "retail_sales":  -0.5,
        },
    }

    # ── Pair → (base, quote) mapping ─────────────────────────────────────────
    PAIR_CURRENCIES: dict = {
        "GBPUSD": ("GBP", "USD"),
        "EURUSD": ("EUR", "USD"),
    }

    # ─────────────────────────────────────────────────────────────────────────

    def score_currency(self, currency: str) -> dict:
        """
        Score a currency's economic health 0-100.
        Higher = stronger economy = stronger currency.
        """
        try:
            data = self.LATEST_DATA.get(currency, {})
            if not data:
                return {
                    "currency": currency,
                    "score": 50,
                    "grade": "?",
                    "bias": "NEUTRAL",
                }

            score = 50  # baseline

            # Interest rate (weight: 30%)
            rate = float(data.get("interest_rate", 2.0))
            if rate >= 5.0:    score += 15
            elif rate >= 3.0:  score += 8
            elif rate >= 1.0:  score += 3
            else:              score -= 5

            # GDP growth (weight: 25%)
            gdp = float(data.get("gdp_growth", 0))
            if gdp >= 2.5:    score += 12
            elif gdp >= 1.0:  score += 7
            elif gdp >= 0:    score += 3
            else:             score -= 8

            # Unemployment (weight: 20%) — lower is better
            unemp = float(data.get("unemployment", 5.0))
            if unemp <= 3.5:   score += 10
            elif unemp <= 5.0: score += 5
            elif unemp <= 7.0: score += 0
            else:              score -= 8

            # CPI inflation (weight: 15%) — 2% is ideal
            cpi = float(data.get("cpi", 2.0))
            if 1.5 <= cpi <= 2.5:  score += 8
            elif cpi <= 4.0:       score += 3
            elif cpi <= 6.0:       score -= 3
            else:                  score -= 10

            # Retail sales (weight: 10%)
            retail = float(data.get("retail_sales", 0))
            if retail >= 0.5:  score += 5
            elif retail >= 0:  score += 2
            else:              score -= 3

            score = max(0, min(100, score))

            if score >= 70:    grade, bias = "A", "STRONG BULLISH"
            elif score >= 60:  grade, bias = "B", "BULLISH"
            elif score >= 45:  grade, bias = "C", "NEUTRAL"
            elif score >= 35:  grade, bias = "D", "BEARISH"
            else:              grade, bias = "F", "STRONG BEARISH"

            return {
                "currency":     currency,
                "score":        score,
                "grade":        grade,
                "bias":         bias,
                "interest_rate": rate,
                "gdp_growth":   gdp,
                "unemployment": unemp,
                "cpi":          cpi,
                "retail_sales": float(data.get("retail_sales", 0)),
                "trade_balance": float(data.get("trade_balance", 0)),
            }
        except Exception as e:
            return {
                "currency": currency,
                "score": 50,
                "grade": "?",
                "bias": "NEUTRAL",
                "error": str(e),
            }

    def score_pair(self, instrument: str) -> dict:
        """
        Compare two currencies and generate pair bias.
        GBPUSD → compare GBP vs USD
        EURUSD → compare EUR vs USD
        """
        try:
            if instrument not in self.PAIR_CURRENCIES:
                return {
                    "instrument":   instrument,
                    "bias":         "N/A",
                    "reason":       "Not a forex pair",
                    "base_score":   0,
                    "quote_score":  0,
                    "confidence":   0,
                }

            base_ccy, quote_ccy = self.PAIR_CURRENCIES[instrument]
            base  = self.score_currency(base_ccy)
            quote = self.score_currency(quote_ccy)

            diff = base["score"] - quote["score"]

            if diff >= 15:
                bias       = "STRONG LONG"
                confidence = 80
                reason     = (f"{base_ccy} economy significantly "
                              f"stronger than {quote_ccy}")
            elif diff >= 7:
                bias       = "LONG"
                confidence = 65
                reason     = (f"{base_ccy} economy moderately "
                              f"stronger than {quote_ccy}")
            elif diff <= -15:
                bias       = "STRONG SHORT"
                confidence = 80
                reason     = (f"{quote_ccy} economy significantly "
                              f"stronger than {base_ccy}")
            elif diff <= -7:
                bias       = "SHORT"
                confidence = 65
                reason     = (f"{quote_ccy} economy moderately "
                              f"stronger than {base_ccy}")
            else:
                bias       = "NEUTRAL"
                confidence = 40
                reason     = "Economies roughly equal — no macro edge"

            rate_diff = (
                float(base.get("interest_rate", 0)) -
                float(quote.get("interest_rate", 0))
            )

            return {
                "instrument":         instrument,
                "bias":               bias,
                "confidence":         confidence,
                "reason":             reason,
                "base_currency":      base_ccy,
                "quote_currency":     quote_ccy,
                "base_score":         base["score"],
                "quote_score":        quote["score"],
                "score_diff":         diff,
                "interest_rate_diff": round(rate_diff, 2),
                "base_details":       base,
                "quote_details":      quote,
            }
        except Exception as e:
            return {"instrument": instrument, "bias": "NEUTRAL", "error": str(e)}

    def get_macro_bias(self, instrument: str) -> str:
        """Quick bias string for confluence engine."""
        try:
            return self.score_pair(instrument).get("bias", "NEUTRAL")
        except Exception:
            return "NEUTRAL"
