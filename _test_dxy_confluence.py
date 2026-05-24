"""Quick test: confluence engine with all 9 checks."""
import sys
sys.path.insert(0, r"c:/Users/alija/Downloads/TradingBotV1")

from confluence_engine import score_confluences
import pandas as pd, numpy as np

n = 60
idx = pd.date_range("2024-01-01", periods=n, freq="h")
close = pd.Series(2000 + np.cumsum(np.random.randn(n)), index=idx)
df = pd.DataFrame({
    "open":   close.shift(1).bfill(),
    "high":   close + 2,
    "low":    close - 2,
    "close":  close,
    "volume": 1000,
}, index=idx)

htf_bias = {
    "overall_bias": "bullish", "bias_strength": "strong",
    "d1_trend": "bullish", "h4_trend": "bullish",
    "h4_structure": {}, "h4_bos": None, "h4_choch": None,
}
dxy_ctx = {"available": True, "dxy_trend": "down", "dxy_rsi": 38.0, "momentum_strength": "strong"}

r = score_confluences(df, "long", htf_bias=htf_bias, dxy_ctx=dxy_ctx)
print(f"total_checks = {r['total_checks']}")
print(f"passed_count = {r['passed_count']}")
print(f"confidence   = {r['confidence']}")
print(f"dxy_result   = {r.get('dxy_result')}")
assert r["total_checks"] == 9, f"Expected 9, got {r['total_checks']}"
print("PASS: 9 checks confirmed")
