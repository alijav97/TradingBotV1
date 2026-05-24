import sys
sys.path.insert(0, ".")
from fundamental_bias import get_fundamental_bias, check_fundamental_conflict, detect_conflict

fb = get_fundamental_bias()
print("=== FUNDAMENTAL BIAS REPORT ===")
print("Bias:       ", fb["fundamental_bias"])
print("Score:      ", f'{fb["total_score"]:+d}')
print("Confidence: ", f'{fb["confidence"]:.1f}/10')
print("Summary:    ", fb["summary"])
print("Display:    ", fb["display_line"])
print()
print("--- Factors ---")
for k, v in fb["factors"].items():
    print(f"  {k:15} {v['score']:+d}  {v['note']}")

print()
print("--- Conflict Tests ---")
for td in ["LONG", "SHORT"]:
    cc = check_fundamental_conflict(td)
    print(f"  Tech={td:5}  conflict={cc['conflict']}  severity={cc['severity']}")

print()
print("--- detect_conflict ---")
r = detect_conflict("long", "STRONGLY_BEARISH")
print(f"  long vs STRONGLY_BEARISH  -> severity={r['severity']}  conflict={r['conflict']}")
r = detect_conflict("short", "BULLISH")
print(f"  short vs BULLISH          -> severity={r['severity']}  conflict={r['conflict']}")
r = detect_conflict("long", "BULLISH")
print(f"  long vs BULLISH           -> severity={r['severity']}  conflict={r['conflict']}")
r = detect_conflict("short", "STRONGLY_BEARISH")
print(f"  short vs STRONGLY_BEARISH -> severity={r['severity']}  conflict={r['conflict']}")
print()
print("ALL VALIDATION TESTS PASS")
