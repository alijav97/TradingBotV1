from auto_trader import (
    INSTRUMENTS, ACCOUNT_BALANCE,
    LEVERAGE, RR_RATIO,
    calculate_adaptive_sl_tp,
    load_state, get_daily_summary,
)
print("=== INSTRUMENT CONFIG VERIFICATION ===")
for instr, cfg in INSTRUMENTS.items():
    price = {
        "XAUUSD": 4500,
        "NAS100": 18000,
        "US30":   39000,
        "GBPUSD": 1.25,
        "EURUSD": 1.08,
        "WTI":    72,
    }[instr]
    sl, tp, sl_pct, tp_pct = (
        calculate_adaptive_sl_tp(instr, price, "LONG"))
    acc_risk   = round(sl_pct * LEVERAGE * 100, 1)
    acc_profit = round(tp_pct * LEVERAGE * 100, 1)
    print(f"{instr} [{cfg['grade']}] "
          f"Priority:{cfg['priority']} | "
          f"SL:{sl_pct*100:.2f}% "
          f"(-{acc_risk}% acc) | "
          f"TP:{tp_pct*100:.2f}% "
          f"(+{acc_profit}% acc) | "
          f"SL price:{sl} TP price:{tp}")

print("\n=== ACCOUNT CONFIG ===")
print(f"Balance: ${ACCOUNT_BALANCE}")
print(f"Leverage: {LEVERAGE}x")
print(f"RR Ratio: 1:{RR_RATIO}")
print(f"Breakeven WR: {100/(1+RR_RATIO):.1f}%")
print("\n✅ Auto trader config verified")
