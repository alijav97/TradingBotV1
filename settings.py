"""
settings.py — Central settings loader and position sizer for TradingBotV1
All files import from here. Never hardcode balance/risk/leverage values.
"""
from __future__ import annotations
import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "user_settings.json")

DEFAULTS: dict = {
    "balance":               300,
    "risk_pct":              10,
    "reward_pct":            30,
    "min_rr":                3.0,
    "leverage":              20,
    "partial_tp":            True,
    "min_confidence":        7.5,
    "daily_loss_limit_pct":  10,
    "weekly_loss_limit_pct": 15,
    "max_open_trades":       3,
    "sessions":              ["London", "NewYork", "Overlap"],
    "min_volume_ratio":      0.5,
    "dead_hours_utc":        [20, 21, 22, 23, 0, 1, 2, 3, 4, 5],
}


def load_settings() -> dict:
    """
    Load settings from data/user_settings.json, merged over DEFAULTS.
    Always recalculates derived fields (risk_usd, reward_usd, min_rr, max_risk_usd).
    """
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            settings = {**DEFAULTS, **saved}
        else:
            settings = DEFAULTS.copy()

        # Derived values — always recalculate so they can't drift out of sync
        balance     = float(settings["balance"])
        risk_pct    = float(settings["risk_pct"])
        reward_pct  = float(settings["reward_pct"])

        settings["risk_usd"]     = round(balance * risk_pct    / 100, 2)
        settings["reward_usd"]   = round(balance * reward_pct  / 100, 2)
        settings["max_risk_usd"] = settings["risk_usd"]          # hard cap
        if settings["risk_usd"] > 0:
            settings["min_rr"]   = round(settings["reward_usd"] / settings["risk_usd"], 2)

        return settings

    except Exception as exc:
        print(f"settings.py load error: {exc}")
        s = DEFAULTS.copy()
        s["risk_usd"]     = round(s["balance"] * s["risk_pct"]    / 100, 2)
        s["reward_usd"]   = round(s["balance"] * s["reward_pct"]  / 100, 2)
        s["max_risk_usd"] = s["risk_usd"]
        return s


def save_settings(updates: dict) -> dict:
    """Persist updates to user_settings.json and return refreshed settings."""
    settings = load_settings()
    settings.update(updates)
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return load_settings()


def calculate_position(
    entry: float,
    sl: float,
    settings: dict | None = None,
) -> dict:
    """
    Calculate position size for XAUUSD (1 lot = 100 oz, $100 P&L per $1 move).

    Returns a dict with:
      tradeable          bool
      lots               float
      sl_distance        float  ($)
      tp1_distance       float  ($ — 2x SL, for 50% close)
      tp2_distance       float  ($ — min_rr × SL, for final close)
      actual_risk_usd    float
      actual_risk_pct    float
      target_reward_usd  float
      target_reward_pct  float
      rr                 float
      balance            float
      leverage           int
    On rejection (not tradeable):
      tradeable  False
      reason     str
    """
    if settings is None:
        settings = load_settings()

    balance     = float(settings["balance"])
    risk_usd    = float(settings["risk_usd"])
    max_risk    = float(settings["max_risk_usd"])
    leverage    = float(settings["leverage"])
    min_rr      = float(settings["min_rr"])

    sl_distance = abs(entry - sl)
    if sl_distance == 0:
        return {
            "tradeable": False,
            "reason": "SL distance is zero — set a valid stop loss",
            "sl_distance": 0,
            "max_risk_usd": max_risk,
        }

    # HARD CHECK: minimum lot (0.01) risk must fit within budget
    # 0.01 lot × sl_distance × $100/lot = minimum possible risk
    min_lot_risk = sl_distance * 0.01 * 100
    if min_lot_risk > max_risk:
        need_balance = min_lot_risk * 10    # 10% of this balance = min_lot_risk
        return {
            "tradeable":   False,
            "reason": (
                f"SL distance ${sl_distance:.2f} is too wide for ${balance:.0f} account.\n"
                f"Minimum risk at 0.01 lots: ${min_lot_risk:.2f} "
                f"(your max: ${max_risk:.2f}).\n"
                f"Needs ${need_balance:.0f}+ account — "
                f"wait for a tighter SL setup."
            ),
            "sl_distance": round(sl_distance, 2),
            "max_risk_usd": max_risk,
        }

    # Lot size from risk budget
    lots_raw = risk_usd / (sl_distance * 100)

    # Leverage cap
    max_lots_leverage = (balance * leverage) / (entry * 100) if entry > 0 else lots_raw
    lots = min(lots_raw, max_lots_leverage)
    lots = round(max(0.01, lots), 2)

    actual_risk      = sl_distance * lots * 100
    actual_risk_pct  = actual_risk / balance * 100
    target_reward    = actual_risk * min_rr
    target_reward_pct = target_reward / balance * 100

    # ── Session profiler adjustment ───────────────────────────────────────────
    _session_adj: dict = {}
    try:
        from session_profiler import get_current_session_profile, get_session_adjusted_position
        _sp  = get_current_session_profile()
        _adj = get_session_adjusted_position(lots, sl_distance, sl_distance * min_rr, _sp)
        lots        = _adj["adjusted_lots"]
        sl_distance = _adj["adjusted_sl_distance"]
        # Recalculate risk/reward with adjusted values
        actual_risk       = sl_distance * lots * 100
        actual_risk_pct   = actual_risk / balance * 100
        target_reward     = actual_risk * min_rr
        target_reward_pct = target_reward / balance * 100
        _session_adj = {
            "lots":         lots,
            "session":      _sp["current_session"],
            "grade":        _sp["session_grade"],
            "lot_change":   _adj["lot_change"],
            "sl_change":    _adj["sl_change"],
            "tp_change":    _adj["tp_change"],
            "lot_multiplier": _adj["lot_multiplier"],
            "sl_multiplier":  _adj["sl_multiplier"],
            "tp_multiplier":  _adj["tp_multiplier"],
            "session_note": _adj["session_note"],
            "trading_recommended": _sp["trading_recommended"],
        }
    except Exception:
        pass   # session_profiler unavailable — use base values unchanged
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "tradeable":          True,
        "lots":               lots,
        "sl_distance":        round(sl_distance, 2),
        "tp1_distance":       round(sl_distance * 2.0, 2),   # 1:2 — close half
        "tp2_distance":       round(sl_distance * min_rr, 2), # 1:3 — close rest
        "actual_risk_usd":    round(actual_risk, 2),
        "actual_risk_pct":    round(actual_risk_pct, 1),
        "target_reward_usd":  round(target_reward, 2),
        "target_reward_pct":  round(target_reward_pct, 1),
        "rr":                 min_rr,
        "balance":            balance,
        "leverage":           int(leverage),
        "session_adjustment": _session_adj,
    }


# Module-level singleton — importers can use S.get('balance') etc.
S = load_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Per-instrument default settings
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENT_SETTINGS: dict = {
    "XAUUSD": {
        "pip_size": 0.10,
        "contract_size": 100,
        "typical_spread": 0.3,
        "session_start_gst": 13,
        "session_end_gst": 22,
        "currency": "USD",
        "asset_class": "commodity",
    },
    "NAS100": {
        "pip_size": 1.0,
        "contract_size": 1,
        "typical_spread": 1.0,
        "session_start_gst": 13,
        "session_end_gst": 22,
        "currency": "USD",
        "asset_class": "index",
    },
    "US30": {
        "pip_size": 1.0,
        "contract_size": 1,
        "typical_spread": 2.0,
        "session_start_gst": 13,
        "session_end_gst": 22,
        "currency": "USD",
        "asset_class": "index",
    },
    "GBPUSD": {
        "pip_size": 0.0001,
        "contract_size": 100000,
        "typical_spread": 0.00008,
        "session_start_gst": 8,
        "session_end_gst": 17,
        "currency": "USD",
        "asset_class": "forex",
    },
    "EURUSD": {
        "pip_size": 0.0001,
        "contract_size": 100000,
        "typical_spread": 0.00006,
        "session_start_gst": 7,
        "session_end_gst": 17,
        "currency": "USD",
        "asset_class": "forex",
    },
    "WTI": {
        "pip_size": 0.01,
        "contract_size": 1000,
        "typical_spread": 0.03,
        "session_start_gst": 13,
        "session_end_gst": 22,
        "currency": "USD",
        "asset_class": "commodity",
    },
}
