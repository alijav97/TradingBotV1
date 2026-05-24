"""
paper_trader.py — Mock trade tracker for TradingBotV1
Tracks signals as paper trades, auto-closes on TP/SL, builds a performance journal.
Storage: data/paper_trades.json
"""
from datetime import datetime, timezone, timedelta
import json
import uuid
import os

GST = timezone(timedelta(hours=4))
PAPER_TRADES_FILE = "data/paper_trades.json"   # legacy fallback (XAUUSD)


def _get_trades_file(instrument: str = "XAUUSD") -> str:
    """Return the per-instrument paper trades JSON file path."""
    safe = instrument.replace("/", "").upper()
    if safe == "XAUUSD":
        return PAPER_TRADES_FILE          # keep backward compat with old file
    return f"data/paper_trades_{safe}.json"


def _load_trades(instrument: str = "XAUUSD") -> list:
    try:
        with open(_get_trades_file(instrument)) as f:
            return json.load(f)
    except Exception:
        return []


def _save_trades(trades: list, instrument: str = "XAUUSD") -> None:
    os.makedirs("data", exist_ok=True)
    with open(_get_trades_file(instrument), "w") as f:
        json.dump(trades, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — Open a new paper trade
# ─────────────────────────────────────────────────────────────────────────────
def open_paper_trade(signal: dict, current_price: float,
                     instrument: str = "XAUUSD") -> dict:
    """Open a new paper trade from a signal dict + current price."""
    direction = signal.get("direction", "short")
    _sl_default = (current_price + 30) if direction == "short" else (current_price - 30)
    _tp_default = (current_price - 60) if direction == "short" else (current_price + 60)

    trade: dict = {
        "trade_id":        str(uuid.uuid4())[:8].upper(),
        "opened_at":       datetime.now(GST).strftime("%I:%M %p UAE | %a %d %b %Y"),
        "opened_at_iso":   datetime.now(GST).isoformat(),
        "direction":       direction,
        "entry":           round(current_price, 2),
        "sl":              round(signal.get("stop_loss",  _sl_default), 2),
        "tp1":             round(signal.get("take_profit", _tp_default), 2),
        "sl_distance":     0.0,
        "tp_distance":     0.0,
        "strategy":        signal.get("pattern_name", "Manual Paper Trade"),
        "confidence":      signal.get("confidence", 0),
        "session":         signal.get("session", "Unknown"),
        "regime":          signal.get("regime",  "Unknown"),
        "status":          "OPEN",
        "outcome":         None,
        "closed_at":       None,
        "close_price":     None,
        "pnl_pips":        0.0,
        "pnl_usd":         0.0,
        "max_profit_pips": 0.0,
        "max_loss_pips":   0.0,
        "updates":         [],
    }

    trade["sl_distance"] = round(abs(trade["entry"] - trade["sl"]),  2)
    trade["tp_distance"] = round(abs(trade["tp1"]   - trade["entry"]), 2)

    # Capture full signal conditions at entry for ML training
    try:
        raw  = signal.get("raw_checks", {}) or {}
        inds = raw.get("indicators", {}) or {}
        _rsi_val = float(signal.get("rsi", 50) or 50)
        conditions = {
            # ── existing fields ───────────────────────────────────────────
            "rsi":               signal.get("rsi", raw.get("rsi", 50)),
            "atr":               signal.get("atr", raw.get("atr", 0)),
            "spread":            signal.get("spread_usd", 0),
            "d1_bias":           raw.get("htf", {}).get("d1_trend", "unknown"),
            "h4_bias":           raw.get("htf", {}).get("h4_trend", "unknown"),
            "session":           signal.get("session", "Unknown"),
            "hour_uae":          signal.get("hour_uae", datetime.now(GST).hour),
            "regime":            signal.get("regime", "Unknown"),
            "in_killzone":       inds.get("killzones", {}).get("in_killzone", False),
            "confidence":        signal.get("confidence", 0),
            "checklist_passed":  signal.get("checklist_results", {}).get("checks_passed", 0),
            "smc_grade":         signal.get("smc_grade", "D"),
            "mtf_score":         signal.get("mtf_score", 0),
            "geo_risk":          signal.get("geo_risk_level", "normal"),
            "fundamental_bias":  signal.get("fundamental_bias", "NEUTRAL"),
            "cot_bias":          signal.get("cot_bias", "NEUTRAL"),
            "macro_bias":        signal.get("macro_bias", "neutral"),
            "dxy_trend":         signal.get("dxy_trend", "unknown"),
            "volume_ratio":      signal.get("volume_ratio", 1.0),
            "volume_class":      signal.get("volume_class", "normal"),
            "alligator_state":   inds.get("alligator", {}).get("state", ""),
            "macd_bias":         inds.get("macd", {}).get("bias", ""),
            "stoch_rsi_k":       inds.get("stoch_rsi", {}).get("k", 50),
            "ichimoku_bias":     inds.get("ichimoku", {}).get("bias", ""),
            "supertrend_bias":   inds.get("supertrend", {}).get("bias", ""),
            "adx":               inds.get("adx", {}).get("adx", 0),
            "adx_trending":      inds.get("adx", {}).get("trending", False),
            "vwap_above":        inds.get("vwap", {}).get("above", False),
            "obv_divergence":    inds.get("obv", {}).get("divergence", ""),
            "squeeze_on":        inds.get("squeeze", {}).get("squeeze_on", False),
            "wyckoff_phase":     inds.get("wyckoff", {}).get("phase", ""),
            "real_rate_bias":    inds.get("real_rate", {}).get("bias", ""),
            "market_cipher":     inds.get("market_cipher", {}).get("bias", ""),
            "news_bias":         signal.get("global_news_bias", "neutral"),
            "volatility_warning": signal.get("volatility_warning", False),
            "rr_ratio":          round(
                signal.get("tp_distance", 1) /
                max(signal.get("sl_distance", 1), 0.1), 2
            ),
            "signal_source":     signal.get("source", "strategy"),
            "pattern_name":      signal.get("pattern_name", ""),
            "is_counter_trend":  (
                (signal.get("direction") == "long"  and raw.get("htf", {}).get("d1_trend") == "bearish") or
                (signal.get("direction") == "short" and raw.get("htf", {}).get("d1_trend") == "bullish")
            ),
            # ── newly added fields ────────────────────────────────────────
            "strategy_tags":     signal.get("strategy_tags",
                                 signal.get("strategies_voted",
                                 signal.get("contributing_strategies", []))),
            "confluence_factors": signal.get("confluence_factors",
                                  signal.get("factors_triggered", [])),
            "confluence_score":  signal.get("confidence",
                                 signal.get("confluence_score", 0)),
            "checklist_gates_passed": signal.get("checklist_results", {})
                                      .get("checks_passed", 0),
            "checklist_gate_details": signal.get("checklist_results",
                                      signal.get("gate_results", {})),
            "d1_trend":          signal.get("d1_trend",
                                 raw.get("htf", {}).get("d1_trend", "unknown")),
            "h4_trend":          signal.get("h4_trend",
                                 raw.get("htf", {}).get("h4_trend", "unknown")),
            "h1_trend":          signal.get("h1_trend", "unknown"),
            "counter_trend":     signal.get("counter_trend", False),
            "killzone_name":     signal.get("session", "Unknown"),
            "rsi_zone": (
                "overbought" if _rsi_val > 70
                else "oversold" if _rsi_val < 30
                else "neutral"
            ),
            "rsi_divergence":    signal.get("rsi_divergence", False),
            "spread_pips":       signal.get("spread_usd", 0),
            "day_of_week":       datetime.now(GST).strftime("%A"),
            "direction":         signal.get("direction", "LONG"),
            "entry_price":       signal.get("entry", 0),
        }
        trade["conditions_at_entry"] = conditions
    except Exception:
        trade["conditions_at_entry"] = {}

    trade["updates"].append({
        "time":          datetime.now(GST).strftime("%I:%M %p UAE"),
        "price":         current_price,
        "floating_pips": 0.0,
        "note":          "✅ Trade opened",
    })

    trades = _load_trades(instrument)
    trades.append(trade)
    _save_trades(trades, instrument)

    # Trigger ML retraining every 5 newly closed trades
    try:
        _closed = [t for t in trades if t.get("status") == "CLOSED"]
        if len(_closed) >= 5 and len(_closed) % 5 == 0:
            from ml_engine import run_ml_training  # type: ignore[import]
            run_ml_training()
    except Exception:
        pass

    return trade


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — Update all open trades with the current price
# ─────────────────────────────────────────────────────────────────────────────
def update_paper_trades(current_price: float,
                        instrument: str = "XAUUSD") -> list:
    """
    Check every OPEN paper trade against current_price.
    Auto-close on TP/SL hit. Returns list of trades closed this call.
    """
    trades     = _load_trades(instrument)
    closed_now = []

    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        entry = trade["entry"]
        sl    = trade["sl"]
        tp1   = trade["tp1"]
        d     = trade["direction"]

        if d == "long":
            floating_pips = round((current_price - entry) / 0.1, 1)
            sl_hit        = current_price <= sl
            tp_hit        = current_price >= tp1
        else:  # short
            floating_pips = round((entry - current_price) / 0.1, 1)
            sl_hit        = current_price >= sl
            tp_hit        = current_price <= tp1

        # Track high-water / low-water marks
        if floating_pips > trade.get("max_profit_pips", 0.0):
            trade["max_profit_pips"] = floating_pips
        if floating_pips < trade.get("max_loss_pips", 0.0):
            trade["max_loss_pips"] = floating_pips

        # Rolling price update (keep last 20)
        trade["updates"].append({
            "time":          datetime.now(GST).strftime("%I:%M %p UAE"),
            "price":         current_price,
            "floating_pips": floating_pips,
            "note":          "Price update",
        })
        trade["updates"] = trade["updates"][-20:]

        if sl_hit:
            try:
                _opened_iso = datetime.fromisoformat(trade["opened_at_iso"])
                _closed_dt  = datetime.now(GST)
                _dur        = _closed_dt - _opened_iso.replace(tzinfo=GST)
                _hrs  = int(_dur.total_seconds() // 3600)
                _mins = int((_dur.total_seconds() % 3600) // 60)
                time_held = f"{_hrs}h {_mins}m" if _hrs > 0 else f"{_mins}m"
            except Exception:
                time_held = "unknown"
            sl_pips = -round(abs(entry - sl) / 0.1, 1)
            trade.update({
                "status":       "CLOSED",
                "outcome":      "LOSS",
                "close_price":  round(sl, 2),
                "closed_at":    datetime.now(GST).strftime("%I:%M %p UAE | %a %d %b %Y"),
                "time_held":    time_held,
                "pnl_pips":     sl_pips,
                "pnl_usd":      round(sl_pips * 0.1, 2),
                "close_reason": "SL_HIT",
                "close_detail": (
                    f"SL hit at ${sl:,.2f} | "
                    f"Held for {time_held} | "
                    f"Max profit seen: +{trade.get('max_profit_pips', 0)} pips | "
                    f"Entered at ${entry:,.2f}"
                ),
            })
            trade["updates"].append({
                "time":          datetime.now(GST).strftime("%I:%M %p UAE"),
                "price":         sl,
                "floating_pips": sl_pips,
                "note": (
                    f"\u274c SL HIT at ${sl:,.2f} | "
                    f"Loss {sl_pips} pips | "
                    f"Time held: {time_held}"
                ),
            })
            closed_now.append(trade)

        elif tp_hit:
            try:
                _opened_iso = datetime.fromisoformat(trade["opened_at_iso"])
                _closed_dt  = datetime.now(GST)
                _dur        = _closed_dt - _opened_iso.replace(tzinfo=GST)
                _hrs  = int(_dur.total_seconds() // 3600)
                _mins = int((_dur.total_seconds() % 3600) // 60)
                time_held = f"{_hrs}h {_mins}m" if _hrs > 0 else f"{_mins}m"
            except Exception:
                time_held = "unknown"
            tp_pips = round(abs(tp1 - entry) / 0.1, 1)
            trade.update({
                "status":       "CLOSED",
                "outcome":      "WIN",
                "close_price":  round(tp1, 2),
                "closed_at":    datetime.now(GST).strftime("%I:%M %p UAE | %a %d %b %Y"),
                "time_held":    time_held,
                "pnl_pips":     tp_pips,
                "pnl_usd":      round(tp_pips * 0.1, 2),
                "close_reason": "TP_HIT",
                "close_detail": (
                    f"TP hit at ${tp1:,.2f} | "
                    f"Held for {time_held} | "
                    f"Max loss seen: {trade.get('max_loss_pips', 0)} pips | "
                    f"Entered at ${entry:,.2f}"
                ),
            })
            trade["updates"].append({
                "time":          datetime.now(GST).strftime("%I:%M %p UAE"),
                "price":         tp1,
                "floating_pips": tp_pips,
                "note": (
                    f"\u2705 TP HIT at ${tp1:,.2f} | "
                    f"Win +{tp_pips} pips | "
                    f"Time held: {time_held}"
                ),
            })
            closed_now.append(trade)

    _save_trades(trades, instrument)
    return closed_now


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — Summary dict
# ─────────────────────────────────────────────────────────────────────────────
def get_paper_summary(instrument: str = "XAUUSD") -> dict:
    """Return aggregate stats for all paper trades."""
    trades  = _load_trades(instrument)
    open_t  = [t for t in trades if t["status"] == "OPEN"]
    closed  = [t for t in trades if t["status"] == "CLOSED"]
    wins    = [t for t in closed  if t["outcome"] == "WIN"]
    losses  = [t for t in closed  if t["outcome"] == "LOSS"]

    total_pnl = sum(t.get("pnl_pips", 0.0) for t in closed)
    wr        = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
    avg_win   = round(sum(t["pnl_pips"] for t in wins)   / len(wins),   1) if wins   else 0.0
    avg_loss  = round(sum(t["pnl_pips"] for t in losses) / len(losses), 1) if losses else 0.0

    return {
        "total":       len(trades),
        "open":        len(open_t),
        "closed":      len(closed),
        "wins":        len(wins),
        "losses":      len(losses),
        "win_rate":    wr,
        "total_pnl":   round(total_pnl, 1),
        "avg_win":     avg_win,
        "avg_loss":    avg_loss,
        "open_trades": open_t,
        "recent":      closed[-10:],
    }


def get_open_trades(instrument: str = "XAUUSD") -> list:
    """Return list of currently open paper trades for the instrument."""
    return [t for t in _load_trades(instrument) if t["status"] == "OPEN"]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — Manual close
# ─────────────────────────────────────────────────────────────────────────────
def close_paper_trade_manually(trade_id: str, current_price: float,
                               instrument: str = "XAUUSD") -> dict:
    """Manually close an open paper trade at current_price."""
    trades = _load_trades(instrument)
    for t in trades:
        if t["trade_id"] == trade_id.upper() and t["status"] == "OPEN":
            d = t["direction"]
            pips = (
                round((current_price - t["entry"]) / 0.1, 1)
                if d == "long"
                else round((t["entry"] - current_price) / 0.1, 1)
            )
            t.update({
                "status":      "CLOSED",
                "outcome":     "WIN" if pips > 0 else "LOSS",
                "close_price": current_price,
                "closed_at":   datetime.now(GST).strftime("%I:%M %p UAE | %a %d %b %Y"),
                "pnl_pips":    pips,
                "pnl_usd":     round(pips * 0.1, 2),
            })
            _save_trades(trades, instrument)
            return t
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — Performance report string
# ─────────────────────────────────────────────────────────────────────────────
def get_paper_performance_report(instrument: str = "XAUUSD") -> str:
    """Return a formatted Markdown performance report."""
    s = get_paper_summary(instrument)
    if s["total"] == 0:
        return (
            "No paper trades yet.\n\n"
            "Type **'paper short'** or **'paper long'** to start tracking without real money."
        )

    pnl_emoji = "📈" if s["total_pnl"] > 0 else "📉"
    pnl_sign  = "+" if s["total_pnl"] > 0 else ""

    lines = [
        "## 📊 PAPER TRADING PERFORMANCE\n",
        f"Total trades:  **{s['total']}** "
        f"({s['open']} open, {s['closed']} closed)\n",
        f"Win rate:      **{s['win_rate']}%**\n",
        f"Total P&L:     {pnl_emoji} **{pnl_sign}{s['total_pnl']} pips**\n",
        f"Avg win:       +{s['avg_win']} pips\n",
        f"Avg loss:      {s['avg_loss']} pips\n",
        "\n" + "─" * 40 + "\n",
        "**RECENT CLOSED TRADES:**\n",
    ]

    for t in s["recent"]:
        emoji    = "✅" if t["outcome"] == "WIN" else "❌"
        pip_sign = "+" if t["pnl_pips"] > 0 else ""
        lines.append(
            f"{emoji} {t['trade_id']} "
            f"{t['direction'].upper()} "
            f"${t['entry']:,.2f} → ${t.get('close_price', 0):,.2f} "
            f"| {pip_sign}{t['pnl_pips']} pips "
            f"| {t.get('closed_at', '')}\n"
        )

    return "\n".join(lines)
