"""
learning.py - Continuous Learning Engine for TradingBotV1

Logs every trade Ali executes, learns from results, updates rule confidence
scores, discovers new patterns via Claude, and produces weekly reports.
"""

import os
import json
import uuid
from datetime import datetime, timezone, timedelta

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(BASE_DIR, "data", "trade_log.json")
RULES_PATH     = os.path.join(BASE_DIR, "data", "rules.json")

# ── Anthropic client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

PIP_SIZE = 0.1   # $0.10 per pip for gold


# ═══════════════════════════════════════════════════════════════════════════════
# Helper I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _load_trade_log() -> list[dict]:
    """Return the full trade log list, or [] if file doesn't exist."""
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    if not os.path.exists(TRADE_LOG_PATH):
        return []
    try:
        with open(TRADE_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_trade_log(trades: list[dict]) -> None:
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    with open(TRADE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2, default=str)


def _load_rules() -> list[dict]:
    if not os.path.exists(RULES_PATH):
        return []
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_rules(rules: list[dict]) -> None:
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, default=str)


def _gst_now() -> str:
    """Return current time in GST (UTC+4) as ISO string."""
    return (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")


def _week_start() -> datetime:
    """Monday 00:00 UTC of the current week."""
    now = datetime.now(timezone.utc)
    return now - timedelta(days=now.weekday(), hours=now.hour,
                           minutes=now.minute, seconds=now.second)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. log_trade
# ═══════════════════════════════════════════════════════════════════════════════

def log_trade(trade_data: dict) -> str:
    """
    Save a new open trade to data/trade_log.json.

    Required keys in trade_data:
        symbol, direction, entry_price, stop_loss, take_profit,
        pattern_name, confidence_score, news_sentiment,
        rsi_at_entry, ema_position, session, day_of_week

    Returns:
        Unique trade_id string assigned to this trade.
    """
    required = [
        "symbol", "direction", "entry_price", "stop_loss", "take_profit",
        "pattern_name", "confidence_score", "news_sentiment",
        "rsi_at_entry", "ema_position", "session", "day_of_week",
    ]
    missing = [k for k in required if k not in trade_data]
    if missing:
        raise ValueError(f"log_trade: missing required fields: {missing}")

    trade_id = str(uuid.uuid4())[:8].upper()

    record = {
        "trade_id":         trade_id,
        "datetime":         trade_data.get("datetime", _gst_now()),
        "symbol":           str(trade_data["symbol"]).upper(),
        "direction":        str(trade_data["direction"]).lower(),
        "entry_price":      float(trade_data["entry_price"]),
        "stop_loss":        float(trade_data["stop_loss"]),
        "take_profit":      float(trade_data["take_profit"]),
        "pattern_name":     str(trade_data["pattern_name"]),
        "confidence_score": float(trade_data["confidence_score"]),
        "news_sentiment":   str(trade_data["news_sentiment"]),
        "rsi_at_entry":     float(trade_data.get("rsi_at_entry", 0)),
        "ema_position":     str(trade_data.get("ema_position", "unknown")),
        "session":          str(trade_data["session"]),
        "day_of_week":      str(trade_data["day_of_week"]),
        "status":           "open",
        # Filled by close_trade:
        "exit_price":       None,
        "exit_reason":      None,
        "result":           None,
        "pips":             None,
        "r_multiple":       None,
        "closed_datetime":  None,
    }

    trades = _load_trade_log()
    trades.append(record)
    _save_trade_log(trades)

    print(f"  ✅ Trade logged  —  ID: {trade_id}  |  {record['symbol']} {record['direction'].upper()}"
          f"  @  {record['entry_price']}  |  SL {record['stop_loss']}  TP {record['take_profit']}")
    return trade_id


# ═══════════════════════════════════════════════════════════════════════════════
# 2. close_trade
# ═══════════════════════════════════════════════════════════════════════════════

def close_trade(trade_id: str, exit_price: float, exit_reason: str = "") -> dict | None:
    """
    Mark an open trade as closed, calculate result metrics, and update rules.

    Args:
        trade_id:    The 8-char ID returned by log_trade().
        exit_price:  The price at which the trade was closed.
        exit_reason: e.g. 'TP hit', 'SL hit', 'manual close'.

    Returns:
        The updated trade record, or None if trade_id not found.
    """
    trades = _load_trade_log()
    target = next((t for t in trades if t["trade_id"] == trade_id), None)

    if target is None:
        print(f"  [ERROR] Trade ID {trade_id} not found.")
        return None
    if target["status"] == "closed":
        print(f"  [WARN] Trade {trade_id} is already closed.")
        return target

    entry     = float(target["entry_price"])
    sl        = float(target["stop_loss"])
    tp        = float(target["take_profit"])
    direction = target["direction"]
    exit_p    = float(exit_price)

    # Pip calculation
    if direction == "long":
        raw_pips = (exit_p - entry) / PIP_SIZE
        sl_pips  = (entry  - sl)    / PIP_SIZE   # distance to SL
    else:
        raw_pips = (entry - exit_p) / PIP_SIZE
        sl_pips  = (sl    - entry)  / PIP_SIZE

    result    = "WIN" if raw_pips > 0 else "LOSS"
    r_mult    = round(raw_pips / sl_pips, 2) if sl_pips > 0 else 0.0

    target.update({
        "exit_price":      round(exit_p, 2),
        "exit_reason":     exit_reason or ("TP hit" if result == "WIN" else "SL hit"),
        "result":          result,
        "pips":            round(raw_pips, 1),
        "r_multiple":      r_mult,
        "closed_datetime": _gst_now(),
        "status":          "closed",
    })

    _save_trade_log(trades)

    icon = "🟢" if result == "WIN" else "🔴"
    print(f"\n  {icon} Trade {trade_id} CLOSED  —  {result}")
    print(f"     Entry: {entry}  →  Exit: {exit_p}")
    print(f"     Pips : {'+' if raw_pips >= 0 else ''}{raw_pips:.1f}  |  R: {r_mult:+.2f}")
    print(f"     Reason: {target['exit_reason']}")

    # Immediately update rule confidence
    update_rule_confidence(target["pattern_name"], result)

    return target


# ═══════════════════════════════════════════════════════════════════════════════
# 3. update_rule_confidence
# ═══════════════════════════════════════════════════════════════════════════════

def update_rule_confidence(pattern_name: str, result: str) -> None:
    """
    Re-calculate live win rate for a rule and adjust its confidence score.

    Args:
        pattern_name: Matches the 'name' or 'pattern_name' field in rules.json.
        result:       "WIN" or "LOSS".
    """
    rules = _load_rules()
    trades = _load_trade_log()

    # Find matching rule (case-insensitive substring match)
    pn_lower = pattern_name.lower()
    rule = next(
        (r for r in rules
         if pn_lower in r.get("name", "").lower()
         or pn_lower in r.get("pattern_name", "").lower()),
        None,
    )

    if rule is None:
        print(f"  [INFO] Rule '{pattern_name}' not found in rules.json — no update made.")
        return

    # All closed trades using this pattern
    pattern_trades = [
        t for t in trades
        if (pn_lower in t.get("pattern_name", "").lower()
            and t["status"] == "closed")
    ]

    total = len(pattern_trades)
    wins  = sum(1 for t in pattern_trades if t["result"] == "WIN")
    live_wr = round(wins / total * 100, 1) if total > 0 else None

    # Backtest win rate stored in rule (may be in 'backtest' sub-dict)
    bt_wr = None
    if "backtest" in rule and isinstance(rule["backtest"], dict):
        bt_wr_raw = rule["backtest"].get("win_rate", None)
        if bt_wr_raw is not None:
            # Could be 0-1 float or 0-100 float
            bt_wr = float(bt_wr_raw) * 100 if float(bt_wr_raw) <= 1 else float(bt_wr_raw)
    if bt_wr is None:
        bt_wr_raw = rule.get("win_rate") or rule.get("success_rate")
        if bt_wr_raw is not None:
            bt_wr = float(bt_wr_raw) * 100 if float(bt_wr_raw) <= 1 else float(bt_wr_raw)

    # Adjust confidence
    old_conf = float(rule.get("confidence_score", rule.get("confidence", 5)) or 5)
    new_conf = old_conf

    if live_wr is not None and bt_wr is not None:
        diff = live_wr - bt_wr
        if diff > 10:
            new_conf = min(10, old_conf + 1)
            direction_label = "↑ increased"
        elif diff < -10:
            new_conf = max(1, old_conf - 1)
            direction_label = "↓ decreased"
        else:
            direction_label = "→ unchanged"

        note = (f"Live performance: {live_wr}% vs Backtest: {bt_wr:.1f}%  "
                f"({total} live trades)")
    elif live_wr is not None:
        note = f"Live performance: {live_wr}% ({total} live trades, no backtest baseline)"
        direction_label = "→ unchanged"
    else:
        note = "No live trades yet."
        direction_label = "→ unchanged"

    # Store updates back into rule
    rule["live_win_rate"]          = live_wr
    rule["live_trade_count"]       = total
    rule["live_performance_note"]  = note
    if "confidence_score" in rule:
        rule["confidence_score"] = new_conf
    elif "confidence" in rule:
        rule["confidence"] = new_conf
    else:
        rule["confidence_score"] = new_conf

    _save_rules(rules)
    print(f"  📊 Rule updated: '{rule.get('name', pattern_name)}'  |  "
          f"Live WR: {live_wr}%  |  Confidence {direction_label} ({old_conf} → {new_conf})")
    print(f"     Note: {note}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. discover_new_patterns
# ═══════════════════════════════════════════════════════════════════════════════

def discover_new_patterns(df=None) -> int:
    """
    Analyse all winning trades, find common pre-conditions, ask Claude to
    identify new rules not already in rules.json, and add them.

    Args:
        df: Optional DataFrame (not used in analysis, reserved for future use).

    Returns:
        Number of new patterns added.
    """
    trades = _load_trade_log()
    winners = [t for t in trades if t.get("result") == "WIN"]

    if len(winners) < 3:
        print(f"  [INFO] Only {len(winners)} winning trade(s) — need at least 3 to discover patterns.")
        return 0

    # Build a concise summary of winning trade conditions
    conditions = []
    for t in winners:
        conditions.append({
            "date":             t.get("datetime", ""),
            "symbol":           t.get("symbol", ""),
            "direction":        t.get("direction", ""),
            "rsi_at_entry":     t.get("rsi_at_entry", ""),
            "ema_position":     t.get("ema_position", ""),
            "session":          t.get("session", ""),
            "day_of_week":      t.get("day_of_week", ""),
            "news_sentiment":   t.get("news_sentiment", ""),
            "pattern_triggered":t.get("pattern_name", ""),
            "pips_gained":      t.get("pips", ""),
            "r_multiple":       t.get("r_multiple", ""),
        })

    rules = _load_rules()
    existing_names = [r.get("name", "") for r in rules]

    prompt = f"""You are a professional trading system analyst.
Below are the conditions present before every winning trade in a real trading account
(symbol: XAUUSD / Gold, MT5, Gulf Standard Time).

WINNING TRADE CONDITIONS ({len(winners)} trades):
{json.dumps(conditions, indent=2)}

EXISTING RULE NAMES (do NOT duplicate these):
{json.dumps(existing_names, indent=2)}

Identify any repeating patterns, edge cases, or high-probability setups
NOT already covered by the existing rules.

Return ONLY a valid JSON array of new rule objects.
Each object must have these exact keys:
  name, description, asset, timeframe, entry_condition,
  stop_loss, take_profit, win_rate, confidence_score,
  source, notes

Set source to "self_discovered" for every new rule.
Return [] if no genuinely new patterns are found.
Return no text outside the JSON array."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])

        new_rules = json.loads(raw)
        if not isinstance(new_rules, list):
            new_rules = []
    except Exception as e:
        print(f"  [ERROR] Claude API call failed: {e}")
        return 0

    # Merge new rules (avoid name duplicates)
    existing_lower = {r.get("name", "").lower() for r in rules}
    added = []
    for r in new_rules:
        if r.get("name", "").lower() not in existing_lower:
            r["source"] = "self_discovered"
            r["discovered_date"] = _gst_now()
            rules.append(r)
            added.append(r)

    if added:
        _save_rules(rules)

    print(f"\n  🔍 Discovered {len(added)} new pattern(s) from your trading history.")
    for r in added:
        print(f"     ✦ {r.get('name', 'Unnamed')} — {r.get('description', '')[:80]}")

    return len(added)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. weekly_performance_report
# ═══════════════════════════════════════════════════════════════════════════════

def weekly_performance_report() -> None:
    """
    Print a full weekly learning report covering trade performance,
    rule updates, newly discovered patterns, and recommendations.
    """
    trades = _load_trade_log()
    week_start = _week_start()

    # Filter to this week's closed trades
    def _in_this_week(t: dict) -> bool:
        raw = t.get("closed_datetime") or t.get("datetime", "")
        try:
            dt = datetime.fromisoformat(str(raw).replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= week_start
        except Exception:
            return False

    week_trades = [t for t in trades if t.get("status") == "closed" and _in_this_week(t)]
    total       = len(week_trades)
    wins        = [t for t in week_trades if t["result"] == "WIN"]
    losses      = [t for t in week_trades if t["result"] == "LOSS"]
    win_rate    = round(len(wins) / total * 100, 1) if total else 0
    total_pips  = sum(t.get("pips", 0) or 0 for t in week_trades)

    # Per-pattern stats
    pattern_stats: dict[str, dict] = {}
    for t in trades:   # use ALL closed trades for live win rate
        if t.get("status") != "closed":
            continue
        pn = t.get("pattern_name", "unknown")
        if pn not in pattern_stats:
            pattern_stats[pn] = {"wins": 0, "total": 0}
        pattern_stats[pn]["total"] += 1
        if t["result"] == "WIN":
            pattern_stats[pn]["wins"] += 1

    for pn, s in pattern_stats.items():
        s["win_rate"] = round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0

    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)
    best_pattern  = sorted_patterns[0]  if sorted_patterns else None
    worst_pattern = sorted_patterns[-1] if sorted_patterns else None

    # Rule updates this week
    rules = _load_rules()
    updated_rules = [r for r in rules if r.get("live_trade_count", 0) > 0]
    improved  = [r for r in updated_rules
                 if (r.get("live_win_rate") or 0) > (r.get("backtest_win_rate") or 0)]
    degraded  = [r for r in updated_rules
                 if (r.get("live_win_rate") or 0) < (r.get("backtest_win_rate") or 0)]

    # Newly discovered patterns
    new_patterns = [r for r in rules if r.get("source") == "self_discovered"]

    # Best session this week
    session_wins: dict[str, int] = {}
    for t in week_trades:
        if t["result"] == "WIN":
            s = t.get("session", "unknown")
            session_wins[s] = session_wins.get(s, 0) + 1
    best_session = max(session_wins, key=lambda k: session_wins[k]) if session_wins else "N/A"

    # Best day this week
    day_wins: dict[str, int] = {}
    for t in week_trades:
        if t["result"] == "WIN":
            d = t.get("day_of_week", "unknown")
            day_wins[d] = day_wins.get(d, 0) + 1
    best_day = max(day_wins, key=lambda k: day_wins[k]) if day_wins else "N/A"

    # ── Print ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("=== WEEKLY LEARNING REPORT ===")
    print("=" * 55)

    print(f"\n  TRADES THIS WEEK : {total}")
    print(f"  WIN RATE         : {win_rate}%  ({len(wins)}W / {len(losses)}L)")
    sign = "+" if total_pips >= 0 else ""
    print(f"  PROFIT / LOSS    : {sign}{total_pips:.1f} pips")

    if best_pattern:
        bp = best_pattern
        print(f"  BEST PATTERN     : {bp[0]}  ({bp[1]['win_rate']}% live, {bp[1]['total']} trades)")
    if worst_pattern and worst_pattern != best_pattern:
        wp = worst_pattern
        print(f"  WORST PATTERN    : {wp[0]}  ({wp[1]['win_rate']}% live, {wp[1]['total']} trades)")

    print(f"\n  RULES UPDATED    : {len(updated_rules)}")
    print(f"  Rules improved   : {len(improved)} patterns")
    print(f"  Rules degraded   : {len(degraded)} patterns")
    for r in improved[:3]:
        print(f"     ↑ {r.get('name', '?')}  live {r.get('live_win_rate')}%")
    for r in degraded[:3]:
        print(f"     ↓ {r.get('name', '?')}  live {r.get('live_win_rate')}%")

    print(f"\n  NEW PATTERNS DISCOVERED : {len(new_patterns)}")
    for r in new_patterns[:5]:
        print(f"     ✦ {r.get('name', 'Unnamed')}: {str(r.get('description', ''))[:70]}")

    print("\n  RECOMMENDATIONS:")
    if best_pattern:
        print(f"  Focus on  : {best_pattern[0]}")
    if worst_pattern and worst_pattern != best_pattern:
        print(f"  Avoid     : {worst_pattern[0]}")
    print(f"  Best session : {best_session}")
    print(f"  Best day     : {best_day}")
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. log_todays_trade  (interactive terminal input)
# ═══════════════════════════════════════════════════════════════════════════════

def log_todays_trade() -> str | None:
    """
    Interactively ask Ali for trade details and log the trade.
    Returns the trade_id on success, None on cancellation.
    """
    print("\n" + "=" * 55)
    print("  LOG TODAY'S TRADE")
    print("=" * 55)

    def _ask(prompt: str, default: str = "") -> str:
        val = input(f"  {prompt}").strip()
        return val if val else default

    def _ask_float(prompt: str, default: float = 0.0) -> float:
        while True:
            raw = input(f"  {prompt}").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                print("    ⚠  Please enter a valid number.")

    symbol     = _ask("Enter symbol traded (e.g. XAUUSD): ", "XAUUSD").upper()
    direction  = ""
    while direction not in ("long", "short"):
        direction = _ask("Enter direction (long/short): ").lower()
        if direction not in ("long", "short"):
            print("    ⚠  Please type 'long' or 'short'.")

    entry_price = _ask_float("Enter entry price: ")
    stop_loss   = _ask_float("Enter stop loss: ")
    take_profit = _ask_float("Enter take profit: ")
    pattern     = _ask("Which pattern triggered this trade?: ", "manual")

    # Optional enrichment
    rsi         = _ask_float("RSI at entry (press Enter to skip): ", 0.0)
    ema_pos     = _ask("EMA position — above/below EMA50/200 (or press Enter): ", "unknown")
    sentiment   = _ask("News sentiment at time of trade (bullish/bearish/neutral): ", "neutral")
    confidence  = _ask_float("Confidence score 1–10 (press Enter to skip): ", 5.0)

    now_utc = datetime.now(timezone.utc)
    session_hour = now_utc.hour
    if 7 <= session_hour < 16:
        session = "London"
    elif 13 <= session_hour < 22:
        session = "NY"
    else:
        session = "Asian"

    trade_data = {
        "symbol":           symbol,
        "direction":        direction,
        "entry_price":      entry_price,
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "pattern_name":     pattern,
        "confidence_score": confidence,
        "news_sentiment":   sentiment,
        "rsi_at_entry":     rsi,
        "ema_position":     ema_pos,
        "session":          session,
        "day_of_week":      now_utc.strftime("%A"),
        "datetime":         _gst_now(),
    }

    print()
    trade_id = log_trade(trade_data)
    print(f"\n  Trade saved with ID: {trade_id}")
    print(f"  Run close_trade('{trade_id}', exit_price) when trade closes.\n")
    return trade_id


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_todays_trade()
