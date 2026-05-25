"""
world_sessions.py — World Market Hours in UAE (GST, UTC+4)
──────────────────────────────────────────────────────────
All times are UAE / Gulf Standard Time (UTC+4).
NY session uses DST-aware windows:
  • Nov-Mar (winter):  18:30 – 02:00 UAE
  • Apr-Oct (summer):  17:30 – 01:00 UAE
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

# ── UAE timezone ───────────────────────────────────────────────────────────────
_GST = timezone(timedelta(hours=4))


def _now_uae() -> datetime:
    return datetime.now(_GST)


# ── Session definitions ────────────────────────────────────────────────────────
# Each session: { "label", "open_h", "open_m", "close_h", "close_m",
#                 "quality", "months" (optional – None means all months) }
# close < open  →  session crosses midnight in UAE time
_SESSIONS_BASE: list[dict[str, Any]] = [
    {
        "key":     "tokyo",
        "label":   "Tokyo",
        "flag":    "🇯🇵",
        "open_h":  4,  "open_m":  0,
        "close_h": 10, "close_m": 0,
        "quality": "LOW",
        "months":  None,
    },
    {
        "key":     "hongkong",
        "label":   "Hong Kong",
        "flag":    "🇭🇰",
        "open_h":  5,  "open_m":  30,
        "close_h": 12, "close_m": 0,
        "quality": "LOW",
        "months":  None,
    },
    {
        "key":     "shanghai",
        "label":   "Shanghai",
        "flag":    "🇨🇳",
        "open_h":  5,  "open_m":  30,
        "close_h": 12, "close_m": 0,
        "quality": "LOW",
        "months":  None,
    },
    {
        "key":     "london",
        "label":   "London",
        "flag":    "🇬🇧",
        "open_h":  12, "open_m": 0,
        "close_h": 20, "close_m": 30,
        "quality": "HIGH",
        "months":  None,
    },
    # NY winter:  Nov-Mar  →  18:30–02:00 UAE (crosses midnight)
    {
        "key":     "newyork",
        "label":   "New York",
        "flag":    "🇺🇸",
        "open_h":  18, "open_m": 30,
        "close_h": 2,  "close_m": 0,
        "quality": "HIGH",
        "months":  [11, 12, 1, 2, 3],
    },
    # NY summer:  Apr-Oct  →  17:30–01:00 UAE (crosses midnight)
    {
        "key":     "newyork",
        "label":   "New York",
        "flag":    "🇺🇸",
        "open_h":  17, "open_m": 30,
        "close_h": 1,  "close_m": 0,
        "quality": "HIGH",
        "months":  [4, 5, 6, 7, 8, 9, 10],
    },
]


def _get_sessions_for_month(month: int) -> list[dict[str, Any]]:
    """Return the correct session list for a given calendar month."""
    result = []
    seen_keys: set[str] = set()
    for s in _SESSIONS_BASE:
        if s["months"] is None or month in s["months"]:
            # For the same key (newyork), keep only the first match per month
            if s["key"] not in seen_keys:
                result.append(s)
                seen_keys.add(s["key"])
    return result


def _is_session_open(s: dict[str, Any], now: datetime) -> bool:
    """Return True if *now* (UAE datetime) falls within session trading hours."""
    open_min  = s["open_h"]  * 60 + s["open_m"]
    close_min = s["close_h"] * 60 + s["close_m"]
    current   = now.hour * 60 + now.minute

    if close_min > open_min:
        # Normal window (no midnight crossover)
        return open_min <= current < close_min
    else:
        # Crosses midnight
        return current >= open_min or current < close_min


def _minutes_until_open(s: dict[str, Any], now: datetime) -> int:
    """Minutes from *now* (UAE) until the session opens."""
    open_min  = s["open_h"]  * 60 + s["open_m"]
    current   = now.hour * 60 + now.minute
    diff = open_min - current
    if diff < 0:
        diff += 24 * 60
    return diff


def _minutes_until_close(s: dict[str, Any], now: datetime) -> int:
    """Minutes from *now* (UAE) until the session closes."""
    close_min = s["close_h"] * 60 + s["close_m"]
    current   = now.hour * 60 + now.minute

    open_min = s["open_h"] * 60 + s["open_m"]
    if close_min <= open_min:
        # Crosses midnight
        if current >= open_min:
            return close_min + (24 * 60 - current)
        else:
            return close_min - current
    return close_min - current


def _fmt_hm(minutes: int) -> str:
    h, m = divmod(abs(int(minutes)), 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


# ── Public API ─────────────────────────────────────────────────────────────────

def get_active_sessions(now: datetime | None = None) -> list[dict[str, Any]]:
    """
    Return a list of currently-open session dicts.
    Each dict has: key, label, flag, quality, open_h, open_m, close_h, close_m, months.
    Empty list means off-hours.
    """
    if now is None:
        now = _now_uae()
    sessions = _get_sessions_for_month(now.month)
    return [s for s in sessions if _is_session_open(s, now)]


def get_session_summary_line(now: datetime | None = None) -> str:
    """
    One-line string describing the current market situation, e.g.
    '🟢 London + NY Active (best window)'
    Used in sidebar and trade-card headers.
    """
    if now is None:
        now = _now_uae()
    active = get_active_sessions(now)
    keys   = {s["key"] for s in active}

    if "london" in keys and "newyork" in keys:
        return "🟢 London + NY Active (best window)"
    if "newyork" in keys:
        return "🟢 New York Active"
    if "london" in keys:
        return "🟢 London Active"
    asian_keys = {"tokyo", "hongkong", "shanghai"}
    if keys & asian_keys:
        labels = " + ".join(s["label"] for s in active if s["key"] in asian_keys)
        return f"🟡 Asian Hours ({labels})"
    return "🔴 Off-Hours"


def get_next_session(now: datetime | None = None) -> dict[str, Any] | None:
    """
    Return the next session to open (or None if something is already open).
    Result dict has: label, flag, minutes_away, open_time_str (HH:MM UAE).
    """
    if now is None:
        now = _now_uae()
    sessions  = _get_sessions_for_month(now.month)
    closed    = [s for s in sessions if not _is_session_open(s, now)]
    if not closed:
        return None

    next_s = min(closed, key=lambda s: _minutes_until_open(s, now))
    mins   = _minutes_until_open(next_s, now)
    return {
        "label":         next_s["label"],
        "flag":          next_s["flag"],
        "minutes_away":  mins,
        "open_time_str": f"{next_s['open_h']:02d}:{next_s['open_m']:02d} UAE",
    }


def get_full_session_board(now: datetime | None = None) -> str:
    """
    Full ASCII session board for display in the chat (morning briefing / sessions command).
    """
    if now is None:
        now = _now_uae()

    sessions = _get_sessions_for_month(now.month)
    date_str = now.strftime("%A %d %B %Y")
    time_str = now.strftime("%I:%M %p UAE")

    lines = [
        f"╔══ 🌍 WORLD MARKET SESSIONS ══════════════════╗",
        f"  📅 {date_str}   🕐 {time_str}",
        f"",
    ]

    for s in sessions:
        is_open   = _is_session_open(s, now)
        dot       = "🟢" if is_open else "⚪"
        open_str  = f"{s['open_h']:02d}:{s['open_m']:02d}"
        close_str = f"{s['close_h']:02d}:{s['close_m']:02d}"
        quality   = s["quality"]

        if is_open:
            eta = _fmt_hm(_minutes_until_close(s, now))
            status = f"[OPEN — closes in {eta}]"
        else:
            eta = _fmt_hm(_minutes_until_open(s, now))
            status = f"[closed — opens in {eta}]"

        lines.append(
            f"  {dot} {s['flag']} {s['label']:<12} "
            f"{open_str}–{close_str} UAE   {quality:<4}  {status}"
        )

    lines.append("")

    # ── Overall summary ──
    active_keys = {s["key"] for s in sessions if _is_session_open(s, now)}
    if "london" in active_keys and "newyork" in active_keys:
        summary = "⭐ BEST WINDOW — London + NY overlap active"
    elif "newyork" in active_keys:
        summary = "✅ New York session active — high volatility"
    elif "london" in active_keys:
        summary = "✅ London session active — good for setups"
    elif active_keys & {"tokyo", "hongkong", "shanghai"}:
        summary = "🟡 Asian hours — lower volatility, watch for breakouts"
    else:
        summary = "🔴 Off-Hours — all major sessions closed"
        nxt = get_next_session(now)
        if nxt:
            summary += f"\n  ⏳ Next: {nxt['flag']} {nxt['label']} opens in {_fmt_hm(nxt['minutes_away'])} ({nxt['open_time_str']})"

    lines.append(f"  {summary}")
    lines.append(f"╚══════════════════════════════════════════════╝")

    return "\n".join(lines)


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(get_full_session_board())
    print()
    print(get_session_summary_line())
