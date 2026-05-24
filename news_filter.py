"""
news_filter.py - Forex Factory Economic Calendar for TradingBotV1

Fetches today's high-impact economic events from the Forex Factory
RSS calendar feed, converts all times to Gulf Standard Time (UTC+4),
and provides print_todays_schedule() for the morning briefing.
"""

import os
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# Gulf Standard Time offset (UTC+4, no DST)
GST = timezone(timedelta(hours=4))

# Forex Factory calendar RSS (returns today's events)
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

# Impact levels to include (filter out low-impact noise)
HIGH_IMPACT_LEVELS = {"High", "Medium"}

# Keywords that warrant a trading warning
WARNING_KEYWORDS = [
    "fed", "federal reserve", "fomc", "powell",
    "nfp", "non-farm", "non farm", "payroll",
    "cpi", "inflation", "interest rate", "rate decision",
    "ecb", "boe", "boj",
]


def fetch_ff_calendar() -> list[dict]:
    """
    Fetch and parse this week's Forex Factory economic calendar.

    Returns:
        List of event dicts with keys:
            title, country, date, time_utc, time_gst,
            impact, forecast, previous
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    events: list[dict] = []

    try:
        response = requests.get(FF_CALENDAR_URL, headers=headers, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for item in root.findall(".//item"):
            def _t(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            title    = _t("title")
            country  = _t("country")
            date_str = _t("date")      # e.g. "05-17-2026"
            time_str = _t("time")      # e.g. "8:30am"
            impact   = _t("impact")    # "High" | "Medium" | "Low"
            forecast = _t("forecast")
            previous = _t("previous")

            # Parse datetime and convert to GST
            time_utc_str = ""
            time_gst_str = ""
            try:
                dt_str = f"{date_str} {time_str}"
                # Try 12-hour format first, then 24-hour
                for fmt in ("%m-%d-%Y %I:%M%p", "%m-%d-%Y %H:%M"):
                    try:
                        dt_utc = datetime.strptime(dt_str.upper(), fmt.upper())
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        dt_gst = dt_utc.astimezone(GST)
                        time_utc_str = dt_utc.strftime("%H:%M UTC")
                        time_gst_str = dt_gst.strftime("%H:%M GST")
                        break
                    except ValueError:
                        continue
            except Exception:
                time_utc_str = time_str
                time_gst_str = time_str

            events.append({
                "title":    title,
                "country":  country,
                "date":     date_str,
                "time_utc": time_utc_str,
                "time_gst": time_gst_str,
                "impact":   impact,
                "forecast": forecast,
                "previous": previous,
            })

    except requests.RequestException as exc:
        print(f"  [ERROR] Could not fetch Forex Factory calendar: {exc}")
    except ET.ParseError as exc:
        print(f"  [ERROR] Could not parse calendar XML: {exc}")

    return events


def get_todays_events(impact_filter: set[str] | None = None) -> list[dict]:
    """
    Return today's economic events filtered by impact level.

    Args:
        impact_filter: Set of impact strings to include.
                       Defaults to HIGH_IMPACT_LEVELS {"High", "Medium"}.

    Returns:
        List of today's matching event dicts, sorted by GST time.
    """
    if impact_filter is None:
        impact_filter = HIGH_IMPACT_LEVELS

    all_events = fetch_ff_calendar()

    today_str = datetime.now(GST).strftime("%m-%d-%Y")
    todays = [
        e for e in all_events
        if e["date"] == today_str and e["impact"] in impact_filter
    ]

    # Sort by GST time string (HH:MM format sorts correctly lexicographically)
    todays.sort(key=lambda e: e["time_gst"])
    return todays


def check_warnings(events: list[dict]) -> list[str]:
    """
    Scan today's event titles for high-risk keywords.

    Returns:
        List of warning strings (empty if no warnings).
    """
    warnings = []
    for event in events:
        title_lower = event["title"].lower()
        for kw in WARNING_KEYWORDS:
            if kw in title_lower:
                warnings.append(
                    f"⚠  HIGH-RISK EVENT: {event['title']} "
                    f"({event['country']}) at {event['time_gst']}"
                )
                break  # one warning per event
    return warnings


def print_todays_schedule(events: list[dict] | None = None) -> list[str]:
    """
    Print today's high/medium impact economic schedule in GST time.

    Args:
        events: Pre-fetched event list. Fetches fresh data if None.

    Returns:
        List of warning strings (may be empty).
    """
    if events is None:
        events = get_todays_events()

    today_label = datetime.now(GST).strftime("%A %d %B %Y")

    print("\n" + "=" * 50)
    print("=== FOREX FACTORY CALENDAR ===")
    print(f"=== {today_label} ===")
    print("=" * 50)

    if not events:
        print("  No high/medium impact events today.")
        print("=" * 50)
        return []

    # Impact colour markers
    IMPACT_ICON = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}

    for e in events:
        icon     = IMPACT_ICON.get(e["impact"], "  ")
        forecast = f"  Forecast: {e['forecast']}" if e["forecast"] else ""
        previous = f"  Prev: {e['previous']}"     if e["previous"] else ""
        print(
            f"  {icon} {e['time_gst']:<10} [{e['country']:<3}] "
            f"{e['title']}{forecast}{previous}"
        )

    warnings = check_warnings(events)
    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")

    print("=" * 50)
    return warnings


if __name__ == "__main__":
    print_todays_schedule()
