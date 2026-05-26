"""
intelligence/news_filter.py — Forex Factory economic calendar for TradingBotV2.
Ported from V1 news_filter.py — logic unchanged, V1 imports removed.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

GST = timezone(timedelta(hours=4))

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

HIGH_IMPACT_LEVELS = {"High", "Medium"}

WARNING_KEYWORDS = [
    "fed", "federal reserve", "fomc", "powell",
    "nfp", "non-farm", "non farm", "payroll",
    "cpi", "inflation", "interest rate", "rate decision",
    "ecb", "boe", "boj",
]

_FF_RATE_LIMITED_UNTIL: float = 0.0


def fetch_ff_calendar() -> list[dict]:
    """
    Fetch and parse this week's Forex Factory economic calendar.
    Respects 429 rate limiting with a 60-minute cooldown.
    """
    global _FF_RATE_LIMITED_UNTIL

    if time.time() < _FF_RATE_LIMITED_UNTIL:
        remaining = int((_FF_RATE_LIMITED_UNTIL - time.time()) / 60)
        logger.debug("FF rate limited — skipping (%dm remaining)", remaining)
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    events: list[dict] = []

    try:
        resp = requests.get(FF_CALENDAR_URL, headers=headers, timeout=15)
        if resp.status_code == 429:
            _FF_RATE_LIMITED_UNTIL = time.time() + 3600
            logger.warning("FF rate limited (429) — cooldown 60m")
            return []
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for item in root.findall(".//item"):
            def _t(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            title    = _t("title")
            country  = _t("country")
            date_str = _t("date")
            time_str = _t("time")
            impact   = _t("impact")
            forecast = _t("forecast")
            previous = _t("previous")

            time_utc_str = time_gst_str = ""
            try:
                dt_str = f"{date_str} {time_str}"
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
                time_utc_str = time_gst_str = time_str

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
        if hasattr(exc, "response") and exc.response is not None and exc.response.status_code == 429:
            _FF_RATE_LIMITED_UNTIL = time.time() + 3600
        else:
            logger.error("FF calendar fetch error: %s", exc)
    except ET.ParseError as exc:
        logger.error("FF calendar XML parse error: %s", exc)

    return events


def get_todays_events(impact_filter: set[str] | None = None) -> list[dict]:
    """Return today's events filtered by impact level, sorted by GST time."""
    if impact_filter is None:
        impact_filter = HIGH_IMPACT_LEVELS

    all_events = fetch_ff_calendar()
    today_str  = datetime.now(GST).strftime("%m-%d-%Y")
    todays     = [e for e in all_events if e["date"] == today_str and e["impact"] in impact_filter]
    todays.sort(key=lambda e: e["time_gst"])
    return todays


def is_high_impact_window(minutes_before: int = 30, minutes_after: int = 30) -> tuple[bool, str]:
    """
    Return (True, event_name) if a high-impact event is within the window.
    Returns (False, "") otherwise.
    Used by entry checklist to block trades near news.
    """
    events = get_todays_events({"High"})
    now    = datetime.now(GST)
    now_t  = now.hour * 60 + now.minute

    for e in events:
        try:
            hh, mm = map(int, e["time_gst"].replace(" GST", "").split(":"))
            event_t = hh * 60 + mm
            if -minutes_after <= now_t - event_t <= minutes_before:
                return True, e["title"]
        except Exception:
            continue

    return False, ""


def check_warnings(events: list[dict]) -> list[str]:
    warnings = []
    for event in events:
        title_lower = event["title"].lower()
        for kw in WARNING_KEYWORDS:
            if kw in title_lower:
                warnings.append(
                    f"HIGH-RISK EVENT: {event['title']} ({event['country']}) at {event['time_gst']}"
                )
                break
    return warnings


def get_calendar_summary() -> dict:
    """Return today's calendar as a structured dict for the morning briefing."""
    events   = get_todays_events()
    warnings = check_warnings(events)
    return {
        "date":     datetime.now(GST).strftime("%A %d %B %Y"),
        "events":   events,
        "warnings": warnings,
        "has_high_risk": any(e["impact"] == "High" for e in events),
        "count":    len(events),
    }
