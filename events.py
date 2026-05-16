"""Earnings dates and other corporate events.

Used to flag when an option expiration spans an upcoming earnings announcement
(issue #3 — earnings awareness drives IV crush and implied-move analysis).
"""

from datetime import date, datetime

try:
    from yahooquery import Ticker as YQTicker
    _YQ_AVAILABLE = True
except ImportError:
    _YQ_AVAILABLE = False


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        s = str(value)[:10]
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def fetch_next_earnings_date(ticker: str) -> date | None:
    """Return the next upcoming earnings date, or None if unavailable."""
    if not _YQ_AVAILABLE:
        return None
    ticker = ticker.upper().strip()
    try:
        t = YQTicker(ticker)
        cal = getattr(t, "calendar_events", None)
        if not isinstance(cal, dict):
            return None
        row = cal.get(ticker) or cal.get(ticker.lower()) or (list(cal.values())[0] if cal else None)
        if not isinstance(row, dict):
            return None
        earnings = row.get("earnings") or {}
        dates = earnings.get("earningsDate") or []
        if isinstance(dates, (list, tuple)) and dates:
            today = date.today()
            candidates = [_as_date(d) for d in dates]
            future = sorted([d for d in candidates if d and d >= today])
            return future[0] if future else None
        return _as_date(dates) if dates else None
    except Exception:
        return None


def earnings_inside_window(earnings_date: date | None, expiry_date: date) -> bool:
    """True if the next earnings announcement falls on or before the option expiration."""
    if earnings_date is None:
        return False
    return earnings_date <= expiry_date
