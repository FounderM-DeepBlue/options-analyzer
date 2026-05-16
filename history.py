"""Historical price data and realized volatility computation."""

import numpy as np

try:
    from yahooquery import Ticker as YQTicker
    _YQ_AVAILABLE = True
except ImportError:
    _YQ_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


def lookback_for_tenor(days_to_expiry: int) -> tuple[int, str]:
    """Choose lookback trading days and label from option tenor."""
    if days_to_expiry <= 45:
        return 30, "30-day (short-dated)"
    if days_to_expiry <= 90:
        return 60, "60-day (near-term)"
    if days_to_expiry <= 180:
        return 90, "90-day (options benchmark)"
    if days_to_expiry <= 400:
        return 252, "1-year (LEAPS)"
    return 504, "2-year (smoothed)"


def _period_for_lookback(lookback_trading_days: int) -> str:
    if lookback_trading_days <= 30:
        return "3mo"
    if lookback_trading_days <= 90:
        return "6mo"
    if lookback_trading_days <= 252:
        return "1y"
    return "2y"


def _closes_via_yq(ticker: str, period: str):
    import pandas as pd
    if not _YQ_AVAILABLE:
        return None
    try:
        t = YQTicker(ticker)
        df = t.history(period=period, interval="1d")
        if df is None or df.empty:
            return None
        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns and len(df.columns):
            close_col = df.columns[0]
        if close_col not in df.columns:
            return None
        closes = df[close_col].dropna()
        if isinstance(closes.index, pd.DatetimeIndex):
            closes = closes.sort_index()
        return closes
    except Exception:
        return None


def _closes_via_yf(ticker: str, period: str):
    import pandas as pd
    if not _YF_AVAILABLE:
        return None
    try:
        yt = yf.Ticker(ticker)
        df = yt.history(period=period, interval="1d")
        if df is None or df.empty:
            return None
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            return None
        closes = df[close_col].dropna()
        if isinstance(closes.index, pd.DatetimeIndex):
            closes = closes.sort_index()
        return closes
    except Exception:
        return None


def fetch_realized_vol(ticker: str, lookback_trading_days: int) -> tuple[float | None, str]:
    """Annualized realized vol from log returns over the lookback window."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return None, "pandas not installed (pip install pandas)"
    ticker = ticker.upper().strip()
    period = _period_for_lookback(lookback_trading_days)
    closes = _closes_via_yq(ticker, period)
    if closes is None or len(closes) < 2:
        closes = _closes_via_yf(ticker, period)
    if closes is None or len(closes) < 2:
        return None, "could not fetch enough price history"
    closes = closes.iloc[-lookback_trading_days:] if len(closes) >= lookback_trading_days else closes
    if len(closes) < 2:
        return None, "not enough trading days in history"
    log_returns = np.log(closes.astype(float).values[1:] / closes.astype(float).values[:-1])
    rv = float(np.std(log_returns) * np.sqrt(252))
    if rv <= 0 or not np.isfinite(rv):
        return None, "realized vol computation failed"
    return rv, f"{lookback_trading_days}-day"
