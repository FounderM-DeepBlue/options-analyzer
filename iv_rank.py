"""IV Rank (IVR) and IV Percentile (IVP) — Issue #1.

True IVR/IVP requires a year of daily IV snapshots, which Yahoo Finance
does not provide for free. We approximate using a 252-day rolling 20-day
realized-vol series as the IV proxy distribution. This matches the method
used by retail platforms without paid data feeds (e.g. tastytrade's
free tier). Results are labelled as 'HV-proxy' so the approximation is clear.

Interpretation (same thresholds used by professional platforms):
  IVR > 50  →  IV elevated vs. recent range  →  favors short premium
  IVR < 30  →  IV compressed                 →  favors long vega / debit
  30–50     →  neutral

IVP > 50  means current IV is above the median of the past year.
"""

import numpy as np

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

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

_LOOKBACK = 252   # trading days (~1 year)
_ROLL_WIN = 20    # rolling window for HV (standard short-term vol)


def _fetch_closes(ticker: str):
    """Return a pandas Series of daily closes for the past 2 years."""
    if not _PD_AVAILABLE:
        return None
    ticker = ticker.upper().strip()
    if _YQ_AVAILABLE:
        try:
            t = YQTicker(ticker)
            df = t.history(period="2y", interval="1d")
            if df is not None and not df.empty:
                col = "close" if "close" in df.columns else "Close"
                if col not in df.columns and df.columns.any():
                    col = df.columns[0]
                closes = df[col].dropna()
                if isinstance(closes.index, pd.DatetimeIndex):
                    closes = closes.sort_index()
                if len(closes) >= _LOOKBACK:
                    return closes
        except Exception:
            pass
    if _YF_AVAILABLE:
        try:
            df = yf.Ticker(ticker).history(period="2y", interval="1d")
            if df is not None and not df.empty:
                col = "Close" if "Close" in df.columns else "close"
                closes = df[col].dropna()
                if isinstance(closes.index, pd.DatetimeIndex):
                    closes = closes.sort_index()
                if len(closes) >= _LOOKBACK:
                    return closes
        except Exception:
            pass
    return None


def _rolling_hv_series(closes) -> "np.ndarray | None":
    """Annualized 20-day realized vol at each trading day over the past 252 days."""
    arr = closes.astype(float).values
    if len(arr) < _LOOKBACK + _ROLL_WIN:
        return None
    # Use the last (LOOKBACK + ROLL_WIN) prices so we get exactly LOOKBACK hv values
    window = arr[-(  _LOOKBACK + _ROLL_WIN):]
    log_ret = np.log(window[1:] / window[:-1])
    # rolling std over ROLL_WIN days, annualized
    hv_series = np.array([
        np.std(log_ret[i: i + _ROLL_WIN]) * np.sqrt(252)
        for i in range(len(log_ret) - _ROLL_WIN + 1)
    ])
    return hv_series  # length = LOOKBACK


def compute_iv_rank(ticker: str, current_iv: float) -> tuple[float | None, float | None, str]:
    """Return (IVR, IVP, bias_label).

    IVR = (current_iv - 52w_low) / (52w_high - 52w_low) * 100  [0-100]
    IVP = % of past-year daily HV values below current_iv        [0-100]
    bias_label = 'Short Premium' | 'Long Premium / Debit' | 'Neutral'

    Returns (None, None, 'N/A') if insufficient data.
    """
    closes = _fetch_closes(ticker)
    if closes is None:
        return None, None, "N/A"
    hv_series = _rolling_hv_series(closes)
    if hv_series is None or len(hv_series) < 20:
        return None, None, "N/A"

    # Use most recent LOOKBACK values (already the case from _rolling_hv_series)
    hist = hv_series[-_LOOKBACK:]
    lo, hi = hist.min(), hist.max()

    if hi <= lo:
        return None, None, "N/A"

    ivr = float(np.clip((current_iv - lo) / (hi - lo) * 100, 0, 100))
    ivp = float(np.mean(hist < current_iv) * 100)

    if ivr >= 50:
        label = "Short Premium"
    elif ivr <= 30:
        label = "Long Premium / Debit"
    else:
        label = "Neutral"

    return ivr, ivp, label


def ivr_display(ivr: float | None, ivp: float | None, label: str) -> str:
    """One-line summary for report headers."""
    if ivr is None:
        return "IVR: N/A  |  IVP: N/A  |  Bias: N/A (insufficient history)"
    return (f"IVR: {ivr:.0f}/100  |  IVP: {ivp:.0f}th pctile  |  "
            f"Bias: {label}  (HV-proxy, 252-day)")
