"""Technical levels — 52-week high/low, 200-day SMA, key strike proximity.

Issue #7. All data sourced from the same 2-year daily closes already used
by the realized-vol calculation; no extra API calls when called alongside
the existing fetch.
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


def _fetch_closes(ticker: str):
    """Return a pandas Series of daily closes for the past year."""
    if not _PD_AVAILABLE:
        return None
    ticker = ticker.upper().strip()
    if _YQ_AVAILABLE:
        try:
            df = YQTicker(ticker).history(period="1y", interval="1d")
            if df is not None and not df.empty:
                col = "close" if "close" in df.columns else "Close"
                if col not in df.columns and df.columns.any():
                    col = df.columns[0]
                closes = df[col].dropna()
                if isinstance(closes.index, pd.DatetimeIndex):
                    closes = closes.sort_index()
                if len(closes) >= 30:
                    return closes
        except Exception:
            pass
    if _YF_AVAILABLE:
        try:
            df = yf.Ticker(ticker).history(period="1y", interval="1d")
            if df is not None and not df.empty:
                col = "Close" if "Close" in df.columns else "close"
                closes = df[col].dropna()
                if isinstance(closes.index, pd.DatetimeIndex):
                    closes = closes.sort_index()
                if len(closes) >= 30:
                    return closes
        except Exception:
            pass
    return None


def compute_technicals(ticker: str, spot: float) -> dict | None:
    """Return key technical levels for the ticker.

    Result:
        {
          'high_52w': float,
          'low_52w': float,
          'sma_200': float|None,
          'pct_from_high': float,   # negative if below high
          'pct_from_low': float,    # positive if above low
          'sma_relative_pct': float|None,  # spot vs SMA
        }
    """
    closes = _fetch_closes(ticker)
    if closes is None or len(closes) < 30:
        return None
    arr = closes.astype(float).values
    high_52w = float(np.max(arr))
    low_52w = float(np.min(arr))
    sma_200 = float(np.mean(arr[-200:])) if len(arr) >= 200 else None
    return {
        "high_52w": high_52w,
        "low_52w": low_52w,
        "sma_200": sma_200,
        "pct_from_high": (spot / high_52w - 1) * 100,
        "pct_from_low": (spot / low_52w - 1) * 100,
        "sma_relative_pct": (spot / sma_200 - 1) * 100 if sma_200 else None,
    }


def _round_increment(spot: float) -> float:
    """Standard option strike increments: $1 under $25, $2.50 to $100, $5 to $200, $10 above."""
    if spot < 25:
        return 1.0
    if spot < 100:
        return 2.50
    if spot < 200:
        return 5.0
    return 10.0


def nearest_round_strikes(spot: float, n: int = 3) -> list[float]:
    """Round-number strikes near spot — high-OI gravity points."""
    inc = _round_increment(spot)
    base = round(spot / inc) * inc
    return sorted({base + i * inc for i in range(-n, n + 1)})


def strike_proximity_flags(strike: float, spot: float, tech: dict,
                            dte: int, tolerance_pct: float = 2.0) -> list[str]:
    """Return list of warning labels if the strike sits on a key technical level."""
    flags = []
    if tech is None:
        return flags
    tol = tolerance_pct / 100.0

    if abs(strike / tech["high_52w"] - 1) <= tol:
        flags.append("AT 52W HIGH")
    if abs(strike / tech["low_52w"] - 1) <= tol:
        flags.append("AT 52W LOW")
    if tech["sma_200"] is not None and abs(strike / tech["sma_200"] - 1) <= tol:
        flags.append("AT 200D SMA")

    rounds = nearest_round_strikes(spot, n=4)
    if any(abs(strike - r) < 0.01 for r in rounds):
        flags.append("ROUND #  (high-OI gravity)")

    if dte <= 5:
        flags.append("PIN RISK  (≤5 DTE)")

    return flags


def print_technicals_block(tech: dict | None, spot: float, strike: float,
                            dte: int) -> None:
    """Print the technicals block for the deep-analysis report."""
    W = 66
    print(f"\n{'━'*W}")
    print("  📍  TECHNICAL LEVELS")
    print(f"{'━'*W}")
    if tech is None:
        print("  No price history available.")
        return

    rows = [
        ("52-Week High",    f"${tech['high_52w']:>8.2f}  ({tech['pct_from_high']:+.1f}% from spot)"),
        ("52-Week Low",     f"${tech['low_52w']:>8.2f}  ({tech['pct_from_low']:+.1f}% from spot)"),
    ]
    if tech["sma_200"] is not None:
        sma_arrow = "↑ above" if tech["sma_relative_pct"] > 0 else "↓ below"
        rows.append((
            "200-Day SMA",
            f"${tech['sma_200']:>8.2f}  ({tech['sma_relative_pct']:+.1f}%, {sma_arrow})",
        ))
    rows.append(("Strike", f"${strike:>8.2f}"))
    rounds = nearest_round_strikes(spot, n=2)
    rows.append(("Nearby round #s", ", ".join(f"${r:.2f}" for r in rounds)))

    for k, v in rows:
        print(f"  {k:<18} {v}")

    flags = strike_proximity_flags(strike, spot, tech, dte)
    if flags:
        print(f"\n  ⚠  Strike proximity flags: {' | '.join(flags)}")
