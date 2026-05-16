"""VIX-based macro regime classification — Issue #6.

Fetches the live VIX level (^VIX) and maps it to a regime bucket that
scales position sizing. Thresholds from professional practice (research
file Section 8):

  VIX < 15     Low vol      scalar 1.00 (full size)
  15 ≤ VIX < 25  Normal     scalar 0.85
  25 ≤ VIX < 35  Elevated   scalar 0.60
  VIX ≥ 35     Crisis       scalar 0.25 (or avoid)
"""

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


def _fetch_vix() -> float | None:
    """Return the current ^VIX level, or None if unavailable."""
    if _YQ_AVAILABLE:
        try:
            t = YQTicker("^VIX")
            q = t.quote
            if isinstance(q, dict):
                row = q.get("^VIX") or (list(q.values())[0] if q else None)
                if isinstance(row, dict):
                    v = row.get("regularMarketPrice") or row.get("previousClose")
                    if v is not None:
                        return float(v)
        except Exception:
            pass
    if _YF_AVAILABLE:
        try:
            df = yf.Ticker("^VIX").history(period="5d", interval="1d")
            if df is not None and not df.empty:
                return float(df["Close"].dropna().iloc[-1])
        except Exception:
            pass
    return None


def classify_vix(vix: float) -> tuple[str, float]:
    """Return (regime_label, position_size_scalar)."""
    if vix < 15:
        return "Low vol", 1.00
    if vix < 25:
        return "Normal", 0.85
    if vix < 35:
        return "Elevated", 0.60
    return "Crisis", 0.25


def get_vix_regime() -> dict:
    """Return {'vix': float|None, 'regime': str, 'scalar': float}."""
    vix = _fetch_vix()
    if vix is None:
        return {"vix": None, "regime": "N/A", "scalar": 1.00}
    label, scalar = classify_vix(vix)
    return {"vix": vix, "regime": label, "scalar": scalar}


def vix_display(regime: dict) -> str:
    """One-line summary for report headers."""
    if regime["vix"] is None:
        return "VIX: N/A  |  Regime: N/A  |  Size scalar: 1.00×"
    return (f"VIX: {regime['vix']:.1f}  |  Regime: {regime['regime']}  |  "
            f"Size scalar: {regime['scalar']:.2f}×")
