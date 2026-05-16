"""Portfolio-level position tracking and aggregate Greeks — Issue #8.

Persists open positions to portfolio.json in the working directory.
Aggregate Greeks are beta-weighted to SPY where beta is available, so
delta/gamma exposure from different tickers can be summed apples-to-apples.

Professional risk thresholds (from research file Section 10):
  Net portfolio vega:  -0.5 to -1.5% of portfolio per 1-vol-point move
  Net portfolio gamma: monitor closely below 21 DTE — gamma dominates P&L
  Beta-weighted delta: aim near zero for market-neutral books
"""

import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

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

PORTFOLIO_FILE = Path("portfolio.json")
RISK_FREE_RATE = 0.043  # matches main.py default; positions revalue at this rate


def _load() -> list[dict]:
    if not PORTFOLIO_FILE.exists():
        return []
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save(positions: list[dict]) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def list_positions() -> list[dict]:
    return _load()


def add_position(
    ticker: str,
    opt_type: str,
    strike: float,
    expiry: date,
    contracts: int,
    entry_premium: float,
    iv_at_entry: float,
    greeks_at_entry: dict,
) -> str:
    positions = _load()
    new_id = str(uuid.uuid4())[:8]
    positions.append({
        "id": new_id,
        "ticker": ticker.upper(),
        "type": opt_type,
        "strike": float(strike),
        "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
        "contracts": int(contracts),
        "entry_premium": float(entry_premium),
        "entry_date": date.today().isoformat(),
        "iv_at_entry": float(iv_at_entry),
        "greeks_at_entry": {k: float(v) for k, v in greeks_at_entry.items()},
    })
    _save(positions)
    return new_id


def remove_position(position_id: str) -> bool:
    positions = _load()
    new_positions = [p for p in positions if p["id"] != position_id]
    if len(new_positions) == len(positions):
        return False
    _save(new_positions)
    return True


def _fetch_spot_and_beta(ticker: str) -> tuple[float | None, float | None]:
    """Return (spot, beta_vs_spy). Beta is None if not available."""
    if _YQ_AVAILABLE:
        try:
            t = YQTicker(ticker)
            spot = None
            fd = getattr(t, "financial_data", None)
            if isinstance(fd, dict):
                row = fd.get(ticker) or fd.get(ticker.lower()) or (list(fd.values())[0] if fd else None)
                if isinstance(row, dict):
                    spot = row.get("currentPrice")
            if spot is None:
                q = getattr(t, "quote", None)
                if isinstance(q, dict):
                    row = q.get(ticker) or q.get(ticker.lower()) or (list(q.values())[0] if q else None)
                    if isinstance(row, dict):
                        spot = row.get("regularMarketPrice")

            beta = None
            sd = getattr(t, "summary_detail", None)
            if isinstance(sd, dict):
                row = sd.get(ticker) or sd.get(ticker.lower()) or (list(sd.values())[0] if sd else None)
                if isinstance(row, dict) and row.get("beta") is not None:
                    beta = float(row["beta"])
            if spot is not None and spot > 0:
                return float(spot), beta
        except Exception:
            pass

    if _YF_AVAILABLE:
        try:
            yt = yf.Ticker(ticker)
            info = yt.info or {}
            spot = info.get("currentPrice") or info.get("regularMarketPrice")
            beta = info.get("beta")
            if spot is not None and spot > 0:
                return float(spot), (float(beta) if beta is not None else None)
        except Exception:
            pass
    return None, None


def _bs_greeks(S, K, T, r, q, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0:
        return None
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1 if opt_type == "C" else -1
    delta = sign * np.exp(-q * T) * norm.cdf(sign * d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -(S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
        - sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2)
        + sign * q * S * np.exp(-q * T) * norm.cdf(sign * d1)
    ) / 365
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def _revalue_position(pos: dict) -> dict:
    """Refresh Greeks at current spot using IV at entry as sigma proxy."""
    spot, beta = _fetch_spot_and_beta(pos["ticker"])
    if spot is None:
        # Fall back to entry-time Greeks; mark as stale
        g = pos["greeks_at_entry"]
        return {**pos, "spot": None, "beta": beta, "current_greeks": g, "stale": True}

    try:
        expiry = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
    except (ValueError, KeyError):
        return {**pos, "spot": spot, "beta": beta,
                "current_greeks": pos["greeks_at_entry"], "stale": True}
    dte = (expiry - date.today()).days
    if dte <= 0:
        return {**pos, "spot": spot, "beta": beta,
                "current_greeks": {"delta": 0, "gamma": 0, "theta": 0, "vega": 0},
                "stale": False, "expired": True}
    T = dte / 365.0
    g = _bs_greeks(spot, pos["strike"], T, RISK_FREE_RATE, 0.0,
                   pos["iv_at_entry"], pos["type"])
    if g is None:
        g = pos["greeks_at_entry"]
    return {**pos, "spot": spot, "beta": beta, "current_greeks": g,
            "dte": dte, "stale": False}


def aggregate_greeks(revalued: list[dict]) -> dict:
    """Sum Greeks across all positions, beta-weighted where beta is available."""
    raw = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    bw_delta = 0.0
    bw_count = 0

    for p in revalued:
        g = p.get("current_greeks") or {}
        n = p["contracts"]
        sign = 1  # tool only supports long options for now
        d  = sign * g.get("delta", 0) * 100 * n
        gm = sign * g.get("gamma", 0) * 100 * n
        th = sign * g.get("theta", 0) * 100 * n
        ve = sign * g.get("vega",  0) * 100 * n

        raw["delta"] += d
        raw["gamma"] += gm
        raw["theta"] += th
        raw["vega"]  += ve

        beta = p.get("beta")
        if beta is not None:
            bw_delta += d * beta
            bw_count += 1

    return {
        "delta": raw["delta"],
        "gamma": raw["gamma"],
        "theta": raw["theta"],
        "vega":  raw["vega"],
        "beta_weighted_delta": bw_delta if bw_count else None,
    }


def run_portfolio_view() -> None:
    """Interactive: print all positions with current Greeks, then aggregates."""
    positions = _load()
    W = 100
    print("\n" + "═" * W)
    print("  📂  PORTFOLIO")
    print("═" * W)
    if not positions:
        print(f"  No positions saved.  ({PORTFOLIO_FILE} does not exist yet.)")
        return

    print(f"  Loading {len(positions)} position(s) and revaluing at current spot...")
    revalued = [_revalue_position(p) for p in positions]

    print(f"\n  {'ID':<10} {'Ticker':<7} {'Type':<5} {'Strike':>8} {'Exp':>11} "
          f"{'DTE':>5} {'Ct':>4} {'Spot':>8} {'Δ$/$1':>8} {'Θ$/d':>8} {'V$/1vol':>8}")
    print("  " + "─" * (W - 2))
    for p in revalued:
        g = p.get("current_greeks") or {}
        n = p["contracts"]
        d_dollar = g.get("delta", 0) * 100 * n
        t_dollar = g.get("theta", 0) * 100 * n
        v_dollar = g.get("vega", 0) * 100 * n
        spot_str = f"${p['spot']:.2f}" if p.get("spot") else "N/A"
        dte = p.get("dte", "—")
        stale_tag = "  (stale)" if p.get("stale") else ("  EXPIRED" if p.get("expired") else "")
        print(f"  {p['id']:<10} {p['ticker']:<7} {p['type']:<5} ${p['strike']:>6.2f} "
              f"{p['expiry']:>11} {str(dte):>5} {n:>4} {spot_str:>8} "
              f"{d_dollar:>+7.0f} {t_dollar:>+7.1f} {v_dollar:>+7.1f}{stale_tag}")

    agg = aggregate_greeks(revalued)
    print("\n  " + "─" * (W - 2))
    print("  AGGREGATE GREEKS  (all positions combined, dollar-denominated)")
    print(f"    Net Delta:    ${agg['delta']:+,.0f}  per $1 move")
    if agg["beta_weighted_delta"] is not None:
        print(f"    Beta-weighted Delta (vs SPY):  ${agg['beta_weighted_delta']:+,.0f}  per $1 SPY move")
    print(f"    Net Gamma:    ${agg['gamma']:+,.2f}  per $1 move (acceleration)")
    print(f"    Net Theta:    ${agg['theta']:+,.0f}/day  →  ${agg['theta']*30:+,.0f}/month")
    print(f"    Net Vega:     ${agg['vega']:+,.0f}  per 1% IV move")

    # Risk threshold warnings
    warnings = []
    near_expiry = [p for p in revalued if isinstance(p.get("dte"), int) and p["dte"] <= 21]
    if near_expiry:
        warnings.append(f"⚠  {len(near_expiry)} position(s) ≤21 DTE — gamma dominates from here.")
    if abs(agg["theta"]) > 0:
        warnings.append(f"ℹ  Daily decay budget: ${agg['theta']:+,.0f}/day")

    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")
    print()


def offer_save_position(ticker: str, opt_type: str, strike: float, expiry: date,
                        contracts: int, entry_premium: float,
                        iv_at_entry: float, greeks: dict) -> None:
    raw = input("\n  Save this position to portfolio? [y/N]: ").strip().lower()
    if raw != "y":
        return
    new_id = add_position(ticker, opt_type, strike, expiry, contracts,
                          entry_premium, iv_at_entry, greeks)
    print(f"  ✅ Saved as position {new_id}.")


def run_position_remove() -> None:
    positions = _load()
    if not positions:
        print("  No positions to remove.")
        return
    print("\n  Current positions:")
    for p in positions:
        print(f"    {p['id']}  {p['ticker']} {p['type']} ${p['strike']:.2f}  exp {p['expiry']}  x{p['contracts']}")
    raw = input("\n  Enter ID to remove (or blank to cancel): ").strip()
    if not raw:
        return
    if remove_position(raw):
        print(f"  ✅ Removed {raw}.")
    else:
        print(f"  ⚠  No position with ID {raw}.")
