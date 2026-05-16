"""Volatility term structure analysis — Issue #4.

For each expiry already scanned, picks the nearest-to-money call's IV and
builds a term structure table. Labels the regime (contango / flat /
backwardation) and computes a slope metric (front-month vs. 90d IV spread).

Contango  (normal): short-dated IV < long-dated IV  — market calm, vol risk
                    priced into the future.
Backwardation:      short-dated IV > long-dated IV  — elevated near-term fear
                    (event risk, systemic stress). Reduce short-premium exposure.
"""

from datetime import date


def build_term_structure(candidates: list[dict], spot: float) -> list[dict]:
    """Return a list of {dte, expiry, atm_iv} sorted by DTE, using the
    nearest-to-money call from each unique expiry in the scanned candidates."""
    by_expiry: dict[date, list[dict]] = {}
    for c in candidates:
        if c.get("iv_contract") and c["type"] == "C":
            by_expiry.setdefault(c["expiry"], []).append(c)

    rows = []
    for exp, contracts in by_expiry.items():
        atm = min(contracts, key=lambda c: abs(c["strike"] - spot))
        iv = atm.get("iv_contract")
        if iv and iv > 0:
            rows.append({
                "expiry": exp,
                "dte": atm["dte"],
                "strike": atm["strike"],
                "atm_iv": iv,
            })
    rows.sort(key=lambda r: r["dte"])
    return rows


def _regime_label(rows: list[dict]) -> tuple[str, float | None]:
    """Returns (regime_label, slope_pp) where slope = front_iv - back_90d_iv in pp."""
    if len(rows) < 2:
        return "Insufficient data", None
    front_iv = rows[0]["atm_iv"]
    # Find the row closest to 90 DTE for the slope benchmark
    back = min(rows, key=lambda r: abs(r["dte"] - 90)) if len(rows) > 1 else rows[-1]
    back_iv = back["atm_iv"]
    slope = (front_iv - back_iv) * 100  # in pp; positive = backwardation

    if slope > 2:
        label = "Backwardation  ⚠  Near-term fear elevated — reduce short-premium exposure"
    elif slope < -2:
        label = "Contango  —  Normal regime, vol risk priced into future expiries"
    else:
        label = "Flat  —  Term structure near-neutral"

    return label, slope


def print_term_structure(rows: list[dict], hist_vol: float) -> None:
    if not rows:
        print("  Term structure: no IV data available from scanned chain.")
        return

    regime, slope = _regime_label(rows)
    W = 66
    print(f"\n{'━'*W}")
    print(f"  📈  VOLATILITY TERM STRUCTURE  (ATM call IV by expiry)")
    print(f"{'━'*W}")
    print(f"  {'Expiry':<14} {'DTE':>5} {'Strike':>8} {'ATM IV':>8} {'vs HV':>8}  Bar")
    print(f"  {'─'*14}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")

    front_iv = rows[0]["atm_iv"] if rows else None
    for r in rows:
        iv = r["atm_iv"]
        vs_hv = (iv - hist_vol) * 100
        bar_len = int(iv * 30)          # 30 chars = 100% IV
        bar = "█" * min(bar_len, 30)
        marker = " ◄ front" if r == rows[0] else ""
        print(f"  {r['expiry'].strftime('%Y-%m-%d'):<14} {r['dte']:>5} "
              f"${r['strike']:>7.2f}  {iv*100:>7.1f}%  {vs_hv:>+7.1f}pp  {bar}{marker}")

    print(f"{'━'*W}")
    if slope is not None:
        print(f"  Slope (front vs. 90d):  {slope:+.1f}pp")
    print(f"  Regime: {regime}")
    print(f"  HV baseline: {hist_vol*100:.1f}%  (all IVs vs. this benchmark)")
