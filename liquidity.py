"""Liquidity quality assessment for the deep-analysis report — Issue #2.

Professional standards (CBOE, Natenberg):
  bid-ask spread < $0.30 OR < 10% of mid          tight
  open interest > 500                              deep
  daily volume > 100                               active
Failing any threshold creates hidden slippage that can eliminate the
modeled edge entirely.
"""


def assess_liquidity(bid: float, ask: float, premium: float,
                     oi: int, volume: int, contracts: int = 1) -> dict:
    """Return a liquidity assessment dict with grades and dollar slippage estimate."""
    if premium <= 0:
        spread = max(ask - bid, 0.0)
        spread_pct = None
    else:
        spread = max(ask - bid, 0.0)
        spread_pct = (spread / premium) * 100 if (bid > 0 and ask > 0) else None

    if spread_pct is None:
        spread_grade = "N/A"
    elif spread <= 0.30 or spread_pct <= 10:
        spread_grade = "Tight"
    elif spread_pct <= 20:
        spread_grade = "Acceptable"
    else:
        spread_grade = "WIDE"

    if oi >= 500:
        oi_grade = "Deep"
    elif oi >= 100:
        oi_grade = "Adequate"
    else:
        oi_grade = "THIN"

    if volume >= 100:
        vol_grade = "Active"
    elif volume >= 10:
        vol_grade = "Light"
    else:
        vol_grade = "DEAD"

    # Round-trip slippage assuming a fill at mid → exit at the wrong side of half-spread
    # for entry AND exit (so full spread total in the worst case)
    slippage_per_contract = spread * 100  # full bid-ask cost both ways
    slippage_total = slippage_per_contract * contracts

    return {
        "spread": spread,
        "spread_pct": spread_pct,
        "spread_grade": spread_grade,
        "oi": oi,
        "oi_grade": oi_grade,
        "volume": volume,
        "volume_grade": vol_grade,
        "slippage_per_contract": slippage_per_contract,
        "slippage_total": slippage_total,
    }


def print_liquidity_block(liq: dict, premium: float) -> None:
    W = 66
    print(f"\n{'━'*W}")
    print("  💧  LIQUIDITY")
    print(f"{'━'*W}")

    spread_pct_str = f"{liq['spread_pct']:.1f}%" if liq.get("spread_pct") is not None else "N/A"
    rows = [
        ("Bid-Ask spread", f"${liq['spread']:.2f}  ({spread_pct_str} of mid)  →  {liq['spread_grade']}"),
        ("Open Interest", f"{liq['oi']:,}  →  {liq['oi_grade']}"),
        ("Daily Volume", f"{liq['volume']:,}  →  {liq['volume_grade']}"),
        ("Round-trip slip", f"${liq['slippage_per_contract']:.0f}/contract  →  ${liq['slippage_total']:.0f} total"),
    ]
    for k, v in rows:
        print(f"  {k:<18} {v}")

    warnings = []
    if liq["spread_grade"] == "WIDE":
        warnings.append("⚠  Wide spread — fills may slip materially.")
    if liq["oi_grade"] == "THIN":
        warnings.append("⚠  Thin open interest — exiting may be difficult.")
    if liq["volume_grade"] == "DEAD":
        warnings.append("⚠  Dead volume — limit orders only, expect partial fills.")

    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")
    elif premium > 0 and (liq["slippage_total"] / (premium * 100)) > 0.05:
        print()
        print(f"  ℹ  Slippage = {(liq['slippage_total']/(premium*100))*100:.1f}% of premium "
              f"— factor into your edge estimate.")
