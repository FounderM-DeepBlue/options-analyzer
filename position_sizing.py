"""Half-Kelly position sizing with VIX and IVR confidence scalars — Issue #5.

Kelly fraction for an options trade (binary outcome approximation):

    f* = (PoP × Avg_Win − (1 − PoP) × |Avg_Loss|) / Avg_Win

Half-Kelly halves f* to capture ~75% of theoretical growth with dramatically
lower drawdown — the professional consensus for fat-tailed P&L distributions.

Final size = Half-Kelly × VIX scalar × IVR confidence × portfolio value,
            capped at 5% of portfolio per position.

For buyers (this tool's orientation), IVR confidence flips relative to sellers:
  IVR < 30 → 1.00 (cheap vol = high conviction long)
  IVR 30-50 → 0.85
  IVR 50-70 → 0.70
  IVR ≥ 70 → 0.50 (expensive vol = caution on debit trades)
"""

from math import floor

MAX_FRACTION = 0.05  # 5% portfolio cap


def _ivr_confidence_long(ivr: float | None) -> float:
    if ivr is None:
        return 0.85
    if ivr < 30:
        return 1.00
    if ivr < 50:
        return 0.85
    if ivr < 70:
        return 0.70
    return 0.50


def compute_size(
    portfolio_value: float,
    premium_per_share: float,
    prob_profit: float,
    avg_win: float,
    avg_loss: float,            # negative number
    vix_scalar: float = 1.00,
    ivr: float | None = None,
) -> dict:
    """Return sizing recommendation dict.

    Inputs:
        portfolio_value   in $
        premium_per_share in $/share (one contract = 100 shares)
        prob_profit       in [0, 1]
        avg_win           in $ (positive)
        avg_loss          in $ (negative)
        vix_scalar        from vix_regime (1.0 for low vol, 0.25 for crisis)
        ivr               IVR 0-100 or None
    """
    if portfolio_value <= 0 or premium_per_share <= 0:
        return {"contracts": 0, "fraction": 0.0, "reason": "Invalid inputs"}

    if avg_win <= 0:
        return {
            "contracts": 0, "fraction": 0.0,
            "reason": "No positive-win outcomes simulated — Kelly = 0",
        }

    loss_abs = abs(avg_loss) if avg_loss < 0 else 0.0
    kelly_full = (prob_profit * avg_win - (1.0 - prob_profit) * loss_abs) / avg_win
    half_kelly = kelly_full * 0.5

    if half_kelly <= 0:
        return {
            "contracts": 0,
            "fraction": half_kelly,
            "kelly_full": kelly_full,
            "half_kelly": half_kelly,
            "vix_scalar": vix_scalar,
            "ivr_scalar": _ivr_confidence_long(ivr),
            "reason": "Negative EV — Kelly says do not trade",
        }

    ivr_scalar = _ivr_confidence_long(ivr)
    raw_fraction = half_kelly * vix_scalar * ivr_scalar
    capped_fraction = min(raw_fraction, MAX_FRACTION)
    cost_per_contract = premium_per_share * 100
    target_dollars = capped_fraction * portfolio_value
    contracts = max(0, floor(target_dollars / cost_per_contract))
    actual_dollars = contracts * cost_per_contract
    actual_fraction = actual_dollars / portfolio_value if portfolio_value else 0.0

    capped = raw_fraction > MAX_FRACTION
    reason = "Sized by Half-Kelly × VIX × IVR confidence"
    if capped:
        reason += f" (capped at {MAX_FRACTION:.0%} of portfolio)"

    return {
        "contracts": contracts,
        "cost_per_contract": cost_per_contract,
        "total_cost": actual_dollars,
        "fraction": actual_fraction,
        "fraction_target": capped_fraction,
        "kelly_full": kelly_full,
        "half_kelly": half_kelly,
        "vix_scalar": vix_scalar,
        "ivr_scalar": ivr_scalar,
        "capped": capped,
        "reason": reason,
    }


def print_sizing_block(sizing: dict, portfolio_value: float,
                        user_contracts: int) -> None:
    W = 66
    print(f"\n{'━'*W}")
    print("  📏  POSITION SIZING  (Half-Kelly × VIX × IVR)")
    print(f"{'━'*W}")
    print(f"  Portfolio value:     ${portfolio_value:,.0f}")
    print(f"  Full Kelly f*:       {sizing.get('kelly_full', 0):.3f}")
    print(f"  Half Kelly:          {sizing.get('half_kelly', 0):.3f}")
    print(f"  VIX scalar:          {sizing.get('vix_scalar', 1.0):.2f}×")
    print(f"  IVR scalar (long):   {sizing.get('ivr_scalar', 1.0):.2f}×")
    print(f"  Target fraction:     {sizing.get('fraction_target', 0):.2%}  "
          f"(cap {MAX_FRACTION:.0%})")
    print()
    print(f"  → Recommended:       {sizing['contracts']} contract(s)  "
          f"(${sizing.get('total_cost', 0):,.0f}, "
          f"{sizing.get('fraction', 0):.2%} of portfolio)")
    print(f"  → User-entered:      {user_contracts} contract(s)")
    if user_contracts > sizing["contracts"] and sizing["contracts"] > 0:
        over = user_contracts - sizing["contracts"]
        print(f"  ⚠  Over-sized by {over} contract(s) vs. Kelly recommendation.")
    elif user_contracts < sizing["contracts"]:
        under = sizing["contracts"] - user_contracts
        print(f"  ℹ  Under Kelly by {under} contract(s) — conservative sizing.")
    print(f"  Note: {sizing['reason']}")
