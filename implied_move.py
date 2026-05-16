"""Implied move from ATM straddle and IV crush estimate — Issue #3.

Implied move (Brenner-Subrahmanyam approximation): for an ATM option,
straddle price ≈ S · σ · √(T/2π), giving the 1-σ expected move over T.
A commonly-used practitioner shortcut is straddle × 0.85, which lines up
well with the theoretical 1-σ band when the chain is not perfectly ATM.

For earnings: empirically stocks land within the implied move ~70-75% of
the time vs. the theoretical 68% (1-σ), evidence that implied vol is
systematically over-priced into binary events. IV crush after the print
typically collapses the event premium 30-50% of the front-month IV
within minutes of the announcement.
"""

from datetime import date


def find_atm_straddle(candidates_all: list[dict], spot: float,
                      expiry: date) -> tuple[dict | None, dict | None]:
    """Return (call, put) at the nearest-to-money strike for the given expiry."""
    same_expiry = [c for c in candidates_all if c["expiry"] == expiry]
    if not same_expiry:
        return None, None

    calls = [c for c in same_expiry if c["type"] == "C"]
    puts = [c for c in same_expiry if c["type"] == "P"]
    if not calls or not puts:
        return None, None

    atm_call = min(calls, key=lambda c: abs(c["strike"] - spot))
    atm_put = min(puts, key=lambda c: abs(c["strike"] - spot))
    return atm_call, atm_put


def compute_implied_move(call: dict, put: dict, spot: float) -> dict:
    """Return implied move metrics from the ATM straddle."""
    straddle = call["premium"] + put["premium"]
    move_dollar = straddle * 0.85
    move_pct = (move_dollar / spot) * 100 if spot > 0 else 0.0
    return {
        "straddle_cost": straddle,
        "move_dollar": move_dollar,
        "move_pct": move_pct,
        "low_band": spot - move_dollar,
        "high_band": spot + move_dollar,
        "atm_strike": call["strike"],
    }


def estimate_iv_crush(front_iv: float | None, back_iv: float | None) -> dict:
    """Estimate post-earnings IV crush from the term structure spread.

    If the front-month IV sits well above the back-month IV, the spread
    is the event premium. Most of that collapses on the announcement.
    """
    if front_iv is None or back_iv is None:
        return {"crush_pp": None, "crush_pct": None}

    spread_pp = (front_iv - back_iv) * 100
    if spread_pp <= 0:
        return {"crush_pp": 0.0, "crush_pct": 0.0}

    # Practitioner rule of thumb: 30-50% of the front-month IV crushes;
    # most of the *event premium* (front - back spread) collapses fully.
    estimated_crush_pp = spread_pp * 0.85
    crush_pct_of_front = (estimated_crush_pp / (front_iv * 100)) * 100
    return {"crush_pp": estimated_crush_pp, "crush_pct": crush_pct_of_front}


def print_implied_move_block(im: dict | None, spot: float,
                              earnings_in_window: bool,
                              crush: dict | None = None) -> None:
    """Print the implied move block for the deep-analysis report."""
    W = 66
    print(f"\n{'━'*W}")
    print("  🎯  IMPLIED MOVE  (ATM straddle × 0.85)")
    print(f"{'━'*W}")
    if im is None:
        print("  ATM straddle not available for this expiry.")
        return

    print(f"  ATM strike:        ${im['atm_strike']:.2f}")
    print(f"  Straddle cost:     ${im['straddle_cost']:.2f}/share  →  "
          f"${im['straddle_cost']*100:.0f}/contract")
    print(f"  Implied move:      ±${im['move_dollar']:.2f}  (±{im['move_pct']:.1f}%)")
    print(f"  Expected range:    ${im['low_band']:.2f}  →  ${im['high_band']:.2f}")

    if earnings_in_window:
        print()
        print("  ⚠  EARNINGS INSIDE EXPIRY  —  IV crush expected post-announcement.")
        if crush and crush.get("crush_pp") is not None:
            print(f"     Estimated IV crush: ~{crush['crush_pp']:.0f}pp  "
                  f"({crush['crush_pct']:.0f}% of front-month IV)")
        print("     Historical: stocks land within implied move ~70-75% of the time.")
