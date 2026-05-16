"""Option ranking algorithm.

Scores each candidate contract on four research-backed criteria from
options_entry_research.md:
  1. Vol regime (IV vs. historical vol) — proxy for IVR until 52w IV history available
  2. Liquidity quality (bid-ask spread %, open interest, volume)
  3. Expected value (BS-MC, normalized to premium cost)
  4. Earnings risk (downgrade if earnings falls inside the expiry window)

The composite is oriented for buying options (long calls/puts): we reward
options whose IV is at or below historical vol, have tight spreads and deep OI,
positive expected value per dollar of premium, and no surprise event exposure.
"""

from datetime import date

from monte_carlo import quick_ev_bs


# Weights for the composite score (sum to 1.0)
W_VOL = 0.30
W_LIQ = 0.25
W_EV = 0.35
W_EARN = 0.10  # multiplier-based, contributes through penalty


def _vol_score(iv_contract: float | None, hist_vol: float) -> float:
    """0-100. Penalize when the contract's IV is well above realized vol (overpaying for vol).

    Score = 100 - 50 * (IV - HV) / max(HV, 0.10) clamped to [0, 100].
    IV ≈ HV → ~100; IV = HV + 10pp on a 30% HV stock → ~83; IV = 2x HV → 50.
    """
    if iv_contract is None or hist_vol <= 0:
        return 50.0  # neutral
    spread = iv_contract - hist_vol
    denom = max(hist_vol, 0.10)
    score = 100.0 - 50.0 * (spread / denom)
    return max(0.0, min(100.0, score))


def _liquidity_score(bid: float, ask: float, oi: int, volume: int, premium: float) -> float:
    """0-100 composite. Bid-ask % of mid is the dominant factor; OI and volume are floors."""
    if premium <= 0:
        return 0.0
    spread_pct = ((ask - bid) / premium) * 100 if (bid > 0 and ask > 0) else 100.0
    # Spread component: 100 at 0% spread, 0 at 25% spread
    spread_score = max(0.0, 100.0 - 4.0 * spread_pct)
    # OI component: log scale, 100 at OI=5000, 50 at OI=500, 0 at OI=0
    if oi <= 0:
        oi_score = 0.0
    else:
        import math
        oi_score = min(100.0, 50.0 * (math.log10(max(oi, 1)) / math.log10(500)))
    # Volume floor: 0 if no volume, 100 if vol >= 100
    vol_score = min(100.0, volume * 1.0)
    return 0.6 * spread_score + 0.3 * oi_score + 0.1 * vol_score


def _ev_score(ev_per_contract: float, premium: float) -> float:
    """0-100. EV per contract normalized to premium cost (×100 shares).

    Mapping: EV/cost = 0% → 50, +20% → 75, +50% → 100, -20% → 25, -50% → 0.
    """
    if premium <= 0:
        return 50.0
    cost = premium * 100  # one contract
    ratio = ev_per_contract / cost if cost > 0 else 0.0
    score = 50.0 + 100.0 * ratio  # linear: each 1% of cost moves score 1 point
    return max(0.0, min(100.0, score))


def _earnings_multiplier(earnings_date: date | None, expiry: date,
                        wants_earnings_exposure: bool) -> float:
    """1.0 if earnings outside window or explicitly desired; 0.7 if inside window without intent."""
    if earnings_date is None or earnings_date > expiry:
        return 1.0
    return 1.0 if wants_earnings_exposure else 0.7


def score_option(
    contract: dict,
    spot: float,
    hist_vol: float,
    r: float,
    q: float,
    earnings_date: date | None = None,
    wants_earnings_exposure: bool = False,
) -> dict:
    """Score a single contract; returns the contract dict augmented with score fields."""
    T = max(contract["dte"], 1) / 365.0
    vol_for_ev = contract.get("iv_contract") or hist_vol
    ev, prob = quick_ev_bs(
        spot, contract["strike"], T, r, q, vol_for_ev,
        contract["premium"], contract["type"],
    )

    vol_s = _vol_score(contract.get("iv_contract"), hist_vol)
    liq_s = _liquidity_score(contract["bid"], contract["ask"],
                             contract["oi"], contract["volume"], contract["premium"])
    ev_s = _ev_score(ev, contract["premium"])
    earn_mult = _earnings_multiplier(earnings_date, contract["expiry"], wants_earnings_exposure)

    composite_raw = W_VOL * vol_s + W_LIQ * liq_s + W_EV * ev_s
    # Earnings acts as a multiplier on the composite, with W_EARN governing how strongly
    earnings_penalty = (1.0 - earn_mult) * 100.0  # 0 if outside, 30 if inside and unwanted
    composite = composite_raw - W_EARN * earnings_penalty
    composite = max(0.0, min(100.0, composite))

    scored = dict(contract)
    scored.update({
        "ev": ev,
        "prob_profit": prob,
        "spread_pct": ((contract["ask"] - contract["bid"]) / contract["premium"] * 100)
                      if (contract["bid"] > 0 and contract["ask"] > 0 and contract["premium"] > 0)
                      else None,
        "vol_score": vol_s,
        "liq_score": liq_s,
        "ev_score": ev_s,
        "earnings_in_window": earn_mult < 1.0,
        "composite": composite,
    })
    return scored


def rank_options(
    contracts: list[dict],
    spot: float,
    hist_vol: float,
    r: float,
    q: float,
    earnings_date: date | None = None,
    wants_earnings_exposure: bool = False,
    top_n: int = 5,
) -> list[dict]:
    """Score and rank all contracts; return top_n by composite score."""
    scored = [
        score_option(c, spot, hist_vol, r, q, earnings_date, wants_earnings_exposure)
        for c in contracts
    ]
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored[:top_n]
