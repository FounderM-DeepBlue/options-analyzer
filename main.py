#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          OPTIONS ANALYZER — BS / HESTON / BATES (SVJ)           ║
║                                                                  ║
║  Ticker-first flow:                                              ║
║    1. Enter ticker                                               ║
║    2. Scan chain across canonical DTE windows (21/30/45/60/90d   ║
║       + 6mo + 1yr LEAPS) and near-money strikes                  ║
║    3. Score each on vol regime, liquidity, EV, earnings risk     ║
║    4. Show top 5 ranked shortlist                                ║
║    5. User picks one → full BS/Heston/Bates/MC deep analysis     ║
║                                                                  ║
║  Usage:  python main.py                                          ║
║  Deps:   pip install numpy scipy pandas yahooquery               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
from datetime import date, datetime

from cli import get_float, get_float_or_default
from data_fetch import fetch_quote_and_profile
from chain_scanner import scan_chain
from history import fetch_realized_vol, lookback_for_tenor
from events import fetch_next_earnings_date
from scorer import rank_options
from shortlist import display_shortlist, pick_from_shortlist
from models import bs_price, bs_greeks, implied_vol, heston_lewis, bates_lewis
from monte_carlo import run_mc
from profiles import get_heston_bates_params_from_data, get_heston_bates_params
from report import print_report

warnings.filterwarnings("ignore")

_DEFAULT_R = 0.043  # risk-free rate fallback
TOP_N_SHORTLIST = 5


def _summarize_profile(profile: dict) -> str:
    cap_b = (profile.get("market_cap") or 0) / 1e9
    cap_str = f"${cap_b:,.1f}B" if cap_b else "N/A"
    sec = profile.get("sector") or "—"
    ind = profile.get("industry") or "—"
    return f"Mkt cap {cap_str}  |  {sec} / {ind}"


def analyze_picked_option(picked: dict, profile: dict, hist_vol: float,
                          earnings_date, contracts: int) -> None:
    """Run full BS / Heston / Bates / MC analysis on the user's selection."""
    S = profile["S"]
    q = profile.get("q", 0.0)
    r = _DEFAULT_R
    K = picked["strike"]
    opt_type = picked["type"]
    expiry_date = picked["expiry"]
    days = picked["dte"]
    T = days / 365.0
    expiry_str = expiry_date.strftime("%b %d, %Y")

    market_price = picked["premium"]
    iv = implied_vol(S, K, T, r, q, market_price, opt_type)
    if iv is None:
        print("  ⚠  Could not solve IV — using 30% fallback.")
        iv = 0.30
    else:
        print(f"\n  ✅ Implied vol from ${market_price:.2f} market price:  {iv*100:.2f}%")
    print(f"  ⚠️  Market is charging {iv*100:.1f}% IV — historical vol {hist_vol*100:.1f}%  "
          f"(spread {(iv - hist_vol)*100:+.1f}%)")

    entry_premium = get_float_or_default(
        f"\n  Price you paid or target entry per share ($)? [Enter = market ${market_price:.2f}]: ",
        market_price, 0.01, 9999.0,
    )

    # Heston / Bates parameters
    if profile.get("market_cap") is not None:
        kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j, profile_label = (
            get_heston_bates_params_from_data(
                hist_vol, profile.get("market_cap"),
                profile.get("sector", ""), profile.get("industry", ""),
            )
        )
        print(f"  Using profile from data: {profile_label}")
    else:
        kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j = get_heston_bates_params(
            hist_vol, profile_override="1",
        )

    print("\n  Running models...")
    bs_fv = bs_price(S, K, T, r, q, hist_vol, opt_type)
    greeks = bs_greeks(S, K, T, r, q, hist_vol, opt_type)
    print("  ├─ Black-Scholes ✅")
    h_fv = heston_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0, opt_type)
    print("  ├─ Heston ✅")
    b_fv = bates_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0,
                       lam_j, mu_j, sigma_j, opt_type)
    print("  ├─ Bates ✅")
    mc = run_mc(S, K, T, r, q, hist_vol, entry_premium, contracts, opt_type,
                kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j)
    print("  └─ Monte Carlo (100k paths) ✅")

    print_report(picked["ticker"], S, K, T, r, q, iv, entry_premium, contracts,
                 opt_type, days, expiry_str,
                 bs_fv, h_fv, b_fv, greeks, mc, earnings_date=earnings_date)


def run_ticker_flow() -> None:
    print("\n── TICKER  ────────────────────────────────────────────────────")
    ticker = input("  Ticker symbol: ").strip().upper()
    if not ticker:
        print("  ⚠  No ticker given.")
        return

    print(f"\n  Fetching quote and profile for {ticker}...")
    profile = fetch_quote_and_profile(ticker)
    if profile is None:
        print(f"  ⚠  Could not fetch market data for {ticker}.")
        return
    S = profile["S"]
    print(f"  ✅ Spot ${S:.2f}   |  {_summarize_profile(profile)}")

    print(f"\n  Fetching upcoming earnings date for {ticker}...")
    earnings_date = fetch_next_earnings_date(ticker)
    if earnings_date is None:
        print("  ⚠  No earnings date available.")
    else:
        dte_earn = (earnings_date - date.today()).days
        print(f"  ✅ Next earnings: {earnings_date.strftime('%Y-%m-%d')}  ({dte_earn} days)")

    wants_earn = False
    if earnings_date is not None:
        raw = input("  Are you trading the earnings event? [y/N]: ").strip().lower()
        wants_earn = raw == "y"

    print(f"\n  Scanning option chain (21/30/45/60/90d + 6mo + 1yr LEAPS, strikes within ±20% of spot)...")
    candidates = scan_chain(ticker, S)
    if not candidates:
        print(f"  ⚠  Chain scan returned no usable contracts (rate limit or no options).")
        return
    print(f"  ✅ {len(candidates)} candidate contracts collected.")

    # Use a 60-day historical vol as the baseline for scoring; deep analysis re-derives
    # the tenor-appropriate window for the chosen option.
    print("\n  Computing 60-day realized vol for scoring baseline...")
    hist_vol, label = fetch_realized_vol(ticker, 60)
    if hist_vol is None:
        print(f"  ⚠  {label}. Using 0.30 fallback.")
        hist_vol = 0.30
    else:
        print(f"  ✅ {label} realized vol: {hist_vol*100:.1f}%")

    scored = rank_options(
        candidates,
        spot=S,
        hist_vol=hist_vol,
        r=_DEFAULT_R,
        q=profile.get("q", 0.0),
        earnings_date=earnings_date,
        wants_earnings_exposure=wants_earn,
        top_n=TOP_N_SHORTLIST,
    )
    if not scored:
        print("  ⚠  No options passed scoring.")
        return
    display_shortlist(scored, S, hist_vol, earnings_date)

    picked = pick_from_shortlist(scored)
    if picked is None:
        print("\n  Skipping deep analysis. Done.")
        return

    # Re-derive the proper tenor-based historical vol for the picked option.
    lookback_days, lookback_label = lookback_for_tenor(picked["dte"])
    if lookback_days != 60:
        print(f"\n  Re-fetching {lookback_label} realized vol for deep analysis...")
        tenor_hv, _ = fetch_realized_vol(ticker, lookback_days)
        if tenor_hv is not None:
            hist_vol = tenor_hv
            print(f"  ✅ {lookback_label} realized vol: {hist_vol*100:.1f}%")

    contracts = int(get_float("\n  Number of contracts [default 1]: ", 1, allow_zero=False))
    analyze_picked_option(picked, profile, hist_vol, earnings_date, contracts)


def main():
    W = 66
    print("\n" + "═" * W)
    print("       OPTIONS ANALYZER  —  BS / HESTON / BATES (SVJ)")
    print("       Ticker-first scan → score → shortlist → deep analysis")
    print("═" * W)

    while True:
        run_ticker_flow()
        again = input("\n  Analyze another ticker? [Y/N]: ").strip().upper()
        if again != "Y":
            print("\n  Done.\n")
            break


if __name__ == "__main__":
    main()
