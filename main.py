#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          OPTIONS ANALYZER — BS / HESTON / BATES (SVJ)           ║
║                                                                  ║
║  Top-level menu:                                                 ║
║    [1] Analyze a ticker     — full ticker-first scan + analysis  ║
║    [2] View portfolio       — aggregate Greeks across positions  ║
║    [3] Remove position      — delete from portfolio.json         ║
║    [4] Refresh VIX regime                                        ║
║    [Q] Quit                                                      ║
║                                                                  ║
║  Usage:  python main.py                                          ║
║  Deps:   pip install numpy scipy pandas yahooquery               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
from datetime import date

from cli import get_float, get_float_or_default
from data_fetch import fetch_quote_and_profile
from chain_scanner import scan_chain
from history import fetch_realized_vol, lookback_for_tenor
from events import fetch_next_earnings_date
from iv_rank import compute_iv_rank
from scorer import rank_options
from shortlist import display_shortlist, pick_from_shortlist
from term_structure import build_term_structure, print_term_structure
from technicals import compute_technicals
from implied_move import find_atm_straddle, compute_implied_move, estimate_iv_crush
from vix_regime import get_vix_regime, vix_display
from liquidity import assess_liquidity
from position_sizing import compute_size
from portfolio import run_portfolio_view, run_position_remove, offer_save_position
from models import bs_price, bs_greeks, implied_vol, heston_lewis, bates_lewis
from monte_carlo import run_mc
from profiles import get_heston_bates_params_from_data, get_heston_bates_params
from report import print_report

warnings.filterwarnings("ignore")

_DEFAULT_R = 0.043
TOP_N_SHORTLIST = 5


def _summarize_profile(profile: dict) -> str:
    cap_b = (profile.get("market_cap") or 0) / 1e9
    cap_str = f"${cap_b:,.1f}B" if cap_b else "N/A"
    sec = profile.get("sector") or "—"
    ind = profile.get("industry") or "—"
    return f"Mkt cap {cap_str}  |  {sec} / {ind}"


def _back_iv_after_earnings(ts_rows: list[dict], earnings_date: date | None) -> float | None:
    """First ATM IV after the earnings date — used as the post-event 'clean' IV."""
    if not earnings_date or not ts_rows:
        return None
    for r in ts_rows:
        if r["expiry"] > earnings_date:
            return r["atm_iv"]
    return None


def analyze_picked_option(picked: dict, profile: dict, hist_vol: float,
                          earnings_date, contracts: int, *,
                          ivr=None, ivp=None, ivr_label="N/A",
                          vix_regime: dict | None = None,
                          tech: dict | None = None,
                          candidates_all: list[dict] | None = None,
                          ts_rows: list[dict] | None = None,
                          portfolio_value: float | None = None) -> None:
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

    # Liquidity (#2)
    liq = assess_liquidity(picked["bid"], picked["ask"], entry_premium,
                           picked["oi"], picked["volume"], contracts)

    # Implied move (#3)
    earnings_in_window = (
        earnings_date is not None and earnings_date <= expiry_date
    )
    im = None
    crush = None
    if candidates_all:
        atm_call, atm_put = find_atm_straddle(candidates_all, S, expiry_date)
        if atm_call and atm_put:
            im = compute_implied_move(atm_call, atm_put, S)
    if earnings_in_window and ts_rows:
        front_iv = ts_rows[0]["atm_iv"]
        back_iv = _back_iv_after_earnings(ts_rows, earnings_date)
        crush = estimate_iv_crush(front_iv, back_iv)

    # Position sizing (#5) — only if portfolio_value provided
    sizing = None
    if portfolio_value and portfolio_value > 0:
        sizing = compute_size(
            portfolio_value=portfolio_value,
            premium_per_share=entry_premium,
            prob_profit=mc["b"]["prob"] / 100.0,
            avg_win=mc["b"]["win"],
            avg_loss=mc["b"]["loss"],
            vix_scalar=(vix_regime or {}).get("scalar", 1.00),
            ivr=ivr,
        )

    print_report(picked["ticker"], S, K, T, r, q, iv, entry_premium, contracts,
                 opt_type, days, expiry_str,
                 bs_fv, h_fv, b_fv, greeks, mc,
                 earnings_date=earnings_date,
                 ivr=ivr, ivp=ivp, ivr_label=ivr_label,
                 vix_regime=vix_regime,
                 tech=tech,
                 liq=liq,
                 im=im,
                 crush=crush,
                 sizing=sizing,
                 portfolio_value=portfolio_value)

    # Portfolio save offer (#8)
    offer_save_position(
        picked["ticker"], opt_type, K, expiry_date, contracts,
        entry_premium, iv, greeks,
    )


def run_ticker_flow(vix_regime: dict) -> None:
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
    print(f"  Macro: {vix_display(vix_regime)}")

    print(f"\n  Fetching upcoming earnings date for {ticker}...")
    earnings_date = fetch_next_earnings_date(ticker)
    if earnings_date is None:
        print("  ⚠  No earnings date available.")
    else:
        dte_earn = (earnings_date - date.today()).days
        print(f"  ✅ Next earnings: {earnings_date.strftime('%Y-%m-%d')}  ({dte_earn} days)")

    while True:
        raw_type = input("  Contract type [C=Calls / P=Puts / B=Both, default B]: ").strip().upper() or "B"
        if raw_type in ("C", "P", "B"):
            break
        print("  ⚠  Enter C, P, or B.")
    opt_filter = raw_type

    wants_earn = False
    if earnings_date is not None:
        raw = input("  Are you trading the earnings event? [y/N]: ").strip().lower()
        wants_earn = raw == "y"

    print(f"\n  Scanning option chain (21/30/45/60/90d + 6mo + 1yr LEAPS, strikes within ±20% of spot)...")
    candidates_all = scan_chain(ticker, S)
    candidates = candidates_all if opt_filter == "B" else [c for c in candidates_all if c["type"] == opt_filter]
    if not candidates:
        print(f"  ⚠  Chain scan returned no usable contracts (rate limit or no options).")
        return
    print(f"  ✅ {len(candidates)} candidate contracts collected.")

    print("\n  Computing 60-day realized vol for scoring baseline...")
    hist_vol, label = fetch_realized_vol(ticker, 60)
    if hist_vol is None:
        print(f"  ⚠  {label}. Using 0.30 fallback.")
        hist_vol = 0.30
    else:
        print(f"  ✅ {label} realized vol: {hist_vol*100:.1f}%")

    ts_rows = build_term_structure(candidates_all, S)
    print_term_structure(ts_rows, hist_vol)

    print("\n  Computing IV Rank / IV Percentile (252-day HV-proxy)...")
    front_atm_iv = ts_rows[0]["atm_iv"] if ts_rows else None
    ivr, ivp, ivr_label = compute_iv_rank(ticker, front_atm_iv or hist_vol)
    if ivr is not None:
        print(f"  ✅ IVR: {ivr:.0f}  |  IVP: {ivp:.0f}th pctile  |  Bias: {ivr_label}")
    else:
        print("  ⚠  IVR unavailable (insufficient price history).")

    print("\n  Fetching technical levels (52w hi/lo, 200d SMA)...")
    tech = compute_technicals(ticker, S)
    if tech:
        print(f"  ✅ 52w hi ${tech['high_52w']:.2f}  /  lo ${tech['low_52w']:.2f}"
              + (f"  /  200d SMA ${tech['sma_200']:.2f}" if tech.get("sma_200") else ""))

    scored = rank_options(
        candidates, spot=S, hist_vol=hist_vol, r=_DEFAULT_R,
        q=profile.get("q", 0.0), earnings_date=earnings_date,
        wants_earnings_exposure=wants_earn, top_n=TOP_N_SHORTLIST,
    )
    if not scored:
        print("  ⚠  No options passed scoring.")
        return
    display_shortlist(scored, S, hist_vol, earnings_date,
                      ivr=ivr, ivp=ivp, ivr_label=ivr_label)

    picked = pick_from_shortlist(scored)
    if picked is None:
        print("\n  Skipping deep analysis. Done.")
        return

    lookback_days, lookback_label = lookback_for_tenor(picked["dte"])
    if lookback_days != 60:
        print(f"\n  Re-fetching {lookback_label} realized vol for deep analysis...")
        tenor_hv, _ = fetch_realized_vol(ticker, lookback_days)
        if tenor_hv is not None:
            hist_vol = tenor_hv
            print(f"  ✅ {lookback_label} realized vol: {hist_vol*100:.1f}%")

    pv_raw = input("\n  Portfolio value $ for Kelly sizing [blank to skip]: ").strip()
    portfolio_value = None
    if pv_raw:
        try:
            portfolio_value = float(pv_raw.replace(",", "").replace("$", ""))
            if portfolio_value <= 0:
                portfolio_value = None
        except ValueError:
            print("  ⚠  Invalid amount — skipping sizing.")

    contracts = int(get_float("\n  Number of contracts [default 1]: ", 1, allow_zero=False))
    analyze_picked_option(
        picked, profile, hist_vol, earnings_date, contracts,
        ivr=ivr, ivp=ivp, ivr_label=ivr_label,
        vix_regime=vix_regime, tech=tech,
        candidates_all=candidates_all, ts_rows=ts_rows,
        portfolio_value=portfolio_value,
    )


def main():
    W = 66
    print("\n" + "═" * W)
    print("       OPTIONS ANALYZER  —  BS / HESTON / BATES (SVJ)")
    print("═" * W)

    print("\n  Fetching VIX regime for session...")
    vix_regime = get_vix_regime()
    print(f"  {vix_display(vix_regime)}")

    while True:
        print("\n── MAIN MENU  ─────────────────────────────────────────────────")
        print("  [1] Analyze a ticker")
        print("  [2] View portfolio")
        print("  [3] Remove position from portfolio")
        print("  [4] Refresh VIX regime")
        print("  [Q] Quit")
        choice = input("\n  Choice: ").strip().upper()

        if choice == "1":
            run_ticker_flow(vix_regime)
        elif choice == "2":
            run_portfolio_view()
        elif choice == "3":
            run_position_remove()
        elif choice == "4":
            print("\n  Refreshing VIX...")
            vix_regime = get_vix_regime()
            print(f"  {vix_display(vix_regime)}")
        elif choice == "Q" or choice == "":
            print("\n  Done.\n")
            break
        else:
            print(f"  ⚠  Unknown choice '{choice}'")


if __name__ == "__main__":
    main()
