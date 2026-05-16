"""Display the ranked shortlist of options to the user and accept their pick."""

from datetime import date


def display_shortlist(scored: list[dict], spot: float, hist_vol: float,
                      earnings_date: date | None = None) -> None:
    W = 108
    print("\n" + "═" * W)
    print(f"  TOP {len(scored)} CANDIDATES  (ranked by composite score: vol regime + liquidity + EV − earnings penalty)")
    print("═" * W)
    print(f"  Spot ${spot:.2f}   |   Historical vol: {hist_vol*100:.1f}%   |   "
          f"Next earnings: {earnings_date.strftime('%Y-%m-%d') if earnings_date else 'N/A'}")
    print("─" * W)
    print(f"  {'#':<3} {'Tier':<6} {'Type':<5} {'Strike':>8} {'Exp':>12} {'DTE':>4} {'Prem':>7} "
          f"{'IV':>6} {'Spd%':>5} {'OI':>6} {'Vol':>6} {'PoP':>5} {'EV':>9} {'Score':>6}  Flags")
    print("─" * W)
    for i, opt in enumerate(scored, 1):
        iv = opt.get("iv_contract")
        iv_str = f"{iv*100:.0f}%" if iv else "  N/A"
        spd = opt.get("spread_pct")
        spd_str = f"{spd:.1f}" if spd is not None else "  N/A"
        flags = []
        if opt.get("earnings_in_window"):
            flags.append("⚠EARN")
        if (opt.get("spread_pct") or 0) > 15:
            flags.append("WIDE")
        if opt.get("oi", 0) < 100:
            flags.append("THIN")
        flag_str = " ".join(flags)
        exp_str = opt["expiry"].strftime("%b-%d-%y")
        ev = opt.get("ev", 0)
        pop = opt.get("prob_profit", 0)
        tier = opt.get("tenor_bucket", "")
        print(f"  {i:<3} {tier:<6} {opt['type']:<5} ${opt['strike']:>6.2f} {exp_str:>12} "
              f"{opt['dte']:>4} ${opt['premium']:>5.2f} {iv_str:>6} {spd_str:>5} "
              f"{opt['oi']:>6} {opt['volume']:>6} {pop:>4.0f}% ${ev:>+7.0f} "
              f"{opt['composite']:>5.1f}  {flag_str}")
    print("═" * W)
    print("  Score components per option:")
    for i, opt in enumerate(scored, 1):
        print(f"  {i}.  Vol regime: {opt['vol_score']:>5.1f}   "
              f"Liquidity: {opt['liq_score']:>5.1f}   "
              f"EV: {opt['ev_score']:>5.1f}   "
              f"{'Earnings inside window' if opt.get('earnings_in_window') else ''}")


def pick_from_shortlist(scored: list[dict]) -> dict | None:
    """Prompt user to pick one option from the shortlist. Returns None to skip deep analysis."""
    if not scored:
        return None
    while True:
        raw = input(f"\n  Pick option [1-{len(scored)}, or 0 to skip deep analysis]: ").strip()
        if raw == "0" or raw.lower() == "skip":
            return None
        try:
            idx = int(raw)
            if 1 <= idx <= len(scored):
                return scored[idx - 1]
            print(f"  ⚠  Enter 1-{len(scored)} or 0 to skip.")
        except ValueError:
            print(f"  ⚠  Enter a number 1-{len(scored)} or 0 to skip.")
