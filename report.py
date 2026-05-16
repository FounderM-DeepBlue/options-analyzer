"""Deep analysis report formatting: Black-Scholes / Heston / Bates / MC output."""

from datetime import date
import numpy as np
from scipy.stats import norm


def print_report(ticker, S, K, T, r, q, iv, premium, contracts,
                 opt_type, days, expiry_str,
                 bs_fv, h_fv, b_fv,
                 greeks, mc,
                 earnings_date=None):

    total_cost = premium * 100 * contracts
    if opt_type == "C":
        breakeven = K + premium
        be_label = f"${breakeven:.2f}  (+{((breakeven/S)-1)*100:.1f}% from spot)"
    else:
        breakeven = K - premium
        be_label = f"${breakeven:.2f}  (-{((1-(breakeven/S))*100):.1f}% from spot)"

    moneyness = (S / K - 1) * 100
    itm_otm = "ITM" if (opt_type == "C" and S > K) or (opt_type == "P" and S < K) else "OTM"

    W = 66

    def header(title):
        print(f"\n{'━'*W}")
        print(f"  {title}")
        print(f"{'━'*W}")

    print("\n" + "=" * W)
    print(f"  {ticker.upper()} ${K:.0f} {'CALL' if opt_type=='C' else 'PUT'}  |  {expiry_str}  |  {days}d")
    print(f"  BS / HESTON / BATES ANALYSIS  —  {date.today().strftime('%b %d, %Y')}")
    print("=" * W)

    # Setup
    header("📌  TRADE SETUP")
    rows = [
        ("Underlying", f"${S:.2f}"),
        ("Strike", f"${K:.2f} {'Call' if opt_type=='C' else 'Put'}  ({moneyness:+.1f}%  {itm_otm})"),
        ("Expiration", f"{expiry_str}  ({days} days / {T:.2f} yrs)"),
        ("Premium Paid", f"${premium:.2f}/share"),
        ("Contracts", f"{contracts}  →  ${total_cost:,.0f} total at risk"),
        ("Breakeven", be_label),
        ("Implied Vol", f"{iv*100:.1f}%"),
        ("Risk-Free Rate", f"{r*100:.1f}%"),
        ("Div Yield", f"{q*100:.1f}%"),
    ]
    if earnings_date is not None:
        inside = (earnings_date.toordinal() - date.today().toordinal()) <= days
        rows.append((
            "Next Earnings",
            f"{earnings_date.strftime('%Y-%m-%d')}"
            f"{'  ⚠ INSIDE EXPIRY' if inside else ''}",
        ))
    for k, v in rows:
        print(f"  {k:<18} {v}")

    # Fair value
    header("📊  FAIR VALUE")
    print(f"  {'Model':<16} │ {'Closed Form':>12} │ {'MC Price':>10} │ {'vs Entry':>12} │  Edge")
    print(f"  {'─'*16}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*12}─┼──────────")
    for label, fv, mc_val in [
        ("Black-Scholes", bs_fv, mc["bs"]["mc"]),
        ("Heston",        h_fv,  mc["h"]["mc"]),
        ("Bates (SVJ)",   b_fv,  mc["b"]["mc"]),
    ]:
        diff = premium - mc_val
        edge = "✅ Underpaid" if diff < -0.50 else ("⚠️  Overpaid" if diff > 0.50 else "✅ At FV")
        fv_str = f"${fv:.2f}" if label == "Black-Scholes" else "(MC only)"
        print(f"  {label:<16} │ {fv_str:>12} │ ${mc_val:>8.2f} │ {diff:>+10.2f}   │  {edge}")

    # Greeks
    header("📐  GREEKS  (Black-Scholes)")
    g = greeks
    print(f"  Delta:  {g['delta']:+.4f}  →  ${g['delta']*100*contracts:+,.0f} per $1 move  ({contracts} contract{'s' if contracts>1 else ''})")
    print(f"  Gamma:  {g['gamma']:.6f}")
    print(f"  Theta:  ${g['theta']*100*contracts:.2f}/day  →  ${g['theta']*100*contracts*30:.0f}/month")
    print(f"  Vega:   ${g['vega']*100*contracts:.2f} per 1% IV increase")
    print(f"  Rho:    ${g['rho']*100*contracts:.2f} per 1% rate change")

    # Monte Carlo
    header("🎲  MONTE CARLO  (100,000 simulations)")
    print(f"  {'Metric':<22} │ {'BS (GBM)':>11} │ {'Heston':>11} │ {'Bates':>11}")
    print(f"  {'─'*22}─┼─{'─'*11}─┼─{'─'*11}─┼─{'─'*11}")
    rows_mc = [
        ("MC Fair Value",    f"${mc['bs']['mc']:.2f}",   f"${mc['h']['mc']:.2f}",   f"${mc['b']['mc']:.2f}"),
        ("Prob of Profit",   f"{mc['bs']['prob']:.1f}%", f"{mc['h']['prob']:.1f}%", f"{mc['b']['prob']:.1f}%"),
        ("Prob of Max Loss", f"{100-mc['bs']['prob']:.1f}%", f"{100-mc['h']['prob']:.1f}%", f"{100-mc['b']['prob']:.1f}%"),
        ("Expected P&L",     f"${mc['bs']['ev']:+,.0f}", f"${mc['h']['ev']:+,.0f}", f"${mc['b']['ev']:+,.0f}"),
        ("Avg Win",          f"${mc['bs']['win']:+,.0f}", f"${mc['h']['win']:+,.0f}", f"${mc['b']['win']:+,.0f}"),
        ("Avg Loss",         f"${mc['bs']['loss']:+,.0f}", f"${mc['h']['loss']:+,.0f}", f"${mc['b']['loss']:+,.0f}"),
        ("Max Loss",         f"-${total_cost:,.0f}", f"-${total_cost:,.0f}", f"-${total_cost:,.0f}"),
    ]
    for row in rows_mc:
        print(f"  {row[0]:<22} │ {row[1]:>11} │ {row[2]:>11} │ {row[3]:>11}")

    rr_bs = mc['bs']['win'] / abs(mc['bs']['loss']) if mc['bs']['loss'] != 0 else 0
    rr_h  = mc['h']['win']  / abs(mc['h']['loss'])  if mc['h']['loss']  != 0 else 0
    rr_b  = mc['b']['win']  / abs(mc['b']['loss'])  if mc['b']['loss']  != 0 else 0
    print(f"  {'Reward:Risk':<22} │ {rr_bs:>9.1f}:1 │ {rr_h:>9.1f}:1 │ {rr_b:>9.1f}:1")

    # Target probabilities
    header("📈  PROBABILITY OF REACHING PRICE TARGETS")
    print(f"  {'Target':<10} │ {'BS':>8} │ {'Heston':>8} │ {'Bates':>8} │  P&L")
    print(f"  {'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼──────────────")

    targets = sorted(set([
        round(S * 0.90), round(S * 0.95), round(K),
        round(breakeven), round(K * 1.05), round(K * 1.10),
        round(K * 1.15), round(K * 1.25)
    ]))

    ST_bs = mc["ST_bs"]; St_h = mc["St_h"]; St_b = mc["St_b"]

    for t in targets:
        d2t = (np.log(S/t) + (r - q - 0.5*iv**2)*T) / (iv*np.sqrt(T))
        if opt_type == "C":
            bp  = norm.cdf(d2t) * 100
            hp  = (St_h >= t).mean() * 100
            btp = (St_b >= t).mean() * 100
            pl  = (max(t - K, 0) - premium) * 100 * contracts
        else:
            bp  = norm.cdf(-d2t) * 100
            hp  = (St_h <= t).mean() * 100
            btp = (St_b <= t).mean() * 100
            pl  = (max(K - t, 0) - premium) * 100 * contracts
        tag = " ← strike" if t == round(K) else (" ← BE" if t == round(breakeven) else "")
        print(f"  ${t:<9} │ {bp:>7.1f}% │ {hp:>7.1f}% │ {btp:>7.1f}% │  ${pl:+,.0f}{tag}")

    # Distribution
    header("📉  STOCK PRICE DISTRIBUTION AT EXPIRY")
    print(f"  {'Percentile':<14} │ {'BS':>10} │ {'Heston':>10} │ {'Bates':>10}")
    print(f"  {'─'*14}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
    for p, label in [(5,"5th  (bear)"),(25,"25th      "),(50,"50th (base)"),(75,"75th      "),(90,"90th (bull)"),(95,"95th      ")]:
        print(f"  {label}  │  ${np.percentile(ST_bs,p):>8.2f} │  ${np.percentile(St_h,p):>8.2f} │  ${np.percentile(St_b,p):>8.2f}")

    # Summary
    header("🔍  SUMMARY")

    prob_pct = mc["b"]["prob"] / 100.0
    loss_pct = 1.0 - prob_pct
    avg_win = mc["b"]["win"]
    avg_loss_signed = mc["b"]["loss"]
    ev_calc = prob_pct * avg_win + loss_pct * avg_loss_signed
    print(f"  Expected profit  =  (Win% × Avg Win) + (Loss% × Avg Loss)")
    print(f"                     =  ({prob_pct:.1%} × ${avg_win:,.0f}) + ({loss_pct:.1%} × ${avg_loss_signed:,.0f})")
    print(f"                     =  ${prob_pct * avg_win:,.0f} + ${loss_pct * avg_loss_signed:,.0f}  =  ${ev_calc:+,.0f}")
    print(f"  (Avg Loss is negative; positive EV = profitable on average over many trades.)")
    print()

    print(f"  What you paid:   ${premium:.2f}/share  →  ${total_cost:,.0f} total ({contracts} contract{'s' if contracts != 1 else ''})")
    b_win = mc["b"]["win"]
    exit_early = total_cost + ev_calc
    exit_max = total_cost + b_win
    exit_mid_4060 = 0.60 * exit_early + 0.40 * exit_max
    exit_mid_55 = 0.45 * exit_early + 0.55 * exit_max
    print(f"  ── 3-tier exit (proceeds to close trade) ──")
    print(f"  Early exit:   ${exit_early:,.0f}   (Cost + EV  =  ${total_cost:,.0f} + ${ev_calc:+,.0f})")
    print(f"  Mid (40/60):  ${exit_mid_4060:,.0f}   (40% toward max / 60% toward early)")
    print(f"  Mid (55/45):  ${exit_mid_55:,.0f}   (55% toward max / 45% toward early)")
    print(f"  Max exit:     ${exit_max:,.0f}   (Cost + Avg Win  =  ${total_cost:,.0f} + ${b_win:+,.0f}  — trade is a winner)")
    print()

    ev = mc['b']['ev']
    prob = mc['b']['prob']
    ev_str = f"${ev:+,.0f}"
    ev_flag = "✅ Positive EV" if ev > 0 else "❌ Negative EV"
    pb_flag = ("✅ Above 30% threshold" if prob >= 30
               else "⚠️  Near threshold (25-30%)" if prob >= 25
               else "❌ Below 25% threshold")
    fv_diff = premium - mc['b']['mc']
    fv_flag = "✅ At / below fair value" if fv_diff <= 0.50 else f"⚠️  Overpaid ${fv_diff:.2f} vs Bates"

    print(f"  Bates Prob of Profit:  {prob:.1f}%   {pb_flag}")
    print(f"  Bates Expected Value:  {ev_str}   {ev_flag}")
    print(f"  Entry vs Bates FV:     {fv_flag}")
    print(f"  Reward:Risk (Bates):   {rr_b:.1f}:1")
    print()
