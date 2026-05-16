"""Heston/Bates parameter profiles selected by market cap and sector."""

from cli import get_float

PROFILES = {
    "1": ("Large-Cap Tech (GOOG, MSFT, META)",   dict(kappa=2.0, xi=0.45, rho=-0.70, lam_j=0.7,  mu_j=-0.055, sigma_j=0.09)),
    "2": ("Mega-Cap Stable (AAPL, AMZN)",        dict(kappa=2.5, xi=0.40, rho=-0.65, lam_j=0.5,  mu_j=-0.04,  sigma_j=0.08)),
    "3": ("Networking / Enterprise (CSCO, ORCL)",dict(kappa=2.0, xi=0.40, rho=-0.65, lam_j=0.6,  mu_j=-0.05,  sigma_j=0.08)),
    "4": ("Restaurant / Consumer (CMG, MCD)",    dict(kappa=2.0, xi=0.50, rho=-0.60, lam_j=0.8,  mu_j=-0.06,  sigma_j=0.10)),
    "5": ("High-Vol / Biotech",                  dict(kappa=1.5, xi=0.60, rho=-0.55, lam_j=1.2,  mu_j=-0.08,  sigma_j=0.15)),
    "6": ("Crypto / Speculative",                dict(kappa=1.0, xi=0.80, rho=-0.40, lam_j=2.0,  mu_j=-0.10,  sigma_j=0.20)),
    "7": ("Custom — enter my own",               None),
}


def _profile_from_market_data(market_cap: int | None, sector: str, industry: str) -> str:
    sector_lower = (sector or "").lower()
    industry_lower = (industry or "").lower()
    cap_b = (market_cap or 0) / 1e9
    if "crypto" in sector_lower or "crypto" in industry_lower or "bitcoin" in industry_lower:
        return "6"
    if "biotech" in industry_lower or "pharmaceutical" in industry_lower or ("healthcare" in sector_lower and cap_b < 50):
        return "5"
    if cap_b >= 500:
        return "1" if ("technology" in sector_lower or "communication" in sector_lower) else "2"
    if cap_b >= 50:
        if "technology" in sector_lower or "communication" in sector_lower:
            return "1"
        if "consumer" in sector_lower or "restaurant" in industry_lower or "retail" in industry_lower:
            return "4"
        return "3"
    if cap_b >= 10:
        return "3" if ("technology" in sector_lower or "communication" in sector_lower) else "4"
    return "5" if cap_b < 2 else "4"


def get_heston_bates_params_from_data(iv: float, market_cap: int | None, sector: str, industry: str) -> tuple:
    """Return (kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j, label)."""
    choice = _profile_from_market_data(market_cap, sector, industry)
    label, params = PROFILES[choice]
    if params is None:
        params = PROFILES["1"][1]
        label = PROFILES["1"][0]
    return (
        params["kappa"], iv**2, params["xi"], params["rho"], iv**2,
        params["lam_j"], params["mu_j"], params["sigma_j"], label,
    )


def get_heston_bates_params(iv: float, profile_override: str | None = None):
    """Interactive: ask user for profile or use override."""
    if profile_override is not None and profile_override in PROFILES:
        choice = profile_override
    else:
        print("\n  Select Heston/Bates parameter profile:")
        for k, (label, _) in PROFILES.items():
            print(f"    {k}.  {label}")
        while True:
            choice = input("\n  Profile [1-7, default 1]: ").strip() or "1"
            if choice in PROFILES:
                break
            print("  ⚠  Choose 1–7.")
    label, params = PROFILES[choice]
    if params is not None:
        print(f"\n  Using profile: {label}")
        kappa, xi, rho = params["kappa"], params["xi"], params["rho"]
        lam_j, mu_j, sigma_j = params["lam_j"], params["mu_j"], params["sigma_j"]
    else:
        print("\n  Enter custom Heston/Bates parameters:")
        kappa   = get_float("    κ  Mean reversion speed    [1.0–4.0, typical 2.0]: ", 0.01)
        xi      = get_float("    ξ  Vol of vol              [0.20–0.80, typical 0.40]: ", 0.01)
        rho     = get_float("    ρ  Spot-vol correlation    [-0.90–0.00, typical -0.65]: ", -0.99, 0.0)
        lam_j   = get_float("    λ  Jump intensity/yr      [0.0–3.0, typical 0.6]: ", 0)
        mu_j    = get_float("    μⱼ Mean jump size         [-0.15–0.0, typical -0.05]: ", -1.0, 0.0)
        sigma_j = get_float("    σⱼ Jump vol               [0.03–0.20, typical 0.08]: ", 0.001)
    return kappa, iv**2, xi, rho, iv**2, lam_j, mu_j, sigma_j
