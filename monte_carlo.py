"""Monte Carlo simulation engine for BS, Heston, and Bates models."""

import numpy as np


def run_mc(S, K, T, r, q, iv, premium, contracts, opt_type,
           kappa, theta, xi, rho, v0,
           lam_j, mu_j, sigma_j,
           n_sims=100_000):

    n_steps = max(int(T * 365), 1)
    dt = T / n_steps
    jump_comp = lam_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

    def payoff(ST):
        return np.maximum(ST - K, 0) if opt_type == "C" else np.maximum(K - ST, 0)

    # Black-Scholes MC
    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    ST_bs = S * np.exp((r - q - 0.5 * iv**2) * T + iv * np.sqrt(T) * Z)
    bs_pay = payoff(ST_bs)
    bs_mc = np.exp(-r * T) * np.mean(bs_pay)
    bs_pnl = (bs_pay - premium) * 100 * contracts
    bs_prob = (bs_pnl > 0).mean() * 100
    bs_ev = bs_pnl.mean()

    # Heston MC
    np.random.seed(42)
    St_h = np.full(n_sims, S)
    vt = np.full(n_sims, v0)
    for _ in range(n_steps):
        Z1 = np.random.standard_normal(n_sims)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sims)
        vp = np.maximum(vt, 0)
        vt = np.abs(vt + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z1)
        St_h = St_h * np.exp((r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * Z2)
    h_pay = payoff(St_h)
    h_mc = np.exp(-r * T) * np.mean(h_pay)
    h_pnl = (h_pay - premium) * 100 * contracts
    h_prob = (h_pnl > 0).mean() * 100
    h_ev = h_pnl.mean()
    h_win = h_pnl[h_pnl > 0].mean() if (h_pnl > 0).any() else 0
    h_loss = h_pnl[h_pnl <= 0].mean() if (h_pnl <= 0).any() else 0

    # Bates MC
    np.random.seed(42)
    St_b = np.full(n_sims, S)
    vt_b = np.full(n_sims, v0)
    for _ in range(n_steps):
        Z1 = np.random.standard_normal(n_sims)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sims)
        vp = np.maximum(vt_b, 0)
        vt_b = np.abs(vt_b + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z1)
        nj = np.random.poisson(lam_j * dt, n_sims)
        J = np.where(nj > 0,
                     np.exp(mu_j * nj + sigma_j * np.sqrt(np.maximum(nj, 1))
                            * np.random.standard_normal(n_sims)),
                     1.0)
        St_b = St_b * np.exp((r - q - jump_comp - 0.5 * vp) * dt
                             + np.sqrt(vp * dt) * Z2) * J
    b_pay = payoff(St_b)
    b_mc = np.exp(-r * T) * np.mean(b_pay)
    b_pnl = (b_pay - premium) * 100 * contracts
    b_prob = (b_pnl > 0).mean() * 100
    b_ev = b_pnl.mean()
    b_win = b_pnl[b_pnl > 0].mean() if (b_pnl > 0).any() else 0
    b_loss = b_pnl[b_pnl <= 0].mean() if (b_pnl <= 0).any() else 0

    return dict(
        ST_bs=ST_bs, St_h=St_h, St_b=St_b,
        bs=dict(mc=bs_mc, prob=bs_prob, ev=bs_ev,
                win=bs_pnl[bs_pnl > 0].mean() if (bs_pnl > 0).any() else 0,
                loss=bs_pnl[bs_pnl <= 0].mean() if (bs_pnl <= 0).any() else 0,
                pnl=bs_pnl),
        h=dict(mc=h_mc, prob=h_prob, ev=h_ev, win=h_win, loss=h_loss, pnl=h_pnl),
        b=dict(mc=b_mc, prob=b_prob, ev=b_ev, win=b_win, loss=b_loss, pnl=b_pnl),
    )


def quick_ev_bs(S, K, T, r, q, sigma, premium, opt_type, n_sims=20_000):
    """Lightweight BS-only Monte Carlo for shortlist scoring. Returns (ev_per_contract, prob_profit)."""
    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0) if opt_type == "C" else np.maximum(K - ST, 0)
    pnl = (payoff - premium) * 100
    return float(pnl.mean()), float((pnl > 0).mean() * 100)
