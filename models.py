"""Closed-form pricing: Black-Scholes, Heston, Bates (SVJ), Greeks, and IV solver.

Heston and Bates use the Lewis (2001) characteristic-function integral.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq


def bs_price(S, K, T, r, q, sigma, opt_type="C"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, opt_type="C"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1 if opt_type == "C" else -1
    delta = sign * np.exp(-q * T) * norm.cdf(sign * d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -(S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
        - sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2)
        + sign * q * S * np.exp(-q * T) * norm.cdf(sign * d1)
    ) / 365
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    rho = sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2) / 100
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_vol(S, K, T, r, q, market_price, opt_type="C"):
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, q, s, opt_type) - market_price,
            1e-6, 20.0, xtol=1e-8,
        )
    except Exception:
        return None


def heston_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0, opt_type="C"):
    i = complex(0, 1)
    x = np.log(S / K) + (r - q) * T

    def integrand(u_real):
        # Lewis (2001): the CF is evaluated at the shifted complex argument
        # w = u_real − i/2. The e^{i u x} factor and the (u² + 1/4) denominator
        # stay in terms of the real integration variable u_real.
        w = complex(u_real, -0.5)
        d = np.sqrt((rho * xi * i * w - kappa) ** 2 + xi**2 * (i * w + w**2))
        g = (kappa - rho * xi * i * w - d) / (kappa - rho * xi * i * w + d)
        C = (
            (r - q) * i * w * T
            + (kappa * theta / xi**2)
            * ((kappa - rho * xi * i * w - d) * T
               - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        )
        D = (kappa - rho * xi * i * w - d) / xi**2 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        cf = np.exp(C + D * v0) * np.exp(complex(0, u_real) * x)
        return np.real(cf / (u_real**2 + 0.25))

    integral, _ = quad(integrand, 0, 300, limit=500, epsabs=1e-8, epsrel=1e-8)
    call = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    if opt_type == "C":
        return max(call, 0)
    return max(call - S * np.exp(-q * T) + K * np.exp(-r * T), 0)


def bates_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j, opt_type="C"):
    i = complex(0, 1)
    x = np.log(S / K) + (r - q) * T
    kj = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def integrand(u_real):
        w = complex(u_real, -0.5)
        d = np.sqrt((rho * xi * i * w - kappa) ** 2 + xi**2 * (i * w + w**2))
        g = (kappa - rho * xi * i * w - d) / (kappa - rho * xi * i * w + d)
        C = (
            (r - q) * i * w * T
            + (kappa * theta / xi**2)
            * ((kappa - rho * xi * i * w - d) * T
               - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        )
        D = (kappa - rho * xi * i * w - d) / xi**2 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        jump = lam_j * T * (np.exp(i * w * mu_j - 0.5 * sigma_j**2 * (w**2)) - 1 - i * w * kj)
        cf = np.exp(C + D * v0 + jump) * np.exp(complex(0, u_real) * x)
        return np.real(cf / (u_real**2 + 0.25))

    integral, _ = quad(integrand, 0, 300, limit=500, epsabs=1e-8, epsrel=1e-8)
    call = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    if opt_type == "C":
        return max(call, 0)
    return max(call - S * np.exp(-q * T) + K * np.exp(-r * T), 0)
