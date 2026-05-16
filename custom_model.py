#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          OPTIONS ANALYZER — BS / HESTON / BATES (SVJ)           ║
║                                                                  ║
║  Models:                                                         ║
║    • Black-Scholes (GBM)                                         ║
║    • Heston Stochastic Volatility                                ║
║    • Bates Stochastic Volatility + Jump Diffusion (SVJ)          ║
║                                                                  ║
║  Usage:  python custom_model.py                                 ║
║  Deps:   pip install numpy scipy pandas yahooquery                 ║
║          (or yfinance for fallback fetch; pandas for realized vol) ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import time
import urllib.request
from datetime import date, datetime
import warnings
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
warnings.filterwarnings('ignore')

try:
    from yahooquery import Ticker as YQTicker
    _YQ_AVAILABLE = True
except ImportError:
    _YQ_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# Yahoo options API; try query2 if query1 rate-limits (429)
_YAHOO_OPTIONS_HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")
_YAHOO_OPTIONS_PATH = "/v7/finance/options"
_YAHOO_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def get_float(prompt, min_val=None, max_val=None, allow_zero=False):
    while True:
        try:
            val = float(input(prompt))
            if not allow_zero and val == 0:
                print("  ⚠  Value cannot be zero.")
                continue
            if min_val is not None and val < min_val:
                print(f"  ⚠  Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  ⚠  Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  ⚠  Please enter a valid number.")


def get_float_or_default(prompt: str, default: float, min_val: float, max_val: float) -> float:
    """Read float from user; if they press Enter, return default. Otherwise validate in [min_val, max_val]."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if val < min_val:
                print(f"  ⚠  Must be >= {min_val}")
                continue
            if val > max_val:
                print(f"  ⚠  Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  ⚠  Please enter a valid number or press Enter for default.")


def get_date(prompt):
    while True:
        try:
            s = input(prompt).strip()
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            print("  ⚠  Use format YYYY-MM-DD  (e.g. 2027-06-17)")


def get_option_type():
    while True:
        t = input("  Option type [C/P]: ").strip().upper()
        if t in ("C", "P"):
            return t
        print("  ⚠  Enter C for Call or P for Put.")


# ══════════════════════════════════════════════════════════════════
#  MARKET DATA FETCH (yahooquery first, then yfinance/direct API)
# ══════════════════════════════════════════════════════════════════

def _fetch_via_yahooquery(ticker: str, strike: float, expiry_date: date, opt_type: str) -> dict | None:
    """
    Fetch stock price and option premium using yahooquery (Yahoo API wrapper).
    Docs: https://yahooquery.dpguthrie.com/guide/ticker/options/
    Returns dict with S, premium, expiry_used, bid, ask; or None on failure.
    """
    if not _YQ_AVAILABLE:
        return None
    ticker = ticker.upper().strip()
    try:
        t = YQTicker(ticker)
        # Price: financial_data has currentPrice (single symbol returns dict keyed by symbol)
        fd = getattr(t, "financial_data", None)
        if isinstance(fd, dict):
            # Symbol key may be upper or lower
            row = fd.get(ticker) or fd.get(ticker.lower()) or (list(fd.values())[0] if fd else None)
        else:
            row = None
        S = None
        if isinstance(row, dict):
            S = row.get("currentPrice")
        if S is None:
            quote = getattr(t, "quote", None)
            if isinstance(quote, dict):
                row = quote.get(ticker) or quote.get(ticker.lower()) or (list(quote.values())[0] if quote else None)
                if isinstance(row, dict):
                    S = row.get("regularMarketPrice") or row.get("currentPrice")
        if S is None or (isinstance(S, (int, float)) and S <= 0):
            return None
        S = float(S)
        # Option chain: DataFrame with MultiIndex (symbol, expiration, optionType)
        chain = getattr(t, "option_chain", None)
        if chain is None or not hasattr(chain, "index"):
            return None
        df = chain
        if df.empty or not hasattr(df.index, "get_level_values"):
            return None
        sym_level = df.index.get_level_values(0).unique()
        if len(sym_level) == 0:
            return None
        sym = sym_level[0]  # use symbol as returned by yahooquery (e.g. 'GOOGL' or 'googl')
        exps = df.index.get_level_values("expiration").unique()
        if len(exps) == 0:
            return None
        # Find expiration nearest to target
        target_ts = datetime(expiry_date.year, expiry_date.month, expiry_date.day)
        def exp_to_date(e):
            if hasattr(e, "date"):
                return e.date() if callable(getattr(e, "date")) else e
            try:
                return datetime.strptime(str(e)[:10], "%Y-%m-%d").date()
            except Exception:
                return None
        exp_dates = [(e, exp_to_date(e)) for e in exps]
        exp_dates = [(e, d) for e, d in exp_dates if d is not None]
        if not exp_dates:
            return None
        best_exp = min(exp_dates, key=lambda x: abs((x[1] - expiry_date).days))
        exp_used = best_exp[0]
        exp_str = best_exp[1].strftime("%Y-%m-%d")
        opt_type_key = "calls" if opt_type == "C" else "puts"
        try:
            sub = df.loc[(sym, exp_used, opt_type_key)]
        except (KeyError, TypeError):
            return None
        if hasattr(sub, "columns") and "strike" in sub.columns and len(sub) > 0:
            # Use integer position to get one row (index may repeat for same sym/exp/type)
            pos = (sub["strike"].astype(float) - float(strike)).abs().argmin()
            row = sub.iloc[pos]
        else:
            row = sub
        def _get(o, k, default=0):
            try:
                v = o.get(k, default) if hasattr(o, "get") else getattr(o, k, default)
                return v if v is not None else default
            except Exception:
                return default
        bid = float(_get(row, "bid", 0))
        ask = float(_get(row, "ask", 0))
        last = _get(row, "lastPrice", None)
        try:
            last = float(last) if last is not None else None
            if last is not None and (last != last or last <= 0):
                last = None  # NaN or invalid
        except (TypeError, ValueError):
            last = None
        if bid > 0 and ask > 0:
            premium = (bid + ask) / 2.0
        elif last is not None and last > 0:
            premium = last
        else:
            return None
        # Dividend yield, market cap, sector from summary_detail / asset_profile
        q = 0.0
        market_cap = None
        sector = ""
        industry = ""
        try:
            sd = getattr(t, "summary_detail", None)
            if isinstance(sd, dict):
                row = sd.get(ticker) or sd.get(ticker.lower()) or (list(sd.values())[0] if sd else None)
                if isinstance(row, dict):
                    if row.get("dividendYield") is not None:
                        q = float(row["dividendYield"])
                        if q > 1:
                            q = q / 100.0
                    if row.get("marketCap") is not None:
                        market_cap = int(row["marketCap"])
            ap = getattr(t, "asset_profile", None)
            if isinstance(ap, dict):
                row = ap.get(ticker) or ap.get(ticker.lower()) or (list(ap.values())[0] if ap else None)
                if isinstance(row, dict):
                    sector = (row.get("sector") or "") or ""
                    industry = (row.get("industry") or "") or ""
        except Exception:
            pass
        return {
            "S": S, "premium": premium, "expiry_used": exp_str, "bid": bid, "ask": ask,
            "q": q, "market_cap": market_cap, "sector": sector, "industry": industry,
        }
    except Exception:
        return None


def _yahoo_get(url: str) -> dict | None:
    """GET URL with User-Agent only; returns JSON or None. May get 401 without cookie/crumb."""
    req = urllib.request.Request(url, headers={"User-Agent": _YAHOO_UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None




def _nearest_expiry_ts(expiration_timestamps: list[int], target: date) -> int | None:
    """Return unix timestamp from list nearest to target date."""
    if not expiration_timestamps:
        return None
    target_ts = int(datetime(target.year, target.month, target.day).timestamp())
    best = expiration_timestamps[0]
    best_diff = abs(best - target_ts)
    for ts in expiration_timestamps[1:]:
        d = abs(ts - target_ts)
        if d < best_diff:
            best_diff = d
            best = ts
    return best


def _ts_to_str(ts: int) -> str:
    """Unix timestamp to YYYY-MM-DD."""
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def _find_option_row(contracts: list[dict], strike: float, opt_type: str) -> dict | None:
    """Find exact or nearest strike in calls/puts list. Returns one contract dict."""
    strike_f = float(strike)
    exact = [c for c in contracts if abs(float(c.get("strike", 0)) - strike_f) < 0.01]
    if exact:
        return exact[0]
    if not contracts:
        return None
    nearest = min(contracts, key=lambda c: abs(float(c.get("strike", 0)) - strike_f))
    return nearest


def _fetch_yahoo_json(url: str, session_get=None):
    """
    Fetch URL and return (parsed_json, status_code).
    status_code is None for urllib path. Prefer session_get(url) when provided (yfinance cookie/crumb).
    """
    if session_get is not None:
        try:
            resp = session_get(url, timeout=20)
            code = getattr(resp, "status_code", 200)
            if code != 200:
                return None, code
            return resp.json(), code
        except Exception:
            return None, None
    data = _yahoo_get(url)
    return data, 200 if data else None


def fetch_market_data(ticker: str, strike: float, expiry_date: date, opt_type: str) -> dict | None:
    """
    Fetch current stock price and option premium from Yahoo Finance.
    Tries yahooquery first (recommended); then yfinance session; then direct API.
    Returns dict with S, premium, expiry_used, bid, ask; or None on failure.
    """
    ticker = ticker.upper().strip()
    # 1) yahooquery (recommended: uses Yahoo API, avoids rate-limit issues)
    out = _fetch_via_yahooquery(ticker, strike, expiry_date, opt_type)
    if out is not None:
        return out
    # 2) yfinance cookie/crumb or direct Yahoo API
    session_get = None
    if _YF_AVAILABLE:
        try:
            yt = yf.Ticker(ticker)
            session_get = lambda u, timeout=20: yt._data.get(url=u, timeout=timeout)  # noqa: E731
        except Exception:
            pass
    time.sleep(1.2)  # reduce rate-limit chance
    data = None
    used_host = _YAHOO_OPTIONS_HOSTS[0]
    for host in _YAHOO_OPTIONS_HOSTS:
        url_all = f"{host}{_YAHOO_OPTIONS_PATH}/{ticker}"
        data, status = _fetch_yahoo_json(url_all, session_get)
        if data and data.get("optionChain", {}).get("result"):
            used_host = host
            break
        if status == 429:
            time.sleep(2.5)
            data, _ = _fetch_yahoo_json(url_all, session_get)
            if data and data.get("optionChain", {}).get("result"):
                used_host = host
                break
        data = None
    if not data or not data.get("optionChain", {}).get("result"):
        print(f"  ⚠  Could not reach Yahoo Finance for {ticker} (rate limit or unavailable).")
        if not _YQ_AVAILABLE:
            print("  →  Install yahooquery for reliable fetch:  pip install yahooquery")
        print("  →  Enter price and premium manually below, or try again in a minute.")
        return None
    results = data.get("optionChain", {}).get("result", [])
    if not results:
        print(f"  ⚠  No option chain result for {ticker}")
        return None
    res = results[0]
    quote = res.get("quote", {})
    S = quote.get("regularMarketPrice") or quote.get("currentPrice")
    if S is None:
        S = quote.get("previousClose")
    if S is None or (isinstance(S, (int, float)) and S <= 0):
        print(f"  ⚠  No stock price for {ticker}")
        return None
    S = float(S)
    exp_dates = res.get("expirationDates") or []
    if not exp_dates:
        print(f"  ⚠  No option expirations for {ticker}")
        return None
    exp_ts = _nearest_expiry_ts(exp_dates, expiry_date)
    if exp_ts is None:
        return None
    time.sleep(0.8)  # avoid back-to-back requests
    url_exp = f"{used_host}{_YAHOO_OPTIONS_PATH}/{ticker}?date={exp_ts}"
    data_exp, _ = _fetch_yahoo_json(url_exp, session_get)
    if not data_exp:
        print(f"  ⚠  Could not fetch option chain for expiry")
        return None
    results_exp = data_exp.get("optionChain", {}).get("result", [])
    if not results_exp:
        return None
    opts = results_exp[0].get("options", [])
    if not opts:
        print(f"  ⚠  No options for that expiry")
        return None
    contracts = opts[0].get("calls" if opt_type == "C" else "puts", [])
    row = _find_option_row(contracts, strike, opt_type)
    if not row:
        print(f"  ⚠  No {opt_type} option for strike {strike}")
        return None
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    last = row.get("lastPrice")
    if last is not None:
        try:
            last = float(last)
            if last != last:
                last = None  # NaN
        except (TypeError, ValueError):
            last = None
    if bid > 0 and ask > 0:
        premium = (bid + ask) / 2.0
    elif last is not None and last > 0:
        premium = last
    else:
        print("  ⚠  No bid/ask/last for this option")
        return None
    expiry_str = _ts_to_str(exp_ts)
    return {
        "S": S,
        "premium": premium,
        "expiry_used": expiry_str,
        "bid": bid,
        "ask": ask,
    }


# ══════════════════════════════════════════════════════════════════
#  REALIZED VOL (historical vol from price returns, by option tenor)
# ══════════════════════════════════════════════════════════════════

def _lookback_for_tenor(days_to_expiry: int) -> tuple[int, str]:
    """
    Choose lookback trading days and label from option tenor.
    30d = short-dated, 60d = near-term, 90d = options benchmark,
    252d = LEAPS, 504d = 2y (smooths extremes).
    """
    if days_to_expiry <= 45:
        return 30, "30-day (short-dated)"
    if days_to_expiry <= 90:
        return 60, "60-day (near-term)"
    if days_to_expiry <= 180:
        return 90, "90-day (options benchmark)"
    if days_to_expiry <= 400:
        return 252, "1-year (LEAPS)"
    return 504, "2-year (smoothed)"


def fetch_realized_vol(ticker: str, lookback_trading_days: int) -> tuple[float | None, str]:
    """
    Fetch daily closes, compute annualized realized vol from log returns.
    Returns (vol, label) or (None, error_msg). Uses yahooquery then yfinance.
    """
    try:
        import pandas as pd
    except ImportError:
        return None, "pandas not installed (pip install pandas)"
    ticker = ticker.upper().strip()
    # Request enough calendar days: ~1.5 * trading days
    if lookback_trading_days <= 30:
        period = "3mo"
    elif lookback_trading_days <= 90:
        period = "6mo"
    elif lookback_trading_days <= 252:
        period = "1y"
    else:
        period = "2y"
    closes = None
    if _YQ_AVAILABLE:
        try:
            t = YQTicker(ticker)
            df = t.history(period=period, interval="1d")
            if df is not None and not df.empty:
                close_col = "close" if "close" in df.columns else "Close"
                if close_col not in df.columns and len(df.columns):
                    close_col = df.columns[0]
                if close_col in df.columns:
                    closes = df[close_col].dropna()
                    if isinstance(closes.index, pd.DatetimeIndex):
                        closes = closes.sort_index()
        except Exception:
            pass
    if (closes is None or len(closes) < 2) and _YF_AVAILABLE:
        try:
            yt = yf.Ticker(ticker)
            df = yt.history(period=period, interval="1d")
            if df is not None and not df.empty:
                close_col = "Close" if "Close" in df.columns else "close"
                if close_col in df.columns:
                    closes = df[close_col].dropna()
                    if isinstance(closes.index, pd.DatetimeIndex):
                        closes = closes.sort_index()
        except Exception:
            pass
    if closes is None or len(closes) < 2:
        return None, "could not fetch enough price history"
    closes = closes.iloc[-lookback_trading_days:] if len(closes) >= lookback_trading_days else closes
    if len(closes) < 2:
        return None, "not enough trading days in history"
    log_returns = np.log(closes.astype(float).values[1:] / closes.astype(float).values[:-1])
    realized_vol = float(np.std(log_returns) * np.sqrt(252))
    if realized_vol <= 0 or not np.isfinite(realized_vol):
        return None, "realized vol computation failed"
    return realized_vol, f"{lookback_trading_days}-day"


def bs_price(S, K, T, r, q, sigma, opt_type="C"):
    """Black-Scholes closed form."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
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
    vega  = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    rho   = sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2) / 100
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_vol(S, K, T, r, q, market_price, opt_type="C"):
    """Solve for IV from market price."""
    try:
        return brentq(lambda s: bs_price(S, K, T, r, q, s, opt_type) - market_price,
                      1e-6, 20.0, xtol=1e-8)
    except Exception:
        return None


def heston_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0, opt_type="C"):
    """Heston model via Lewis (2001) formula."""
    i  = complex(0, 1)
    x  = np.log(S / K) + (r - q) * T

    def integrand(u):
        u = complex(u, -0.5)
        d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
        g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
        C = ((r - q) * i * u * T
             + (kappa * theta / xi**2)
             * ((kappa - rho * xi * i * u - d) * T
                - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
        D = ((kappa - rho * xi * i * u - d) / xi**2
             * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        cf = np.exp(C + D * v0 + i * u * x)
        return np.real(cf / (u**2 + 0.25))

    integral, _ = quad(integrand, 0, 300, limit=500, epsabs=1e-8, epsrel=1e-8)
    call = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    if opt_type == "C":
        return max(call, 0)
    else:
        return max(call - S * np.exp(-q * T) + K * np.exp(-r * T), 0)  # put-call parity


def bates_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0,
                lam_j, mu_j, sigma_j, opt_type="C"):
    """Bates (SVJ) model via Lewis formula."""
    i  = complex(0, 1)
    x  = np.log(S / K) + (r - q) * T
    kj = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def integrand(u):
        u = complex(u, -0.5)
        d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
        g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
        C = ((r - q) * i * u * T
             + (kappa * theta / xi**2)
             * ((kappa - rho * xi * i * u - d) * T
                - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
        D = ((kappa - rho * xi * i * u - d) / xi**2
             * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        jump = lam_j * T * (np.exp(i * u * mu_j - 0.5 * sigma_j**2 * (u**2)) - 1 - i * u * kj)
        cf   = np.exp(C + D * v0 + i * u * x + jump)
        return np.real(cf / (u**2 + 0.25))

    integral, _ = quad(integrand, 0, 300, limit=500, epsabs=1e-8, epsrel=1e-8)
    call = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    if opt_type == "C":
        return max(call, 0)
    else:
        return max(call - S * np.exp(-q * T) + K * np.exp(-r * T), 0)


# ══════════════════════════════════════════════════════════════════
#  MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════

def run_mc(S, K, T, r, q, iv, premium, contracts, opt_type,
           kappa, theta, xi, rho, v0,
           lam_j, mu_j, sigma_j,
           n_sims=100_000):

    n_steps   = max(int(T * 365), 1)
    dt        = T / n_steps
    jump_comp = lam_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    sign      = 1  # payoff always based on call; put handled via put-call later

    def payoff(ST):
        if opt_type == "C":
            return np.maximum(ST - K, 0)
        else:
            return np.maximum(K - ST, 0)

    # ── Black-Scholes MC ─────────────────────────────────────────
    np.random.seed(42)
    Z     = np.random.standard_normal(n_sims)
    ST_bs = S * np.exp((r - q - 0.5 * iv**2) * T + iv * np.sqrt(T) * Z)
    bs_pay = payoff(ST_bs)
    bs_mc  = np.exp(-r * T) * np.mean(bs_pay)
    bs_pnl = (bs_pay - premium) * 100 * contracts
    bs_prob = (bs_pnl > 0).mean() * 100
    bs_ev   = bs_pnl.mean()

    # ── Heston MC ────────────────────────────────────────────────
    np.random.seed(42)
    St_h = np.full(n_sims, S)
    vt   = np.full(n_sims, v0)
    for _ in range(n_steps):
        Z1  = np.random.standard_normal(n_sims)
        Z2  = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sims)
        vp  = np.maximum(vt, 0)
        vt  = np.abs(vt + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z1)
        St_h = St_h * np.exp((r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * Z2)
    h_pay  = payoff(St_h)
    h_mc   = np.exp(-r * T) * np.mean(h_pay)
    h_pnl  = (h_pay - premium) * 100 * contracts
    h_prob = (h_pnl > 0).mean() * 100
    h_ev   = h_pnl.mean()
    h_win  = h_pnl[h_pnl > 0].mean() if (h_pnl > 0).any() else 0
    h_loss = h_pnl[h_pnl <= 0].mean() if (h_pnl <= 0).any() else 0

    # ── Bates MC ─────────────────────────────────────────────────
    np.random.seed(42)
    St_b = np.full(n_sims, S)
    vt_b = np.full(n_sims, v0)
    for _ in range(n_steps):
        Z1   = np.random.standard_normal(n_sims)
        Z2   = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sims)
        vp   = np.maximum(vt_b, 0)
        vt_b = np.abs(vt_b + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z1)
        nj   = np.random.poisson(lam_j * dt, n_sims)
        J    = np.where(nj > 0,
                        np.exp(mu_j * nj + sigma_j
                               * np.sqrt(np.maximum(nj, 1))
                               * np.random.standard_normal(n_sims)),
                        1.0)
        St_b = St_b * np.exp((r - q - jump_comp - 0.5 * vp) * dt
                              + np.sqrt(vp * dt) * Z2) * J
    b_pay  = payoff(St_b)
    b_mc   = np.exp(-r * T) * np.mean(b_pay)
    b_pnl  = (b_pay - premium) * 100 * contracts
    b_prob = (b_pnl > 0).mean() * 100
    b_ev   = b_pnl.mean()
    b_win  = b_pnl[b_pnl > 0].mean() if (b_pnl > 0).any() else 0
    b_loss = b_pnl[b_pnl <= 0].mean() if (b_pnl <= 0).any() else 0

    return dict(
        ST_bs=ST_bs, St_h=St_h, St_b=St_b,
        bs=dict(mc=bs_mc, prob=bs_prob, ev=bs_ev,
                win=bs_pnl[bs_pnl>0].mean() if (bs_pnl>0).any() else 0,
                loss=bs_pnl[bs_pnl<=0].mean() if (bs_pnl<=0).any() else 0,
                pnl=bs_pnl),
        h=dict(mc=h_mc, prob=h_prob, ev=h_ev, win=h_win, loss=h_loss, pnl=h_pnl),
        b=dict(mc=b_mc, prob=b_prob, ev=b_ev, win=b_win, loss=b_loss, pnl=b_pnl),
    )


# ══════════════════════════════════════════════════════════════════
#  PRINT REPORT
# ══════════════════════════════════════════════════════════════════

def print_report(ticker, S, K, T, r, q, iv, premium, contracts,
                 opt_type, days, expiry_str,
                 bs_fv, h_fv, b_fv,
                 greeks, mc):

    total_cost = premium * 100 * contracts
    if opt_type == "C":
        breakeven = K + premium
        be_label  = f"${breakeven:.2f}  (+{((breakeven/S)-1)*100:.1f}% from spot)"
    else:
        breakeven = K - premium
        be_label  = f"${breakeven:.2f}  (-{((1-(breakeven/S))*100):.1f}% from spot)"

    moneyness = (S / K - 1) * 100
    itm_otm   = "ITM" if (opt_type == "C" and S > K) or (opt_type == "P" and S < K) else "OTM"

    W = 66

    def header(title):
        print(f"\n{'━'*W}")
        print(f"  {title}")
        print(f"{'━'*W}")

    print("\n" + "=" * W)
    print(f"  {ticker.upper()} ${K:.0f} {'CALL' if opt_type=='C' else 'PUT'}  |  {expiry_str}  |  {days}d")
    print(f"  BS / HESTON / BATES ANALYSIS  —  {date.today().strftime('%b %d, %Y')}")
    print("=" * W)

    # ── Setup ────────────────────────────────────────────────────
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
    for k, v in rows:
        print(f"  {k:<18} {v}")

    # ── Fair Value ───────────────────────────────────────────────
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

    # ── Greeks ───────────────────────────────────────────────────
    header("📐  GREEKS  (Black-Scholes)")
    g = greeks
    print(f"  Delta:  {g['delta']:+.4f}  →  ${g['delta']*100*contracts:+,.0f} per $1 move  ({contracts} contract{'s' if contracts>1 else ''})")
    print(f"  Gamma:  {g['gamma']:.6f}")
    print(f"  Theta:  ${g['theta']*100*contracts:.2f}/day  →  ${g['theta']*100*contracts*30:.0f}/month")
    print(f"  Vega:   ${g['vega']*100*contracts:.2f} per 1% IV increase")
    print(f"  Rho:    ${g['rho']*100*contracts:.2f} per 1% rate change")

    # ── Monte Carlo ──────────────────────────────────────────────
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

    # Reward:Risk
    for label, w, l in [("BS", mc['bs']['win'], mc['bs']['loss']),
                         ("Heston", mc['h']['win'], mc['h']['loss']),
                         ("Bates",  mc['b']['win'], mc['b']['loss'])]:
        pass  # computed inline below
    rr_bs = mc['bs']['win'] / abs(mc['bs']['loss']) if mc['bs']['loss'] != 0 else 0
    rr_h  = mc['h']['win']  / abs(mc['h']['loss'])  if mc['h']['loss']  != 0 else 0
    rr_b  = mc['b']['win']  / abs(mc['b']['loss'])  if mc['b']['loss']  != 0 else 0
    print(f"  {'Reward:Risk':<22} │ {rr_bs:>9.1f}:1 │ {rr_h:>9.1f}:1 │ {rr_b:>9.1f}:1")

    # ── Target Probabilities ─────────────────────────────────────
    header("📈  PROBABILITY OF REACHING PRICE TARGETS")
    print(f"  {'Target':<10} │ {'BS':>8} │ {'Heston':>8} │ {'Bates':>8} │  P&L")
    print(f"  {'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼──────────────")

    # Build target list around key levels
    targets = sorted(set([
        round(S * 0.90), round(S * 0.95), round(K),
        round(breakeven), round(K * 1.05), round(K * 1.10),
        round(K * 1.15), round(K * 1.25)
    ]))

    ST_bs = mc["ST_bs"]; St_h = mc["St_h"]; St_b = mc["St_b"]

    for t in targets:
        d2t  = (np.log(S/t) + (r - q - 0.5*iv**2)*T) / (iv*np.sqrt(T))
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

    # ── Price Distribution ───────────────────────────────────────
    header("📉  STOCK PRICE DISTRIBUTION AT EXPIRY")
    print(f"  {'Percentile':<14} │ {'BS':>10} │ {'Heston':>10} │ {'Bates':>10}")
    print(f"  {'─'*14}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
    for p, label in [(5,"5th  (bear)"),(25,"25th      "),(50,"50th (base)"),(75,"75th      "),(90,"90th (bull)"),(95,"95th      ")]:
        print(f"  {label}  │  ${np.percentile(ST_bs,p):>8.2f} │  ${np.percentile(St_h,p):>8.2f} │  ${np.percentile(St_b,p):>8.2f}")

    # ── Summary ──────────────────────────────────────────────────
    header("🔍  SUMMARY")

    # Expected profit formula: use signed Avg Loss so EV = Win%*AvgWin + Loss%*AvgLoss
    prob_pct = mc["b"]["prob"] / 100.0
    loss_pct = 1.0 - prob_pct
    avg_win = mc["b"]["win"]
    avg_loss_signed = mc["b"]["loss"]  # negative number
    ev_calc = prob_pct * avg_win + loss_pct * avg_loss_signed
    print(f"  Expected profit  =  (Win% × Avg Win) + (Loss% × Avg Loss)")
    print(f"                     =  ({prob_pct:.1%} × ${avg_win:,.0f}) + ({loss_pct:.1%} × ${avg_loss_signed:,.0f})")
    print(f"                     =  ${prob_pct * avg_win:,.0f} + ${loss_pct * avg_loss_signed:,.0f}  =  ${ev_calc:+,.0f}")
    print(f"  (Avg Loss is negative; positive EV = profitable on average over many trades.)")
    print()

    # What you paid and 3-tier exit plan (Bates)
    print(f"  What you paid:   ${premium:.2f}/share  →  ${total_cost:,.0f} total ({contracts} contract{'s' if contracts != 1 else ''})")
    b_win = mc["b"]["win"]
    exit_early = total_cost + ev_calc
    exit_max = total_cost + b_win
    # Mid: 40% toward max / 60% toward early (replaces even 50/50 blend)
    exit_mid_4060 = 0.60 * exit_early + 0.40 * exit_max
    exit_mid_55 = 0.45 * exit_early + 0.55 * exit_max
    print(f"  ── 3-tier exit (proceeds to close trade) ──")
    print(f"  Early exit:   ${exit_early:,.0f}   (Cost + EV  =  ${total_cost:,.0f} + ${ev_calc:+,.0f})")
    print(f"  Mid (40/60):  ${exit_mid_4060:,.0f}   (40% toward max / 60% toward early)")
    print(f"  Mid (55/45):  ${exit_mid_55:,.0f}   (55% toward max / 45% toward early)")
    print(f"  Max exit:     ${exit_max:,.0f}   (Cost + Avg Win  =  ${total_cost:,.0f} + ${b_win:+,.0f}  — trade is a winner)")
    print()

    ev      = mc['b']['ev']
    prob    = mc['b']['prob']
    ev_str  = f"${ev:+,.0f}"
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


# ══════════════════════════════════════════════════════════════════
#  HESTON / BATES PARAMETER PROFILES
# ══════════════════════════════════════════════════════════════════

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
    """Pick Heston/Bates profile 1–6 from market cap and sector. Never returns 7."""
    sector_lower = (sector or "").lower()
    industry_lower = (industry or "").lower()
    cap_b = (market_cap or 0) / 1e9
    if "crypto" in sector_lower or "crypto" in industry_lower or "bitcoin" in industry_lower:
        return "6"
    if "biotech" in industry_lower or "pharmaceutical" in industry_lower or "healthcare" in sector_lower and cap_b < 50:
        return "5"
    if cap_b >= 500:
        return "2" if "technology" not in sector_lower and "communication" not in sector_lower else "1"
    if cap_b >= 50:
        if "technology" in sector_lower or "communication" in sector_lower:
            return "1"
        if "consumer" in sector_lower or "restaurant" in industry_lower or "retail" in industry_lower:
            return "4"
        return "3"
    if cap_b >= 10:
        return "3" if "technology" in sector_lower or "communication" in sector_lower else "4"
    return "5" if cap_b < 2 else "4"


def get_heston_bates_params_from_data(iv: float, market_cap: int | None, sector: str, industry: str) -> tuple:
    """Return (kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j) using profile chosen from market data."""
    choice = _profile_from_market_data(market_cap, sector, industry)
    label, params = PROFILES[choice]
    if params is None:
        params = PROFILES["1"][1]
        label = PROFILES["1"][0]
    kappa = params["kappa"]
    xi = params["xi"]
    rho = params["rho"]
    lam_j = params["lam_j"]
    mu_j = params["mu_j"]
    sigma_j = params["sigma_j"]
    theta = iv**2
    v0 = iv**2
    return kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j, label


def get_heston_bates_params(iv, profile_override: str | None = None):
    """Interactive: ask user for profile. If profile_override is set (e.g. from data), use it and do not ask."""
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
    theta = iv**2
    v0 = iv**2
    return kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

_DEFAULT_R = 0.043  # risk-free rate (4.3%), used when data is fetched


def main():
    print("\n" + "═" * 66)
    print("       OPTIONS ANALYZER  —  BS / HESTON / BATES (SVJ)")
    print("═" * 66)

    while True:
        print("\n── OPTION (ticker, type, strike, expiry) ──────────────────────")
        ticker   = input("  Ticker symbol: ").strip().upper() or "?"
        opt_type = get_option_type()
        K        = get_float("  Strike price ($): ", 0.01)
        exp_date = get_date("  Expiration date (YYYY-MM-DD): ")
        data     = fetch_market_data(ticker, K, exp_date, opt_type)
        if data:
            S       = data["S"]
            premium = data["premium"]
            q       = data.get("q", 0.0)
            r       = _DEFAULT_R
            exp_used = data["expiry_used"]
            try:
                exp_date = datetime.strptime(exp_used, "%Y-%m-%d").date()
            except ValueError:
                pass
            profile_label = ""
        else:
            data = None
            print("  ⚠  Fetch failed. Enter manually:")
            S       = get_float("  Current stock price ($): ", 0.01)
            premium = get_float("  Premium per share ($): ", 0.01)
            print("\n  Market parameters:")
            r = get_float("  Risk-free rate [e.g. 0.043 for 4.3%]: ", 0, 1)
            q = get_float("  Dividend yield [0 if none]: ", 0, 1, allow_zero=True)
            profile_label = ""
        today     = date.today()
        days      = (exp_date - today).days
        if days <= 0:
            print("  ⚠  Expiration must be in the future.")
            continue
        T         = days / 365
        expiry_str = exp_date.strftime("%b %d, %Y")
        contracts = int(get_float("  Number of contracts [default 1]: ", 1, allow_zero=False))

        # Solve IV from market price
        iv = implied_vol(S, K, T, r, q, premium, opt_type)
        if iv is None:
            print("  ⚠  Could not solve implied vol — using 30% as fallback.")
            iv = 0.30
        else:
            print(f"\n  ✅ Implied vol from ${premium:.2f} market price:  {iv*100:.2f}%")
        print(f"  ⚠️  Market is charging {iv*100:.1f}% IV — historical vol used to benchmark fair value")
        lookback_days, lookback_label = _lookback_for_tenor(days)
        realized_vol, realized_label = fetch_realized_vol(ticker, lookback_days)
        if realized_vol is not None:
            print(f"  {lookback_label} realized vol: {realized_vol*100:.2f}%")
            hist_vol = realized_vol
        else:
            print(f"  ⚠  {realized_label} — enter historical vol manually.")
            hist_vol = get_float("  Historical vol [e.g. 0.285 for 28.5%]: ", 0.01, 5.0)
        print(f"  Vol premium you're paying: {(iv - hist_vol)*100:+.1f}%")

        entry_premium = get_float_or_default(
            f"  Price you paid or target entry per share ($)? [Enter = market ${premium:.2f}]: ", premium, 0.01, 9999.0
        )

        # Heston / Bates parameters: from market cap & sector when available, else default profile 1 (no prompt)
        if data and data.get("market_cap") is not None:
            kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j, profile_label = get_heston_bates_params_from_data(
                hist_vol, data.get("market_cap"), data.get("sector", ""), data.get("industry", "")
            )
            print(f"  Using profile from data: {profile_label}")
        else:
            kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j = get_heston_bates_params(hist_vol, profile_override="1")

        # ── Run models ───────────────────────────────────────────
        print("\n  Running models...")

        bs_fv   = bs_price(S, K, T, r, q, hist_vol, opt_type)
        greeks  = bs_greeks(S, K, T, r, q, hist_vol, opt_type)

        print("  ├─ Black-Scholes ✅")
        h_fv = heston_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0, opt_type)
        print("  ├─ Heston ✅")
        b_fv = bates_lewis(S, K, T, r, q, kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j, opt_type)
        print("  ├─ Bates ✅")
        mc   = run_mc(S, K, T, r, q, hist_vol, entry_premium, contracts, opt_type,
                      kappa, theta, xi, rho, v0, lam_j, mu_j, sigma_j)
        print("  └─ Monte Carlo (100k paths) ✅")

        # ── Print report ─────────────────────────────────────────
        print_report(ticker, S, K, T, r, q, iv, entry_premium, contracts,
                     opt_type, days, expiry_str,
                     bs_fv, h_fv, b_fv, greeks, mc)

        again = input("  Analyze another option? [Y/N]: ").strip().upper()
        if again != "Y":
            print("\n  Done.\n")
            break


if __name__ == "__main__":
    main()