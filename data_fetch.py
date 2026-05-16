"""Market data fetching: yahooquery -> yfinance -> direct Yahoo Finance API.

Returns dicts with price, premium, bid, ask, dividend yield, market cap, sector,
and industry — used by both the single-option flow and the chain scanner.
"""

import json
import time
import urllib.request
from datetime import date, datetime

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

_YAHOO_OPTIONS_HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")
_YAHOO_OPTIONS_PATH = "/v7/finance/options"
_YAHOO_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _yahoo_get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": _YAHOO_UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _fetch_yahoo_json(url: str, session_get=None):
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


def _nearest_expiry_ts(expiration_timestamps: list[int], target: date) -> int | None:
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
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def _find_option_row(contracts: list[dict], strike: float) -> dict | None:
    strike_f = float(strike)
    exact = [c for c in contracts if abs(float(c.get("strike", 0)) - strike_f) < 0.01]
    if exact:
        return exact[0]
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(float(c.get("strike", 0)) - strike_f))


def get_yf_session_get(ticker: str):
    """Return a callable for fetching Yahoo JSON using yfinance's cookie/crumb session."""
    if not _YF_AVAILABLE:
        return None
    try:
        yt = yf.Ticker(ticker)
        return lambda u, timeout=20: yt._data.get(url=u, timeout=timeout)
    except Exception:
        return None


def fetch_quote_and_profile(ticker: str) -> dict | None:
    """Fetch current price, dividend yield, market cap, sector, industry via yahooquery."""
    if not _YQ_AVAILABLE:
        return None
    ticker = ticker.upper().strip()
    try:
        t = YQTicker(ticker)
        S = None
        fd = getattr(t, "financial_data", None)
        if isinstance(fd, dict):
            row = fd.get(ticker) or fd.get(ticker.lower()) or (list(fd.values())[0] if fd else None)
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
        return {"S": S, "q": q, "market_cap": market_cap, "sector": sector, "industry": industry}
    except Exception:
        return None


def _fetch_via_yahooquery(ticker: str, strike: float, expiry_date: date, opt_type: str) -> dict | None:
    """Fetch stock price and option premium using yahooquery for a single contract."""
    if not _YQ_AVAILABLE:
        return None
    ticker = ticker.upper().strip()
    profile = fetch_quote_and_profile(ticker)
    if profile is None:
        return None
    try:
        t = YQTicker(ticker)
        chain = getattr(t, "option_chain", None)
        if chain is None or not hasattr(chain, "index") or chain.empty:
            return None
        df = chain
        if not hasattr(df.index, "get_level_values"):
            return None
        sym_level = df.index.get_level_values(0).unique()
        if len(sym_level) == 0:
            return None
        sym = sym_level[0]
        exps = df.index.get_level_values("expiration").unique()
        if len(exps) == 0:
            return None

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
                last = None
        except (TypeError, ValueError):
            last = None
        if bid > 0 and ask > 0:
            premium = (bid + ask) / 2.0
        elif last is not None and last > 0:
            premium = last
        else:
            return None
        oi = int(_get(row, "openInterest", 0) or 0)
        volume = int(_get(row, "volume", 0) or 0)
        iv_contract = _get(row, "impliedVolatility", None)
        try:
            iv_contract = float(iv_contract) if iv_contract is not None else None
        except (TypeError, ValueError):
            iv_contract = None

        return {
            "S": profile["S"],
            "premium": premium,
            "expiry_used": exp_str,
            "bid": bid,
            "ask": ask,
            "oi": oi,
            "volume": volume,
            "iv_contract": iv_contract,
            "q": profile["q"],
            "market_cap": profile["market_cap"],
            "sector": profile["sector"],
            "industry": profile["industry"],
        }
    except Exception:
        return None


def fetch_market_data(ticker: str, strike: float, expiry_date: date, opt_type: str) -> dict | None:
    """Fetch stock price + option premium. yahooquery -> yfinance session -> direct API."""
    ticker = ticker.upper().strip()
    out = _fetch_via_yahooquery(ticker, strike, expiry_date, opt_type)
    if out is not None:
        return out

    session_get = get_yf_session_get(ticker)
    time.sleep(1.2)
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
        return None
    results = data.get("optionChain", {}).get("result", [])
    if not results:
        return None
    res = results[0]
    quote = res.get("quote", {})
    S = quote.get("regularMarketPrice") or quote.get("currentPrice") or quote.get("previousClose")
    if S is None or (isinstance(S, (int, float)) and S <= 0):
        return None
    S = float(S)
    exp_dates = res.get("expirationDates") or []
    if not exp_dates:
        return None
    exp_ts = _nearest_expiry_ts(exp_dates, expiry_date)
    if exp_ts is None:
        return None
    time.sleep(0.8)
    url_exp = f"{used_host}{_YAHOO_OPTIONS_PATH}/{ticker}?date={exp_ts}"
    data_exp, _ = _fetch_yahoo_json(url_exp, session_get)
    if not data_exp:
        return None
    results_exp = data_exp.get("optionChain", {}).get("result", [])
    if not results_exp:
        return None
    opts = results_exp[0].get("options", [])
    if not opts:
        return None
    contracts = opts[0].get("calls" if opt_type == "C" else "puts", [])
    row = _find_option_row(contracts, strike)
    if not row:
        return None
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    last = row.get("lastPrice")
    if last is not None:
        try:
            last = float(last)
            if last != last:
                last = None
        except (TypeError, ValueError):
            last = None
    if bid > 0 and ask > 0:
        premium = (bid + ask) / 2.0
    elif last is not None and last > 0:
        premium = last
    else:
        return None
    return {
        "S": S,
        "premium": premium,
        "expiry_used": _ts_to_str(exp_ts),
        "bid": bid,
        "ask": ask,
        "oi": int(row.get("openInterest") or 0),
        "volume": int(row.get("volume") or 0),
        "iv_contract": float(row.get("impliedVolatility")) if row.get("impliedVolatility") else None,
    }
