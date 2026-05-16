"""Scan the option chain across canonical DTE windows and a near-money strike range.

Used by the ticker-first flow to gather candidate options before scoring.
Targets the windows professionals actually watch (21/30/45/60/90 DTE)
and restricts strikes to ±20% of spot to keep API load bounded.
"""

from datetime import date, datetime, timedelta

try:
    from yahooquery import Ticker as YQTicker
    _YQ_AVAILABLE = True
except ImportError:
    _YQ_AVAILABLE = False

DEFAULT_TARGET_DTES = (21, 30, 45, 60, 90)
STRIKE_RANGE_PCT = 0.20  # ±20% of spot


def _to_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _pick_expiries(all_expiries: list[date], target_dtes: tuple[int, ...]) -> list[date]:
    today = date.today()
    chosen: list[date] = []
    for tdte in target_dtes:
        target = today + timedelta(days=tdte)
        future = [d for d in all_expiries if d >= today]
        if not future:
            continue
        best = min(future, key=lambda d: abs((d - target).days))
        if best not in chosen:
            chosen.append(best)
    return chosen


def scan_chain(ticker: str, spot: float, target_dtes: tuple[int, ...] = DEFAULT_TARGET_DTES,
               strike_range_pct: float = STRIKE_RANGE_PCT) -> list[dict]:
    """Return list of candidate contracts across the chosen expiries and strike range.

    Each contract dict has: ticker, type ('C'/'P'), strike, expiry (date), dte,
    bid, ask, last, premium, oi, volume, iv_contract.
    """
    if not _YQ_AVAILABLE:
        return []
    ticker = ticker.upper().strip()
    lo = spot * (1.0 - strike_range_pct)
    hi = spot * (1.0 + strike_range_pct)
    out: list[dict] = []
    try:
        t = YQTicker(ticker)
        chain = getattr(t, "option_chain", None)
        if chain is None or not hasattr(chain, "index") or chain.empty:
            return []
        df = chain
        if not hasattr(df.index, "get_level_values"):
            return []
        sym_level = df.index.get_level_values(0).unique()
        if len(sym_level) == 0:
            return []
        sym = sym_level[0]
        exps_raw = df.index.get_level_values("expiration").unique()
        exp_pairs = [(e, _to_date(e)) for e in exps_raw]
        exp_pairs = [(e, d) for e, d in exp_pairs if d is not None]
        if not exp_pairs:
            return []
        all_dates = sorted({d for _, d in exp_pairs})
        chosen_dates = _pick_expiries(all_dates, target_dtes)
        if not chosen_dates:
            return []
        chosen_raw = [e for e, d in exp_pairs if d in chosen_dates]
        today = date.today()
        for raw_exp in chosen_raw:
            exp_d = _to_date(raw_exp)
            if exp_d is None:
                continue
            dte = (exp_d - today).days
            for opt_type, key in (("C", "calls"), ("P", "puts")):
                try:
                    sub = df.loc[(sym, raw_exp, key)]
                except (KeyError, TypeError):
                    continue
                if not hasattr(sub, "columns") or "strike" not in sub.columns:
                    continue
                strikes = sub["strike"].astype(float)
                mask = (strikes >= lo) & (strikes <= hi)
                sub_f = sub[mask]
                if sub_f.empty:
                    continue
                for _, row in sub_f.iterrows():
                    bid = float(row.get("bid") or 0)
                    ask = float(row.get("ask") or 0)
                    last_v = row.get("lastPrice")
                    try:
                        last_v = float(last_v) if last_v is not None else None
                        if last_v is not None and (last_v != last_v or last_v <= 0):
                            last_v = None
                    except (TypeError, ValueError):
                        last_v = None
                    if bid > 0 and ask > 0:
                        premium = (bid + ask) / 2.0
                    elif last_v is not None and last_v > 0:
                        premium = last_v
                    else:
                        continue  # no usable price
                    iv_c = row.get("impliedVolatility")
                    try:
                        iv_c = float(iv_c) if iv_c is not None else None
                        if iv_c is not None and (iv_c != iv_c or iv_c <= 0):
                            iv_c = None
                    except (TypeError, ValueError):
                        iv_c = None
                    out.append({
                        "ticker": ticker,
                        "type": opt_type,
                        "strike": float(row["strike"]),
                        "expiry": exp_d,
                        "dte": dte,
                        "bid": bid,
                        "ask": ask,
                        "last": last_v,
                        "premium": premium,
                        "oi": int(row.get("openInterest") or 0),
                        "volume": int(row.get("volume") or 0),
                        "iv_contract": iv_c,
                    })
    except Exception:
        return out
    return out
