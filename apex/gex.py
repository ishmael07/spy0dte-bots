"""Live dealer-gamma map for SPY (recorded, displayed, NOT traded on).

Backtest (lab5/gex_test.py, 676 days, 0DTE contracts, start-of-day OI): the regime did not predict trend vs chop in
either period, and gating Apex on it would have hurt. Limitation: start-of-day OI misses same-day 0DTE positioning.
So the live version uses every expiry out to 7 days, is recorded every 15 minutes, and gets scored forward before
any rule uses it.

Convention (the common "naive" one): dealers long calls / short puts -> call gamma +, put gamma -.
  net    $ gamma per 1% SPY move;  tilt = net / gross in [-1, 1]
  flip   SPY level where net gamma changes sign (nearest to spot)
  walls  strikes with the largest call / put gamma exposure
"""
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
TRADE = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs(S, K, T, v, call):
    d1 = (math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    return S * _cdf(d1) - K * _cdf(d2) if call else K * _cdf(-d2) - S * _cdf(-d1)


def _iv(price, S, K, T, call):
    intr = max(0.0, (S - K) if call else (K - S))
    if price <= intr + 0.005 or T <= 0:
        return None
    lo, hi = 0.01, 5.0
    for _ in range(40):
        m = (lo + hi) / 2
        if _bs(S, K, T, m, call) > price:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def _gamma(S, K, T, v):
    d1 = (math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
    return _pdf(d1) / (S * v * math.sqrt(T))


def open_interest(H, today, spot, days=7):
    """{occ symbol: (side, strike, expiry, OI)} for SPY contracts expiring within `days`, strikes within ±5%"""
    end = (datetime.fromisoformat(today) + timedelta(days=days)).strftime("%Y-%m-%d")
    out, tok = {}, None
    while True:
        p = dict(underlying_symbols="SPY", expiration_date_gte=today, expiration_date_lte=end, status="active",
                 strike_price_gte=round(spot * 0.95), strike_price_lte=round(spot * 1.05), limit=10000)
        if tok:
            p["page_token"] = tok
        j = requests.get(f"{TRADE}/v2/options/contracts", headers=H, params=p, timeout=30).json()
        for c in j.get("option_contracts") or []:
            if c.get("open_interest"):
                out[c["symbol"]] = (c["type"][0].upper(), float(c["strike_price"]), c["expiration_date"], int(c["open_interest"]))
        tok = j.get("next_page_token")
        if not tok:
            return out


def snapshot_mids(H, symbols):
    mids = {}
    for k in range(0, len(symbols), 100):
        r = requests.get(f"{DATA}/v1beta1/options/snapshots", headers=H,
                         params=dict(symbols=",".join(symbols[k:k + 100]), feed="indicative"), timeout=20)
        for s, sn in ((r.json().get("snapshots") or {}) if r.ok else {}).items():
            q = sn.get("latestQuote") or {}
            if q.get("ap") and q.get("bp"):
                mids[s] = (q["bp"] + q["ap"]) / 2
    return mids


def compute(oi, mids, spot, now=None):
    now = now or datetime.now(ET)
    rows = []
    for sym, (side, K, exp, n) in oi.items():
        px = mids.get(sym)
        if not px:
            continue
        close = datetime.fromisoformat(exp + "T16:00:00").replace(tzinfo=ET)
        T = max((close - now).total_seconds(), 60) / (365 * 24 * 3600)
        v = _iv(px, spot, K, T, side == "C")
        if v:
            rows.append((side, K, n, v, T))
    if len(rows) < 10:
        return None

    def net_at(S):
        net = gross = 0.0
        per = {}
        for side, K, n, v, T in rows:
            g = _gamma(S, K, T, v) * n * 100 * S * S * 0.01
            net += g if side == "C" else -g
            gross += g
            per[(side, K)] = per.get((side, K), 0) + g
        return net, gross, per

    net, gross, per = net_at(spot)
    flips, prev = [], None
    for k in range(61):
        S2 = spot * (0.985 + 0.0005 * k)
        n2 = net_at(S2)[0]
        if prev and (prev[1] > 0) != (n2 > 0):
            flips.append((prev[0] + S2) / 2)
        prev = (S2, n2)
    flip = min(flips, key=lambda x: abs(x - spot)) if flips else None
    cw = max(((K, g) for (s, K), g in per.items() if s == "C"), key=lambda x: x[1], default=(None, 0))[0]
    pw = max(((K, g) for (s, K), g in per.items() if s == "P"), key=lambda x: x[1], default=(None, 0))[0]
    return dict(t=now.strftime("%H:%M"), spot=round(spot, 2), net_bn=round(net / 1e9, 3), tilt=round(net / gross, 3) if gross else 0,
                flip=round(flip, 2) if flip else None, call_wall=cw, put_wall=pw, contracts=len(rows),
                regime=("short gamma" if (flip and spot < flip) or (not flip and net < 0) else "long gamma"))
