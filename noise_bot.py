"""Noise Area bot: the published SPY intraday-momentum strategy, traded with 0DTE options. No Supertrend anywhere.

Source: Zarattini, Aziz & Barbon, "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"
(SSRN 4824172, 2024) and Concretum Group's public Python code + "Intraday Trend Following With 0DTE Options".
On SPY shares 2007–2024 the paper reports ~19.6% net CAGR, Sharpe 1.33, ~43% hit rate.

Rules (as published, volatility multiplier 1):
  sigma(t)  mean over the previous 14 sessions of |close at time-of-day t / that day's 9:30 open − 1|
  UB / LB   max(open, prev close) × (1 + sigma)  /  min(open, prev close) × (1 − sigma)
  checks    only on the half hour (10:00, 10:30, … 15:00) — between checks the noise is ignored
  entry     close > UB and > VWAP → buy a call;  close < LB and < VWAP → buy a put (nearest affordable to ATM)
  exit      at a half-hour check: a call exits if close < max(UB, VWAP), a put if close > min(LB, VWAP);
            flat at 3:45 (0DTE). One position at a time; after an exit the next check may enter again.
No price stop, no target, no time cap — the band/VWAP trail is the only exit, as in the paper.
"""
import sys
import warnings

import numpy as np

from acct import Account

warnings.filterwarnings("ignore")
LOOKBACK = 14
CHECK = {955, 1025, 1055, 1125, 1155, 1225, 1255, 1325, 1355, 1425, 1455, 1525}   # 5m bars closing on :00/:30
LAST_ENTRY = 1455


def bands(T):
    """Per-5m-bar UB / LB (NaN until 10 sessions of history exist)."""
    f5 = T.f5
    if "noise_ub" not in f5:
        move = (f5.close / f5.day_open - 1).abs()
        sigma = move.groupby(f5.hm).transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=10).mean())
        pdc = f5.pdc.fillna(f5.day_open)
        f5["noise_ub"] = np.maximum(f5.day_open, pdc) * (1 + sigma)
        f5["noise_lb"] = np.minimum(f5.day_open, pdc) * (1 - sigma)
    return f5.noise_ub.values, f5.noise_lb.values


def run(T, days, capital, frac, detail=False, acct_rule="none", sessions=None, budget=None):
    """budget=None → size off the account (frac of equity); budget=x → flat $x per trade (backtests)."""
    ub, lb = bands(T)
    c, hm, vw = T.sim.c, T.sim.hm, T.f5.vwap.values

    def trail(j, k):
        if hm[j] not in CHECK:
            return None
        if (k == 1 and c[j] < max(ub[j], vw[j])) or (k == -1 and c[j] > min(lb[j], vw[j])):
            return j, c[j], "NOISE TRAIL"
        return None

    ex = dict(stop=99, be=None, prove=None, half=False, trail=False, cap=None, exit_fn=trail)
    A = Account(capital, acct_rule, sorted(sessions or days))
    out = []
    for d in sorted(days):
        if d not in T.sim.base:
            continue
        b, n = T.sim.base[d], len(T.sim.models[d].hm)
        j = b
        while j < b + n - 1:
            if hm[j] > LAST_ENTRY:
                break
            side = None
            if hm[j] in CHECK and not np.isnan(ub[j]):
                side = "C" if (c[j] > ub[j] and c[j] > vw[j]) else "P" if (c[j] < lb[j] and c[j] < vw[j]) else None
            if not side or not A.allowed(d):
                j += 1
                continue
            t = T.trade(d, j, side, ex, budget=budget or A.budget(d, frac), detail=True)
            if t is None:
                j += 1
                continue
            A.book(d, t["cost"], t["pnl"])
            t["eq"] = round(A.equity(), 2)
            out.append(t)
            j = t["xi"] if t["xi"] > j else j + 1      # the exit bar is itself a check: it may reverse at once
    return out, A.equity()


def stats(tr):
    p = np.array([x["pnl"] for x in tr])
    if not len(p):
        return "no trades"
    eq = np.concatenate([[0], np.cumsum(p)])
    return (f"n={len(p):3d}  net ${p.sum():+8.0f}  win {100 * (p > 0).mean():3.0f}%  avg {np.mean([x['pct'] for x in tr]):+6.1f}%  "
            f"best ${p.max():+.0f}  minus best ${p.sum() - p.max():+.0f}  maxDD ${np.max(np.maximum.accumulate(eq) - eq):.0f}")


if __name__ == "__main__":
    # python3 noise_bot.py            → 90-session backtest (real-anchored option model), flat $100 per trade
    from data2 import build
    from sniperpp import Trader
    D = build()
    T, days = Trader(D), D["days"]
    tr, _ = run(T, days, 1e6, None, budget=float(sys.argv[1]) if len(sys.argv) > 1 else 100.0)
    print(f"Noise Area · {days[0]} → {days[-1]} · flat ${sys.argv[1] if len(sys.argv) > 1 else 100}/trade")
    print("  all     ", stats(tr))
    for q in range(3):
        blk = set(days[30 * q:30 * q + 30])
        print(f"  block {q + 1} ", stats([x for x in tr if x["day"] in blk]))
    print("  1st/day ", stats([x for i, x in enumerate(tr) if i == 0 or tr[i - 1]["day"] != x["day"]]))
    for cap in (100, 500):
        _, eq = run(T, days, cap, 0.5)
        print(f"  ${cap} @50% compounding → ${eq:,.0f}")
    from collections import Counter
    print("  exits", Counter(x["why"] for x in tr), " per day", Counter(Counter(x["day"] for x in tr).values()))
