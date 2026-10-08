"""Sniper4 search: tiered conviction sizing, per-tier exits, pyramiding, re-entries, strike leverage.

Accepted only if it beats the best existing strategy on 90-day flat P&L, is positive in every 30-day block,
and the selection procedure holds up walk-forward.
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd

from data2 import build
from sniperpp import Trader, load_events

warnings.filterwarnings("ignore")
RIDE = dict(stop=1.25, be=1.5, prove=(6, 0.5), half=False)
RUNNER = dict(stop=1.25, be=None, prove=(6, 0.5), half=False, half_atr=1.5, be_on_half=True)
SCALP = dict(stop=0.75, be=None, prove=None, half=False, tp=1.5, cap=6, trail=False)
EXITS = {"ride": RIDE, "runner": RUNNER, "scalp": SCALP}


def tiers(ev, a_window):
    core = (ev.q_5m + ev.q_15m + ev["q_1h"] + ev.q_vol == 4)
    a = core & (ev["q_50d"] == 1) & (ev.mins <= a_window)
    b = core & (ev.mins <= 120) & ~a
    c = (ev.score >= 7) & (ev.mins <= 120) & ~a & ~b
    return np.where(a, "A", np.where(b, "B", np.where(c, "C", "")))


def run(T, ev, cfg, days):
    ev = ev[ev.day.isin(days)].sort_values("i")
    tier = tiers(ev, cfg["a_window"])
    out, busy, cnt = [], {}, {}
    for r, t in zip(ev.itertuples(), tier):
        if not t:
            continue
        mult = cfg["size"][t]
        if mult <= 0 or cnt.get(r.day, 0) >= cfg["max_day"] or r.i <= busy.get(r.day, -1):
            continue
        ex = dict(EXITS[cfg["exit"][t]])
        if t == "A" and cfg["pyramid"]:
            ex["add_at"] = 1.0
        x = T.trade(r.day, int(r.i), r.side, ex, budget=100.0 * mult, rule=cfg["strike"] if t == "A" else "near")
        if x is None:
            continue
        x["tier"] = t
        out.append(x)
        busy[r.day] = r.i + x["held"]
        cnt[r.day] = cnt.get(r.day, 0) + 1
    return out


def blocks(trades, days):
    b = [days[0:30], days[30:60], days[60:90]]
    return [round(sum(x["pnl"] for x in trades if x["day"] in set(bb)), 0) for bb in b]


GRID = dict(
    a_exit=("ride", "runner"), b_exit=("ride", "runner", "scalp"), c_exit=("scalp", "runner"),
    b_size=(0.0, 0.5, 1.0), c_size=(0.0, 0.25, 0.5), a_size=(1.0, 2.0),
    max_day=(1, 2, 3), pyramid=(False, True), strike=("near", "otm1"), a_window=(120, 180),
)


def main():
    t0 = time.time()
    D = build()
    T = Trader(D)
    days = D["days"]
    ev = load_events()
    rows = []
    keys = list(GRID)
    for n, combo in enumerate(itertools.product(*GRID.values())):
        g = dict(zip(keys, combo))
        cfg = dict(exit={"A": g["a_exit"], "B": g["b_exit"], "C": g["c_exit"]},
                   size={"A": g["a_size"], "B": g["b_size"], "C": g["c_size"]},
                   max_day=g["max_day"], pyramid=g["pyramid"], strike=g["strike"], a_window=g["a_window"])
        if g["c_size"] == 0 and g["c_exit"] != "scalp":
            continue   # C exit irrelevant when C is off
        if g["b_size"] == 0 and g["b_exit"] != "ride":
            continue
        tr = run(T, ev, cfg, days)
        pnl = np.array([x["pnl"] for x in tr]) if tr else np.zeros(1)
        eq = np.concatenate([[0], np.cumsum(pnl)])
        bl = blocks(tr, days)
        risked = sum(100 * cfg["size"][x["tier"]] for x in tr)
        rows.append(dict(**g, n=len(tr), net=round(float(pnl.sum()), 0), b1=bl[0], b2=bl[1], b3=bl[2],
                         win=round(100 * float((pnl > 0).mean()), 0), mdd=round(float(np.max(np.maximum.accumulate(eq) - eq)), 0),
                         ret_per_100=round(100 * float(pnl.sum()) / max(risked, 1), 1)))
        if n % 1000 == 0:
            print(n, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle("data/sniper4_grid.pkl")
    print(f"{len(df)} configs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()


SNIPER4 = dict(exit={"A": "ride", "B": "scalp", "C": "scalp"}, size={"A": 2.0, "B": 1.0, "C": 0.0},
               max_day=1, pyramid=True, strike="otm1", a_window=120)


def account(T, ev, days, capital, frac, cfg=SNIPER4, detail=False, acct_rule="none", sessions=None):
    """frac=None: flat $100 unit. Otherwise unit = frac × equity; A-tier base = min(2 units, buying power),
    and the +1 ATR pyramid add uses whatever buying power is left (none when all-in).
    acct_rule: none | cash (T+1 settlement) | pdt (3 day trades per 5 sessions) — see acct.py."""
    from acct import Account
    ev = ev[ev.day.isin(days)].sort_values("i")
    tier = tiers(ev, cfg["a_window"])
    A = Account(capital, acct_rule, sessions or sorted(set(days)))
    cash, out, busy, cnt = capital, [], {}, {}
    for r, t in zip(ev.itertuples(), tier):
        if not t or cfg["size"][t] <= 0 or cnt.get(r.day, 0) >= cfg["max_day"] or r.i <= busy.get(r.day, -1):
            continue
        if frac is not None and not A.allowed(r.day):
            continue
        ex = dict(EXITS[cfg["exit"][t]])
        if frac is None:
            base = cfg["size"][t] * 100.0
        else:
            bp = A.buying_power(r.day)
            base = min(cfg["size"][t] * frac * A.equity(), bp)
        if t == "A" and cfg["pyramid"]:
            ex["add_at"] = 1.0
            ex["add_budget"] = None if frac is None else bp - base
        x = T.trade(r.day, int(r.i), r.side, ex, budget=base, rule=cfg["strike"] if t == "A" else "near", detail=detail)
        if x is None:
            if detail:
                out.append(dict(day=r.day, side=r.side, skipped=True, eq=round(cash, 2)))
            continue
        x.update(tier=t, kind=(cfg["exit"]["A"] if t == "A" else "scalp"))
        if frac is None:
            cash += x["pnl"]
        else:
            A.book(r.day, x["cost"], x["pnl"])
            cash = A.equity()
        x["eq"] = round(cash, 2)
        out.append(x)
        busy[r.day] = r.i + x["held"]
        cnt[r.day] = cnt.get(r.day, 0) + 1
        if cash < 20:
            break
    return out, cash
