"""Sniper++ lab: loss-control exits × signal-quality selection × trade frequency.

Quality score (0-10) for any candidate signal, using only what is known at the signal bar:
  5m ST, 15m ST, 1h ST agree · volume > avg · EMA 8/20 agree · RSI leaning (>8 pts) · in gap direction ·
  not stretched (0 < distance past VWAP < 1.5 ATR) · prior close on trade side of 50-day SMA · first hour.
Exits add: breakeven stop, "prove it" time stop, half off at +100%.
"""
import itertools
import warnings

import numpy as np
import pandas as pd

import lab2
from data2 import build
from lab2 import FEE, FLAT_AT
from lab3 import Sim3

warnings.filterwarnings("ignore")

CHECKS = {
    "5m": lambda e: e.a_st5 > 0, "15m": lambda e: e.a_st15 > 0, "1h": lambda e: e.a_st60 > 0,
    "vol": lambda e: e.relvol > 1.0, "ema": lambda e: e.a_ema > 0, "rsi": lambda e: e.a_rsi > 8,
    "gap": lambda e: e.a_gap > 0, "fresh": lambda e: (e.a_vwap > 0) & (e.a_vwap < 1.5),
    "50d": lambda e: e.a_d50 > 0, "open": lambda e: e.mins <= 60,
}
EXITS = [dict(stop=s, be=b, prove=p, half=h) for s, b, p, h in itertools.product(
    (1.0, 1.25, 1.5), (None, 1.0, 1.5), (None, (3, 0.3), (6, 0.5)), (False, True))]


def load_events():
    ev = pd.read_csv("data/events.csv")
    for k, f in CHECKS.items():
        ev[f"q_{k}"] = f(ev).astype(int)
    ev["score"] = ev[[f"q_{k}" for k in CHECKS]].sum(axis=1)
    return ev


def select(ev, mode):
    """Causal selection: walk each day's signals in time order, take the first that qualifies."""
    ev = ev.sort_values("i")
    if mode == "sniper":
        m = (ev.q_5m + ev.q_15m + ev["q_1h"] + ev.q_vol == 4) & (ev.mins <= 120)
        return ev[m].groupby("day").head(1)
    if mode == "sniper_plus":
        m = (ev.q_5m + ev.q_15m + ev["q_1h"] + ev.q_vol + ev["q_50d"] == 5) & (ev.mins <= 120)
        return ev[m].groupby("day").head(1)
    if mode.startswith("score"):           # e.g. score7 = any signal with ≥7 checks before 11:30
        k = int(mode[5:])
        return ev[(ev.score >= k) & (ev.mins <= 120)].groupby("day").head(1)
    if mode.startswith("daily"):           # daily{k}: best-effort every day — ≥k before 11:30, else ≥5 until 14:30
        k = int(mode[5:])
        prime = ev[(ev.score >= k) & (ev.mins <= 120)].groupby("day").head(1)
        later = ev[(ev.score >= 5) & (ev.mins > 120) & (ev.mins <= 300) & ~ev.day.isin(prime.day)].groupby("day").head(1)
        return pd.concat([prime, later]).sort_values("i")
    if mode == "every":                    # the first signal of every day, no filter
        return ev.groupby("day").head(1)
    raise ValueError(mode)


class Trader:
    def __init__(self, D):
        self.sim = Sim3(D)
        self.f5 = self.sim.f5
        self.st5 = self.f5.st_trend.values
        self.atr = self.f5.atr.values

    def trade(self, d, i, side, ex, budget=100.0, rule="near", detail=False):
        s = self.sim
        o, h, l, c, hm = s.o, s.h, s.l, s.c, s.hm
        b, n = s.base[d], len(s.models[d].hm)
        k = 1 if side == "C" else -1
        atr = self.atr[i]
        ref = c[i]
        stop = ref - k * ex["stop"] * atr
        ei = i + 1
        pk = s.pick3(d, side, ei, o[ei], budget, rule, ref + 3 * k * atr, stop, 6)
        if not pk:
            return None
        K, key, fill, qty = pk
        dm = s.models[d]
        best = 0.0
        half_px, moved, add_px = None, False, None
        xi = xs = why = None
        for j in range(ei, b + n):
            fav = (h[j] - ref) if k == 1 else (ref - l[j])
            best = max(best, fav / atr)
            if (l[j] <= stop) if k == 1 else (h[j] >= stop):
                xi, xs, why = j, stop, ("BREAKEVEN" if moved else "STOP"); break
            if ex["half"] and half_px is None:
                hi_px = dm.price(key, j - b, float(h[j] if k == 1 else l[j]))
                if hi_px >= 2 * fill:
                    half_px = 2 * fill - lab2.SLIP           # sell half at +100%
            if ex.get("add_at") and add_px is None and best >= ex["add_at"]:
                add_px = dm.price(key, j - b, float(ref + k * ex["add_at"] * atr)) + lab2.SLIP
                stop, moved = ref, True                    # protect the add: stop to breakeven
            if ex.get("half_atr") and half_px is None and best >= ex["half_atr"]:   # bank half at +x ATR, ride the rest
                half_px = max(dm.price(key, j - b, float(ref + k * ex["half_atr"] * atr)) - lab2.SLIP, 0.0)
                if ex.get("be_on_half"):
                    stop, moved = ref, True
            if ex.get("tp") and best >= ex["tp"]:
                xi, xs, why = j, ref + k * ex["tp"] * atr, "TARGET"; break
            if ex["prove"] and j - ei + 1 == ex["prove"][0] and best < ex["prove"][1]:
                xi, xs, why = j, c[j], "NO FOLLOW-THROUGH"; break
            if ex.get("trail", True) and j > ei and self.st5[j] == -k:
                xi = min(j + 1, b + n - 1); xs, why = o[xi], "TREND FLIP"; break
            if j - ei + 1 >= ex.get("cap", 24):
                xi, xs, why = j, c[j], ("2H CAP" if ex.get("cap", 24) == 24 else "TIME LIMIT"); break
            if hm[j] >= FLAT_AT or j == b + n - 1:
                xi, xs, why = j, c[j], "3:45 EXIT"; break
            if ex["be"] and not moved and best >= ex["be"]:
                stop, moved = ref, True
        out = max(dm.price(key, xi - b, float(xs)) - lab2.SLIP, 0.0)
        if half_px is not None:
            out = (half_px + out) / 2
        cost = qty * (fill * 100 + FEE)
        pnl = qty * (out * 100 - FEE) - cost
        add_qty = 0
        if add_px is not None:                              # second unit bought at add_px (limited by add_budget)
            add_qty = qty if ex.get("add_budget") is None else min(qty, int(ex["add_budget"] // (add_px * 100 + FEE)))
        if add_qty:
            cost2 = add_qty * (add_px * 100 + FEE)
            pnl += add_qty * (max(dm.price(key, xi - b, float(xs)) - lab2.SLIP, 0.0) * 100 - FEE) - cost2
            cost += cost2
        r = dict(day=d, side=side, pnl=round(pnl, 2), pct=round(100 * pnl / cost, 1), why=why, held=xi - ei + 1, added=bool(add_qty), addQty=add_qty, addPx=round(add_px, 2) if add_qty else None, cost=round(cost, 2))
        if detail:
            r.update(si=i, ei=ei, xi=xi, strike=K, qty=qty, optIn=round(fill, 2), optOut=round(out, 2),
                     spyIn=round(float(o[ei]), 2), spyOut=round(float(xs), 2), stopLvl=round(float(ref - k * ex["stop"] * atr), 2),
                     cost=round(cost, 2), path=[dm.price(key, j - b, float(c[j])) for j in range(ei, xi + 1)], half=half_px is not None)
        return r


def evaluate(trades, days):
    t = [x for x in trades if x]
    if not t:
        return dict(n=0, net=0, avg=0, win=0, avgloss=0, worst=0, mdd=0, blocks=[0, 0, 0], pf=0)
    pnl = np.array([x["pnl"] for x in t])
    eq = np.cumsum(pnl)
    mdd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])))
    loss = pnl[pnl <= 0]
    gl = -loss.sum()
    blocks = [round(float(sum(x["pnl"] for x in t if days[20 * q] <= x["day"] < (days[20 * q + 20] if q < 2 else "9999"))), 0) for q in range(3)]
    return dict(n=len(t), net=round(float(pnl.sum()), 0), avg=round(float(pnl.mean()), 1), win=round(100 * float((pnl > 0).mean()), 0),
                avgloss=round(float(loss.mean()), 1) if len(loss) else 0, worst=round(float(pnl.min()), 0), mdd=round(mdd, 0),
                blocks=blocks, pf=round(float(pnl[pnl > 0].sum() / gl), 2) if gl else 99.0)


def main():
    D = build()
    T = Trader(D)
    days = D["days"]
    ev = load_events()
    modes = ["sniper", "sniper_plus", "score6", "score7", "score8", "score9", "daily7", "daily8", "every"]
    rows = []
    for mode in modes:
        sel = select(ev, mode)
        for e, ex in enumerate(EXITS):
            tr = [T.trade(r.day, int(r.i), r.side, ex) for r in sel.itertuples()]
            rows.append(dict(mode=mode, exit=e, **evaluate(tr, days)))
    df = pd.DataFrame(rows)
    df["b_min"] = df.blocks.apply(min)
    df.to_pickle("data/sniperpp_grid.pkl")
    base = df[(df.exit == EXITS.index(dict(stop=1.5, be=None, prove=None, half=False)))]
    print("== Current exit (stop 1.5, ride trend) by selection mode ==")
    print(base[["mode", "n", "net", "avg", "win", "avgloss", "worst", "mdd", "pf", "blocks"]].to_string(index=False))
    print("\n== Best exit per mode (all 3 blocks positive, ranked by net / max drawdown) ==")
    ok = df[df.b_min > 0].copy()
    ok["score"] = ok.net / ok.mdd.clip(lower=50)
    for mode in modes:
        sub = ok[ok["mode"] == mode].sort_values("score", ascending=False).head(1)
        for _, r in sub.iterrows():
            print(f"{mode:12s} {EXITS[int(r.exit)]}  n={r.n} net {r.net:+.0f} avg {r.avg:+.1f} win {r.win:.0f}% avgloss {r.avgloss} worst {r.worst} mdd {r.mdd} pf {r.pf} blocks {r.blocks}")


if __name__ == "__main__":
    main()
