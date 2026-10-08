"""Paper trading: run every bot on recent/live sessions with REAL option prices wherever Robinhood still has them.

Inputs per real-data day D (written by the fetch step):
  data/live/spy_1m_D.json          SPY 1-minute bars for D
  data/live/opt_D_*.json           get_option_historicals minute results for that day's 0DTE contracts
Days without real minute option bars fall back to the backtest's real-anchored model and are flagged "model".

Usage: python3 paper.py 2026-10-05 2026-10-06 2026-10-07 2026-10-08 [--start 2026-10-09]
  Days before --start are a dry-run replay; accounts reset to $100 / $500 at --start.
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import lab2
from data2 import build
from pricing import DayModel, bs
from sniper import signal_rows
from sniper3 import RUNNER, SCALP, plan
from sniper4 import account as s4_account
from sniped1 import SNIPED1, OneMin
from sniperpp import CHECKS, Trader, select
from combined import Combined

warnings.filterwarnings("ignore")
LIVE = "data/live"
OLD_EXIT = dict(stop=1.5, be=None, prove=None, half=False)
NEW_EXIT = dict(stop=1.25, be=1.5, prove=(6, 0.5), half=False)
BOTS = {
    "sniper": ("Sniper", "sniper", OLD_EXIT),
    "sniper_plus": ("Sniper+", "sniper_plus", OLD_EXIT),
    "sniper_pp": ("Sniper++", "sniper_plus", NEW_EXIT),
    "sniper_plus_scalp": ("Sniper+ scalp", "sniper_plus", dict(stop=0.75, be=None, prove=None, half=False, tp=1.5, cap=6, trail=False)),
    "sniper_pp_scalp": ("Sniper++ scalp", "sniper_plus", dict(stop=0.75, be=0.75, prove=None, half=False, tp=1.5, cap=4, trail=False)),
    "sniper_ppp": ("Sniper+++", "ppp", None),
    "sniper4": ("Sniper4", "s4", "s4"),
    "sniped1": ("Sniped1", "s1", "s1"),
    "combined": ("CombinedBot", "cb", "cb"),
    "daily": ("Every day", "daily7", NEW_EXIT),
}
SIZING = {"all": 1.0, "half": 0.5, "quarter": 0.25}
SLEEVES = {"all": (1.0, 0.5), "half": (0.5, 0.25), "quarter": (0.25, 0.1)}


# ------------------------------------------------------------------ data refresh
def merge_live_spy():
    """Fold data/live/spy_1m_*.json into the main 1m file and the daily file used for trend context."""
    main = {b["begins_at"]: b for b in json.load(open("data/spy_1m.json"))}
    daily = {b["begins_at"][:10]: b for b in json.load(open("data/spy_daily.json"))}
    now = pd.Timestamp.now(tz="UTC")
    for f in sorted(glob.glob(f"{LIVE}/spy_1m_*.json")):
        bars = [b for b in json.load(open(f)) if not b.get("interpolated")]
        # drop the minute that is still forming (live polls)
        bars = [b for b in bars if pd.Timestamp(b["begins_at"]) + pd.Timedelta(minutes=1) <= now]
        for b in bars:
            main[b["begins_at"]] = b
        d = os.path.basename(f)[7:17]
        if bars and d not in daily:
            daily[d] = dict(begins_at=f"{d}T00:00:00Z", open_price=bars[0]["open_price"],
                            high_price=str(max(float(b["high_price"]) for b in bars)),
                            low_price=str(min(float(b["low_price"]) for b in bars)),
                            close_price=bars[-1]["close_price"], volume=sum(b.get("volume", 0) for b in bars))
    json.dump([main[k] for k in sorted(main)], open("data/spy_1m.json", "w"))
    json.dump([daily[k] for k in sorted(daily)], open("data/spy_daily.json", "w"))


def quote_minutes(day, minutes):
    """Alpaca live quotes recorded each minute → {key: {close: mid per SPY minute (ffill), o,h,l,c}}."""
    path = f"{LIVE}/quotes_{day}.json"
    if not os.path.exists(path):
        return {}
    log = json.load(open(path))
    out = {}
    for key, by_min in log.items():
        mids, last = [], None
        for m in minutes:
            q = by_min.get(m)
            if q and q[1]:
                last = (q[0] + q[1]) / 2 if q[0] else q[1]
            mids.append(last)
        first = next((v for v in mids if v is not None), None)
        if first is None:
            continue
        arr = np.array([first if v is None else v for v in mids], float)
        out[key] = dict(close=arr, o=arr[0], h=float(arr.max()), l=float(arr.min()), c=float(arr[-1]))
    return out


def real_minutes(day, minutes=None):
    """{key: np.array of 1m close prices} from Alpaca quote logs (live) or saved Robinhood minute bars."""
    if minutes is not None:
        q = quote_minutes(day, minutes)
        if q:
            return q
    out = {}
    for f in glob.glob(f"{LIVE}/opt_{day}_*.json"):
        for r in json.load(open(f))["data"]["results"]:
            occ = r["occ_symbol"].split()[-1]
            key = f"{occ[6]}{int(occ[7:]) // 1000}"
            bars = [b for b in r["bars"] if not b.get("interpolated")]
            if bars:
                out[key] = dict(close=np.array([float(b["close_price"]) for b in bars]),
                                o=float(bars[0]["open_price"]), h=max(float(b["high_price"]) for b in bars),
                                l=min(float(b["low_price"]) for b in bars), c=float(bars[-1]["close_price"]))
    return out


class RealDay:
    """Same interface as DayModel, but prices come from real 1m option bars. A fill at SPY level S inside bar i
    uses the real option price in the minute whose SPY close is nearest S, adjusted by the option's delta."""

    def __init__(self, base, real, spy_close, step):
        self.base, self.real, self.spy, self.step = base, real, spy_close, step
        self.hm, self.T, self.atm_iv, self.b = base.hm, base.T, base.atm_iv, base.b
        self.daily = {k: dict(o=v["o"], h=v["h"], l=v["l"], c=v["c"], interp=False) for k, v in real.items()}

    def has(self, key):
        return key in self.real

    def iv_at(self, key, i):
        return self.base.iv_at(key, min(i, len(self.base.T) - 1))

    def price(self, key, i, S):
        if key not in self.real:
            return self.base.price(key, i, S)
        arr = self.real[key]["close"]
        lo = min(i * self.step, len(self.spy) - 1)
        hi = min(lo + self.step, len(self.spy), len(arr))
        if hi <= lo:
            m = min(lo, len(arr) - 1)
        else:
            m = lo + int(np.argmin(np.abs(self.spy[lo:hi] - S)))
        m = min(m, len(arr) - 1)
        K, side = float(key[1:]), key[0]
        ii = min(i, len(self.base.T) - 1)
        T, iv, ref = self.base.T[ii], self.iv_at(key, ii), self.spy[m]
        delta = (bs(ref + 0.05, K, T, iv, side) - bs(ref - 0.05, K, T, iv, side)) / 0.1
        return max(round(float(arr[m] + delta * (S - ref)), 2), 0.01)

    raw_price = price


def install_real(models, strikes, frames_by_day, step, m1, days):
    """Swap in RealDay models on days that have real minute bars. Returns {day: 'real'|'model'}."""
    source = {}
    for d in days:
        mins = [t.strftime("%H:%M") for t in m1[m1.day == d].t]
        real = real_minutes(d, mins)
        if not real:
            source[d] = "model"
            continue
        daily = {k: dict(o=v["o"], h=v["h"], l=v["l"], c=v["c"], interp=False) for k, v in real.items()}
        base = DayModel(d, frames_by_day(d), daily)
        spy = m1[m1.day == d].close.values
        models[d] = RealDay(base, real, spy, step)
        strikes[d] = {s: sorted(int(k[1:]) for k in real if k[0] == s) for s in "CP"}
        source[d] = "real"
    return source


# ------------------------------------------------------------------ bot runners
def scored(ev):
    if ev.empty:   # no candidate signals at all (e.g. first minutes of the day)
        ev = pd.DataFrame(columns=["day", "i", "side", "mins", "fams", "a_st5", "a_st15", "a_st60", "relvol", "a_ema",
                                   "a_rsi", "a_gap", "a_vwap", "a_d50", "n_fams"])
    for k, f in CHECKS.items():
        ev[f"q_{k}"] = f(ev).astype(int)
    ev["score"] = ev[[f"q_{k}" for k in CHECKS]].sum(axis=1)
    return ev


def run_bot(key, T, S1, CB, ev, days, capital, size):
    name, mode, ex = BOTS[key]
    frac = SIZING[size]
    if mode == "cb":
        return CB.run_sleeves(capital, *SLEEVES[size], detail=True)[0]
    if mode == "s1":
        C = SNIPED1
        return S1.run(C["family"], C["confirm"], C["window"], C["exit"], C["max_day"], C["rule"],
                      [d for d in S1.days if d in set(days)], frac=frac, capital=capital, detail=True)
    if mode == "s4":
        return [x for x in s4_account(T, ev, set(days), capital, frac, detail=True)[0] if not x.get("skipped")]
    if mode == "ppp":
        cash, out, busy, cnt = capital, [], {}, {}
        for r, kind in plan(ev, "all"):
            if r.day not in set(days) or cnt.get(r.day, 0) >= 2 or r.i <= busy.get(r.day, -1):
                continue
            t = T.trade(r.day, int(r.i), r.side, RUNNER if kind == "A" else SCALP, budget=frac * cash, detail=True)
            if t is None:
                continue
            cash += t["pnl"]
            t.update(eq=round(cash, 2), kind="runner" if kind == "A" else "scalp")
            out.append(t)
            busy[r.day], cnt[r.day] = r.i + t["held"], cnt.get(r.day, 0) + 1
        return out
    cash, out = capital, []
    for r in select(ev, mode).itertuples():
        if r.day not in set(days):
            continue
        t = T.trade(r.day, int(r.i), r.side, ex, budget=frac * cash, detail=True)
        if t is None:
            continue
        cash += t["pnl"]
        t["eq"] = round(cash, 2)
        out.append(t)
    return out


def main():
    argv = sys.argv[1:]
    start = None
    if "--start" in argv:
        k = argv.index("--start")
        start = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    days = sorted(set(a for a in argv if not a.startswith("--")))
    merge_live_spy()
    D = build()
    f5 = D["f5"]
    D["days"] = [d for d in days if d in set(f5.day)]
    T = Trader(D)
    S1 = OneMin()
    S1.days = [d for d in D["days"] if d in S1.base or d in set(S1.m1.day)]
    for d in S1.days:
        if d not in S1.base:
            S1.base[d] = int(S1.m1.index[S1.m1.day == d][0])
            S1.models[d] = DayModel(d, S1.m1[S1.m1.day == d], {})
            S1.strikes[d] = {"C": [], "P": []}
    m1 = S1.m1
    src5 = install_real(T.sim.models, T.sim.strikes, lambda d: T.f5[T.f5.day == d], 5, m1, D["days"])
    install_real(S1.models, S1.strikes, lambda d: m1[m1.day == d], 1, m1, S1.days)
    ev = scored(signal_rows(D, T.sim, D["days"], outcomes=False))
    CB = Combined(T=T, S1=S1, ev=ev, days=S1.days)
    phases = {"dry": [d for d in D["days"] if not start or d < start], "live": [d for d in D["days"] if start and d >= start]}
    results = {}
    for phase, pdays in phases.items():
        if not pdays:
            continue
        for key in BOTS:
            for cap in (100, 500):
                for size in SIZING:
                    if key in ("sniped1", "combined"):
                        CB.days = S1.days = [d for d in pdays if d in set(m1.day)]
                        CB.__init__(T=T, S1=S1, ev=ev, days=CB.days)
                    tr = run_bot(key, T, S1, CB, ev, pdays, cap, size)
                    results[f"{phase}|{key}|{cap}|{size}"] = tr
    json.dump(dict(days=D["days"], source=src5, start=start), open("data/paper_meta.json", "w"))
    pd.to_pickle(dict(results=results, days=D["days"], source=src5, start=start, f5=T.f5, m1=m1),
                 "data/paper_results.pkl")
    for phase in phases:
        if not phases[phase]:
            continue
        print(f"\n=== {phase.upper()} {phases[phase][0]} → {phases[phase][-1]} · fills: " +
              ", ".join(f"{d[5:]} {src5.get(d, '?')}" for d in phases[phase]) + " · 50% per trade ===")
        for key, (name, _, _) in BOTS.items():
            line = []
            for cap in (100, 500):
                tr = [x for x in results.get(f"{phase}|{key}|{cap}|half", []) if not x.get("skipped")]
                pnl = sum(x["pnl"] for x in tr)
                line.append(f"${cap}→${cap + pnl:,.2f}")
            n = len([x for x in results.get(f"{phase}|{key}|100|half", []) if not x.get("skipped")])
            print(f"  {name:15s} trades {n:2d}  " + "  ".join(line))


if __name__ == "__main__":
    main()
