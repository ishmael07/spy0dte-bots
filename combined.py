"""CombinedBot: Sniper4 (5m, runners) + Sniped1 (1m, open scalps) sharing one account.

Each bot keeps its own entry rules and schedule. Positions can overlap; every new trade is sized from the
cash not tied up in open trades, and cash returns when a trade closes. Span: Aug 27 – Oct 7 (1m data).
"""
import warnings

import numpy as np
import pandas as pd

from data2 import build
from sniped1 import SNIPED1, OneMin
from sniper4 import EXITS as S4_EXITS, SNIPER4, account
from sniperpp import Trader, load_events

warnings.filterwarnings("ignore")


class Combined:
    def __init__(self, T=None, S1=None, ev=None, days=None):
        if T is None:
            T = Trader(build())
        self.T = T
        self.S1 = S1 or OneMin()
        self.ev = load_events() if ev is None else ev
        self.days = days or self.S1.days
        f5, m1 = self.T.f5, self.S1.m1
        C = SNIPED1
        s4, _ = account(self.T, self.ev, set(self.days), 100, None, detail=True)
        s1 = self.S1.run(C["family"], C["confirm"], C["window"], C["exit"], C["max_day"], C["rule"], self.days, detail=True)
        sched = []
        for x in s4:
            if x.get("skipped"):
                continue
            sched.append(dict(bot="Sniper4", day=x["day"], side=x["side"], sig=x["si"], tier=x["tier"],
                              t_in=f5.t[x["ei"]], t_out=f5.t[x["xi"]] + pd.Timedelta(minutes=5)))
        for x in s1:
            sched.append(dict(bot="Sniped1", day=x["day"], side=x["side"], sig=x["i"],
                              t_in=m1.t[x["ei"]], t_out=m1.t[x["xi"]] + pd.Timedelta(minutes=1)))
        self.sched = sorted(sched, key=lambda s: s["t_in"])
        self.flat = {"Sniper4": s4, "Sniped1": s1}

    def _trade(self, s, budget, avail, detail):
        if s["bot"] == "Sniper4":
            ex = dict(S4_EXITS[SNIPER4["exit"][s["tier"]]])
            base = budget * SNIPER4["size"][s["tier"]]
            if avail is not None:
                base = min(base, avail)
            if s["tier"] == "A":
                ex["add_at"] = 1.0
                ex["add_budget"] = None if avail is None else max(avail - base, 0.0)
            x = self.T.trade(s["day"], s["sig"], s["side"], ex, budget=base, rule=SNIPER4["strike"] if s["tier"] == "A" else "near", detail=detail)
        else:
            C = SNIPED1
            ex = dict(C["exit"], add_budget=(max(avail - budget, 0.0) if avail is not None else None))
            x = self.S1.trade(s["day"], s["sig"], s["side"], ex, budget, C["rule"], detail)
        return x

    def run_sleeves(self, capital, f4, f1, detail=False, acct_rule="none", sessions=None):
        """Each bot sizes off total equity (cash + money in open trades): Sniper4 unit = f4 × equity,
        Sniped1 unit = f1 × equity, both capped by usable cash under the account rule (acct.py)."""
        from acct import Account
        A = Account(capital, acct_rule, sessions or list(self.days))
        open_, out = [], []
        for s in self.sched:
            day = s["t_in"].tz_convert("America/New_York").strftime("%Y-%m-%d")
            for p in [p for p in open_ if p["t_out"] <= s["t_in"]]:
                A.close(p["day"], p["cost"] + p["pnl"])
                open_.remove(p)
            if not A.allowed(day):
                continue
            free = A.buying_power(day)
            if free < 5:
                continue
            equity = A.equity() + sum(p["cost"] for p in open_)
            unit = (f4 if s["bot"] == "Sniper4" else f1) * equity
            x = self._trade(s, min(unit, free), free, detail)
            if x is None:
                continue
            cost = x["cost"]
            A.open(day, cost)
            open_.append(dict(t_out=s["t_out"], cost=cost, pnl=x["pnl"], day=day))
            x.update(bot=s["bot"], t_in=s["t_in"], t_out=s["t_out"])
            out.append(x)
        for p in open_:
            A.close(p["day"], p["cost"] + p["pnl"])
        A.roll("9999-12-31")
        cash = A.equity()
        out.sort(key=lambda q: q["t_out"])     # list in close order so the running balance reads correctly
        running = capital
        for x in out:
            running += x["pnl"]
            x["eq"] = round(running, 2)
        return out, cash

    def run(self, capital=100.0, frac=None, detail=False):
        """frac=None: flat $100 unit per bot trade (no cash limit). Otherwise unit = frac × free cash."""
        cash, open_, out = capital, [], []
        for s in self.sched:
            for p in [p for p in open_ if p["t_out"] <= s["t_in"]]:
                cash += p["cost"] + p["pnl"]
                open_.remove(p)
            free = cash
            if frac is None:
                x = self._trade(s, 100.0, None, detail)
            else:
                if free < 5:
                    continue
                x = self._trade(s, frac * free, free, detail)
            if x is None:
                continue
            cost = x["cost"]
            if frac is not None:
                cash -= cost
                open_.append(dict(t_out=s["t_out"], cost=cost, pnl=x["pnl"]))
            x.update(bot=s["bot"], t_in=s["t_in"], t_out=s["t_out"])
            out.append(x)
        for p in open_:
            cash += p["cost"] + p["pnl"]
        if frac is None:
            cash = capital + sum(x["pnl"] for x in out)
        out.sort(key=lambda q: q["t_out"])
        eq, running = [], capital   # realized balance after each trade closes (in exit order)
        for x in out:
            running += x["pnl"]
            x["eq"] = round(running, 2)
        return out, cash


def summary(name, trades, days, capital=100.0):
    blocks = [days[:10], days[10:20], days[20:]]
    p = np.array([x["pnl"] for x in trades]) if trades else np.zeros(1)
    order = sorted(trades, key=lambda q: q["t_out"])
    eq = np.array([capital] + [capital + c for c in np.cumsum([x["pnl"] for x in order])])
    bl = [round(sum(x["pnl"] for x in trades if x["day"] in set(b))) for b in blocks]
    return (f"{name:26s} n={len(trades):3d} ({len(trades) / len(days) * 5:.1f}/wk) net {p.sum():+7.0f}  blocks {bl}  "
            f"win {100 * (p > 0).mean():.0f}%  avg loss {p[p <= 0].mean():+.1f}  max drawdown ${np.max(np.maximum.accumulate(eq) - eq):.0f}")


if __name__ == "__main__":
    C = Combined()
    days = C.days
    flat, _ = C.run()
    s4 = [x for x in flat if x["bot"] == "Sniper4"]
    s1 = [x for x in flat if x["bot"] == "Sniped1"]
    print(f"{len(days)} sessions {days[0]} → {days[-1]} · $100 unit per trade")
    print(summary("Sniper4 alone", s4, days))
    print(summary("Sniped1 alone", s1, days))
    print(summary("CombinedBot", flat, days))
    overlap = sum(1 for a in s4 for b in s1 if a["day"] == b["day"] and a["t_in"] < b["t_out"] and b["t_in"] < a["t_out"])
    print(f"overlapping Sniper4/Sniped1 positions: {overlap}; days with both: {len({x['day'] for x in s4} & {x['day'] for x in s1})}")
    for cap in (100, 200, 500):
        line = []
        for frac in (1.0, 0.5, 0.25):
            tr, cash = C.run(cap, frac)
            order = sorted(tr, key=lambda q: q["t_out"])
            eq = np.array([cap] + [cap + c for c in np.cumsum([x["pnl"] for x in order])])
            mdd = 100 * np.max(1 - eq / np.maximum.accumulate(eq))
            line.append(f"{int(frac * 100)}%: ${cash:,.0f} (mdd {mdd:.0f}%)")
        print(f"CombinedBot ${cap}: " + " | ".join(line))
