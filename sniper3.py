"""Sniper+++ — aggressive: two trade types, up to 2 trades/day, runner exits, growth-optimal sizing."""
import warnings

import numpy as np
import pandas as pd

from data2 import build
from sniperpp import Trader, load_events

warnings.filterwarnings("ignore")
RIDE = dict(stop=1.25, be=1.5, prove=(6, 0.5), half=False)
RUNNER = dict(stop=1.25, be=None, prove=(6, 0.5), half=False, half_atr=1.5, be_on_half=True)
SCALP = dict(stop=0.75, be=None, prove=None, half=False, tp=1.5, cap=6, trail=False)


def plan(ev, variant):
    """Causal: walk each day's signals in time; A = Sniper+ quality, B = Sniper quality (scalp)."""
    ev = ev.sort_values("i")
    core = (ev.q_5m + ev.q_15m + ev["q_1h"] + ev.q_vol == 4) & (ev.mins <= 120)
    a = core & (ev["q_50d"] == 1)
    rows = []
    for d, g in ev.groupby("day"):
        for r in g.itertuples():
            if a[r.Index]:
                rows.append((r, "A"))
            elif core[r.Index] and variant != "A_only":
                rows.append((r, "B"))
    return rows


def simulate(T, rows, days, capital, frac, a_exit, b_exit, max_day=2, rule_big="near", big_at=None):
    cash, out, busy, cnt = capital, [], {}, {}
    for r, kind in rows:
        if r.day not in days or cnt.get(r.day, 0) >= max_day or r.i <= busy.get(r.day, -1):
            continue
        budget = 100.0 if frac is None else cash * frac
        rule = rule_big if (big_at and budget >= big_at) else "near"
        t = T.trade(r.day, int(r.i), r.side, a_exit if kind == "A" else b_exit, budget=budget, rule=rule)
        if t is None:
            continue
        cash += t["pnl"]
        t.update(kind=kind, eq=cash)
        out.append(t)
        busy[r.day] = r.i + t["held"]
        cnt[r.day] = cnt.get(r.day, 0) + 1
        if cash < 5:
            break
    return out, cash


def week_odds(trades, days, frac, n=20000, seed=3):
    """Block bootstrap of trading days: draw 5 random sessions (with their real trades) and compound."""
    rng = np.random.default_rng(seed)
    by_day = {d: [t["pct"] / 100 for t in trades if t["day"] == d] for d in days}
    mult = []
    for _ in range(n):
        m = 1.0
        for d in rng.choice(days, 5):
            for x in by_day[d]:
                m *= max(1 + frac * x, 0)
        mult.append(m)
    mult = np.array(mult)
    return dict(x10=100 * (mult >= 10).mean(), x2=100 * (mult >= 2).mean(), up=100 * (mult > 1).mean(),
                bust=100 * (mult <= 0.2).mean(), med=float(np.median(mult)))


def main():
    D = build()
    T = Trader(D)
    ev = load_events()
    days = D["days"]
    unseen = [d for d in days if d < "2026-07-15"]
    design = [d for d in days if d >= "2026-07-15"]
    configs = {
        "Sniper++ (baseline)": ("A_only", RIDE, RIDE, 1),
        "A ride + B scalp, 2/day": ("all", RIDE, SCALP, 2),
        "A runner + B scalp, 2/day": ("all", RUNNER, SCALP, 2),
        "A runner only, 2/day": ("A_only", RUNNER, SCALP, 2),
        "A runner + B scalp, 1/day": ("all", RUNNER, SCALP, 1),
    }
    print(f"{'config ($100 flat per trade)':30s} {'unseen Jun1-Jul14':>20s} {'design Jul15-Oct7':>20s}  win  avg/trade")
    for name, (var, ae, be, md) in configs.items():
        rows = plan(ev, var)
        u, _ = simulate(T, rows, set(unseen), 100, None, ae, be, md)
        g, _ = simulate(T, rows, set(design), 100, None, ae, be, md)
        allt = u + g
        print(f"{name:30s} n{len(u):2d} {sum(x['pnl'] for x in u):+7.0f}      n{len(g):2d} {sum(x['pnl'] for x in g):+7.0f}    "
              f"{100 * np.mean([x['pnl'] > 0 for x in allt]):3.0f}%  {np.mean([x['pnl'] for x in allt]):+6.1f}")
    # sizing on the most promising aggressive config: Kelly-style growth vs all-in, real 90-day path + weekly odds
    for name in ("Sniper++ (baseline)", "A runner + B scalp, 2/day"):
        var, ae, be, md = configs[name]
        rows = plan(ev, var)
        flat, _ = simulate(T, rows, set(days), 100, None, ae, be, md)
        r = np.array([x["pct"] / 100 for x in flat])
        fs = np.linspace(0.05, 1.0, 20)
        g = [np.mean(np.log(np.maximum(1 + f * r, 1e-9))) for f in fs]
        kelly = fs[int(np.argmax(g))]
        print(f"\n{name}: growth-optimal fraction ≈ {kelly:.2f} of balance per trade")
        for frac in (kelly, 0.5, 1.0):
            tr, cash = simulate(T, rows, set(days), 100, frac, ae, be, md)
            o = week_odds(flat, days, frac)
            print(f"  bet {frac:.2f}: 90-day path $100→${cash:,.0f} · any 5-day week: 10x {o['x10']:.1f}% | 2x {o['x2']:.1f}% | up {o['up']:.0f}% | lose ≥80% {o['bust']:.1f}% | median ×{o['med']:.2f}")


if __name__ == "__main__":
    main()
