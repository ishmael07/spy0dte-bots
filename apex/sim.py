"""Apex v2 on recorded days for the leaderboard (paper.py): SPY 1m bars + the minute-by-minute 0DTE bid/ask the runner
records (data/live/quotes_D.json). Same core.Session as the live engine; buys at the ask, sells at the bid.

run(m1, days, capital, frac, acct_rule, sessions) -> trades in paper.py's format (ei/xi index the m1 frame).
"""
import glob
import gzip
import json
import os

import numpy as np

from acct import Account
from apex import core

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = 0.0          # quotes are real bid/ask: buy at ask, sell at bid


def _minutes_from_live(d, live="data/live"):
    p = f"{live}/spy_1m_{d}.json"
    if not os.path.exists(p):
        return None
    out = {}
    import pandas as pd
    for b in json.load(open(p)):
        if b.get("interpolated"):
            continue
        t = pd.Timestamp(b["begins_at"]).tz_convert("America/New_York")
        i = t.hour * 60 + t.minute - 570
        if 0 <= i < 390:
            out[i] = (float(b["open_price"]), float(b["high_price"]), float(b["low_price"]), float(b["close_price"]), float(b.get("volume") or 0))
    return out


def history_days(live="data/live"):
    """{day: {i: (o,h,l,c,v)}} for warmup (SIP, from the lab) + every recorded live day"""
    H = {}
    for d, rows in json.load(gzip.open(os.path.join(HERE, "warmup_1m.json.gz"), "rt")).items():
        H[d] = {r[0]: tuple(r[1:6]) for r in rows}
    for f in glob.glob(f"{live}/spy_1m_*.json"):
        d = os.path.basename(f)[7:17]
        m = _minutes_from_live(d, live)
        if m and len(m) >= 200:
            H[d] = m
    return H


def blocks_before(H, day, m, keep=400):
    out = []
    for d in sorted(x for x in H if x < day):
        bars = H[d]
        for b0 in range(0, 390, m):
            seg = [bars[i] for i in range(b0, b0 + m) if i in bars]
            if len(seg) == m:
                out.append((max(s[1] for s in seg), min(s[2] for s in seg), seg[-1][3]))
    return out[-keep:]


def setup(H, day):
    prior = [d for d in sorted(H) if d < day]
    hist = []
    for d in prior[-core.LOOKBACK:]:
        bars = H[d]
        hist.append((bars[min(bars)][0], [bars[i][3] if i in bars else None for i in range(390)]))
    sig = core.sigma_profile(hist)
    pb = H[prior[-1]]
    pc = pb[max(pb)][3]
    return sig, pc, {m: blocks_before(H, day, m) for m in (5, 15)}


class Quotes:
    def __init__(self, day, live="data/live"):
        p = f"{live}/quotes_{day}.json"
        self.q = json.load(open(p)) if os.path.exists(p) else {}
        import pandas as pd
        self.off = int(pd.Timestamp(day + " 12:00", tz="America/New_York").utcoffset().total_seconds() // 3600)

    def at(self, side, k, i):
        """(bid, ask) at the close of minute i (latest recorded at or before it)"""
        m = self.q.get(f"{side}{k}")
        if not m:
            return None
        tot = 570 + i + 1 - self.off * 60
        key = f"{tot // 60:02d}:{tot % 60:02d}"
        ks = [t for t in m if t <= key]
        if not ks:
            return None
        b, a = m[max(ks)]
        return (b, a) if a else None


def pick(Q, side, spot, i, budget):
    base = round(spot)
    for k in range(0, 13):
        kk = base + k if side == "C" else base - k
        ba = Q.at(side, kk, i)
        if ba and ba[1] > 0 and ba[1] * 100 <= budget:
            return kk, ba
    return None


def run(m1, days, capital, frac, acct_rule="none", sessions=None, live="data/live"):
    H = history_days(live)
    A = Account(capital, acct_rule, sorted(sessions or days))
    out = []
    for d in sorted(days):
        if d not in H or not any(x < d for x in H):
            continue
        rows = m1.index[m1.day == d]
        if not len(rows):
            continue
        base = int(rows[0])
        hm = m1.hm.values[rows]
        bars = H[d]
        Q = Quotes(d, live)
        sig, pc, prior = setup(H, d)
        if sig is None:
            continue
        end = max(bars)
        if end < 385 and d == max(H):          # today, still in progress: don't treat the latest bar as the close
            end = 389
        o0 = bars[0][0] if 0 in bars else bars[min(bars)][0]
        lanes = {k: core.Session(o0, pc, sig, prior, end=end, books=L["books"], max_trades=L["max_trades"]) for k, L in core.LANES.items()}
        imp = None
        open_cost = 0.0
        day_out = []
        for i in range(390):
            if i not in bars:
                continue
            o, h, l, c, v = bars[i]
            if i == 3:
                k = round(c)
                a, b = Q.at("C", k, 3), Q.at("P", k, 3)
                imp = ((a[0] + a[1]) / 2 + (b[0] + b[1]) / 2) / o0 * 100 if a and b else None
            for lane, S in lanes.items():
                S.imp = imp
                for kind, t in S.on_bar(i, o, h, l, c, v):
                    if kind == "enter":
                        ref = Q.at(t["side"], round(c), i)          # ATM reference (as in the lab) for the loss counter
                        if not ref:
                            S.cancel(t)
                            continue
                        t["ref_in"] = ref[1]
                        w = core.dial(imp) * core.LANES[lane]["weight"]
                        budget = min(A.budget(d, frac) * w, A.buying_power(d) - open_cost) if A.allowed(d) else 0
                        got = pick(Q, t["side"], c, i, budget) if budget > 0 else None
                        t["fill"] = None
                        if got:
                            k, (bid, ask) = got
                            qty = max(1, int(budget // (ask * 100)))
                            t["fill"] = dict(k=k, qty=qty, px=ask)
                            open_cost += ask * 100 * qty
                    else:
                        ref = Q.at(t["side"], round(t["spy_in"]), i)
                        S.result(t, (ref[0] if ref else 0.0) / t["ref_in"] - 1 if t["ref_in"] else -1)
                        f = t.get("fill")
                        if not f:
                            continue
                        ba = Q.at(t["side"], f["k"], i)
                        px_out = ba[0] if ba else 0.0
                        cost = f["px"] * 100 * f["qty"]
                        open_cost -= cost
                        pnl = (px_out - f["px"]) * 100 * f["qty"]
                        A.book(d, cost, pnl)
                        ei = base + min(t["i"] + 1, len(rows) - 1)
                        xi = base + min(t["j"] + 1, len(rows) - 1)
                        path = []
                        for q in range(t["i"] + 1, t["j"] + 2):
                            z = Q.at(t["side"], f["k"], min(q, 389))
                            path.append(round((z[0] + z[1]) / 2, 2) if z else None)
                        ex = t["ex"]
                        day_out.append(dict(day=d, side=t["side"], strike=f["k"], qty=f["qty"], optIn=f["px"], optOut=round(px_out, 2),
                                            pnl=round(pnl, 2), pct=round(100 * (px_out / f["px"] - 1), 1), why=t["why"], cost=round(cost, 2),
                                            si=ei, ei=ei, xi=xi, held=t["j"] - t["i"], added=False, addQty=0, addPx=None,
                                            spyIn=round(t["spy_in"], 2), spyOut=round(c, 2), stopLvl=round(ex["sl"], 2) if ex.get("sl") else None,
                                            path=[p for p in path if p is not None], kind=None, book=t["book"], lane=lane,
                                            eq=round(A.equity(), 2), dial=core.dial(imp), imp=round(imp, 3) if imp else None))
        out += day_out
    return out
