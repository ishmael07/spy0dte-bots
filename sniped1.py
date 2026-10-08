"""Sniped1 — market-open 1-minute scalper (Aug 27 – Oct 7, the span with real 1m SPY bars).

Signals on the 1m chart, optional 5m/15m Supertrend confirmation, short holds (≤15 min), several trades a day.
Contracts priced with the same real-anchored model (validated at 1m: ±3.6% median).
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd

import lab2
from data2 import build, load_daily_all
from lab2 import FEE, MIN_PREMIUM
from pricing import DayModel
from signals import ema, rma, supertrend, true_range

warnings.filterwarnings("ignore")
START_DAY = "2026-08-27"
FAMILIES = ("orb5", "st1_flip", "ema_x", "vwap_x", "pullback")
CONFIRM = ("both", "mtf3", "mtf3_50d", "mtf3_gap", "gap_both")
WINDOWS = {"30m": 1000, "60m": 1030, "90m": 1100}
EXITS = [dict(stop=s, tp=t, cap=c, be=b) for s, t, c, b in itertools.product((1.5, 2.5), (3.0, 5.0, 8.0), (10, 15, 20), (False, True))]


class OneMin:
    def __init__(self):
        D = build()
        f5, f15, m1 = D["f5"], D["f15"], D["m1"].copy()
        supertrend(m1)
        c, h, l, o, v = (m1[k].values for k in ("close", "high", "low", "open", "volume"))
        m1["ema_f"], m1["ema_s"] = ema(c, 8), ema(c, 20)
        m1["atr"] = rma(true_range(m1), 14)
        hlc3 = (h + l + c) / 3
        m1["vwap"] = (pd.Series(hlc3 * v).groupby(m1.day).cumsum() / pd.Series(v).groupby(m1.day).cumsum()).values
        m1["relvol"] = v / pd.Series(v).rolling(20).mean().values
        first5 = m1[(m1.hm >= 930) & (m1.hm < 935)].groupby("day")
        m1["orh"] = np.where(m1.hm >= 934, m1.day.map(first5.high.max()), np.nan)
        m1["orl"] = np.where(m1.hm >= 934, m1.day.map(first5.low.min()), np.nan)
        for name, f in (("st5", f5), ("st15", f15)):   # last completed higher-timeframe bar
            src = pd.DataFrame({"avail": f.t_close, "v": f.st_trend.values}).sort_values("avail")
            m1[name] = pd.merge_asof(pd.DataFrame({"done": m1.t_close}), src, left_on="done", right_on="avail")["v"].fillna(0).values
        from signals import resample
        h1 = resample(f5, 60)
        supertrend(h1)
        src = pd.DataFrame({"avail": h1.t_close, "v": h1.st_trend.values}).sort_values("avail")
        m1["st60"] = pd.merge_asof(pd.DataFrame({"done": m1.t_close}), src, left_on="done", right_on="avail")["v"].fillna(0).values
        from sniper import daily_context
        ctx = daily_context()
        m1["d50"] = m1.day.map(ctx.d_above50).fillna(0).values
        lv = D["levels"]
        m1["gap"] = m1.day.map(np.sign(lv.gap)).fillna(0).values
        self.m1 = m1.reset_index(drop=True)
        self.days = sorted(d for d in self.m1.day.unique() if d >= START_DAY)
        self.base = {d: int(self.m1.index[self.m1.day == d][0]) for d in self.days}
        self.models = {d: DayModel(d, self.m1[self.m1.day == d], load_daily_all(d)) for d in self.days}
        self.strikes = {d: {s: sorted(int(k[1:]) for k in self.models[d].daily if k[0] == s) for s in "CP"} for d in self.days}
        self.o, self.h, self.l, self.c, self.hm = (self.m1[k].values for k in ("open", "high", "low", "close", "hm"))
        self.cache = {}

    def entries(self, fam):
        if fam in self.cache:
            return self.cache[fam]
        f = self.m1
        c, o, h, l = f.close, f.open, f.high, f.low
        cp = c.shift(1)
        if fam == "orb5":
            call = (c > f.orh) & (cp <= f.orh.shift(1))
            put = (c < f.orl) & (cp >= f.orl.shift(1))
        elif fam == "st1_flip":
            call, put = f.st_buy, f.st_sell
        elif fam == "ema_x":
            call = (f.ema_f > f.ema_s) & (f.ema_f.shift(1) <= f.ema_s.shift(1)) & (c > f.vwap)
            put = (f.ema_f < f.ema_s) & (f.ema_f.shift(1) >= f.ema_s.shift(1)) & (c < f.vwap)
        elif fam == "vwap_x":
            call = (c > f.vwap) & (cp <= f.vwap.shift(1)) & (f.relvol > 1.5)
            put = (c < f.vwap) & (cp >= f.vwap.shift(1)) & (f.relvol > 1.5)
        elif fam == "pullback":   # 1m dips to EMA20/VWAP inside a 5m uptrend, then closes back up (mirror for puts)
            call = (f.st5 == 1) & (l <= np.maximum(f.ema_s, f.vwap) + 0.02) & (c > f.ema_f) & (c > o) & (c > f.vwap)
            put = (f.st5 == -1) & (h >= np.minimum(f.ema_s, f.vwap) - 0.02) & (c < f.ema_f) & (c < o) & (c < f.vwap)
        self.cache[fam] = (np.nan_to_num(call.values).astype(bool), np.nan_to_num(put.values).astype(bool))
        return self.cache[fam]

    def confirm(self, side_k, i, how):
        if how == "none":
            return True
        r = self.m1
        ok5, ok15, ok60 = r.st5.iat[i] == side_k, r.st15.iat[i] == side_k, r.st60.iat[i] == side_k
        mtf3 = ok5 and ok15 and ok60
        return {"5m": ok5, "15m": ok15, "both": ok5 and ok15, "mtf3": mtf3,
                "mtf3_50d": mtf3 and r.d50.iat[i] == side_k, "mtf3_gap": mtf3 and r.gap.iat[i] == side_k,
                "gap_both": ok5 and ok15 and r.gap.iat[i] == side_k}[how]

    def pick(self, d, side, i, S, budget, rule):
        dm, ii = self.models[d], i - self.base[d]
        sgn = 1 if side == "C" else -1
        first_otm = int(np.ceil(S)) if side == "C" else int(np.floor(S))
        best = None
        for K in self.strikes[d][side]:
            key = f"{side}{K}"
            fill = dm.price(key, ii, S) + lab2.SLIP
            qty = int(budget // (fill * 100 + FEE))
            if fill - lab2.SLIP < MIN_PREMIUM or qty < 1:
                continue
            steps = sgn * (K - first_otm)
            want = 0 if rule == "near" else 1
            score = (abs(steps - want), -steps) if rule != "near" else (abs(K - S), -K * sgn)
            if best is None or score < best[0]:
                best = (score, K, key, fill, qty)
        return best[1:] if best else None

    def trade(self, d, i, side, ex, budget=100.0, rule="near", detail=False):
        o, h, l, c, hm = self.o, self.h, self.l, self.c, self.hm
        b, n = self.base[d], len(self.models[d].hm)
        k = 1 if side == "C" else -1
        atr = self.m1.atr.iat[i]
        ref, ei = c[i], i + 1
        if ei >= b + n - 1 or np.isnan(atr):
            return None
        pk = self.pick(d, side, ei, o[ei], budget, rule)
        if not pk:
            return None
        K, key, fill, qty = pk
        stop = ref - k * ex["stop"] * atr
        tp = ref + k * ex["tp"] * atr if ex.get("tp") else None
        moved, add_px, best = False, None, 0.0
        st1 = self.m1.st_trend.values
        dm = self.models[d]
        for j in range(ei, b + n):
            if (l[j] <= stop) if k == 1 else (h[j] >= stop):
                xi, xs, why = j, stop, ("BREAKEVEN" if moved else "STOP"); break
            best = max(best, ((h[j] - ref) if k == 1 else (ref - l[j])) / atr)
            if ex.get("add_at") and add_px is None and best >= ex["add_at"]:
                add_px = dm.price(key, j - b, float(ref + k * ex["add_at"] * atr)) + lab2.SLIP
                stop, moved = ref, True
            if tp is not None and ((h[j] >= tp) if k == 1 else (l[j] <= tp)):
                xi, xs, why = j, tp, "TARGET"; break
            if ex.get("trail") and j > ei and st1[j] == -k:
                xi = min(j + 1, b + n - 1); xs, why = o[xi], "TREND FLIP"; break
            if j - ei + 1 >= ex["cap"] or hm[j] >= 1544 or j == b + n - 1:
                xi, xs, why = j, c[j], "TIME LIMIT"; break
            if ex["be"] and not moved and ((h[j] - ref) if k == 1 else (ref - l[j])) >= 1.5 * atr:
                stop, moved = ref, True
        out = max(dm.price(key, xi - b, float(xs)) - lab2.SLIP, 0.0)
        cost = qty * (fill * 100 + FEE)
        pnl = qty * (out * 100 - FEE) - cost
        add_qty = 0
        if add_px is not None:
            add_qty = qty if ex.get("add_budget") is None else min(qty, int(ex["add_budget"] // (add_px * 100 + FEE)))
        if add_qty:
            c2 = add_qty * (add_px * 100 + FEE)
            pnl += add_qty * (out * 100 - FEE) - c2
            cost += c2
        r = dict(day=d, side=side, i=i, xi=xi, pnl=pnl, pct=100 * pnl / cost, why=why, held=xi - ei + 1,
                 addQty=add_qty, addPx=round(add_px, 2) if add_qty else None, cost=round(cost, 2))
        if detail:
            r.update(ei=ei, strike=K, qty=qty, optIn=round(fill, 2), optOut=round(out, 2), spyIn=round(float(o[ei]), 2),
                     spyOut=round(float(xs), 2), stopLvl=round(float(ref - k * ex["stop"] * atr), 2), cost=round(cost, 2),
                     path=[dm.price(key, j - b, float(c[j])) for j in range(ei, xi + 1)])
        return r

    def run(self, fam, how, win, ex, max_day, rule, days, budget=100.0, frac=None, capital=100.0, detail=False,
            acct_rule="none", sessions=None):
        calls, puts = self.entries(fam)
        from acct import Account
        A = Account(capital, acct_rule, sessions or list(days))
        cash, out = capital, []
        for d in days:
            b, n = self.base[d], len(self.models[d].hm)
            busy, cnt = -1, 0
            for i in range(b + 1, b + n - 1):
                if self.hm[i] >= WINDOWS[win] or cnt >= max_day:
                    break
                if i <= busy or not (calls[i] or puts[i]):
                    continue
                side = "C" if calls[i] else "P"
                if not self.confirm(1 if side == "C" else -1, i, how):
                    continue
                if frac is not None and not A.allowed(d):
                    break
                bp = A.buying_power(d) if frac is not None else None
                bud = budget if frac is None else min(frac * A.equity(), bp)
                exx = dict(ex, add_budget=(None if frac is None else max(bp - bud, 0.0))) if ex.get("add_at") else ex
                t = self.trade(d, i, side, exx, bud, rule, detail)
                if t is None:
                    continue
                if frac is None:
                    cash += t["pnl"]
                else:
                    A.book(d, t["cost"], t["pnl"])
                    cash = A.equity()
                t["eq"] = round(cash, 2)
                out.append(t)
                busy, cnt = t["xi"], cnt + 1
        return out


def main():
    t0 = time.time()
    S = OneMin()
    days = S.days
    blocks = [days[:10], days[10:20], days[20:]]
    print(f"{len(days)} sessions {days[0]}→{days[-1]} · setup {time.time() - t0:.0f}s", flush=True)
    rows = []
    grid = list(itertools.product(FAMILIES, CONFIRM, WINDOWS, range(len(EXITS)), (2, 3, 5), ("near", "otm1")))
    for fam, how, win, e, md, rule in grid:
        tr = S.run(fam, how, win, EXITS[e], md, rule, days)
        bl = [sum(x["pnl"] for x in tr if x["day"] in set(bk)) for bk in blocks]
        pnl = np.array([x["pnl"] for x in tr]) if tr else np.zeros(1)
        eq = np.concatenate([[0], np.cumsum(pnl)])
        rows.append(dict(fam=fam, how=how, win=win, exit=e, max_day=md, rule=rule, n=len(tr), net=round(float(pnl.sum()), 1),
                         b1=round(bl[0], 1), b2=round(bl[1], 1), b3=round(bl[2], 1), win_pct=round(100 * float((pnl > 0).mean()), 1),
                         avg_loss=round(float(pnl[pnl <= 0].mean()), 1) if (pnl <= 0).any() else 0,
                         mdd=round(float(np.max(np.maximum.accumulate(eq) - eq)), 1),
                         held=round(float(np.mean([x["held"] for x in tr])), 1) if tr else 0))
    df = pd.DataFrame(rows)
    df.to_pickle("data/sniped1_grid.pkl")
    print(f"{len(df)} configs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()


SNIPED1 = dict(family="pullback", confirm="mtf3_50d", window="90m", max_day=3, rule="otm1",
               exit=dict(stop=1.5, tp=12.0, cap=30, be=True, trail=False, add_at=3.0))
