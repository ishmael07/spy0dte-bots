"""Strategy lab: many entry families x filters x exits x timeframes x session windows on 60 sessions,
executed on the 5m clock with real-anchored 0DTE option pricing.

Honesty rule: configs are ranked on the first 40 sessions (train) only; the last 20 (test) are
reported for whatever the ranking picks, never used to choose.
"""
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from data2 import build, load_daily_all
from pricing import DayModel

START, SLIP, FEE, MIN_PREMIUM = 100.0, 0.02, 0.03, 0.05
FLAT_AT = 1540          # exit at the close of the 15:40 5m bar (15:45, Robinhood 0DTE sellout)
N_TRAIN = 40
WINDOWS = {"am": 1130, "all": 1500}

FAMILIES = ("sc", "storb", "orb5", "orb15", "orb30", "vwap_pb", "vwap_x", "rsi_rev", "ema_x", "pd_brk", "gap_go", "first_bar")
FILTERS = ("none", "st15", "vwap", "ema200")
EXITS = [dict(stop=s, tp=t, tcap=c, flip=f)
         for (s, t), c, f in itertools.product([(0.75, 1.5), (1.0, 1.5), (1.0, 2.0), (1.2, 1.8), (1.5, 3.0)],
                                               (None, 6, 12), (False, True))]


# ---------------------------------------------------------------- entries
def entries(f, fam):
    c, o, h, l = f.close, f.open, f.high, f.low
    cp = c.shift(1)
    up_bar, dn_bar = c > o, c < o
    if fam == "sc":
        return f.sc_call.values, f.sc_put.values
    if fam == "storb":
        hi, lo = f.orh15, f.orl15
        up, dn = f.st_trend == 1, f.st_trend == -1
        return (((f.st_buy & (c > hi)) | (up & (c > hi) & (cp <= hi.shift(1)))).values,
                ((f.st_sell & (c < lo)) | (dn & (c < lo) & (cp >= lo.shift(1)))).values)
    if fam.startswith("orb"):
        n = fam[3:]
        hi, lo = f[f"orh{n}"], f[f"orl{n}"]
        return ((c > hi) & (cp <= hi.shift(1))).values, ((c < lo) & (cp >= lo.shift(1))).values
    if fam == "vwap_pb":   # trend pullback to VWAP / slow EMA, then resume
        trend_up = (c > f.vwap) & (f.ema_f > f.ema_s)
        trend_dn = (c < f.vwap) & (f.ema_f < f.ema_s)
        touch_up = (l <= np.maximum(f.vwap, f.ema_s) + 0.05) | (l.shift(1) <= np.maximum(f.vwap, f.ema_s).shift(1) + 0.05)
        touch_dn = (h >= np.minimum(f.vwap, f.ema_s) - 0.05) | (h.shift(1) >= np.minimum(f.vwap, f.ema_s).shift(1) - 0.05)
        return (trend_up & touch_up & up_bar & (c > f.ema_f)).values, (trend_dn & touch_dn & dn_bar & (c < f.ema_f)).values
    if fam == "vwap_x":    # VWAP reclaim / loss on volume
        return (((c > f.vwap) & (cp <= f.vwap.shift(1)) & (f.relvol > 1.3)).values,
                ((c < f.vwap) & (cp >= f.vwap.shift(1)) & (f.relvol > 1.3)).values)
    if fam == "rsi_rev":   # stretched beyond 2σ VWAP band with RSI extreme, first reversal candle
        lo_band, hi_band = f.vwap - 2 * f.vsd, f.vwap + 2 * f.vsd
        return (((f.rsi.shift(1) < 30) & (l.shift(1) < lo_band.shift(1)) & up_bar).values,
                ((f.rsi.shift(1) > 70) & (h.shift(1) > hi_band.shift(1)) & dn_bar).values)
    if fam == "ema_x":
        xu = (f.ema_f > f.ema_s) & (f.ema_f.shift(1) <= f.ema_s.shift(1))
        xd = (f.ema_f < f.ema_s) & (f.ema_f.shift(1) >= f.ema_s.shift(1))
        return (xu & (c > f.vwap) & (f.rsi > 50)).values, (xd & (c < f.vwap) & (f.rsi < 50)).values
    if fam == "pd_brk":
        return (((c > f.pdh) & (cp <= f.pdh)).values, ((c < f.pdl) & (cp >= f.pdl)).values)
    if fam == "gap_go":    # gap day, opening range breaks in the gap's direction
        gap_pct = f.gap / f.pdc
        hi, lo = f.orh5, f.orl5
        return (((gap_pct > 0.0015) & (c > hi) & (cp <= hi.shift(1)) & (c > f.day_open)).values,
                ((gap_pct < -0.0015) & (c < lo) & (cp >= lo.shift(1)) & (c < f.day_open)).values)
    if fam == "first_bar":  # strong opening candle continuation (signal at the first bar's close)
        body = (c - o).abs() / (h - l).replace(0, np.nan)
        first = f.bar_n == 0
        strong = first & (body > 0.6) & (f.relvol > 1.0)
        return (strong & up_bar).values, (strong & dn_bar).values
    raise ValueError(fam)


def direction_filter(f, f15_on_f, name):
    if name == "none":
        return np.ones(len(f), bool), np.ones(len(f), bool)
    if name == "st15":
        return f15_on_f == 1, f15_on_f == -1
    if name == "vwap":
        return (f.close > f.vwap).values, (f.close < f.vwap).values
    if name == "ema200":
        return (f.close > f.ema200).values, (f.close < f.ema200).values
    raise ValueError(name)


# ---------------------------------------------------------------- simulation
class Sim:
    def __init__(self, D):
        self.D = D
        f5, f15 = D["f5"], D["f15"]
        self.f5 = f5.reset_index(drop=True)
        self.day = self.f5.day.values
        self.hm = self.f5.hm.values
        self.o, self.h, self.l, self.c = (self.f5[k].values for k in ("open", "high", "low", "close"))
        self.days = D["days"]
        self.dayset = set(self.days)
        self.base = {d: int(self.f5.index[self.f5.day == d][0]) for d in self.days}
        # place 15m bars on the 5m bar where they close; last completed 15m Supertrend for each 5m bar
        self.pos15 = pd.Index(self.f5.t).get_indexer(f15.t_close - pd.Timedelta(minutes=5))
        src = pd.DataFrame({"avail": f15.t_close, "st": f15.st_trend.values}).sort_values("avail")
        self.st15_on5 = pd.merge_asof(pd.DataFrame({"done": self.f5.t_close}), src, left_on="done",
                                      right_on="avail")["st"].fillna(0).values
        self.st15_on15 = f15.st_trend.shift(1).fillna(0).values  # previous completed 15m bar for 15m signals
        self.models = {d: DayModel(d, self.f5[self.f5.day == d], load_daily_all(d)) for d in self.days}
        self.strikes = {d: {s: sorted(int(k[1:]) for k in self.models[d].daily if k[0] == s) for s in "CP"}
                        for d in self.days}
        self.cache = {}

    def signals(self, fam, filt, tf):
        key = (fam, filt, tf)
        if key not in self.cache:
            f = self.f5 if tf == 5 else self.D["f15"]
            calls, puts = entries(f, fam)
            st15 = self.st15_on5 if tf == 5 else f.st_trend.values
            fc, fp = direction_filter(f, st15, filt)
            calls, puts = np.nan_to_num(calls & fc).astype(bool), np.nan_to_num(puts & fp).astype(bool)
            N = len(self.f5)
            ev_c = np.zeros(N, bool); ev_p = np.zeros(N, bool); ev_atr = np.full(N, np.nan); ev_st = np.zeros(N, int)
            idx = np.arange(N) if tf == 5 else self.pos15
            ok = idx >= 0
            ev_c[idx[ok]] = calls[ok]; ev_p[idx[ok]] = puts[ok]
            ev_atr[idx[ok]] = f.atr.values[ok]; ev_st[idx[ok]] = f.st_trend.values[ok]
            self.cache[key] = (ev_c, ev_p, ev_atr, ev_st)
        return self.cache[key]

    def pick(self, d, side, i, S, budget):
        dm, ii = self.models[d], i - self.base[d]
        for K in sorted(self.strikes[d][side], key=lambda k: (abs(k - S), -k if side == "C" else k)):
            key = f"{side}{K}"
            p = dm.price(key, ii, S)
            fill = p + SLIP
            qty = int(budget // (fill * 100 + FEE))
            if p >= MIN_PREMIUM and qty >= 1:
                return K, key, fill, qty
        return None

    def run(self, fam, filt, tf, win, ex, days=None, flat=True, max_trades=2, keep_path=False):
        ev_c, ev_p, ev_atr, ev_st = self.signals(fam, filt, tf)
        days = set(days or self.days)
        last_entry = WINDOWS[win]
        o, h, l, c, hm, day = self.o, self.h, self.l, self.c, self.hm, self.day
        cash, trades, pos, count = START, [], None, {}
        for d in sorted(days):
            b = self.base[d]
            n = len(self.models[d].hm)
            for i in range(b, b + n - 1):
                if pos:
                    reason = None
                    if i >= pos["ei"]:
                        if (l[i] <= pos["stop"]) if pos["side"] == "C" else (h[i] >= pos["stop"]):
                            reason, xi, xs = "STOP", i, pos["stop"]
                        elif (h[i] >= pos["tp"]) if pos["side"] == "C" else (l[i] <= pos["tp"]):
                            reason, xi, xs = "TP", i, pos["tp"]
                        elif ex["tcap"] and i - pos["ei"] + 1 >= ex["tcap"]:
                            reason, xi, xs = "TIME", i, c[i]
                        elif ex["flip"] and ev_st[i] == (-1 if pos["side"] == "C" else 1):
                            reason, xi, xs = "FLIP", i + 1, o[i + 1]
                        elif hm[i] >= FLAT_AT:
                            reason, xi, xs = "EOD", i, c[i]
                    if reason:
                        trades.append(self.settle(pos, xi, xs, reason, keep_path))
                        if not flat:
                            cash += trades[-1].get("pnl", 0)
                        pos = None
                        continue
                if pos is None and hm[i] < last_entry and count.get(d, 0) < max_trades and (ev_c[i] or ev_p[i]):
                    side = "C" if ev_c[i] else "P"
                    k = 1 if side == "C" else -1
                    atr = ev_atr[i]
                    if np.isnan(atr):
                        continue
                    count[d] = count.get(d, 0) + 1
                    pk = self.pick(d, side, i + 1, o[i + 1], START if flat else cash)
                    pos = dict(day=d, side=side, si=i, ei=i + 1, spy_in=o[i + 1], pick=pk,
                               stop=c[i] - k * ex["stop"] * atr, tp=c[i] + k * ex["tp"] * atr)
            if pos:   # safety: never carry overnight
                trades.append(self.settle(pos, b + n - 1, c[b + n - 1], "EOD", keep_path))
                pos = None
        return trades

    def settle(self, pos, xi, xs, reason, keep_path):
        d = pos["day"]
        rec = dict(day=d, side=pos["side"], reason=reason, ei=pos["ei"], xi=xi,
                   spyIn=round(float(pos["spy_in"]), 2), spyOut=round(float(xs), 2))
        if not pos["pick"]:
            rec["skipped"] = True
            return rec
        K, key, fill, qty = pos["pick"]
        dm, b = self.models[d], self.base[d]
        out = max(dm.price(key, xi - b, float(xs)) - SLIP, 0.0)
        cost = qty * (fill * 100 + FEE)
        pnl = qty * (out * 100 - FEE) - cost
        rec.update(strike=K, qty=qty, optIn=round(fill, 2), optOut=round(out, 2), pnl=round(pnl, 2),
                   pct=round(100 * pnl / cost, 1))
        if keep_path:
            rec["path"] = [dm.price(key, j - b, float(self.c[j])) for j in range(pos["ei"], xi + 1)]
        return rec


def summarize(trades):
    t = [x for x in trades if not x.get("skipped")]
    if not t:
        return dict(n=0, net=0.0, win=0.0, pf=0.0, avg=0.0)
    pnl = np.array([x["pnl"] for x in t])
    gl = -pnl[pnl <= 0].sum()
    return dict(n=len(t), net=round(float(pnl.sum()), 2), win=round(100 * float((pnl > 0).mean()), 1),
                pf=round(float(pnl[pnl > 0].sum() / gl), 2) if gl else 99.0, avg=round(float(pnl.mean()), 2))


def main():
    t0 = time.time()
    D = build()
    sim = Sim(D)
    train, test = D["days"][:N_TRAIN], D["days"][N_TRAIN:]
    print(f"built in {time.time() - t0:.1f}s · train {train[0]}→{train[-1]} · test {test[0]}→{test[-1]}", flush=True)
    rows = []
    grid = list(itertools.product(FAMILIES, FILTERS, (5, 15), WINDOWS, range(len(EXITS))))
    for n, (fam, filt, tf, win, e) in enumerate(grid):
        ex = EXITS[e]
        tr = sim.run(fam, filt, tf, win, ex, days=train)
        te = sim.run(fam, filt, tf, win, ex, days=test)
        rows.append(dict(fam=fam, filt=filt, tf=tf, win=win, **{k: ex[k] for k in ex},
                         **{f"tr_{k}": v for k, v in summarize(tr).items()},
                         **{f"te_{k}": v for k, v in summarize(te).items()}))
        if n % 500 == 0:
            print(f"{n}/{len(grid)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("data/lab2_grid.csv", index=False)
    print(f"done {len(df)} configs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
