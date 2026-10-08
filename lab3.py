"""Round 3 lab: smarter contract choice + morning-focused entries + breakeven stops, graded walk-forward.

Walk-forward: the 60 sessions are three 20-day blocks. The setup traded in block 2 is picked using
block 1 only; the setup traded in block 3 is picked using blocks 1-2 only. Blocks 2+3 are the
out-of-sample record. Contract selection never looks at the real daily high/low (raw model price only).
"""
import itertools
import time

import numpy as np
import pandas as pd

import lab2
from data2 import build
from lab2 import FLAT_AT, MIN_PREMIUM, SLIP, FEE, START, Sim, entries as base_entries

FAMILIES = ("sc", "storb", "orb5", "orb15", "orb30", "orb_rt15", "open_drive", "vwap_x", "first_bar", "gap_go", "ema_x")
FILTERS = ("none", "vwap", "st15", "rv")
WINDOWS = {"open": 1030, "am": 1130, "all": 1500}
STRIKES = ("near", "otm1", "otm2", "otm3", "maxup", "rr")
EXITS = [dict(stop=s, tp=t, tcap=c, be=b) for s, t, c, b in
         itertools.product((1.0, 1.5, 2.0), (2.0, 3.0, 4.0), (3, 6, 12), (None, 1.0))]
BLOCK = 20


def entries(f, fam):
    c, o, h, l = f.close, f.open, f.high, f.low
    if fam == "orb_rt15":   # break of the 15m opening range, then the first retest that holds
        hi, lo = f.orh15, f.orl15
        broke_up = (c > hi).groupby(f.day).cummax().shift(1).fillna(False)
        broke_dn = (c < lo).groupby(f.day).cummax().shift(1).fillna(False)
        call = broke_up & (l <= hi + 0.05) & (c > hi) & (c > o)
        put = broke_dn & (h >= lo - 0.05) & (c < lo) & (c < o)
        first_c = call & ~call.groupby(f.day).cummax().shift(1).fillna(False)
        first_p = put & ~put.groupby(f.day).cummax().shift(1).fillna(False)
        return first_c.values, first_p.values
    if fam == "open_drive":  # first 15 minutes move > 0.6 ATR one way, closing near the extreme
        tf_bars = 3 if f.tf.iloc[0] == 5 else 1
        at = f.bar_n == tf_bars - 1
        move = c - f.day_open
        rng = f.groupby("day").high.cummax() - f.groupby("day").low.cummin()
        strong = at & (move.abs() > 0.6 * f.atr) & (move.abs() > 0.6 * rng)
        return (strong & (move > 0)).values, (strong & (move < 0)).values
    return base_entries(f, fam)


class Sim3(Sim):
    def signals(self, fam, filt, tf):
        key = (fam, filt, tf)
        if key not in self.cache:
            f = self.f5 if tf == 5 else self.D["f15"]
            calls, puts = entries(f, fam)
            if filt == "rv":
                fc = fp = (f.relvol > 1.2).values
            else:
                fc, fp = lab2.direction_filter(f, self.st15_on5 if tf == 5 else f.st_trend.values, filt)
            calls = np.nan_to_num(calls & fc).astype(bool)
            puts = np.nan_to_num(puts & fp).astype(bool)
            N = len(self.f5)
            ev_c = np.zeros(N, bool); ev_p = np.zeros(N, bool); ev_atr = np.full(N, np.nan)
            idx = np.arange(N) if tf == 5 else self.pos15
            ok = idx >= 0
            ev_c[idx[ok]] = calls[ok]; ev_p[idx[ok]] = puts[ok]; ev_atr[idx[ok]] = f.atr.values[ok]
            self.cache[key] = (ev_c, ev_p, ev_atr)
        return self.cache[key]

    def pick3(self, d, side, i, S, budget, rule, tp_lvl, stop_lvl, horizon):
        dm, ii = self.models[d], i - self.base[d]
        ks = self.strikes[d][side]
        sgn = 1 if side == "C" else -1
        first_otm = int(np.ceil(S)) if side == "C" else int(np.floor(S))
        cands = []
        for K in ks:
            key = f"{side}{K}"
            raw = dm.raw_price(key, ii, S)
            fill = dm.price(key, ii, S) + SLIP          # actual fill (model clamped to the real range)
            qty = int(budget // (fill * 100 + FEE))
            if raw < MIN_PREMIUM or qty < 1:
                continue
            otm_steps = sgn * (K - first_otm)
            up = dm.raw_price(key, ii + horizon, tp_lvl) / (raw + SLIP) - 1
            dn = 1 - dm.raw_price(key, ii + horizon, stop_lvl) / (raw + SLIP)
            cands.append((K, key, fill, qty, otm_steps, up, dn))
        if not cands:
            return None
        if rule == "near":
            c = min(cands, key=lambda x: (abs(x[0] - S), -x[0] * sgn))
        elif rule.startswith("otm"):
            want = int(rule[3:]) - 1
            c = min(cands, key=lambda x: (abs(x[4] - want), -x[4]))
        elif rule == "maxup":
            c = max(cands, key=lambda x: x[5])
        else:  # rr: best modeled upside per unit of downside
            c = max(cands, key=lambda x: x[5] / max(x[6], 0.05))
        return c[:4]

    def run3(self, fam, filt, tf, win, ex, rule, days, max_trades=2):
        ev_c, ev_p, ev_atr = self.signals(fam, filt, tf)
        last_entry = WINDOWS[win]
        o, h, l, c, hm = self.o, self.h, self.l, self.c, self.hm
        trades = []
        for d in days:
            b = self.base[d]
            n = len(self.models[d].hm)
            pos, count = None, 0
            for i in range(b, b + n - 1):
                if pos:
                    reason = None
                    if i >= pos["ei"]:
                        call = pos["side"] == "C"
                        if (l[i] <= pos["stop"]) if call else (h[i] >= pos["stop"]):
                            reason, xi, xs = ("BE" if pos["moved"] else "STOP"), i, pos["stop"]
                        elif (h[i] >= pos["tp"]) if call else (l[i] <= pos["tp"]):
                            reason, xi, xs = "TP", i, pos["tp"]
                        elif i - pos["ei"] + 1 >= ex["tcap"]:
                            reason, xi, xs = "TIME", i, c[i]
                        elif hm[i] >= FLAT_AT:
                            reason, xi, xs = "EOD", i, c[i]
                        elif ex["be"] and not pos["moved"]:
                            fav = (h[i] - pos["ref"]) if call else (pos["ref"] - l[i])
                            if fav >= ex["be"] * pos["atr"]:
                                pos["stop"], pos["moved"] = pos["ref"], True
                    if reason:
                        trades.append(self.settle(pos, xi, xs, reason, False))
                        pos = None
                    continue
                if hm[i] < last_entry and count < max_trades and (ev_c[i] or ev_p[i]) and not np.isnan(ev_atr[i]):
                    side = "C" if ev_c[i] else "P"
                    k = 1 if side == "C" else -1
                    atr = ev_atr[i]
                    stop, tp = c[i] - k * ex["stop"] * atr, c[i] + k * ex["tp"] * atr
                    count += 1
                    pk = self.pick3(d, side, i + 1, o[i + 1], START, rule, tp, stop, max(1, ex["tcap"] // 2))
                    pos = dict(day=d, side=side, si=i, ei=i + 1, spy_in=o[i + 1], pick=pk, stop=stop, tp=tp,
                               ref=c[i], atr=atr, moved=False)
            if pos:
                trades.append(self.settle(pos, b + n - 1, c[b + n - 1], "EOD", False))
        return trades


def main():
    t0 = time.time()
    D = build()
    sim = Sim3(D)
    days = D["days"]
    blocks = [days[i:i + BLOCK] for i in range(0, len(days), BLOCK)]
    block_of = {d: k for k, blk in enumerate(blocks) for d in blk}
    grid = list(itertools.product(FAMILIES, FILTERS, (5, 15), WINDOWS, range(len(EXITS)), STRIKES))
    print(f"{len(grid)} setups · built {time.time() - t0:.1f}s", flush=True)
    rows = []
    for n, (fam, filt, tf, win, e, rule) in enumerate(grid):
        tr = [t for t in sim.run3(fam, filt, tf, win, EXITS[e], rule, days) if not t.get("skipped")]
        net = [0.0, 0.0, 0.0]; cnt = [0, 0, 0]; gw = [0.0, 0.0, 0.0]; gl = [0.0, 0.0, 0.0]
        for t in tr:
            k = block_of[t["day"]]
            net[k] += t["pnl"]; cnt[k] += 1
            if t["pnl"] > 0:
                gw[k] += t["pnl"]
            else:
                gl[k] -= t["pnl"]
        rows.append(dict(fam=fam, filt=filt, tf=tf, win=win, exit=e, rule=rule,
                         **{f"net{k}": round(net[k], 2) for k in range(3)}, **{f"n{k}": cnt[k] for k in range(3)},
                         **{f"gw{k}": round(gw[k], 2) for k in range(3)}, **{f"gl{k}": round(gl[k], 2) for k in range(3)}))
        if n % 5000 == 0:
            print(f"{n}/{len(grid)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv("data/lab3_grid.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
