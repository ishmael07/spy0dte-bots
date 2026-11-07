"""Apex v2 rules — a bar-by-bar port of ~/spy0dte-backtest/lab5/playbooks.py (validated on 677 sessions of real
0DTE option prints; parity checked by lab5/parity2.py).

Minute index i = 0 is the 9:30 bar, 389 the 15:59 bar. Everything is decided when bar i CLOSES; orders go out at once.

Playbooks (checked in this order whenever flat; one position at a time):
  ORB    bars 5-60   (9:35-10:30) close crosses the 5-minute opening range with the 5m Supertrend agreeing.
                     Target 1x OR range, stop at the OR midpoint, 30-min cap; if the target hits while SPY is already
                     outside the noise band it converts into a ride.
  RIDE   bars 30-325 (10:00-14:55, every 5 min) first close beyond the noise band AND VWAP -> hold until SPY closes back
                     inside max(UB, VWAP) / min(LB, VWAP); flat at 15:45.
  PULL   bars 30-330 on a day that has already closed outside the band: 1m bar dips to the EMA20 and closes back in trend
                     direction, 5m+15m Supertrend agree, price on the trend side of VWAP -> scalp: +3 ATR target, -2 ATR
                     stop, 20-min cap (max 2 a day).
  FADE   a RIDE/ORB that failed (structure exit within 30 min) and SPY crosses VWAP the other way within 30 min -> scalp.
Two independent lanes run side by side (LANES): A = RIDE alone (never blocked by scalps), B = ORB/PULL/FADE with at
most 2 trades and a stop after 2 losers -> at most 3 trades a day. (Lab: lanes2.py.) Sizing dial from the 9:33 ATM straddle (implied move % of SPY):
  <0.45% full, <0.60% half, <0.80% quarter, else no trades; lane B trades at half of that.
"""
import numpy as np

LOOKBACK = 14
T_EXIT = 375
LAST_SCAN = 370
MAX_TRADES, MAX_LOSSES = 3, 2
P = dict(orb_from=5, orb_to=60, orb_tp=1.0, orb_cap=30, pull_max=2, pull_tp=3.0, pull_sl=2.0, pull_cap=20,
         fade_tp=3.0, fade_sl=2.0, fade_cap=20)


def sigma_profile(history):
    """history: list of (9:30 open, [390 closes or None]) for past sessions -> 390 sigmas (mean of last 14)."""
    rows = []
    for o, closes in history[-LOOKBACK:]:
        a = np.array([np.nan if c is None else c for c in closes], dtype=float)
        for k in range(1, len(a)):
            if np.isnan(a[k]):
                a[k] = a[k - 1]
        rows.append(np.abs(a / o - 1))
    if len(rows) < 10:
        return None
    return np.nanmean(np.vstack(rows), axis=0)


def supertrend(h, l, c, n=10, k=3.0):
    h, l, c = map(np.asarray, (h, l, c))
    if len(c) == 0:
        return np.array([])
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.empty(len(tr))
    atr[0] = tr[0]
    for t in range(1, len(tr)):
        atr[t] = atr[t - 1] + (tr[t] - atr[t - 1]) / n
    mid = (h + l) / 2
    up, dn = mid - k * atr, mid + k * atr
    d = np.ones(len(c))
    fu, fd = up.copy(), dn.copy()
    for t in range(1, len(c)):
        fu[t] = max(up[t], fu[t - 1]) if c[t - 1] > fu[t - 1] else up[t]
        fd[t] = min(dn[t], fd[t - 1]) if c[t - 1] < fd[t - 1] else dn[t]
        d[t] = 1 if c[t] > fd[t - 1] else -1 if c[t] < fu[t - 1] else d[t - 1]
    return d


LANES = {"A": dict(books=("ride",), max_trades=1), "B": dict(books=("orb", "pull", "fade"), max_trades=2)}

# Risk modes: fraction of the account's equity per trade, by lane (x the 9:33 size dial). Chosen 2026-11-06 from
# lab5 compounding tests (design / calm test / live month, $500): balanced $3,254 / $116 / $1,953;
# survival $2,402 / $293 / $1,062; aggressive $7,121 / $0 / $9,081.
MODES = {"balanced": {"A": 0.15, "B": 0.05}, "survival": {"A": 0.10, "B": 0.0}, "aggressive": {"A": 0.50, "B": 0.10}}
PRIMARY = "balanced"
# Profit-taking (lab5/scaleout.py): sell half when the option doubles, then a breakeven stop on the rest.
# Balanced sizing: design +278% -> +180%, calm test -59% -> -36%, live month +171% -> +104% (smaller swings both ways).
# Quicker targets (+20/30/50%) raised the green-trade rate to 46-57% but lost much more money in every period.
SCALE_OUT = {"balanced": (1.0, 0.5), "survival": (1.0, 0.5), "aggressive": None}      # (gain, fraction sold)


def dial(imp_pct):
    if imp_pct is None or imp_pct != imp_pct:
        return 0.5
    return 1.0 if imp_pct < 0.45 else 0.5 if imp_pct < 0.6 else 0.25 if imp_pct < 0.8 else 0.0


class Session:
    """One trading day. Feed bars in order with on_bar(); act on the returned events.

    prior_blocks: {5: [(h, l, c), ...], 15: [...]} completed 5m/15m bars before today (continuous history for the
    Supertrend). end: index of the last bar of the session (389, or 209 on a 1 pm half day)."""

    def __init__(self, open_, prev_close, sigma, prior_blocks, end=389, books=("orb", "ride", "pull", "fade"), max_trades=MAX_TRADES):
        self.o0, self.pc, self.sig, self.end = open_, prev_close, sigma, end
        self.books, self.max_trades = books, max_trades
        self.blocks = {m: list(prior_blocks.get(m, [])) for m in (5, 15)}
        self.st_prev = {m: (supertrend(*zip(*self.blocks[m]))[-1] if self.blocks[m] else 0) for m in (5, 15)}
        self.st = {5: [], 15: []}
        self.c, self.h, self.l, self.vw, self.up, self.dn, self.atr, self.e20 = ([] for _ in range(8))
        self._pv = self._v = 0.0
        self._an = self._ad = 0.0
        self.pos = None
        self.trades = []          # closed + open trades: dict(book, side, i, j, why, ret)
        self.losses = 0
        self.next_scan = 5
        self.ctx = dict(orb_done=False, ride_done=False, pulls=0, fade_done=False, failed=None)
        self.imp = None           # set by the caller after bar 3 (ATM straddle mid / open * 100)
        self.halted = False

    # ---------------------------------------------------------------- indicators (identical to the lab)
    def _ub(self, i):
        return max(self.o0, self.pc) * (1 + self.sig[i])

    def _lb(self, i):
        return min(self.o0, self.pc) * (1 - self.sig[i])

    def _update(self, i, o, h, l, c, v):
        self.c.append(c), self.h.append(h), self.l.append(l)
        self._pv += (h + l + c) / 3 * v
        self._v += v
        self.vw.append(self._pv / self._v if self._v else c)
        self.up.append(self._ub(i)), self.dn.append(self._lb(i))
        prev = self.c[-2] if i else c
        tr = max(h - l, abs(h - prev))                         # lab feats.F: max(h-l, |h - prev close|)
        a = 1 / 14                                             # pandas ewm(alpha=1/14), adjust=True
        self._an = tr + (1 - a) * self._an
        self._ad = 1 + (1 - a) * self._ad
        self.atr.append(self._an / self._ad)
        k = 2 / 21                                             # ewm(span=20, adjust=False)
        self.e20.append(c if not i else self.e20[-1] + k * (c - self.e20[-1]))
        for m in (5, 15):
            if (i + 1) % m == 0:                                # a block just completed
                s = i + 1 - m
                self.blocks[m].append((max(self.h[s:i + 1]), min(self.l[s:i + 1]), c))
                blk = self.blocks[m][-400:]
                self.st_prev[m] = supertrend(*zip(*blk))[-1]
            self.st[m].append(self.st_prev[m])

    def levels(self):
        if not self.c:
            return {}
        i = len(self.c) - 1
        return dict(ub=round(self.up[i], 2), lb=round(self.dn[i], 2), vwap=round(self.vw[i], 2),
                    call_exit=round(max(self.up[i], self.vw[i]), 2), put_exit=round(min(self.dn[i], self.vw[i]), 2),
                    orh=round(max(self.h[:5]), 2) if i >= 4 else None, orl=round(min(self.l[:5]), 2) if i >= 4 else None,
                    atr=round(self.atr[i], 3), ema20=round(self.e20[i], 2), st5=int(self.st[5][i]), st15=int(self.st[15][i]))

    # ---------------------------------------------------------------- playbooks (lab5/playbooks.py)
    def _orb(self, i):
        if not (P["orb_from"] <= i <= P["orb_to"]) or self.ctx["orb_done"]:
            return None
        hi, lo = max(self.h[:5]), min(self.l[:5])
        rng = hi - lo
        c, cp, s5 = self.c[i], self.c[i - 1], self.st[5][i]
        if c > hi and cp <= hi and s5 >= 0:
            self.ctx["orb_done"] = True
            return "C", dict(tp=c + P["orb_tp"] * rng, sl=(hi + lo) / 2, cap=P["orb_cap"], upgrade=True)
        if c < lo and cp >= lo and s5 <= 0:
            self.ctx["orb_done"] = True
            return "P", dict(tp=c - P["orb_tp"] * rng, sl=(hi + lo) / 2, cap=P["orb_cap"], upgrade=True)
        return None

    def _ride(self, i):
        if self.ctx["ride_done"] or not (30 <= i < 330) or (i - 30) % 5:
            return None
        c = self.c[i]
        if c > self.up[i] and c > self.vw[i]:
            self.ctx["ride_done"] = True
            return "C", dict(struct="band")
        if c < self.dn[i] and c < self.vw[i]:
            self.ctx["ride_done"] = True
            return "P", dict(struct="band")
        return None

    def _pull(self, i):
        if not (30 <= i <= 330) or self.ctx["pulls"] >= P["pull_max"]:
            return None
        c, a, e = self.c[i], self.atr[i], self.e20[i]
        out_up = any(x > u for x, u in zip(self.c, self.up))
        out_dn = any(x < d for x, d in zip(self.c, self.dn))
        s5, s15 = self.st[5][i], self.st[15][i]
        if out_up and s5 > 0 and s15 > 0 and c > self.vw[i] and self.l[i] <= e and c > e and c > self.c[i - 1]:
            self.ctx["pulls"] += 1
            return "C", dict(tp=c + P["pull_tp"] * a, sl=c - P["pull_sl"] * a, cap=P["pull_cap"])
        if out_dn and s5 < 0 and s15 < 0 and c < self.vw[i] and self.h[i] >= e and c < e and c < self.c[i - 1]:
            self.ctx["pulls"] += 1
            return "P", dict(tp=c - P["pull_tp"] * a, sl=c + P["pull_sl"] * a, cap=P["pull_cap"])
        return None

    def _fade(self, i):
        f = self.ctx["failed"]
        if not f or i > 300 or self.ctx["fade_done"] or i - f["j"] > 30:
            return None
        c, a = self.c[i], self.atr[i]
        if f["side"] == "C" and c < self.vw[i]:
            self.ctx["fade_done"] = True
            return "P", dict(tp=c - P["fade_tp"] * a, sl=c + P["fade_sl"] * a, cap=P["fade_cap"])
        if f["side"] == "P" and c > self.vw[i]:
            self.ctx["fade_done"] = True
            return "C", dict(tp=c + P["fade_tp"] * a, sl=c - P["fade_sl"] * a, cap=P["fade_cap"])
        return None

    BOOKS = (("orb", "_orb"), ("ride", "_ride"), ("pull", "_pull"), ("fade", "_fade"))

    # ---------------------------------------------------------------- main step
    def on_bar(self, i, o, h, l, c, v):
        """Returns a list of events: ("exit", trade) and/or ("enter", trade)."""
        if i < len(self.c):                       # duplicate bar
            return []
        while len(self.c) < i:                    # minute with no prints: carry the last close forward (lab does ffill)
            pc = self.c[-1] if self.c else o
            self._update(len(self.c), pc, pc, pc, pc, 0.0)
        self._update(i, o, h, l, c, v)
        ev = []
        p = self.pos
        if p:
            sg = 1 if p["side"] == "C" else -1
            ex = p["ex"]
            why = None
            t_exit = min(T_EXIT, self.end - 1)
            if i >= t_exit:
                why = "TIME"
            else:
                if ex.get("sl") is not None and sg * (c - ex["sl"]) <= 0:
                    why = "STOP"
                elif ex.get("tp") is not None and sg * (c - ex["tp"]) >= 0:
                    if ex.get("upgrade") and (c > self.up[i] if sg > 0 else c < self.dn[i]):
                        p["ex"] = ex = dict(struct="band")
                        p["book"] = p["book"] + "→ride"
                    else:
                        why = "TARGET"
                if not why and ex.get("struct") == "band":
                    ref = max(self.up[i], self.vw[i]) if sg > 0 else min(self.dn[i], self.vw[i])
                    if sg * (c - ref) < 0:
                        why = "BAND"
                if not why and ex.get("cap") and i - p["i"] >= ex["cap"]:
                    why = "CAP"
            if why:
                p.update(j=i, why=why)
                self.pos = None
                self.next_scan = i + 1
                base = p["book"].split("→")[0]
                if base in ("ride", "orb") and why in ("BAND", "STOP") and i - p["i"] <= 30:
                    self.ctx["failed"] = dict(side=p["side"], j=i)
                ev.append(("exit", p))
            return ev
        if self.halted or i < self.next_scan or i >= min(LAST_SCAN, self.end - 5):
            return ev
        if len(self.trades) >= self.max_trades or self.losses >= MAX_LOSSES:
            return ev
        for name, fn in self.BOOKS:
            if name not in self.books:
                continue
            r = getattr(self, fn)(i)
            if r:
                side, ex = r
                t = dict(book=name, side=side, i=i, j=None, ex=ex, spy_in=c, n=len(self.trades) + 1)
                self.pos = t
                self.trades.append(t)
                ev.append(("enter", t))
                break
        return ev

    def force_exit(self, why):
        """exit decided between bar closes (tick-level target/stop). Same bookkeeping as a bar-close exit."""
        p = self.pos
        if not p:
            return None
        i = len(self.c) - 1
        p.update(j=i, why=why)
        self.pos = None
        self.next_scan = i + 1
        if p["book"].split("→")[0] in ("ride", "orb") and why.startswith(("BAND", "STOP")) and i - p["i"] <= 30:
            self.ctx["failed"] = dict(side=p["side"], j=i)
        return p

    def result(self, trade, ret):
        """caller reports the realised option return of a closed trade (drives the 2-loser stop)"""
        trade["ret"] = ret
        self.losses = sum(1 for t in self.trades if t.get("ret") is not None and t["ret"] <= 0)

    def cancel(self, trade):
        """entry could not be filled (no contract / no quote): forget it, keep scanning from the next bar"""
        if self.pos is trade:
            self.pos = None
        if trade in self.trades:
            self.trades.remove(trade)
        self.next_scan = trade["i"] + 1
