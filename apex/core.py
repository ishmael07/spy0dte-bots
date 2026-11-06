"""Apex core rules: the exact logic validated in ~/spy0dte-backtest/lab5 (lab.py entries_noise + sim.manage band exit).

Minute index i = 0 is the 9:30 bar, 389 the 15:59 bar. A decision on bar i is made when bar i closes.

  sigma(i)  mean over the previous 14 sessions of |close(i)/open(0) - 1|
  UB / LB   max(open, prev close) * (1 + sigma) / min(open, prev close) * (1 - sigma)
  entry     checked on bars i = 30, 35, ..., 325 (10:00 -> 14:55 closes), FIRST signal of the day only:
            close > UB and > VWAP -> buy the ATM call; close < LB and < VWAP -> buy the ATM put
  exit      every bar: call exits when close < max(UB, VWAP), put when close > min(LB, VWAP); else flat after bar 375 (15:45)
"""
from dataclasses import dataclass, field

import numpy as np

LOOKBACK = 14
FIRST, LAST, STRIDE = 30, 330, 5
T_EXIT = 375


def sigma_profile(history):
    """history: list of past sessions, each a list of 390 closes (None where missing) plus its 9:30 open.
    Returns 390 sigmas from the last LOOKBACK sessions."""
    rows = []
    for o, closes in history[-LOOKBACK:]:
        a = np.array([np.nan if c is None else c for c in closes], dtype=float)
        # forward-fill missing minutes like the lab does
        for k in range(1, len(a)):
            if np.isnan(a[k]):
                a[k] = a[k - 1]
        rows.append(np.abs(a / o - 1))
    if len(rows) < 10:
        return None
    return np.nanmean(np.vstack(rows), axis=0)


@dataclass
class Day:
    open: float
    prev_close: float
    sigma: np.ndarray
    closes: list = field(default_factory=list)
    pv: float = 0.0
    vol: float = 0.0
    position: dict = None          # {"side": "C"/"P", "i": entry bar, ...}
    traded: bool = False

    def ub(self, i):
        return max(self.open, self.prev_close) * (1 + self.sigma[i])

    def lb(self, i):
        return min(self.open, self.prev_close) * (1 - self.sigma[i])

    def vwap(self):
        return self.pv / self.vol if self.vol else self.closes[-1]

    def on_bar(self, i, o, h, l, c, v):
        """Feed bar i (must be called in order). Returns ("enter", side) / ("exit", reason) / None."""
        self.closes.append(c)
        self.pv += (h + l + c) / 3 * v
        self.vol += v
        vw = self.vwap()
        if self.position:
            side = self.position["side"]
            if i >= T_EXIT:
                return ("exit", "TIME 15:45")
            if side == "C" and c < max(self.ub(i), vw):
                return ("exit", "BAND")
            if side == "P" and c > min(self.lb(i), vw):
                return ("exit", "BAND")
            return None
        if self.traded or i < FIRST or i >= LAST or (i - FIRST) % STRIDE:
            return None
        if c > self.ub(i) and c > vw:
            return ("enter", "C")
        if c < self.lb(i) and c < vw:
            return ("enter", "P")
        return None

    def levels(self, i):
        vw = self.vwap()
        return dict(ub=round(self.ub(i), 2), lb=round(self.lb(i), 2), vwap=round(vw, 2),
                    call_exit=round(max(self.ub(i), vw), 2), put_exit=round(min(self.lb(i), vw), 2))
