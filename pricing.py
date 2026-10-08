"""Real-anchored 0DTE option pricing.

Robinhood keeps only daily OHLC for expired SPY contracts, so each contract's intraday path is
rebuilt from: its real daily open (solves the contract's implied vol at 9:30), real SPY minute
bars (drive the price), an intraday IV drift calibrated on the real minute bars Robinhood still
had (Oct 7 full day, Oct 8 morning; see validate.py), and a final clamp into the contract's real
daily [low, high] range.
"""
import json
import math
import os

import numpy as np

YEAR_MIN = 252 * 390          # trading-time year: variance accrues only during the session
CLOSE_HM = 1600
IV_DRIFT = -0.4  # annualized IV(t) = IV_open * exp(IV_DRIFT * fraction of session elapsed)


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, iv, side):
    intrinsic = max(S - K, 0) if side == "C" else max(K - S, 0)
    if T <= 0 or iv <= 0:
        return intrinsic
    v = iv * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if side == "C":
        return S * ncdf(d1) - K * ncdf(d2)
    return K * ncdf(-d2) - S * ncdf(-d1)


def implied_vol(price, S, K, T, side):
    intrinsic = max(S - K, 0) if side == "C" else max(K - S, 0)
    if price <= intrinsic + 1e-4:
        return None
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs(S, K, T, mid, side) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def minutes_left(hm):
    return (CLOSE_HM // 100 * 60 + CLOSE_HM % 100) - (hm // 100 * 60 + hm % 100)


class DayModel:
    """All modeled contracts for one expiration day."""

    def __init__(self, day, spy_day, daily):
        self.day = day
        self.hm = spy_day["hm"].values                 # minute starts
        self.o = spy_day["open"].values
        self.h = spy_day["high"].values
        self.l = spy_day["low"].values
        self.c = spy_day["close"].values
        self.daily = {k: v for k, v in daily.items() if v.get("o") and not v.get("interp")}
        self.T = np.array([minutes_left(x) for x in self.hm]) / YEAR_MIN   # at minute start
        self.elapsed = (390 - np.array([minutes_left(x) for x in self.hm])) / 390
        S0, T0 = self.o[0], self.T[0]
        raw = {}
        for key, bar in self.daily.items():
            iv = implied_vol(bar["o"], S0, float(key[1:]), T0, key[0])
            if iv and 0.03 < iv < 3:
                raw[key] = iv
        # one vol per strike, read off the out-of-the-money side (ITM prints are stale/wide)
        smile = {}
        for K in sorted({int(k[1:]) for k in raw}):
            otm, itm = ("C", "P") if K >= S0 else ("P", "C")
            smile[K] = raw.get(f"{otm}{K}", raw.get(f"{itm}{K}"))
        # replace outliers with the median of neighbouring strikes
        ks = sorted(smile)
        self.smile = {}
        for j, K in enumerate(ks):
            near = [smile[x] for x in ks[max(0, j - 2): j + 3] if x != K]
            med = float(np.median(near)) if near else smile[K]
            self.smile[K] = med if abs(smile[K] / med - 1) > 0.25 else smile[K]
        self.atm_iv = self.smile[min(ks, key=lambda K: abs(K - S0))] if ks else 0.15
        self.b = IV_DRIFT

    def iv_at(self, key, i):
        K = int(key[1:])
        iv0 = self.smile.get(K) or self.smile[min(self.smile, key=lambda x: abs(x - K))] if self.smile else self.atm_iv
        return iv0 * math.exp(self.b * self.elapsed[i])

    def price(self, key, i, S):
        """Modeled price of contract `key` at minute index i with SPY at S, clamped to the real range."""
        p = bs(S, float(key[1:]), self.T[i], self.iv_at(key, i), key[0])
        bar = self.daily.get(key)
        if bar:
            p = min(max(p, bar["l"]), bar["h"])
        return max(round(p, 2), 0.01)

    def raw_price(self, key, i, S):
        """Model price without the clamp to the real daily range — what a trader could know at time i."""
        return max(bs(S, float(key[1:]), self.T[min(i, len(self.T) - 1)], self.iv_at(key, min(i, len(self.T) - 1)), key[0]), 0.01)

    def has(self, key):
        return key in self.daily


def load_daily(day):
    path = f"data/daily/{day}.json"
    return json.load(open(path)) if os.path.exists(path) else {}
