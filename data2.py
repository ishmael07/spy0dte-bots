"""60-session dataset: 5m clock (Robinhood 5m before Aug 26, resampled real 1m after), 15m frames,
and every indicator the strategy lab uses."""
import json
import os

import numpy as np
import pandas as pd

from signals import ET, SC, _frame, _stamp, ema, load_1m, resample, rma, rsi, scalper, supertrend, true_range

N_TEST = 90


def load_5m():
    """Continuous 5m RTH bars: real 5m from Robinhood, then 1m→5m once 1m history exists."""
    m1 = load_1m()
    parts = [json.load(open(p)) for p in ("data/spy_5m_may.json", "data/spy_5m_jul.json", "data/spy_5m_warmup.json") if os.path.exists(p)]
    early = pd.concat([_frame(x) for x in parts if x]) if any(parts) else _frame(json.load(open("data/spy_1m.json")))[:0]
    early = early[early.t < m1.t.min()].drop_duplicates("t").sort_values("t")
    late = resample(m1, 5)[["t", "open", "high", "low", "close", "volume"]]
    df = pd.concat([early, late]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return _stamp(df, 5), m1


def add_features(df):
    supertrend(df)
    scalper(df)
    c, h, l, v = (df[k].values for k in ("close", "high", "low", "volume"))
    df["ema50"] = ema(c, 50)
    df["rsi"] = rsi(c, 14)
    # session VWAP and its volume-weighted standard deviation bands
    hlc3 = (h + l + c) / 3
    g = df["day"]
    pv = pd.Series(hlc3 * v).groupby(g).cumsum()
    vv = pd.Series(v).groupby(g).cumsum()
    p2v = pd.Series(hlc3 * hlc3 * v).groupby(g).cumsum()
    vwap = pv / vv
    sd = np.sqrt(np.maximum(p2v / vv - vwap * vwap, 0))
    df["vwap"], df["vsd"] = vwap.values, sd.values
    df["relvol"] = v / pd.Series(v).rolling(20).mean().values
    df["bar_n"] = df.groupby("day").cumcount()
    return df


def day_levels(f5):
    d = f5.groupby("day").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    d["pdh"], d["pdl"], d["pdc"] = d.h.shift(1), d.l.shift(1), d.c.shift(1)
    d["gap"] = d.o - d.pdc
    return d


def add_levels(df, lv):
    for k in ("pdh", "pdl", "pdc", "gap", "o"):
        df[k if k != "o" else "day_open"] = df["day"].map(lv[k]).values
    return df


def add_orb(df, f5, minutes):
    first = f5[(f5.hm >= 930) & (f5.hm < 930 + minutes if minutes < 30 else f5.hm < 1000)].groupby("day")
    hi, lo = first["high"].max(), first["low"].min()
    ready = (df["t_close"].dt.tz_convert(ET).dt.hour * 100 + df["t_close"].dt.tz_convert(ET).dt.minute) >= 930 + minutes if minutes < 30 else \
        (df["t_close"].dt.tz_convert(ET).dt.hour * 100 + df["t_close"].dt.tz_convert(ET).dt.minute) >= 1000
    df[f"orh{minutes}"] = np.where(ready, df["day"].map(hi), np.nan)
    df[f"orl{minutes}"] = np.where(ready, df["day"].map(lo), np.nan)


def build():
    f5, m1 = load_5m()
    w5 = None
    f15 = resample(f5, 15)
    lv = day_levels(f5)
    for f in (f5, f15):
        add_features(f)
        add_levels(f, lv)
        for mins in (5, 15, 30):
            add_orb(f, f5, mins)
    days = sorted(d for d in f5.day.unique() if d <= "2026-10-07")[-N_TEST:]
    return dict(f5=f5, f15=f15, m1=m1, days=days, levels=lv)


def load_daily_all(day):
    out = {}
    for folder in ("data/daily", "data/daily_add"):
        p = f"{folder}/{day}.json"
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


if __name__ == "__main__":
    D = build()
    print(len(D["days"]), D["days"][0], D["days"][-1], len(D["f5"]), len(D["f15"]))
    x = D["f5"][D["f5"].day == "2026-09-22"].iloc[[0, 3, 12]]
    print(x[["hm", "close", "vwap", "vsd", "rsi", "orh15", "orl15", "pdh", "gap", "st_trend", "ema200"]])
