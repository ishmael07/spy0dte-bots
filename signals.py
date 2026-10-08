"""Port of both TradingView indicators to Python, on SPY RTH bars at 1m / 5m / 15m.

Settings match the user's chart (screenshot inputs):
  Supertrend: ATR 10, hl2, x3.0, changeATR=true
  Scalper v2: EMA 8/20/200, RSI 14 (70/30), vol 1.5x/20, ATR 10, stop 1.2, TP 1.8,
              sessions 0930-1200 + 1200-1500, min score 5, lookback 3, min bars 5
"""
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

ST = dict(period=10, mult=3.0)
SC = dict(fast=8, slow=20, trend=200, rsi=14, rsi_ob=70, rsi_os=30, vol_mult=1.5, vol_len=20,
          atr=10, stop=1.2, tp=1.8, min_score=5, lookback=3, min_bars=5)


def _frame(raw):
    df = pd.DataFrame({
        "t": pd.to_datetime([b["begins_at"] for b in raw], utc=True),
        "open": [float(b["open_price"]) for b in raw],
        "high": [float(b["high_price"]) for b in raw],
        "low": [float(b["low_price"]) for b in raw],
        "close": [float(b["close_price"]) for b in raw],
        "volume": [float(b["volume"]) for b in raw],
    })
    return df


def _stamp(df, tf):
    df["et"] = df["t"].dt.tz_convert(ET)
    df["day"] = df["et"].dt.strftime("%Y-%m-%d")
    df["hm"] = df["et"].dt.hour * 100 + df["et"].dt.minute
    df["tf"] = tf
    df["t_close"] = df["t"] + pd.Timedelta(minutes=tf)  # moment the bar is final
    return df.reset_index(drop=True)


def load_1m():
    return _stamp(_frame(json.load(open("data/spy_1m.json"))), 1)


def resample(m1, tf, warm5=None):
    """Aggregate 1m (plus older 5m warmup) into tf-minute bars anchored at 9:30 ET."""
    parts = [m1[["t", "open", "high", "low", "close", "volume"]]]
    if warm5 is not None:
        parts.insert(0, warm5)
    src = pd.concat(parts).sort_values("t")
    et = src["t"].dt.tz_convert(ET)
    mins = (et.dt.hour * 60 + et.dt.minute) - 570
    bucket = et.dt.normalize() + pd.to_timedelta(570 + (mins // tf) * tf, unit="m")
    g = src.groupby(bucket.dt.tz_convert("UTC").values)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out.index.name = "t"
    out = out.reset_index()
    out["t"] = pd.to_datetime(out["t"], utc=True)
    # live data: a trailing bar that hasn't finished yet is dropped, so signals never use a forming bar
    data_end = src["t"].max() + pd.Timedelta(minutes=1)
    if len(out) and out["t"].iloc[-1] + pd.Timedelta(minutes=tf) > data_end and (et.dt.hour * 60 + et.dt.minute).iloc[-1] < 959:
        out = out.iloc[:-1]
    return _stamp(out, tf)


# ---- Pine built-ins -------------------------------------------------------
def rma(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def ema(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    a = 2 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def true_range(df):
    pc = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr.values


def rsi(close, n):
    d = np.diff(np.asarray(close, float))
    up, dn = rma(np.maximum(d, 0), n), rma(np.maximum(-d, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    r[np.isnan(up)] = np.nan
    return np.concatenate([[np.nan], r])


def barssince(cond):
    out = np.full(len(cond), np.inf)
    last = None
    for i, c in enumerate(cond):
        if c:
            last = i
        if last is not None:
            out[i] = i - last
    return out


# ---- Indicator #1: Supertrend + 15m ORB -----------------------------------
def supertrend(df):
    n, m = ST["period"], ST["mult"]
    atr = rma(true_range(df), n)
    src = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    N = len(df)
    up = np.full(N, np.nan); dn = np.full(N, np.nan); trend = np.ones(N, int)
    for i in range(N):
        if np.isnan(atr[i]):
            continue
        u, d = src[i] - m * atr[i], src[i] + m * atr[i]
        u1 = up[i - 1] if not np.isnan(up[i - 1]) else u
        d1 = dn[i - 1] if not np.isnan(dn[i - 1]) else d
        if c[i - 1] > u1:
            u = max(u, u1)
        if c[i - 1] < d1:
            d = min(d, d1)
        up[i], dn[i] = u, d
        t = trend[i - 1]
        if t == -1 and c[i] > d1:
            t = 1
        elif t == 1 and c[i] < u1:
            t = -1
        trend[i] = t
    df["st_up"], df["st_dn"], df["st_trend"] = up, dn, trend
    prev = np.roll(trend, 1); prev[0] = trend[0]
    df["st_buy"] = (trend == 1) & (prev == -1)
    df["st_sell"] = (trend == -1) & (prev == 1)


def orb(df, m1):
    first = m1[(m1.hm >= 930) & (m1.hm < 945)].groupby("day")
    hi, lo = first["high"].max(), first["low"].min()
    ready = pd.Timestamp("09:45").time()
    live = (df["t_close"].dt.tz_convert(ET).dt.time >= ready).values
    df["orb_hi"] = np.where(live, df["day"].map(hi), np.nan)
    df["orb_lo"] = np.where(live, df["day"].map(lo), np.nan)


# ---- Indicator #2: Options Scalper v2 -------------------------------------
def scalper(df):
    c, h, l, o, v = (df[k].values for k in ("close", "high", "low", "open", "volume"))
    ef, es, e200 = ema(c, SC["fast"]), ema(c, SC["slow"]), ema(c, SC["trend"])
    r = rsi(c, SC["rsi"])
    hlc3 = (h + l + c) / 3
    vwap = (pd.Series(hlc3 * v).groupby(df["day"]).cumsum() / pd.Series(v).groupby(df["day"]).cumsum()).values
    volma = pd.Series(v).rolling(SC["vol_len"]).mean().values
    spike = v > volma * SC["vol_mult"]
    atr = rma(true_range(df), SC["atr"])
    macd = ema(c, 12) - ema(c, 26)
    k0 = np.where(~np.isnan(macd))[0][0]
    sig = np.full(len(c), np.nan); sig[k0:] = ema(macd[k0:], 9)
    hist = macd - sig
    histp = np.concatenate([[np.nan], hist[:-1]])
    mbull = (macd > sig) & (hist > histp)
    mbear = (macd < sig) & (hist < histp)
    ll = pd.Series(l).rolling(14).min().values
    hh = pd.Series(h).rolling(14).max().values
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100 * (c - ll) / (hh - ll)
    d = pd.Series(k).rolling(3).mean().values
    sos, sob = (k < 25) & (d < 25), (k > 75) & (d > 75)
    efp = np.concatenate([[np.nan], ef[:-1]]); esp = np.concatenate([[np.nan], es[:-1]])
    xup = (ef > es) & (efp <= esp)
    xdn = (ef < es) & (efp >= esp)
    rup = barssince(xup) <= SC["lookback"]
    rdn = barssince(xdn) <= SC["lookback"]
    bull_c = (c > o) & ((c - o) > (h - l) * 0.5)
    bear_c = (c < o) & ((o - c) > (h - l) * 0.5)
    hhp = pd.Series(h).shift(1).rolling(5).max().values
    llp = pd.Series(l).shift(1).rolling(5).min().values
    hm = df["hm"].values
    in_sess = (hm >= 930) & (hm < 1500)

    cs = (rup * 2 + (ef > es) + ((r > 45) & (r < SC["rsi_ob"])) + (c > vwap) + mbull
          + (spike & bull_c) + (sos | (k > d)) + (c > e200) + (h > hhp)).astype(int)
    ps = (rdn * 2 + (ef < es) + ((r < 55) & (r > SC["rsi_os"])) + (c < vwap) + mbear
          + (spike & bear_c) + (sob | (k < d)) + (c < e200) + (l < llp)).astype(int)
    call = rup & (cs >= SC["min_score"]) & in_sess & (cs > ps)
    put = rdn & (ps >= SC["min_score"]) & in_sess & (ps > cs)

    fc = np.zeros(len(c), bool); fp = np.zeros(len(c), bool)
    since = 0
    for i in range(len(c)):
        since += 1
        ok = since >= SC["min_bars"]
        fc[i], fp[i] = call[i] and ok, put[i] and ok
        if fc[i] or fp[i]:
            since = 0
    df["ema_f"], df["ema_s"], df["ema200"], df["vwap"], df["atr"] = ef, es, e200, vwap, atr
    df["call_score"], df["put_score"] = cs, ps
    df["sc_call"], df["sc_put"] = fc, fp


def frames():
    """{1: df1m, 5: df5m, 15: df15m} with both indicators computed."""
    m1 = load_1m()
    w5 = _frame(json.load(open("data/spy_5m_warmup.json")))
    out = {1: m1}
    for tf in (5, 15):
        out[tf] = resample(m1, tf, w5)
    for tf, df in out.items():
        supertrend(df)
        orb(df, m1)
        scalper(df)
    return out


if __name__ == "__main__":
    F = frames()
    test = sorted(F[1].day.unique())[8:]
    print("test days", len(test), test[0], test[-1])
    for tf, df in F.items():
        t = df[df.day.isin(test)]
        print(tf, len(t), "bars | ST flips", int(t.st_buy.sum() + t.st_sell.sum()),
              "| scalper calls", int(t.sc_call.sum()), "puts", int(t.sc_put.sum()),
              "| ema200 nan in test", int(t.ema200.isna().sum()))
