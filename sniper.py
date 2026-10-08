"""Sniper research: every candidate entry from every strategy family, with ~25 features known at entry time
and the outcome of the option trade under several let-winners-run exit policies.

Writes data/events.csv. No feature looks past the signal bar's close.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

from data2 import build
from lab2 import FLAT_AT, SLIP, FEE, START
from lab3 import FAMILIES, Sim3, entries
from signals import _frame, _stamp, resample, supertrend, true_range, rma

warnings.filterwarnings("ignore")

# exit policies: SPY-ATR stop, target (None = let it run), trailing on 5m Supertrend flip, time cap (5m bars)
POLICIES = {
    "fixed3": dict(stop=1.0, tp=3.0, trail=False, cap=12),
    "run": dict(stop=1.0, tp=None, trail=True, cap=18),
    "run_wide": dict(stop=1.5, tp=None, trail=True, cap=24),
    "quick": dict(stop=0.75, tp=2.0, trail=False, cap=6),
}


def daily_context():
    s = pd.DataFrame(json.load(open("data/spy_daily.json")))
    s["day"] = s.begins_at.str[:10]
    for k in ("open", "high", "low", "close"):
        s[k] = s[f"{k}_price"].astype(float)
    s = s.set_index("day")
    st = s[["open", "high", "low", "close"]].reset_index(drop=True).copy()
    supertrend(st)
    s["dst"] = st.st_trend.values
    s["sma20"] = s.close.rolling(20).mean()
    s["sma50"] = s.close.rolling(50).mean()
    s["ret5"] = s.close / s.close.shift(5) - 1
    s["rv10"] = np.log(s.close / s.close.shift(1)).rolling(10).std() * np.sqrt(252)
    vix = json.load(open("data/vix_daily.json")) if os.path.exists("data/vix_daily.json") else []
    if vix:
        v = pd.DataFrame(vix)
        v["day"] = v.begins_at.str[:10]
        v = v.set_index("day")
        s["vix_open"] = v.open_value.astype(float).reindex(s.index)
        s["vix_close"] = v.close_value.astype(float).reindex(s.index)
    else:   # VIX is only used for research features, never by the bots
        s["vix_open"] = s["vix_close"] = np.nan
    prev = s.shift(1)   # everything about prior days is known before today's open
    ctx = pd.DataFrame(index=s.index)
    ctx["d_st"] = prev.dst
    ctx["d_above20"] = np.sign(prev.close - prev.sma20)
    ctx["d_above50"] = np.sign(prev.close - prev.sma50)
    ctx["d_ret5"] = prev.ret5
    ctx["rv10"] = prev.rv10
    ctx["vix_prev"] = prev.vix_close
    ctx["vix_gap"] = s.vix_open - prev.vix_close     # known at 9:30
    ctx["vix_5d"] = prev.vix_close - s.vix_close.shift(6)
    return ctx


def htf_trend_on5(f5, minutes):
    h = resample(f5, minutes)
    supertrend(h)
    src = pd.DataFrame({"avail": h.t_close, "st": h.st_trend.values}).sort_values("avail")
    return pd.merge_asof(pd.DataFrame({"done": f5.t_close}), src, left_on="done", right_on="avail")["st"].fillna(0).values


def simulate(sim, d, i, side, atr, pol, rule="near"):
    """One trade from signal bar i (5m clock). Returns (pnl, pct, mfe_pct, bars_held, reason)."""
    o, h, l, c, hm = sim.o, sim.h, sim.l, sim.c, sim.hm
    b = sim.base[d]
    n = len(sim.models[d].hm)
    k = 1 if side == "C" else -1
    stop = c[i] - k * pol["stop"] * atr
    tp = c[i] + k * pol["tp"] * atr if pol["tp"] else None
    ei = i + 1
    if ei >= b + n - 1:
        return None
    pk = sim.pick3(d, side, ei, o[ei], START, rule, c[i] + k * 3 * atr, stop, 6)
    if not pk:
        return None
    K, key, fill, qty = pk
    dm = sim.models[d]
    st5 = sim.f5.st_trend.values
    mfe = 0.0
    xi = xs = None
    reason = "EOD"
    for j in range(ei, b + n):
        mfe = max(mfe, dm.price(key, j - b, float(h[j] if side == "C" else l[j])) / fill - 1)
        if (l[j] <= stop) if side == "C" else (h[j] >= stop):
            xi, xs, reason = j, stop, "STOP"; break
        if tp is not None and ((h[j] >= tp) if side == "C" else (l[j] <= tp)):
            xi, xs, reason = j, tp, "TP"; break
        if pol["trail"] and j > ei and st5[j] == -k:
            xi, xs, reason = min(j + 1, b + n - 1), o[min(j + 1, b + n - 1)], "TRAIL"; break
        if j - ei + 1 >= pol["cap"]:
            xi, xs, reason = j, c[j], "TIME"; break
        if hm[j] >= FLAT_AT or j == b + n - 1:
            xi, xs, reason = j, c[j], "EOD"; break
    out = max(dm.price(key, xi - b, float(xs)) - SLIP, 0.0)
    cost = qty * (fill * 100 + FEE)
    pnl = qty * (out * 100 - FEE) - cost
    return round(pnl, 2), round(100 * pnl / cost, 1), round(100 * mfe, 1), xi - ei + 1, reason, K, round(fill, 2), qty


def main():
    D = build()
    sim = Sim3(D)
    ev = signal_rows(D, sim, D["days"], outcomes=True)
    ev.to_csv("data/events.csv", index=False)
    print(len(ev), "events on", ev.day.nunique(), "days")
    for name in POLICIES:
        print(f"{name:9s} mean pct {ev[f'{name}_pct'].mean():+6.1f}  win {100 * (ev[f'{name}_pnl'] > 0).mean():4.0f}%  "
              f"≥+100% {100 * (ev[f'{name}_pct'] >= 100).mean():4.1f}%  mfe≥100% {100 * (ev[f'{name}_mfe'] >= 100).mean():4.1f}%")


def signal_rows(D, sim, day_list, outcomes=True):
    """Every candidate entry on the given days with its entry-time features (and outcomes if asked)."""
    f5, f15 = sim.f5, D["f15"]
    ctx = daily_context()
    st15 = sim.st15_on5
    st30 = htf_trend_on5(f5, 30)
    st60 = htf_trend_on5(f5, 60)
    days = set(day_list)

    # which families fire on each 5m bar (5m and 15m signal timeframes), per side
    fire = {}
    for tf in (5, 15):
        f = f5 if tf == 5 else f15
        idx = np.arange(len(f5)) if tf == 5 else sim.pos15
        for fam in FAMILIES:
            calls, puts = entries(f, fam)
            calls, puts = np.nan_to_num(calls).astype(bool), np.nan_to_num(puts).astype(bool)
            for side, arr in (("C", calls), ("P", puts)):
                for j in np.where(arr & (idx >= 0))[0]:
                    fire.setdefault((int(idx[j]), side), []).append(f"{fam}@{tf}")

    rows = []
    atr5 = f5.atr.values
    for (i, side), fams in sorted(fire.items()):
        d = f5.day.iat[i]
        if d not in days or f5.hm.iat[i] >= 1500 or np.isnan(atr5[i]):
            continue
        k = 1 if side == "C" else -1
        r = f5.iloc[i]
        cx = ctx.loc[d]
        day_bars = f5[(f5.day == d) & (f5.index <= i)]
        feat = dict(
            day=d, i=i, side=side, hm=int(r.hm), mins=int((r.hm // 100 - 9) * 60 + r.hm % 100 - 30),
            fams="+".join(sorted(set(fams))), n_fams=len(set(fams)),
            a_st5=k * r.st_trend, a_st15=k * st15[i], a_st30=k * st30[i], a_st60=k * st60[i],
            a_dst=k * cx.d_st, a_d20=k * cx.d_above20, a_d50=k * cx.d_above50, a_ret5=k * cx.d_ret5,
            a_vwap=k * (r.close - r.vwap) / atr5[i], a_ema=k * np.sign(r.ema_f - r.ema_s), a_ema200=k * np.sign(r.close - r.ema200),
            a_gap=k * r.gap / r.pdc * 100, a_open=k * (r.close - r.day_open) / atr5[i],
            a_rsi=k * (r.rsi - 50), relvol=r.relvol, atr_pct=atr5[i] / r.close * 100,
            range_atr=(day_bars.high.max() - day_bars.low.min()) / atr5[i],
            orb_atr=(r.orh15 - r.orl15) / atr5[i] if r.orh15 == r.orh15 else np.nan,
            iv=getattr(sim.models.get(d), 'atm_iv', np.nan), iv_rv=getattr(sim.models.get(d), 'atm_iv', np.nan) / cx.rv10 if cx.rv10 else np.nan,
            vix=cx.vix_prev, vix_gap=cx.vix_gap, a_vix=-k * cx.vix_gap, vix_5d=cx.vix_5d,
            dow=pd.Timestamp(d).dayofweek,
        )
        if not outcomes:
            rows.append(feat)
            continue
        for name, pol in POLICIES.items():
            res = simulate(sim, d, i, side, atr5[i], pol)
            if res is None:
                break
            pnl, pct, mfe, held, reason, K, fill, qty = res
            feat.update({f"{name}_pnl": pnl, f"{name}_pct": pct, f"{name}_mfe": mfe, f"{name}_held": held, f"{name}_why": reason})
            if name == "run":
                feat.update(strike=K, fill=fill, qty=qty)
        else:
            rows.append(feat)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
