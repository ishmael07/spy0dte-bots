"""Time bot: Sniper4's entries (tiers, sizing, pyramid) with no fixed holding clock.

Instead of "out after 2 hours", the trade decides when it's done:
  LOCK PROFIT    once the contract is up ≥ arm, exit if it gives back `giveback` of its peak gain
  MOMENTUM FADE  while green, a 5m close back through the 8 EMA (momentum gone) → take it
  STALLED        no new high in the contract for n bars (fewer bars after 1 pm, when theta speeds up) and gain < x
  THETA          late in the day, if the trade still isn't up ≥ y, get out before decay eats the premium
Hard rules kept: SPY-based stop, breakeven after +1.5 ATR, trend-flip exit, flat at 3:45 (never held to expiry).

v2 (Oct 16): B-tier trades (against the 50-day trend) ride with the same stop and adaptive exits as A-tier instead of
a 0.75-ATR scalp stop, which shook every one of them out within 5–10 minutes in the first paper week; profit lock
arms at +200% (was +100%) and gives back up to 40% — cheap far-OTM contracts swing ±50% as noise, so a +100% lock
sold the Oct 16 766P at $0.49 on its way to $2.48. time_lab.py, flat $100 units, net minus best trade:
pick 768 (v1 909) · test 985 (v1 588) · real-quote paper days +318 (v1 −98). An SPY-ATR trailing stop was tested
and rejected (it cut the big winners: pick ~600–790, test ~600–780).
"""
import itertools

import numpy as np

from sniper4 import SNIPER4, EXITS, account

LOCK = [None, (0.3, 0.4), (0.5, 0.4), (0.5, 0.5), (1.0, 0.35), (1.0, 0.5)]
FADE = [None, 0.0, 0.25]
STALL = [None, (6, 3, 0.0), (8, 4, 0.1), (4, 2, 0.0)]
LATE = [None, (1400, 0.25), (1330, 0.0)]
CAP = [None, 24]
SCALP_B = ["scalp", "scalp_open"]


def make_cfg(lock, fade, stall, late, cap, b_exit):
    ride = dict(EXITS["ride"], cap=cap, adaptive=dict(lock=lock, fade=fade, stall=stall, late=late))
    scalp_open = dict(EXITS["scalp"], cap=None, adaptive=dict(lock=(0.3, 0.4), fade=0.0, stall=(3, 2, 0.0), late=None))
    exits = dict(EXITS, time_ride=ride, time_ride_b=ride, scalp_open=scalp_open)
    return dict(SNIPER4, exit={"A": "time_ride", "B": b_exit, "C": b_exit}), exits


def run_cfg(T, ev, days, cfg_exits, capital=100, frac=None, **kw):
    import sniper4
    cfg, exits = cfg_exits
    saved = sniper4.EXITS
    sniper4.EXITS = exits
    try:
        return account(T, ev, set(days), capital, frac, cfg=cfg, **kw)
    finally:
        sniper4.EXITS = saved


def stats(tr):
    p = np.array([x["pnl"] for x in tr]) if tr else np.zeros(0)
    if not len(p):
        return dict(n=0, net=0.0, win=0, mdd=0.0)
    eq = np.concatenate([[0], np.cumsum(p)])
    return dict(n=len(p), net=round(float(p.sum()), 0), win=round(100 * float((p > 0).mean())),
                mdd=round(float(np.max(np.maximum.accumulate(eq) - eq)), 0))


GRID = list(itertools.product(LOCK, FADE, STALL, LATE, CAP, SCALP_B))


# v1 (picked on the first 60 of 90 days): lock +100% / give back 35%, B-tier scalps — ((1.0, 0.35), ..., "scalp_open")
# v2: lock +200% / give back 40% · B-tier rides like A · late-day theta check at 2 pm · no fixed holding clock anywhere.
TIME = ((2.0, 0.4), None, None, (1400, 0.25), None, "time_ride_b")


def run(T, ev, days, capital, frac, detail=False, acct_rule="none", sessions=None):
    return run_cfg(T, ev, days, make_cfg(*TIME), capital, frac, detail=detail, acct_rule=acct_rule, sessions=sessions)
