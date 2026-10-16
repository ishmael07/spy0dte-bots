"""Time bot: Sniper4's entries (tiers, sizing, pyramid) with no fixed holding clock.

Instead of "out after 2 hours", the trade decides when it's done:
  LOCK PROFIT    once the contract is up ≥ arm, exit if it gives back `giveback` of its peak gain
  MOMENTUM FADE  while green, a 5m close back through the 8 EMA (momentum gone) → take it
  STALLED        no new high in the contract for n bars (fewer bars after 1 pm, when theta speeds up) and gain < x
  THETA          late in the day, if the trade still isn't up ≥ y, get out before decay eats the premium
Hard rules kept: SPY-based stop, breakeven after +1.5 ATR, trend-flip exit, flat at 3:45 (never held to expiry).
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
    exits = dict(EXITS, time_ride=ride, scalp_open=scalp_open)
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


# Picked on the first 60 of 90 days only (best result with the single best trade removed); last 30 days untouched.
# Lock profit at +100% / give back 35% · late-day theta check at 2 pm · no fixed holding clock anywhere.
TIME = ((1.0, 0.35), None, None, (1400, 0.25), None, "scalp_open")


def run(T, ev, days, capital, frac, detail=False, acct_rule="none", sessions=None):
    return run_cfg(T, ev, days, make_cfg(*TIME), capital, frac, detail=detail, acct_rule=acct_rule, sessions=sessions)
