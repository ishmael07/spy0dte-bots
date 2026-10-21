"""Noise Area variants vs the published original — parameters fixed before looking (no grid search).

  python3 noise_lab.py                     90-session backtest (real-anchored option model), flat $100/trade + compounding
  python3 noise_lab.py real D1 D2 ...      paper days with real option quotes (run from the runtime dir, PYTHONPATH=repo)
Blocks: first 60 / last 30 sessions. Keep a variant only if it beats the original in both blocks and on the real days.
"""
import sys
import warnings
from collections import Counter

import noise_bot
from noise_bot import VARIANTS, run, stats

warnings.filterwarnings("ignore")


def report(T, days, blocks):
    for key in VARIANTS:
        tr, _ = run(T, days, 1e6, None, budget=100.0, variant=key)
        print(f"\n{key}")
        print("   all     ", stats(tr))
        for name, blk in blocks.items():
            print(f"   {name:8s}", stats([x for x in tr if x["day"] in set(blk)]))
        comp = "  ".join(f"${cap}→${run(T, days, cap, 0.5, variant=key)[1]:,.0f}" for cap in (100, 500))
        print(f"   50% compounding {comp}   exits {dict(Counter(x['why'] for x in tr))}")


if __name__ == "__main__":
    from data2 import build
    from sniperpp import Trader
    if sys.argv[1:2] == ["real"]:
        import paper
        from sniped1 import OneMin
        paper.merge_live_spy()
        D = build()
        D["days"] = [d for d in sys.argv[2:] if d in set(D["f5"].day)]
        T = Trader(D)
        paper.install_real(T.sim.models, T.sim.strikes, lambda d: T.f5[T.f5.day == d], 5, OneMin().m1, D["days"])
        report(T, D["days"], {})
    else:
        D = build()
        T, days = Trader(D), D["days"]
        report(T, days, {"first 60": days[:60], "last 30": days[60:]})
