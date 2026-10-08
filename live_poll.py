"""One live paper-trading poll: rerun every bot on the replay days + all live days so far, rebuild the page,
and print what changed since the previous poll (new entries / exits)."""
import glob
import json
import os
import subprocess
import sys

import pandas as pd

START = "2026-10-09"
DRY = ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
SNAP = "data/live/last_snapshot.json"

live_days = sorted({os.path.basename(f)[7:17] for f in glob.glob("data/live/spy_1m_*.json")} - set(DRY))
live_days = [d for d in live_days if d >= START]
if "--summary-only" not in sys.argv:
    cmd = [sys.executable, "-W", "ignore", "paper.py", *DRY, *live_days, "--start", START]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-W", "ignore", "export_paper.py"], check=True, stdout=subprocess.DEVNULL)

R = pd.read_pickle("data/paper_results.pkl")
names = {"sniper": "Sniper", "sniper_plus": "Sniper+", "sniper_pp": "Sniper++", "sniper_plus_scalp": "Sniper+ scalp",
         "sniper_pp_scalp": "Sniper++ scalp", "sniper_ppp": "Sniper+++", "sniper4": "Sniper4", "sniped1": "Sniped1",
         "combined": "CombinedBot", "daily": "Every day"}
snap, lines = {}, []
for key, name in names.items():
    tr = [x for x in R["results"].get(f"live|{key}|100|half", []) if not x.get("skipped") and "ei" in x]
    snap[key] = [f"{x['day']}|{x['side']}{x.get('strike')}|{x['ei']}|{x['xi']}|{round(x['pnl'], 2)}" for x in tr]
    bal100 = 100 + sum(x["pnl"] for x in tr)
    bal500 = 500 + sum(x["pnl"] for x in R["results"].get(f"live|{key}|500|half", []) if not x.get("skipped"))
    lines.append((bal100, f"{name:15s} $100→${bal100:,.2f}  $500→${bal500:,.2f}  trades {len(tr)}"))
old = json.load(open(SNAP)) if os.path.exists(SNAP) else {}
changes = []
for key, items in snap.items():
    new = [i for i in items if i not in old.get(key, [])]
    for i in new:
        day, contract, ei, xi, pnl = i.split("|")
        changes.append(f"{names[key]}: {contract} ({day}) P&L {float(pnl):+.2f}")
json.dump(snap, open(SNAP, "w"))
print(f"live days: {live_days or 'none yet'}")
print("CHANGES:" if changes else "no changes since last poll")
for c in changes:
    print("  " + c)
for _, l in sorted(lines, reverse=True):
    print("  " + l)
