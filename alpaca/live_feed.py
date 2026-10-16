"""Live view feed: today's SPY bars + every bot's trades with fresh option quotes and the Alpaca paper positions,
written to live.json and force-pushed as a single commit to the `live` branch (no history, no Pages build).
The page reads it through the GitHub contents API about once a minute."""
import json
import os
import subprocess
import time

from alpaca import orders


def build(H, today, path="data/live/live_today.json"):
    if not os.path.exists(path):
        return None
    L = json.load(open(path))
    if L.get("day") != today:
        L.update(trades=[], spy=[])
    marks = {}
    for t in L["trades"]:
        if not t.get("open"):
            continue
        sym = orders.occ(today, t["side"], t["strike"])
        if sym not in marks:
            marks[sym] = orders.quote(H, sym)
        bid, ask = marks[sym]
        if bid and ask:
            mid = round((bid + ask) / 2, 2)
            q1, q2 = t.get("qty") or 0, t.get("addQty") or 0
            cost = q1 * t["optIn"] * 100 + q2 * (t.get("addPx") or 0) * 100
            t.update(mark=mid, bid=bid, ask=ask, pnl=round(mid * 100 * (q1 + q2) - cost, 2),
                     pct=round(100 * (mid * 100 * (q1 + q2) - cost) / cost, 1) if cost else 0)
    code, pos = orders._req(H, "GET", "/v2/positions")
    L["positions"] = [dict(symbol=p["symbol"], qty=int(float(p["qty"])), avg=float(p["avg_entry_price"]),
                           price=float(p["current_price"]), pnl=float(p["unrealized_pl"]))
                      for p in (pos if code == 200 else []) if p.get("symbol", "").startswith("SPY")]
    L["account"] = orders.account_summary(H)
    L["pushed"] = int(time.time())
    return L


def push(root, L):
    """One orphan commit holding live.json, force-pushed to refs/heads/live (working tree and main untouched)."""
    git = ["git", "-C", root]
    blob = subprocess.run(git + ["hash-object", "-w", "--stdin"], input=json.dumps(L, separators=(",", ":")),
                          capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(git + ["mktree"], input=f"100644 blob {blob}\tlive.json\n", capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(git + ["commit-tree", tree, "-m", "live"], capture_output=True, text=True, check=True).stdout.strip()
    return subprocess.run(git + ["push", "-f", "-q", "origin", f"{commit}:refs/heads/live"], capture_output=True, text=True).returncode == 0
