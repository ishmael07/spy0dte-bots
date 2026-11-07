"""Build the Apex tab (web/apex.html) from the runtime data. Run from alpaca/runtime (like export_paper.py).
Static part: Apex v2's record on the recorded option prices since Oct 9 (paper.py 'live|apex|{cap}|half') and the
real-fill ledger; live part: the page polls live.json (apex state + today's SPY bars) every ~70 s."""
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_URL = "https://api.github.com/repos/ishmael07/spy0dte-bots/contents/live.json?ref=live"


def main():
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    R = pd.read_pickle("data/paper_results.pkl")
    m1 = R["m1"]
    sys.path.insert(0, ROOT)
    from apex import core, sim as apex_sim
    live_days = [d for d in R["days"] if d >= "2026-10-09"]
    sim = {}
    for mode in core.MODES:
        sim[mode] = {}
        for cap in (100, 500):
            sim[mode][cap] = convert(apex_sim.run(m1, live_days, cap, mode=mode), m1)
    ledger = {}
    led_p = "data/live/apex_ledger.json"
    if os.path.exists(led_p):
        for k, v in json.load(open(led_p)).items():
            ledger[k if "|" in k else f"aggressive|{k}"] = dict(cash=v["cash"])
    _rest(sim, ledger, today)


def convert(trades, m1):
    out = []
    for x in trades:
        if "ei" not in x:
            continue
        out.append(dict(day=x["day"], book=x.get("book", "ride"), side=x["side"], strike=x["strike"], qty=x["qty"], optIn=x["optIn"],
                        optOut=x["optOut"], pct=x["pct"], pnl=x["pnl"], eq=x["eq"], why=x.get("why"),
                        tIn=int(m1.t[x["ei"]].timestamp()), tOut=int(m1.t[x["xi"]].timestamp()) + 60))
    out.sort(key=lambda t: t["tOut"])
    return out


def _rest(sim, ledger, today):
    ap = f"data/live/apex_{today}.json"
    days = sorted(f[5:15] for f in os.listdir("data/live") if re.match(r"apex_\d{4}-\d\d-\d\d\.json$", f))
    if not os.path.exists(ap) and days:
        ap = f"data/live/apex_{days[-1]}.json"
    apex = json.load(open(ap)) if os.path.exists(ap) else None
    lt = json.load(open("data/live/live_today.json")) if os.path.exists("data/live/live_today.json") else {}
    from apex import core
    data = dict(liveUrl=LIVE_URL, sim=sim, ledger=ledger, apex=apex, spy=lt.get("spy") or [], day=(apex or {}).get("day", today),
                marketOpen=bool(lt.get("marketOpen")), modes={m: core.MODES[m] for m in core.MODES}, primary=core.PRIMARY)
    tpl = open(os.path.join(ROOT, "web", "apex_template.html")).read()
    main_t = open(os.path.join(ROOT, "web", "template4.html")).read()
    tokens = main_t[main_t.index("<style>") + 7: main_t.index("* { box-sizing")]
    page = tpl.replace("/*TOKENS*/", tokens).replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n')
    os.makedirs("web", exist_ok=True)
    open("web/apex.html", "w").write(head + page + "\n</body></html>")
    print(f"apex page {len(page) / 1e3:.0f} KB · " + ", ".join(f"{m} {len(v[500])}" for m, v in sim.items()))


if __name__ == "__main__":
    main()
