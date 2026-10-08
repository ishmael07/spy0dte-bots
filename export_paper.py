"""Build the paper-trading page from data/paper_results.pkl (run paper.py first)."""
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from paper import BOTS, SIZING

ET = ZoneInfo("America/New_York")
URL_NOTE = "Paper money only · fills at real Robinhood option minute prices where available (+$0.02 slippage/side, $0.03/contract fees)"


def ts(t):
    return int(t.timestamp() + t.tz_convert(ET).utcoffset().total_seconds())


def main():
    R = pd.read_pickle("data/paper_results.pkl")
    res, f5, m1, source, start = R["results"], R["f5"], R["m1"], R["source"], R["start"]
    days = R["days"]
    now_et = pd.Timestamp.now(tz=ET)
    last_day = days[-1] if days else None
    last_bar = m1[m1.day == last_day].t.max() if last_day else None
    last_bar_et = last_bar.tz_convert(ET) if last_bar is not None else None
    market_open = bool(start and last_day and last_day >= start and last_day == now_et.strftime("%Y-%m-%d") and now_et.hour * 100 + now_et.minute < 1600
                       and last_bar_et is not None and last_bar_et.hour * 100 + last_bar_et.minute < 1559)
    last_end = {d: ts(m1[m1.day == d].t.max() + pd.Timedelta(minutes=1)) for d in days}
    phase_days = {"dry": [d for d in days if not start or d < start], "live": [d for d in days if start and d >= start]}
    phases = (["dry", "live"] if phase_days["live"] else ["live", "dry"]) if start else ["dry"]   # last = default view
    labels = {"dry": "Replay Mon–Thu (Oct 5–8)", "live": f"Live paper (from {pd.Timestamp(start).strftime('%a %b %-d')})" if start else ""}
    runs, chart, chart1 = {}, {}, {}
    order_state = json.load(open("data/live/orders.json")) if os.path.exists("data/live/orders.json") else {}
    for k, tr in res.items():
        phase, key, cap, size = k.split("|")
        out = []
        for x in tr:
            if x.get("skipped") or "ei" not in x:
                continue
            x = dict(x)
            one = key == "sniped1" or x.get("bot") == "Sniped1"
            fr = m1 if one else f5
            step = 1 if one else 5
            sig = x.get("si", x.get("i"))
            tag = f"{key}-{cap}-{x['day'][2:].replace('-', '')}-{x['side']}{x.get('strike')}-{x['ei']}"
            o = order_state.get(tag)
            if o:
                x["alpaca"] = dict(buy=(o.get("buy") or {}).get("fill"), add=(o.get("add") or {}).get("fill"),
                                   sell=(o.get("sell") or {}).get("fill"), pnl=o.get("pnl"),
                                   status=(o.get("sell") or o.get("buy") or {}).get("status"))
            x["tIn"], x["tOut"] = ts(fr.t[x["ei"]]), ts(fr.t[x["xi"]] + pd.Timedelta(minutes=step))
            x["path"] = [[ts(fr.t[x["ei"] + j]), p] for j, p in enumerate(x["path"])]
            x["vol"] = round(float(fr.relvol.iat[sig]), 2) if "relvol" in fr else 0
            x["tf"] = step
            x["fills"] = source.get(x["day"], "model")
            x["reason"] = x.pop("why", x.get("reason"))
            x["pnl"], x["pct"] = round(x["pnl"], 2), round(x["pct"], 1)
            x["score"] = int(x.get("score", 0))
            if market_open and x["day"] == last_day and x["tOut"] >= last_end[last_day] - 300 and x["reason"] in ("3:45 EXIT", "EOD", "TIME LIMIT", "2H CAP"):
                x["open"], x["reason"] = True, "OPEN"
            if one and key != "sniped1":
                x["kind"] = "pullback1m"
            elif key == "sniped1":
                x["kind"] = "pullback1m"
            for q in ("i", "si", "ei", "xi", "t_in", "t_out"):
                x.pop(q, None)
            (chart1 if one else chart).setdefault(x["day"], None)
            out.append(x)
        out.sort(key=lambda q: q["tOut"])
        cash = int(cap)
        for x in out:
            cash += x["pnl"]
            x["eq"] = round(cash, 2)
        pdays = phase_days[phase]
        eq = [int(cap)] + [x["eq"] for x in out]
        peak, mdd = int(cap), 0.0
        for e in eq:
            peak = max(peak, e)
            mdd = max(mdd, (peak - e) / peak)
        wins = [x for x in out if x["pnl"] > 0]
        losses = [x for x in out if x["pnl"] <= 0]
        runs[f"{key}|{phase}|{cap}|{size}"] = dict(trades=out, stats=dict(
            start=int(cap), final=round(eq[-1], 2), ret=round(100 * (eq[-1] / int(cap) - 1), 1), trades=len(out), skipped=0,
            wins=len(wins), winRate=round(100 * len(wins) / len(out), 1) if out else 0,
            avgWin=round(float(np.mean([x["pct"] for x in wins])), 1) if wins else 0,
            avgLoss=round(float(np.mean([x["pct"] for x in losses])), 1) if losses else 0,
            worst=min((x["pct"] for x in out), default=0), best=max((x["pct"] for x in out), default=0),
            mdd=round(100 * mdd, 1), perWeek=round(len(out) / max(len(pdays), 1) * 5, 1), odds=None,
            week=dict(x2=0, x10=0, up=0), realDays=sum(1 for d in pdays if source.get(d) == "real"),
            modelDays=sum(1 for d in pdays if source.get(d) != "real"), tradeDays=len({x["day"] for x in out}), sessions=len(pdays),
            bestUsd=max((x["pnl"] for x in out), default=0), worstUsd=min((x["pnl"] for x in out), default=0),
            todayPnl=round(sum(x["pnl"] for x in out if pdays and x["day"] == pdays[-1]), 2), open=sum(1 for x in out if x.get("open")),
            alpacaPnl=round(sum(r["pnl"] for r in order_state.values() if r.get("bot") == key and r.get("cap") == int(cap) and r.get("pnl") is not None), 2)
            if phase == "live" and size == "half" else None))
    for store, fr in ((chart, f5), (chart1, m1)):
        for d in list(store):
            g = fr[fr.day == d]
            st = g.st_trend.values
            col = lambda n: [None if v != v else round(float(v), 2) for v in g[n].values]
            store[d] = dict(t=[ts(t) for t in g.t], o=col("open"), h=col("high"), l=col("low"), c=col("close"),
                            stUp=[u if s == 1 else None for u, s in zip(col("st_up"), st)],
                            stDn=[v if s == -1 else None for v, s in zip(col("st_dn"), st)], vwap=col("vwap"))
    table = []
    for key, (name, _, _) in BOTS.items():
        row = dict(key=key, name=name)
        for slot, phase in (("a", "dry"), ("b", "live")):
            r = runs.get(f"{key}|{phase}|100|half")
            t = r["trades"] if r else []
            row[slot] = [len(t), round(sum(x["pnl"] for x in t), 2), round(100 * np.mean([x["pnl"] > 0 for x in t])) if t else 0]
        allt = [x for ph in ("dry", "live") for x in (runs.get(f"{key}|{ph}|100|half") or {"trades": []})["trades"]]
        row["worst"] = min((x["pnl"] for x in allt), default=0)
        table.append(row)
    # make sure every key/phase/cap/size exists so the page never hits a missing run
    for key in BOTS:
        for phase in phases:
            for cap in (100, 500):
                for size in SIZING:
                    runs.setdefault(f"{key}|{phase}|{cap}|{size}", dict(trades=[], stats=dict(
                        start=cap, final=cap, ret=0, trades=0, skipped=0, wins=0, winRate=0, avgWin=0, avgLoss=0, worst=0, best=0,
                        mdd=0, perWeek=0, odds=None, week=dict(x2=0, x10=0, up=0), realDays=0, modelDays=0, tradeDays=0,
                        sessions=len(phase_days[phase]), bestUsd=0, worstUsd=0, todayPnl=0, open=0)))
    # replay (Mon–Thu, Robinhood data) is exported once from the main project and merged into the Alpaca page
    if phase_days["dry"]:
        dry_days = phase_days["dry"]
        json.dump(dict(runs={k: v for k, v in runs.items() if "|dry|" in k}, days=dry_days, label=labels["dry"],
                       chart={d: chart[d] for d in dry_days if d in chart}, chart1={d: chart1[d] for d in dry_days if d in chart1},
                       table={r["key"]: r["a"] for r in table}, worst={r["key"]: r["worst"] for r in table}),
                  open("data/replay_page.json", "w"), separators=(",", ":"))
    elif os.path.exists("data/replay_page.json"):
        rp = json.load(open("data/replay_page.json"))
        runs.update(rp["runs"])
        chart.update(rp["chart"]); chart1.update(rp["chart1"])
        phase_days["dry"] = rp["days"]; labels["dry"] = rp["label"]
        phases = ["dry"] + [p for p in phases if p != "dry"]
        for r in table:
            r["a"] = rp["table"].get(r["key"], r["a"])
            r["worst"] = min(r["worst"], rp["worst"].get(r["key"], 0))
        days = rp["days"] + [d for d in days if d not in rp["days"]]
    data = dict(paper=True, marketOpen=market_open, liveStart=pd.Timestamp(start).strftime("%a %b %-d") if start else "",
                updated=(last_bar_et + pd.Timedelta(minutes=1)).strftime("%a %-I:%M %p") if last_bar_et is not None and start and last_day >= start else "", days=days, days1=days, chart=chart, chart1=chart1, periods=phases, capitals=[100, 500],
                sizing=list(SIZING), names={k: v[0] for k, v in BOTS.items()}, runs=runs, table=table,
                tf={k: (1 if k in ("sniped1", "combined") else 5) for k in BOTS}, periodsBy={k: phases for k in BOTS},
                periodLabels=labels, phaseDays=phase_days, designStart=start or days[-1],
                tableHead=["Replay Mon–Thu · $100 at 50%", "Live paper · $100 at 50%"], footer=URL_NOTE,
                validation=json.load(open("data/validation.json")))
    page = open("web/template4.html").read().replace("<title>SPY 0DTE Sniper</title>", "<title>SPY 0DTE Paper Week</title>")
    page = page.replace("<h1>SPY 0DTE Sniper</h1>", "<h1>SPY 0DTE Paper Week</h1>").replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open("web/paper_artifact.html", "w").write(page)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>body{margin:0}</style></head><body>\n')
    open("web/paper.html", "w").write(head + page + "\n</body></html>")
    print("phases", phases, {p: phase_days[p] for p in phases}, f"page {len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
