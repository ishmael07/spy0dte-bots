"""Mirror every bot's trades into the Alpaca PAPER account as real paper orders.

Each bot × account size ($100 / $500 at 50% sizing) is a separate ledger inside one paper account, tagged with a
deterministic client_order_id, so restarts never double-submit. State: data/live/orders.json (committed).
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

TRADE = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
EDGE = 0.03          # marketable limit: buy at ask + 3¢, sell at bid − 3¢ (floor $0.01)
STATE = "data/live/orders.json"
SIZE = "half"
CAPS = (100, 500)
FRESH_MIN = 5        # never place an entry that is more than 5 minutes old (e.g. after a restart)
FLATTEN_HM = 1550    # close anything left in today's 0DTE at 3:50 pm ET


def occ(day, side, strike):
    return f"SPY{day[2:4]}{day[5:7]}{day[8:10]}{side}{int(round(strike * 1000)):08d}"


def _req(H, method, path, **kw):
    r = requests.request(method, TRADE + path, headers=H, timeout=20, **kw)
    return r.status_code, (r.json() if r.content else {})


def quote(H, symbol):
    r = requests.get(f"{DATA}/v1beta1/options/snapshots", headers=H, params=dict(symbols=symbol, feed="indicative"), timeout=20)
    q = ((r.json().get("snapshots") or {}).get(symbol) or {}).get("latestQuote") or {} if r.ok else {}
    return q.get("bp") or 0.0, q.get("ap") or 0.0


def submit(H, symbol, qty, side, cid):
    """Marketable limit order (Alpaca rejects market orders on contracts with no bid)."""
    bid, ask = quote(H, symbol)
    if side == "buy":
        if not ask:
            return None, "no_quote", {}
        limit = round(ask + EDGE, 2)
    else:
        limit = max(round(bid - EDGE, 2), 0.01)
    code, j = _req(H, "POST", "/v2/orders", json=dict(symbol=symbol, qty=str(int(qty)), side=side, type="limit",
                                                       limit_price=f"{limit:.2f}", time_in_force="day", client_order_id=cid))
    if code == 422 and "client_order_id" in json.dumps(j).lower():   # already submitted in an earlier run
        code, j = _req(H, "GET", "/v2/orders:by_client_order_id", params=dict(client_order_id=cid))
    return j.get("id"), j.get("status"), j


def refresh(H, rec):
    """Pull fill info for any submitted-but-unfilled legs."""
    for leg in ("buy", "add", "sell"):
        o = rec.get(leg)
        if o and o.get("id") and o.get("status") not in ("filled", "canceled", "rejected", "expired"):
            code, j = _req(H, "GET", f"/v2/orders/{o['id']}")
            if code == 200:
                o.update(status=j.get("status"), fill=float(j["filled_avg_price"]) if j.get("filled_avg_price") else None,
                         filled_qty=int(float(j.get("filled_qty") or 0)))


def sync(H, today, now_et):
    """Compare the engine's trades for today with submitted orders; place what's missing."""
    if not os.path.exists("data/paper_results.pkl"):
        return []
    R = pd.read_pickle("data/paper_results.pkl")
    f5, m1 = R["f5"], R["m1"]
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    last5 = f5.index[f5.day == today].max() if (f5.day == today).any() else None
    last1 = m1.index[m1.day == today].max() if (m1.day == today).any() else None
    nowmin = now_et.hour * 60 + now_et.minute
    log = []
    for k, trades in R["results"].items():
        phase, bot, cap, size = k.split("|")
        if phase != "live" or size != SIZE or int(cap) not in CAPS:
            continue
        for t in trades:
            if t.get("skipped") or t["day"] != today or "ei" not in t:
                continue
            one = bot == "sniped1" or t.get("bot") == "Sniped1"
            fr, last = (m1, last1) if one else (f5, last5)
            t_in = fr.t[t["ei"]].tz_convert(now_et.tzinfo)
            still_open = t["xi"] >= last and now_et.hour * 100 + now_et.minute < 1545
            sym = occ(today, t["side"], t["strike"])
            tag = f"{bot}-{cap}-{today[2:].replace('-', '')}-{t['side']}{t['strike']}-{t['ei']}"
            rec = state.setdefault(tag, dict(bot=bot, cap=int(cap), symbol=sym, day=today, sim_in=t["optIn"], qty=t["qty"]))
            refresh(H, rec)
            age = nowmin - (t_in.hour * 60 + t_in.minute)
            if "buy" not in rec:
                if age > FRESH_MIN:
                    rec["buy"] = dict(status="missed", note=f"entry {age} min old when first seen")
                else:
                    oid, st, _ = submit(H, sym, t["qty"], "buy", tag + "-b")
                    rec["buy"] = dict(id=oid, status=st)
                    log.append(f"BUY  {bot} ${cap}: {t['qty']}× {sym}")
            if t.get("addQty") and "add" not in rec and rec["buy"].get("id"):
                oid, st, _ = submit(H, sym, t["addQty"], "buy", tag + "-a")
                rec["add"] = dict(id=oid, status=st)
                log.append(f"ADD  {bot} ${cap}: {t['addQty']}× {sym}")
            if not still_open and "sell" not in rec and rec["buy"].get("id"):
                refresh(H, rec)
                held = sum(rec[l].get("filled_qty") or 0 for l in ("buy", "add") if l in rec)
                if held:
                    oid, st, _ = submit(H, sym, held, "sell", tag + "-s")
                    rec["sell"] = dict(id=oid, status=st)
                    rec["sim_out"] = t["optOut"]
                    log.append(f"SELL {bot} ${cap}: {held}× {sym} ({t.get('why', '')})")
            refresh(H, rec)
    if now_et.hour * 100 + now_et.minute >= FLATTEN_HM:
        log += flatten(H, today, state)
    for rec in state.values():
        legs = [rec.get(l) for l in ("buy", "add", "sell")]
        b = [(o.get("fill"), o.get("filled_qty")) for o in legs[:2] if o and o.get("fill")]
        s = legs[2]
        if b and s and s.get("fill"):
            cost = sum(p * q * 100 for p, q in b)
            rec["pnl"] = round(s["fill"] * (s.get("filled_qty") or 0) * 100 - cost, 2)
    json.dump(state, open(STATE, "w"), indent=0)
    return log


def flatten(H, today, state):
    """Safety net: close every remaining position in today's 0DTE contracts and mark open ledgers closed."""
    out = []
    code, positions = _req(H, "GET", "/v2/positions")
    prefix = f"SPY{today[2:4]}{today[5:7]}{today[8:10]}"
    for p in positions if code == 200 else []:
        if p.get("symbol", "").startswith(prefix):
            _req(H, "DELETE", f"/v2/positions/{p['symbol']}")
            out.append(f"FLATTEN {p['symbol']} {p.get('qty')}")
    for rec in state.values():
        if rec.get("day") == today and rec.get("buy", {}).get("id") and "sell" not in rec:
            rec["sell"] = dict(status="flattened", note="closed by 3:50 pm safety net")
    return out
