"""Apex live engine: websocket streams in, paper orders out, decisions the instant a bar closes.

Streams (Alpaca, free plan):
  stocks  wss://stream.data.alpaca.markets/v2/iex          SPY trades (tick price) + 1-minute bars
  options wss://stream.data.alpaca.markets/v1beta1/indicative  live bid/ask for the strikes around the money (msgpack)
  news    wss://stream.data.alpaca.markets/v1beta1/news    Benzinga headlines -> Claude news desk (ai.py)

Rules: core.py (validated in ~/spy0dte-backtest/lab5). Orders: two ledgers in the shared Alpaca paper account,
$100 and $500, each risking 50% of its equity per trade (premium = max loss), client_order_id
apex-{cap}-{yymmdd}-{b|s} so a restart can never double-buy. Restart-safe: on start it replays today's bars
through the rules (deterministic) and resumes from the persisted ledger.

The AI desk is advisory: its reads, the pre-market brief, and the paper-only "shock" ledger are logged and scored
forward. APEX_AI_VETO=1 lets a fresh, large, opposite headline block a new entry (off by default until scored).

  python3 apex/stream.py --env ~/dawn-raid/.env [--max-minutes 340] [--dry]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpaca import feed, orders  # noqa: E402
from apex import core  # noqa: E402
from apex.ai import Desk  # noqa: E402

ET = ZoneInfo("America/New_York")
LIVE = os.path.join(ROOT, "alpaca", "runtime", "data", "live")
DATA = "https://data.alpaca.markets"
WS_STOCK = "wss://stream.data.alpaca.markets/v2/iex"
WS_OPT = "wss://stream.data.alpaca.markets/v1beta1/indicative"
WS_NEWS = "wss://stream.data.alpaca.markets/v1beta1/news"
CAPS = (100, 500)
FRAC = 0.5
EDGE = 0.03
SHOCK_MOVE = 0.0015          # shock ledger: SPY must confirm the headline by 0.15% within 2 minutes
SHOCK_HOLD = 30              # minutes


def now_et():
    return datetime.now(ET)


def minute_index(t):
    return (t.hour * 60 + t.minute) - (9 * 60 + 30)


class Apex:
    def __init__(self, H, dry=False):
        self.H, self.dry = H, dry
        self.today = now_et().strftime("%Y-%m-%d")
        self.path = os.path.join(LIVE, f"apex_{self.today}.json")
        self.ledger_path = os.path.join(LIVE, "apex_ledger.json")
        self.S = json.load(open(self.path)) if os.path.exists(self.path) else dict(day=self.today, events=[], ai=[], trades=[], shock=[])
        self.L = json.load(open(self.ledger_path)) if os.path.exists(self.ledger_path) else {str(c): dict(cash=float(c), trades=[]) for c in CAPS}
        self.desk = Desk(self.log_ai)
        self.day = None
        self.last_i = -1
        self.tick = None             # last SPY trade price
        self.tick_t = 0.0
        self.quotes = {}             # occ symbol -> (bid, ask, ts)
        self.opt_ws = None
        self.opt_subs = set()
        self.reads = []              # recent AI reads (for veto / shock)
        self.dirty = True

    # ------------------------------------------------------------------ persistence
    def event(self, msg, **kw):
        rec = dict(t=now_et().strftime("%H:%M:%S"), msg=msg, **kw)
        self.S["events"].append(rec)
        self.S["events"] = self.S["events"][-300:]
        print(rec["t"], msg, flush=True)
        self.dirty = True

    def log_ai(self, rec):
        rec = dict(rec, t=now_et().strftime("%H:%M:%S"))
        self.S["ai"].append(rec)
        self.S["ai"] = self.S["ai"][-200:]
        print(rec["t"], "AI", json.dumps(rec)[:300], flush=True)
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        st = dict(self.S)
        if self.day:
            i = max(self.last_i, 0)
            st["levels"] = self.day.levels(min(i, 389)) if self.day.closes else None
            st["spy"] = self.tick or (self.day.closes[-1] if self.day.closes else None)
            st["bar"] = self.last_i
            st["position"] = self.day.position
        st["ledger"] = {c: dict(cash=round(v["cash"], 2), n=len(v["trades"])) for c, v in self.L.items()}
        st["ai_on"] = self.desk.on
        st["updated"] = int(time.time())
        for p, obj in ((self.path, st), (self.ledger_path, self.L)):
            tmp = p + ".tmp"
            json.dump(obj, open(tmp, "w"), separators=(",", ":"))
            os.replace(tmp, p)
        self.dirty = False

    # ------------------------------------------------------------------ setup
    def history(self):
        """sigma profile from the last 14 full sessions (SIP 1m, history is >15 min old) + prior close."""
        start = (now_et() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00Z")
        end = (now_et().replace(hour=0, minute=0, second=0, microsecond=0)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bars = feed.bars(self.H, "SPY", "1Min", start, end, feed="sip")
        sess = {}
        for b in bars:
            t = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            i = minute_index(t)
            if 0 <= i < 390:
                d = t.strftime("%Y-%m-%d")
                sess.setdefault(d, {})[i] = b
        days = [d for d in sorted(sess) if len(sess[d]) >= 300 and d < self.today]
        hist = []
        for d in days:
            m = sess[d]
            o = m[min(m)]["o"]
            hist.append((o, [m[i]["c"] if i in m else None for i in range(390)]))
        prev = sess[days[-1]]
        prev_close = prev[max(prev)]["c"]
        return core.sigma_profile(hist), prev_close

    def today_bars(self):
        start = now_et().replace(hour=9, minute=30, second=0, microsecond=0).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = {}
        for b in feed.bars(self.H, "SPY", "1Min", start, feed="iex"):
            t = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            i = minute_index(t)
            if 0 <= i < 390:
                out[i] = b
        return out

    def start_day(self, first_open):
        sig, pc = self.history()
        self.day = core.Day(open=first_open, prev_close=pc, sigma=sig)
        self.event(f"day ready: open {first_open:.2f}, prev close {pc:.2f}, 14-day noise at 10:00 ±{sig[30] * 100:.2f}%")

    # ------------------------------------------------------------------ bars -> rules
    def feed_bar(self, i, b, replay=False):
        if self.day is None:
            if i != 0:
                # joined late: rebuild from the 9:30 bar
                return self.catch_up()
            self.start_day(b["o"])
        if i <= self.last_i:
            return
        if i > self.last_i + 1:          # missed bars (disconnect): fill from REST first
            have = self.today_bars()
            for k in range(self.last_i + 1, i):
                if k in have:
                    self._bar(k, have[k], replay=True)
        self._bar(i, b, replay)

    def _bar(self, i, b, replay):
        self.last_i = i
        sig = self.day.on_bar(i, b["o"], b["h"], b["l"], b["c"], b.get("v", 0))
        self.dirty = True
        if not sig:
            return
        kind, x = sig
        if kind == "enter":
            self.day.traded = True
            self.day.position = dict(side=x, i=i, spy=b["c"], t=now_et().strftime("%H:%M:%S"))
            lv = self.day.levels(i)
            self.event(f"SIGNAL {'CALL' if x == 'C' else 'PUT'} at bar {i} close {b['c']:.2f} (band {lv['ub']}/{lv['lb']}, vwap {lv['vwap']})",
                       kind="enter", replay=replay)
            if not replay or self._fresh(i):
                self.enter(x, b["c"], i)
        elif kind == "exit":
            pos = self.day.position
            self.day.position = None
            self.event(f"EXIT {x} at bar {i} close {b['c']:.2f}", kind="exit", replay=replay)
            self.exit(pos, x)

    def _fresh(self, i):
        return minute_index(now_et()) - i <= 3

    def catch_up(self):
        bars = self.today_bars()
        if 0 not in bars:
            self.event("waiting for the 9:30 bar")
            return
        self.start_day(bars[0]["o"])
        for i in sorted(bars):
            if t_closed(i):
                self._bar(i, bars[i], replay=True)
        self.event(f"replayed {self.last_i + 1} bars; position {self.day.position}")

    # ------------------------------------------------------------------ execution
    def chain(self, side, spot):
        """strikes from ATM outward (OTM direction) with live quotes"""
        base = round(spot)
        ks = [base + k if side == "C" else base - k for k in range(0, 13)]
        syms = [orders.occ(self.today, side, k) for k in ks]
        q = {}
        for s in syms:
            v = self.quotes.get(s)
            if v and time.time() - v[2] < 5 and v[1] > 0:
                q[s] = (v[0], v[1])
        missing = [s for s in syms if s not in q]
        if missing:
            r = requests.get(f"{DATA}/v1beta1/options/snapshots", headers=self.H,
                             params=dict(symbols=",".join(missing), feed="indicative"), timeout=10)
            for s, snap in ((r.json().get("snapshots") or {}) if r.ok else {}).items():
                lq = snap.get("latestQuote") or {}
                if lq.get("ap"):
                    q[s] = (lq.get("bp") or 0.0, lq["ap"])
        return [(k, s, q[s]) for k, s in zip(ks, syms) if s in q]

    def enter(self, side, spot, i):
        ch = self.chain(side, spot)
        for cap, led in self.L.items():
            tag = f"apex-{cap}-{self.today[2:].replace('-', '')}"
            if any(t["tag"] == tag for t in led["trades"]):
                continue
            budget = led["cash"] * FRAC
            pick = next(((k, s, ba) for k, s, ba in ch if (ba[1] + EDGE) * 100 <= budget), None)
            if not pick:
                self.event(f"${cap}: nothing affordable within ${budget:.0f}", kind="skip")
                continue
            k, sym, (bid, ask) = pick
            qty = max(1, int(budget // ((ask + EDGE) * 100)))
            veto = self.ai_veto(side)
            rec = dict(tag=tag, cap=int(cap), day=self.today, side=side, strike=k, symbol=sym, qty=qty, bar=i, spy_in=spot,
                       quote_in=[bid, ask], t_in=now_et().strftime("%H:%M:%S"), ai_veto=veto)
            if veto and os.environ.get("APEX_AI_VETO") == "1":
                rec["status"] = "vetoed"
                led["trades"].append(rec)
                self.event(f"${cap}: entry vetoed by news desk ({veto})", kind="veto")
                continue
            led["trades"].append(rec)
            if self.dry:
                rec.update(fill_in=ask, status="dry")
            else:
                oid, st, j = orders.submit(self.H, sym, qty, "buy", tag + "-b")
                rec.update(buy_id=oid, status=st)
            self.event(f"${cap}: BUY {qty}x {sym} ask {ask:.2f} → {rec['status']}", kind="order")
        self.sub_options()

    def exit(self, pos, why):
        for cap, led in self.L.items():
            rec = next((t for t in led["trades"] if t["day"] == self.today and "sell_status" not in t and t.get("status") not in ("vetoed",)), None)
            if not rec:
                continue
            rec.update(t_out=now_et().strftime("%H:%M:%S"), why=why, spy_out=self.tick or self.day.closes[-1])
            if self.dry:
                b = self.quotes.get(rec["symbol"], (0, 0, 0))[0]
                rec.update(sell_status="dry", fill_out=b)
            else:
                self.fill(rec)
                held = rec.get("filled_qty") or 0
                if not held:
                    if rec.get("buy_id") and not rec.get("buy_done"):
                        orders._req(self.H, "DELETE", f"/v2/orders/{rec['buy_id']}")
                    rec.update(sell_status="nothing filled", booked=True, pnl=0.0, pct=0.0)
                    continue
                oid, st, j = orders.submit(self.H, rec["symbol"], held, "sell", rec["tag"] + "-s")
                rec.update(sell_id=oid, sell_status=st)
            self.event(f"${cap}: SELL {rec['symbol']} ({why}) → {rec['sell_status']}", kind="order")
        self.settle()

    def fill(self, rec):
        for leg, key in (("buy", "buy_id"), ("sell", "sell_id")):
            oid = rec.get(key)
            if not oid or rec.get(f"{leg}_done"):
                continue
            code, j = orders._req(self.H, "GET", f"/v2/orders/{oid}")
            if code != 200:
                continue
            st = j.get("status")
            if leg == "buy":
                rec["status"] = st
                rec["filled_qty"] = int(float(j.get("filled_qty") or 0))
                if j.get("filled_avg_price"):
                    rec["fill_in"] = float(j["filled_avg_price"])
            else:
                rec["sell_status"] = st
                if j.get("filled_avg_price"):
                    rec["fill_out"] = float(j["filled_avg_price"])
            if st in ("filled", "canceled", "rejected", "expired"):
                rec[f"{leg}_done"] = True

    def settle(self):
        """book realised P&L into each ledger once both legs filled; re-price unfilled sells"""
        for cap, led in self.L.items():
            for rec in led["trades"]:
                if rec.get("booked") or rec.get("status") == "vetoed":
                    continue
                if not self.dry:
                    self.fill(rec)
                if rec.get("sell_status") in ("new", "accepted", "partially_filled") and rec.get("sell_id"):
                    # still working after a while: cancel and resend lower (marketable)
                    if time.time() - rec.get("sell_ts", time.time()) > 8:
                        orders._req(self.H, "DELETE", f"/v2/orders/{rec['sell_id']}")
                    rec.setdefault("sell_ts", time.time())
                if rec.get("fill_in") is not None and rec.get("fill_out") is not None:
                    q = rec.get("filled_qty") or rec["qty"]
                    rec["pnl"] = round((rec["fill_out"] - rec["fill_in"]) * 100 * q, 2)
                    rec["pct"] = round(100 * (rec["fill_out"] / rec["fill_in"] - 1), 1) if rec["fill_in"] else 0
                    led["cash"] += rec["pnl"]
                    rec["booked"] = True
                    self.event(f"${cap}: closed {rec['symbol']} {rec['pnl']:+.2f} ({rec['pct']:+.1f}%) → ledger ${led['cash']:.2f}", kind="pnl")
                elif rec.get("sell_status") in ("canceled",) and rec.get("filled_qty"):
                    oid, st, j = orders.submit(self.H, rec["symbol"], rec["filled_qty"], "sell", rec["tag"] + f"-s{int(time.time()) % 1000}")
                    rec.update(sell_id=oid, sell_status=st, sell_ts=time.time(), sell_done=False)
        self.dirty = True

    # ------------------------------------------------------------------ AI
    def ai_veto(self, side):
        want = "up" if side == "C" else "down"
        cutoff = time.time() - 600
        for r in self.reads:
            if r["ts"] >= cutoff and r.get("relevant") and r.get("magnitude") == "large" and r.get("urgency") == "minutes" \
                    and r.get("direction") not in (want, "none"):
                return r.get("headline", "")[:120]
        return None

    async def on_news(self, n):
        n["et"] = datetime.fromisoformat(n["created_at"].replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M:%S")
        spy = self.tick or (self.day.closes[-1] if self.day and self.day.closes else 0.0)
        ctx = ""
        if self.day and self.day.closes:
            ctx = f"SPY open {self.day.open:.2f}, prior close {self.day.prev_close:.2f}, VWAP {self.day.vwap():.2f}"
        rec = await asyncio.to_thread(self.desk.read, n, spy, ctx)
        if not rec:
            return
        rec["ts"] = time.time()
        self.reads = [r for r in self.reads if r["ts"] > time.time() - 3600] + [rec]
        if rec.get("relevant") and rec.get("urgency") == "minutes" and rec.get("magnitude") in ("medium", "large") \
                and rec.get("direction") in ("up", "down") and 0 <= minute_index(now_et()) < 370:
            asyncio.create_task(self.shock(rec, spy))

    async def shock(self, rec, spy0):
        """paper-only: did a confirmed headline move pay? ATM option bought at the ask once SPY confirms, sold at bid."""
        sgn = 1 if rec["direction"] == "up" else -1
        side = "C" if sgn > 0 else "P"
        t_end = time.time() + 120
        while time.time() < t_end:
            if self.tick and sgn * (self.tick / spy0 - 1) >= SHOCK_MOVE:
                ch = self.chain(side, self.tick)
                if not ch:
                    return
                k, sym, (bid, ask) = ch[0]
                pos = dict(headline=rec.get("headline"), side=side, symbol=sym, t_in=now_et().strftime("%H:%M:%S"), px_in=ask, spy_in=self.tick)
                self.S["shock"].append(pos)
                self.opt_subs.add(sym)
                self.event(f"SHOCK (paper) {side} {sym} @ {ask:.2f} on: {rec.get('headline', '')[:80]}", kind="shock")
                await asyncio.sleep(SHOCK_HOLD * 60)
                q = self.chain(side, self.tick or spy0)
                b = next((ba[0] for kk, s, ba in q if s == sym), None)
                pos.update(t_out=now_et().strftime("%H:%M:%S"), px_out=b, spy_out=self.tick,
                           pct=round(100 * (b / ask - 1), 1) if b and ask else None)
                self.event(f"SHOCK (paper) closed {sym} {pos['pct']}%", kind="shock")
                return
            await asyncio.sleep(1)
        self.log_ai(dict(kind="shock_unconfirmed", headline=rec.get("headline"), spy0=spy0, spy=self.tick))

    async def brief(self):
        try:
            start = (now_et() - timedelta(hours=18)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = requests.get(f"{DATA}/v1beta1/news", headers=self.H, params=dict(start=start, limit=50, sort="desc"), timeout=20)
            items = []
            for a in (r.json().get("news") or []) if r.ok else []:
                a["et"] = datetime.fromisoformat(a["created_at"].replace("Z", "+00:00")).astimezone(ET).strftime("%m-%d %H:%M")
                if self.desk.wanted(a):
                    items.append(a)
            sig, pc = self.history()
            out = await asyncio.to_thread(self.desk.brief, list(reversed(items)), pc)
            if out:
                self.S["brief"] = out
                self.event(f"brief: {out.get('regime')} / lean {out.get('lean')} / trend-day {out.get('trend_day_odds')}")
        except Exception as e:
            self.event(f"brief error {e!r}")

    # ------------------------------------------------------------------ websockets
    def auth(self):
        return {"action": "auth", "key": self.H["APCA-API-KEY-ID"], "secret": self.H["APCA-API-SECRET-KEY"]}

    async def stock_stream(self):
        import websockets
        while True:
            try:
                async with websockets.connect(WS_STOCK, ping_interval=20) as ws:
                    await ws.send(json.dumps(self.auth()))
                    await ws.send(json.dumps({"action": "subscribe", "trades": ["SPY"], "bars": ["SPY"]}))
                    self.event("stock stream connected")
                    async for raw in ws:
                        for m in json.loads(raw):
                            if m.get("T") == "t":
                                self.tick, self.tick_t = m["p"], time.time()
                                self.fast_check()
                            elif m.get("T") == "b":
                                t = datetime.fromisoformat(m["t"].replace("Z", "+00:00")).astimezone(ET)
                                i = minute_index(t)
                                if 0 <= i < 390:
                                    self.feed_bar(i, m)
                                    self.sub_options()
                            elif m.get("T") == "error":
                                self.event(f"stock stream error {m}")
            except Exception as e:
                self.event(f"stock stream dropped {e!r}; reconnecting")
                await asyncio.sleep(2)

    def fast_check(self):
        """tick-level guard between bar closes: if SPY blows through the exit level by 0.15%, don't wait for the close"""
        d = self.day
        if not d or not d.position or not d.closes:
            return
        lv = d.levels(min(self.last_i, 389))
        p = self.tick
        side = d.position["side"]
        if (side == "C" and p < lv["call_exit"] * (1 - 0.0015)) or (side == "P" and p > lv["put_exit"] * (1 + 0.0015)):
            pos = d.position
            d.position = None
            self.event(f"FAST EXIT tick {p:.2f} through {'call' if side == 'C' else 'put'} exit level", kind="exit")
            self.exit(pos, "FAST")

    def sub_options(self):
        if not self.tick and not (self.day and self.day.closes):
            return
        spot = self.tick or self.day.closes[-1]
        base = round(spot)
        want = {orders.occ(self.today, s, k) for s in "CP" for k in range(base - 4, base + 5)}
        for led in self.L.values():
            want |= {t["symbol"] for t in led["trades"] if t["day"] == self.today}
        if want - self.opt_subs and self.opt_ws is not None:
            self.opt_subs |= want
            asyncio.ensure_future(self._opt_send({"action": "subscribe", "quotes": sorted(want)}))

    async def _opt_send(self, msg):
        import msgpack
        try:
            await self.opt_ws.send(msgpack.packb(msg))
        except Exception as e:
            self.event(f"option subscribe failed {e!r}")

    async def option_stream(self):
        import msgpack
        import websockets
        while True:
            try:
                async with websockets.connect(WS_OPT, additional_headers={"Content-Type": "application/msgpack"}, ping_interval=20) as ws:
                    await ws.send(msgpack.packb(self.auth()))
                    self.opt_ws, self.opt_subs = ws, set()
                    self.event("option stream connected")
                    self.sub_options()
                    async for raw in ws:
                        for m in msgpack.unpackb(raw, raw=False):
                            if m.get("T") == "q":
                                self.quotes[m["S"]] = (m.get("bp") or 0.0, m.get("ap") or 0.0, time.time())
                            elif m.get("T") == "error":
                                self.event(f"option stream error {m}")
            except Exception as e:
                self.opt_ws = None
                self.event(f"option stream dropped {e!r}; reconnecting")
                await asyncio.sleep(3)

    async def news_stream(self):
        import websockets
        while True:
            try:
                async with websockets.connect(WS_NEWS, ping_interval=20) as ws:
                    await ws.send(json.dumps(self.auth()))
                    await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
                    self.event(f"news stream connected (AI desk {'on' if self.desk.on else 'off: no ANTHROPIC_API_KEY'})")
                    async for raw in ws:
                        for m in json.loads(raw):
                            if m.get("T") == "n":
                                asyncio.create_task(self.on_news(m))
            except Exception as e:
                self.event(f"news stream dropped {e!r}; reconnecting")
                await asyncio.sleep(3)

    async def housekeeping(self, deadline):
        briefed = "brief" in self.S
        while time.time() < deadline:
            n = now_et()
            hm = n.hour * 100 + n.minute
            if not briefed and hm >= 920:
                briefed = True
                asyncio.create_task(self.brief())
            if self.day and self.day.position is None:
                self.settle()
            elif self.day and self.day.position:
                for led in self.L.values():
                    for rec in led["trades"]:
                        if rec["day"] == self.today and not rec.get("booked") and not self.dry:
                            self.fill(rec)
                            rec.setdefault("buy_ts", time.time())
                            if not rec.get("buy_done") and rec.get("buy_id") and time.time() - rec["buy_ts"] > 15:
                                orders._req(self.H, "DELETE", f"/v2/orders/{rec['buy_id']}")
                                self.fill(rec)
            # mark open trades
            for led in self.L.values():
                for rec in led["trades"]:
                    q = self.quotes.get(rec.get("symbol"))
                    if rec["day"] == self.today and not rec.get("booked") and q:
                        rec["mark"] = round((q[0] + q[1]) / 2, 2)
            self.save()
            if hm >= 1600 or (hm >= 1547 and self.day and not self.day.position and all(
                    t.get("booked") or t.get("status") in ("vetoed",) for led in self.L.values() for t in led["trades"] if t["day"] == self.today)):
                if hm >= 1547:
                    self.event("session done")
                    self.save()
                    return
            await asyncio.sleep(1)
        self.event("job time limit reached; state saved for the next job")
        self.save()


def t_closed(i):
    return minute_index(now_et()) > i


async def amain(a):
    H = feed.load_keys(os.path.expanduser(a.env) if a.env else None)
    c = feed.clock(H)
    nxt = datetime.fromisoformat(c["next_open"]).astimezone(ET)
    if not c["is_open"] and nxt.date() != now_et().date():
        print("no session today; next open", c["next_open"])
        return
    deadline = time.time() + a.max_minutes * 60
    while not c["is_open"] and (nxt - now_et()).total_seconds() > 20 * 60:     # start streams ~20 min before the bell
        await asyncio.sleep(30)
    os.makedirs(LIVE, exist_ok=True)
    ap = Apex(H, dry=a.dry)
    if c["is_open"]:
        ap.catch_up()
    tasks = [asyncio.create_task(x) for x in (ap.stock_stream(), ap.option_stream(), ap.news_stream())]
    await ap.housekeeping(deadline)
    for t in tasks:
        t.cancel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None)
    p.add_argument("--max-minutes", type=float, default=400)
    p.add_argument("--dry", action="store_true", help="no orders (fills = quotes)")
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()
