"""Apex v2 live engine: websocket streams in, paper orders out, decisions the instant a bar closes.

Streams (Alpaca, free plan):
  stocks  wss://stream.data.alpaca.markets/v2/iex              SPY trades (tick price) + 1-minute bars
  options wss://stream.data.alpaca.markets/v1beta1/indicative  live bid/ask for the strikes around the money (msgpack)
  news    wss://stream.data.alpaca.markets/v1beta1/news        Benzinga headlines -> Claude news desk (ai.py)

Rules: core.py, two lanes (A = trend ride, B = opening range / pullback / fade scalps), at most 3 trades a day,
sized by the 9:33 straddle dial. Same Session code as the lab parity test and the leaderboard sim (sim.py).
Orders: $100 and $500 ledgers in the shared Alpaca paper account, 50% of ledger equity x dial x lane weight,
client_order_id apex-{cap}-{yymmdd}-{lane}{n}-{b|s} so a restart can never double-buy. Restart-safe: on start it
replays today's bars through the rules and resumes from the persisted ledger.
Seconds-level exits: scalp targets/stops act on the SPY trade print itself; rides exit on a 0.15% tick break of the band.

AI desk (advisory): pre-market brief, a read on every market-moving headline, a paper-only "shock" ledger, all logged
and scored forward. APEX_AI_VETO=1 lets a fresh, large, opposite headline block a new entry (off by default).

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
TICK_GUARD = 0.0015
SHOCK_MOVE = 0.0015
SHOCK_HOLD = 30


def now_et():
    return datetime.now(ET)


def minute_index(t):
    return (t.hour * 60 + t.minute) - (9 * 60 + 30)


def t_closed(i):
    return minute_index(now_et()) > i


class Apex:
    def __init__(self, H, dry=False):
        self.H, self.dry = H, dry
        self.today = now_et().strftime("%Y-%m-%d")
        self.ymd = self.today[2:].replace("-", "")
        self.path = os.path.join(LIVE, f"apex_{self.today}.json")
        self.ledger_path = os.path.join(LIVE, "apex_ledger.json")
        self.S = json.load(open(self.path)) if os.path.exists(self.path) else dict(day=self.today, events=[], ai=[], shock=[])
        self.S.setdefault("shock", [])
        self.L = json.load(open(self.ledger_path)) if os.path.exists(self.ledger_path) else {str(c): dict(cash=float(c), trades=[]) for c in CAPS}
        # v1 -> v2 hand-over: a v1 trade from today is the ride lane's first trade (same rule), so v2 manages its exit
        for led in self.L.values():
            for rec in led["trades"]:
                if rec.get("day") == self.today and "lane" not in rec:
                    rec.update(lane="A", n=1, book="ride", tag=rec["tag"] + "-A1")
        self.desk = Desk(self.log_ai)
        self.lanes = None
        self.imp = self.S.get("imp")            # 9:33 straddle, persisted so a restart keeps the same size dial
        self.last_i = -1
        self.tick = None
        self.quotes = {}
        self.opt_ws = None
        self.opt_subs = set()
        self.reads = []
        self.dirty = True

    # ------------------------------------------------------------------ persistence
    def event(self, msg, **kw):
        rec = dict(t=now_et().strftime("%H:%M:%S"), msg=msg, **kw)
        self.S["events"] = (self.S["events"] + [rec])[-300:]
        print(rec["t"], msg, flush=True)
        self.dirty = True

    def log_ai(self, rec):
        rec = dict(rec, t=now_et().strftime("%H:%M:%S"))
        self.S["ai"] = (self.S["ai"] + [rec])[-200:]
        print(rec["t"], "AI", json.dumps(rec)[:300], flush=True)
        self.dirty = True

    def today_trades(self):
        return [t for led in self.L.values() for t in led["trades"] if t["day"] == self.today]

    def save(self):
        if not self.dirty:
            return
        st = dict(self.S)
        A = self.lanes["A"] if self.lanes else None
        if A and A.c:
            st["levels"] = A.levels()
            st["spy"] = self.tick or A.c[-1]
            st["bar"] = self.last_i
            st["positions"] = {k: (dict(book=S.pos["book"], side=S.pos["side"], i=S.pos["i"], spy_in=S.pos["spy_in"],
                                        stop=S.pos["ex"].get("sl"), target=S.pos["ex"].get("tp")) if S.pos else None)
                               for k, S in self.lanes.items()}
            st["imp"], st["dial"] = self.imp, core.dial(self.imp) if self.last_i >= 3 else None
            st["lane_trades"] = {k: [dict(n=t["n"], book=t["book"], side=t["side"], i=t["i"], j=t.get("j"), why=t.get("why"),
                                          ret=t.get("ret")) for t in S.trades] for k, S in self.lanes.items()}
        st["ledger"] = {c: dict(cash=round(v["cash"], 2), n=len(v["trades"])) for c, v in self.L.items()}
        st["trades"] = self.today_trades()
        st["ai_on"] = self.desk.on
        st["updated"] = int(time.time())
        for p, obj in ((self.path, st), (self.ledger_path, self.L)):
            json.dump(obj, open(p + ".tmp", "w"), separators=(",", ":"))
            os.replace(p + ".tmp", p)
        self.dirty = False

    # ------------------------------------------------------------------ setup
    def history(self):
        """sigma (last 14 sessions), prior close, completed 5m/15m blocks (SIP 1m, >15 min old), today's last bar index"""
        start = (now_et() - timedelta(days=45)).strftime("%Y-%m-%dT00:00:00Z")
        end = now_et().replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sess = {}
        for b in feed.bars(self.H, "SPY", "1Min", start, end, feed="sip"):
            t = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            i = minute_index(t)
            if 0 <= i < 390:
                sess.setdefault(t.strftime("%Y-%m-%d"), {})[i] = b
        days = [d for d in sorted(sess) if len(sess[d]) >= 200 and d < self.today]
        hist = [(sess[d][min(sess[d])]["o"], [sess[d][i]["c"] if i in sess[d] else None for i in range(390)]) for d in days]
        prior = {}
        for m in (5, 15):
            blk = []
            for d in days:
                bars = sess[d]
                for b0 in range(0, 390, m):
                    seg = [bars[i] for i in range(b0, b0 + m) if i in bars]
                    if len(seg) == m:
                        blk.append((max(s["h"] for s in seg), min(s["l"] for s in seg), seg[-1]["c"]))
            prior[m] = blk[-400:]
        last = sess[days[-1]]
        cal = feed.calendar(self.H, self.today, self.today)
        close = (cal[0].get("close") if cal else "16:00") or "16:00"
        endi = (int(close[:2]) * 60 + int(close[3:5])) - 570 - 1
        return core.sigma_profile(hist), last[max(last)]["c"], prior, endi

    def today_bars(self):
        start = now_et().replace(hour=9, minute=30, second=0, microsecond=0).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out = {}
        for b in feed.bars(self.H, "SPY", "1Min", start, feed="iex"):
            i = minute_index(datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET))
            if 0 <= i < 390:
                out[i] = b
        return out

    def start_day(self, first_open):
        sig, pc, prior, endi = self.history()
        self.lanes = {k: core.Session(first_open, pc, sig, prior, end=endi, books=L["books"], max_trades=L["max_trades"])
                      for k, L in core.LANES.items()}
        self.event(f"day ready: open {first_open:.2f}, prev close {pc:.2f}, noise band at 10:00 ±{sig[30] * 100:.2f}%, "
                   f"5m trend {'up' if self.lanes['A'].st_prev[5] > 0 else 'down'}, 15m {'up' if self.lanes['A'].st_prev[15] > 0 else 'down'}")

    def catch_up(self):
        bars = self.today_bars()
        if 0 not in bars:
            self.event("waiting for the 9:30 bar")
            return
        self.start_day(bars[0]["o"])
        for i in sorted(bars):
            if t_closed(i):
                self._bar(i, bars[i], replay=True)
        self.event(f"replayed {self.last_i + 1} bars")

    # ------------------------------------------------------------------ bars -> rules
    def feed_bar(self, i, b):
        if self.lanes is None:
            if i != 0:
                return self.catch_up()
            self.start_day(b["o"])
        if i <= self.last_i:
            return
        if i > self.last_i + 1:
            have = self.today_bars()
            for k in range(self.last_i + 1, i):
                if k in have:
                    self._bar(k, have[k], replay=True)
        self._bar(i, b, replay=False)

    def _bar(self, i, b, replay):
        self.last_i = i
        if i == 3 and self.imp is None:
            self.set_implied(b["c"], replay)
        for lane, S in self.lanes.items():
            S.imp = self.imp
            for kind, t in S.on_bar(i, b["o"], b["h"], b["l"], b["c"], b.get("v", 0)):
                if kind == "enter":
                    self.on_enter(lane, S, t, b["c"], replay)
                else:
                    self.on_exit(lane, S, t, b["c"], replay)
        self.dirty = True

    def set_implied(self, spot, replay=False):
        k = round(spot)
        syms = [orders.occ(self.today, x, k) for x in "CP"]
        if replay:
            # started after 9:33: use the real 9:33 option prints, not the current quotes
            c = p = None
            try:
                t0 = now_et().replace(hour=9, minute=33, second=0, microsecond=0).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                t1 = now_et().replace(hour=9, minute=40, second=0, microsecond=0).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                r = requests.get(f"{DATA}/v1beta1/options/bars", headers=self.H, params=dict(symbols=",".join(syms), timeframe="1Min",
                                 start=t0, end=t1, limit=1000), timeout=10)
                bars = (r.json().get("bars") or {}) if r.ok else {}
                c = (bars.get(syms[0]) or [{}])[0].get("c")
                p = (bars.get(syms[1]) or [{}])[0].get("c")
            except requests.RequestException:
                pass
            c, p = ((c, c), (p, p)) if c and p else (None, None)
        else:
            ch = self.snap(syms)
            c, p = ch.get(syms[0]), ch.get(syms[1])
        if c and p:
            self.imp = ((c[0] + c[1]) / 2 + (p[0] + p[1]) / 2) / self.lanes["A"].o0 * 100
            self.S["imp"] = self.imp
            d = core.dial(self.imp)
            self.event(f"9:33 straddle prices a {self.imp:.2f}% move → size dial {d:.2f}" + (" (no trades today: options too expensive)" if not d else ""))
        else:
            self.event("9:33 straddle quote missing → size dial 0.50")

    # ------------------------------------------------------------------ quotes
    def snap(self, syms):
        q, missing = {}, []
        for s in syms:
            v = self.quotes.get(s)
            if v and time.time() - v[2] < 5 and v[1] > 0:
                q[s] = (v[0], v[1])
            else:
                missing.append(s)
        for k in range(0, len(missing), 50):
            try:
                r = requests.get(f"{DATA}/v1beta1/options/snapshots", headers=self.H,
                                 params=dict(symbols=",".join(missing[k:k + 50]), feed="indicative"), timeout=10)
                for s, sn in ((r.json().get("snapshots") or {}) if r.ok else {}).items():
                    lq = sn.get("latestQuote") or {}
                    if lq.get("ap"):
                        q[s] = (lq.get("bp") or 0.0, lq["ap"])
            except requests.RequestException:
                pass
        return q

    def chain(self, side, spot):
        base = round(spot)
        ks = [base + k if side == "C" else base - k for k in range(0, 13)]
        syms = [orders.occ(self.today, side, k) for k in ks]
        q = self.snap(syms)
        return [(k, s, q[s]) for k, s in zip(ks, syms) if s in q]

    # ------------------------------------------------------------------ execution
    def tag(self, cap, lane, n):
        return f"apex-{cap}-{self.ymd}-{lane}{n}"

    def on_enter(self, lane, S, t, spot, replay):
        side = t["side"]
        name = {"orb": "OPENING RANGE", "ride": "TREND RIDE", "pull": "PULLBACK SCALP", "fade": "FAILED-BREAKOUT FADE"}[t["book"]]
        ex = t["ex"]
        plan = (f"target {ex['tp']:.2f} / stop {ex['sl']:.2f} / {ex['cap']} min" if ex.get("tp") else "hold while outside the band, flat 15:45")
        self.event(f"[{lane}] {name} {'CALL' if side == 'C' else 'PUT'} at {spot:.2f} — {plan}", kind="enter", replay=replay)
        ch = self.chain(side, spot)
        atm = next((ba for k, s, ba in ch if k == round(spot)), None)
        if not atm:
            S.cancel(t)
            self.event(f"[{lane}] no ATM quote; skipped")
            return
        t["ref_in"] = atm[1]
        if replay and minute_index(now_et()) - t["i"] > 3:
            return                                  # stale signal after a restart: track it, don't chase it
        w = core.dial(self.imp) * core.LANES[lane]["weight"]
        veto = self.ai_veto(side)
        for cap, led in self.L.items():
            tg = self.tag(cap, lane, t["n"])
            if any(x["tag"] == tg for x in led["trades"]):
                continue
            open_cost = sum(x.get("cost", 0) for x in led["trades"] if x["day"] == self.today and not x.get("booked"))
            budget = min(led["cash"] * FRAC * w, led["cash"] - open_cost)
            pick = next(((k, s, ba) for k, s, ba in ch if (ba[1] + orders.EDGE) * 100 <= budget), None) if budget > 0 else None
            if not pick:
                self.event(f"${cap}: {'dial is 0' if w == 0 else f'nothing affordable within ${budget:.0f}'} — not traded", kind="skip")
                continue
            k, sym, (bid, ask) = pick
            qty = max(1, int(budget // ((ask + orders.EDGE) * 100)))
            rec = dict(tag=tg, cap=int(cap), day=self.today, lane=lane, n=t["n"], book=t["book"], side=side, strike=k, symbol=sym,
                       qty=qty, bar=t["i"], spy_in=spot, quote_in=[bid, ask], cost=round((ask + orders.EDGE) * 100 * qty, 2),
                       t_in=now_et().strftime("%H:%M:%S"), ai_veto=veto, stop=ex.get("sl"), target=ex.get("tp"))
            if veto and os.environ.get("APEX_AI_VETO") == "1":
                rec.update(status="vetoed", booked=True, pnl=0.0)
                led["trades"].append(rec)
                self.event(f"${cap}: vetoed by the news desk ({veto})", kind="veto")
                continue
            led["trades"].append(rec)
            if self.dry:
                rec.update(fill_in=ask, filled_qty=qty, status="dry", buy_done=True)
            else:
                oid, st, _ = orders.submit(self.H, sym, qty, "buy", tg + "-b")
                rec.update(buy_id=oid, status=st, buy_ts=time.time())
            self.event(f"${cap}: BUY {qty}× {sym} @ ask {ask:.2f} → {rec['status']}", kind="order")
        self.sub_options()

    def on_exit(self, lane, S, t, spot, replay):
        done = next((x for led in self.L.values() for x in led["trades"] if x["tag"].endswith(f"-{lane}{t['n']}") and x["day"] == self.today
                     and x.get("pct") is not None), None)
        if replay and done:
            S.result(t, done["pct"] / 100)          # restart: use the real result we already booked
        else:
            atm = self.snap([orders.occ(self.today, t["side"], round(t["spy_in"]))]).get(orders.occ(self.today, t["side"], round(t["spy_in"])))
            S.result(t, (atm[0] / t["ref_in"] - 1) if atm and t.get("ref_in") else -1.0)
        self.event(f"[{lane}] EXIT {t['book']} ({t['why']}) at {spot:.2f}", kind="exit", replay=replay)
        for cap, led in self.L.items():
            rec = next((x for x in led["trades"] if x["tag"] == self.tag(cap, lane, t["n"])), None)
            if not rec or rec.get("booked") or rec.get("sell_id") or rec.get("sell_status"):
                continue
            rec.update(t_out=now_et().strftime("%H:%M:%S"), why=t["why"], spy_out=spot)
            self.sell(rec)

    def sell(self, rec):
        if self.dry:
            q = self.quotes.get(rec["symbol"]) or (*self.snap([rec["symbol"]]).get(rec["symbol"], (0.0, 0.0)), 0)
            rec.update(sell_status="dry", fill_out=q[0], sell_done=True)
            self.settle()
            return
        self.fill(rec)
        held = rec.get("filled_qty") or 0
        if not held:
            if rec.get("buy_id") and not rec.get("buy_done"):
                orders._req(self.H, "DELETE", f"/v2/orders/{rec['buy_id']}")
            rec.update(sell_status="nothing filled", booked=True, pnl=0.0, pct=0.0)
            return
        oid, st, _ = orders.submit(self.H, rec["symbol"], held, "sell", rec["tag"] + "-s")
        rec.update(sell_id=oid, sell_status=st, sell_ts=time.time(), sell_done=False)
        self.event(f"${rec['cap']}: SELL {held}× {rec['symbol']} ({rec.get('why')}) → {st}", kind="order")

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
        for cap, led in self.L.items():
            for rec in led["trades"]:
                if rec.get("booked") or rec.get("day") != self.today:
                    continue
                if not self.dry:
                    self.fill(rec)
                    # entry still working after 15 s: cancel the rest (keep any partial fill)
                    if rec.get("buy_id") and not rec.get("buy_done") and time.time() - rec.get("buy_ts", time.time()) > 15:
                        orders._req(self.H, "DELETE", f"/v2/orders/{rec['buy_id']}")
                        self.fill(rec)
                    # exit still working after 8 s: cancel and resend at the new bid
                    if rec.get("sell_id") and not rec.get("sell_done") and time.time() - rec.get("sell_ts", time.time()) > 8:
                        orders._req(self.H, "DELETE", f"/v2/orders/{rec['sell_id']}")
                        self.fill(rec)
                    if rec.get("sell_status") in ("canceled", "expired", "rejected") and (rec.get("filled_qty") or 0) and rec.get("fill_out") is None:
                        oid, st, _ = orders.submit(self.H, rec["symbol"], rec["filled_qty"], "sell", rec["tag"] + f"-s{int(time.time()) % 10000}")
                        rec.update(sell_id=oid, sell_status=st, sell_ts=time.time(), sell_done=False)
                if rec.get("fill_in") is not None and rec.get("fill_out") is not None:
                    q = rec.get("filled_qty") or rec["qty"]
                    rec["pnl"] = round((rec["fill_out"] - rec["fill_in"]) * 100 * q, 2)
                    rec["pct"] = round(100 * (rec["fill_out"] / rec["fill_in"] - 1), 1) if rec["fill_in"] else 0.0
                    led["cash"] += rec["pnl"]
                    rec["booked"] = True
                    self.event(f"${cap}: closed {rec['symbol']} {rec['pnl']:+.2f} ({rec['pct']:+.1f}%) → ledger ${led['cash']:.2f}", kind="pnl")
        self.dirty = True

    # ------------------------------------------------------------------ tick guard
    def fast_check(self):
        """seconds-level exits on every SPY trade print:
        scalps/ORB: target or stop touched -> sell now (an ORB target hit while SPY is breaking out is left to the bar close,
        where it converts into a ride); rides: SPY 0.15% through the band/VWAP exit level -> sell now."""
        if not self.lanes or self.last_i < 0 or not self.tick:
            return
        p = self.tick
        for lane, S in self.lanes.items():
            pos = S.pos
            if not pos or not S.c:
                continue
            sg = 1 if pos["side"] == "C" else -1
            ex, i = pos["ex"], len(S.c) - 1
            why = None
            if ex.get("sl") is not None and sg * (p - ex["sl"]) <= 0:
                why, lv = "STOP (tick)", ex["sl"]
            elif ex.get("tp") is not None and sg * (p - ex["tp"]) >= 0:
                breaking_out = p > S.up[i] if sg > 0 else p < S.dn[i]
                if not (ex.get("upgrade") and breaking_out):
                    why, lv = "TARGET (tick)", ex["tp"]
            elif ex.get("struct") == "band":
                lv = max(S.up[i], S.vw[i]) if sg > 0 else min(S.dn[i], S.vw[i])
                if sg * (p - lv * (1 - sg * TICK_GUARD)) < 0:
                    why = "BAND (tick)"
            if why:
                t = S.force_exit(why)
                self.event(f"[{lane}] {why}: SPY {p:.2f} vs {lv:.2f} — selling now", kind="exit")
                self.on_exit(lane, S, t, p, False)

    # ------------------------------------------------------------------ AI
    def ai_veto(self, side):
        want = "up" if side == "C" else "down"
        for r in self.reads:
            if r["ts"] >= time.time() - 600 and r.get("relevant") and r.get("magnitude") == "large" and r.get("urgency") == "minutes" \
                    and r.get("direction") not in (want, "none"):
                return r.get("headline", "")[:120]
        return None

    async def on_news(self, n):
        n["et"] = datetime.fromisoformat(n["created_at"].replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M:%S")
        A = self.lanes["A"] if self.lanes else None
        spy = self.tick or (A.c[-1] if A and A.c else 0.0)
        ctx = f"SPY open {A.o0:.2f}, prior close {A.pc:.2f}, VWAP {A.vw[-1]:.2f}" if A and A.c else ""
        rec = await asyncio.to_thread(self.desk.read, n, spy, ctx)
        if not rec:
            return
        rec["ts"] = time.time()
        self.reads = [r for r in self.reads if r["ts"] > time.time() - 3600] + [rec]
        if rec.get("relevant") and rec.get("urgency") == "minutes" and rec.get("magnitude") in ("medium", "large") \
                and rec.get("direction") in ("up", "down") and 0 <= minute_index(now_et()) < 370 and spy:
            asyncio.create_task(self.shock(rec, spy))

    async def shock(self, rec, spy0):
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
                self.event(f"SHOCK (paper only) {side} {sym} @ {ask:.2f} on: {rec.get('headline', '')[:80]}", kind="shock")
                await asyncio.sleep(SHOCK_HOLD * 60)
                q = self.snap([sym]).get(sym)
                b = q[0] if q else None
                pos.update(t_out=now_et().strftime("%H:%M:%S"), px_out=b, spy_out=self.tick, pct=round(100 * (b / ask - 1), 1) if b and ask else None)
                self.event(f"SHOCK (paper only) closed {sym} {pos['pct']}%", kind="shock")
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
            sig, pc, _, _ = self.history()
            out = await asyncio.to_thread(self.desk.brief, list(reversed(items)), pc)
            if out:
                self.S["brief"] = out
                self.event(f"brief: {out.get('regime')} / lean {out.get('lean')} / trend-day odds {out.get('trend_day_odds')}")
        except Exception as e:
            self.event(f"brief error {e!r}")

    # ------------------------------------------------------------------ websockets
    def auth(self):
        return {"action": "auth", "key": self.H["APCA-API-KEY-ID"], "secret": self.H["APCA-API-SECRET-KEY"]}

    async def stock_stream(self):
        import websockets
        backoff = 2
        while True:
            try:
                async with websockets.connect(WS_STOCK, ping_interval=20) as ws:
                    await ws.send(json.dumps(self.auth()))
                    await ws.send(json.dumps({"action": "subscribe", "trades": ["SPY"], "bars": ["SPY"]}))
                    self.event("stock stream connected")
                    backoff = 2
                    async for raw in ws:
                        for m in json.loads(raw):
                            if m.get("T") == "t":
                                self.tick = m["p"]
                                self.fast_check()
                            elif m.get("T") == "b":
                                i = minute_index(datetime.fromisoformat(m["t"].replace("Z", "+00:00")).astimezone(ET))
                                if 0 <= i < 390:
                                    self.feed_bar(i, m)
                                    self.sub_options()
                            elif m.get("T") == "error":
                                self.event(f"stock stream error {m}")
            except Exception as e:
                self.event(f"stock stream dropped {e!r}; reconnecting")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def sub_options(self):
        A = self.lanes["A"] if self.lanes else None
        spot = self.tick or (A.c[-1] if A and A.c else None)
        if not spot or self.opt_ws is None:
            return
        base = round(spot)
        want = {orders.occ(self.today, s, k) for s in "CP" for k in range(base - 6, base + 7)}
        want |= {t["symbol"] for t in self.today_trades()}
        new = want - self.opt_subs
        if new:
            self.opt_subs |= new
            asyncio.ensure_future(self._opt_send({"action": "subscribe", "quotes": sorted(new)}))

    async def _opt_send(self, msg):
        import msgpack
        try:
            await self.opt_ws.send(msgpack.packb(msg))
        except Exception as e:
            self.event(f"option subscribe failed {e!r}")

    async def option_stream(self):
        import msgpack
        import websockets
        backoff = 3
        while True:
            try:
                async with websockets.connect(WS_OPT, additional_headers={"Content-Type": "application/msgpack"}, ping_interval=20) as ws:
                    await ws.send(msgpack.packb(self.auth()))
                    self.opt_ws, self.opt_subs = ws, set()
                    self.event("option stream connected")
                    backoff = 3
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
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def news_stream(self):
        import websockets
        backoff = 3
        while True:
            try:
                async with websockets.connect(WS_NEWS, ping_interval=20) as ws:
                    await ws.send(json.dumps(self.auth()))
                    await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
                    self.event(f"news stream connected (AI desk {'on' if self.desk.on else 'off: no ANTHROPIC_API_KEY'})")
                    backoff = 3
                    async for raw in ws:
                        for m in json.loads(raw):
                            if m.get("T") == "n":
                                asyncio.create_task(self.on_news(m))
            except Exception as e:
                self.event(f"news stream dropped {e!r}; reconnecting")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def housekeeping(self, deadline):
        briefed = "brief" in self.S
        while time.time() < deadline:
            hm = now_et().hour * 100 + now_et().minute
            if not briefed and hm >= 920:
                briefed = True
                asyncio.create_task(self.brief())
            if any(not t.get("booked") for t in self.today_trades()):
                self.settle()
            for t in self.today_trades():
                q = self.quotes.get(t.get("symbol"))
                if not t.get("booked") and q:
                    t["mark"] = round((q[0] + q[1]) / 2, 2)
            self.save()
            busy = (self.lanes and any(S.pos for S in self.lanes.values())) or any(not t.get("booked") for t in self.today_trades())
            if hm >= 1600 or (hm >= 1547 and not busy):
                self.event("session done")
                self.save()
                return
            await asyncio.sleep(1)
        self.event("job time limit reached; state saved for the next job")
        self.save()


async def amain(a):
    H = feed.load_keys(os.path.expanduser(a.env) if a.env else None)
    c = feed.clock(H)
    nxt = datetime.fromisoformat(c["next_open"]).astimezone(ET)
    if not c["is_open"] and nxt.date() != now_et().date():
        print("no session today; next open", c["next_open"])
        return
    deadline = time.time() + a.max_minutes * 60
    while not c["is_open"] and (nxt - now_et()).total_seconds() > 20 * 60:
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
