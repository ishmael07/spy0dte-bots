"""Alpaca data feed for the live paper bots.

Writes the same file formats the backtest engine reads, into the runtime folder's data/:
  data/spy_1m.json          SPY 1-minute bars (IEX feed — real time on the free plan), last ~10 sessions + today
  data/spy_5m_jul.json      SPY 5-minute bars (IEX) for ~45 sessions before that (indicator warmup)
  data/spy_daily.json       SPY daily bars (SIP) for the 50-day trend
  data/live/spy_1m_D.json   today's 1-minute bars
  data/live/quotes_D.json   0DTE option bid/ask recorded every minute {key: {"HH:MM": [bid, ask]}}
Keys come from ALPACA_KEY_ID / ALPACA_SECRET_KEY in the environment (or an env file passed to load_keys).
"""
import json
import os
from datetime import datetime, timedelta, timezone

import requests

DATA = "https://data.alpaca.markets"
TRADE = "https://paper-api.alpaca.markets"


def load_keys(env_file=None):
    if env_file and os.path.exists(env_file):
        for line in open(env_file):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    return {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}


def _get(H, url, params):
    r = requests.get(url, headers=H, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def bars(H, symbol, timeframe, start, end=None, feed="iex"):
    out, token = [], None
    while True:
        p = dict(timeframe=timeframe, start=start, limit=10000, feed=feed, adjustment="split")
        if end:
            p["end"] = end
        if token:
            p["page_token"] = token
        j = _get(H, f"{DATA}/v2/stocks/{symbol}/bars", p)
        out += j.get("bars") or []
        token = j.get("next_page_token")
        if not token:
            return out


def rh_format(b):
    """Alpaca bar → the Robinhood-style dict the engine reads."""
    return dict(begins_at=b["t"].replace("+00:00", "Z"), open_price=str(b["o"]), high_price=str(b["h"]),
                low_price=str(b["l"]), close_price=str(b["c"]), volume=int(b["v"]), session="reg")


def rth(bar_list):
    """Keep regular-hours bars only (9:30–15:59 ET); IEX returns pre/post market too."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    keep = []
    for b in bar_list:
        t = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
        if 570 <= t.hour * 60 + t.minute < 960:
            keep.append(b)
    return keep


def clock(H):
    return _get(H, f"{TRADE}/v2/clock", {})


def calendar(H, start, end):
    return _get(H, f"{TRADE}/v2/calendar", dict(start=start, end=end))


def refresh_history(H, today):
    """Daily warm-up: 1m for the last ~10 sessions, 5m for ~45 before that, daily for a year."""
    os.makedirs("data/live", exist_ok=True)
    cal = [c["date"] for c in calendar(H, (datetime.fromisoformat(today) - timedelta(days=110)).date().isoformat(), today)]
    past = [d for d in cal if d < today]
    one_min_from, five_min_from = past[-10], past[-55]
    m1 = rth(bars(H, "SPY", "1Min", f"{one_min_from}T13:00:00Z", f"{today}T00:00:00Z"))
    m5 = rth(bars(H, "SPY", "5Min", f"{five_min_from}T13:00:00Z", f"{one_min_from}T00:00:00Z"))
    day = bars(H, "SPY", "1Day", (datetime.fromisoformat(today) - timedelta(days=400)).date().isoformat(),
               f"{today}T00:00:00Z", feed="sip")
    json.dump([rh_format(b) for b in m1], open("data/spy_1m.json", "w"))
    json.dump([rh_format(b) for b in m5], open("data/spy_5m_jul.json", "w"))
    json.dump([dict(rh_format(b), begins_at=b["t"][:10] + "T00:00:00Z") for b in day], open("data/spy_daily.json", "w"))
    for f in ("data/spy_5m_may.json", "data/spy_5m_warmup.json"):
        json.dump([], open(f, "w"))
    return past


def refresh_today(H, today):
    """Today's 1m bars so far (IEX, real time) → data/live/spy_1m_D.json."""
    b = rth(bars(H, "SPY", "1Min", f"{today}T13:30:00Z"))
    json.dump([rh_format(x) for x in b], open(f"data/live/spy_1m_{today}.json", "w"))
    return b


def record_quotes(H, today, spot, width=10):
    """Snapshot every 0DTE contract within ±width strikes of spot; store bid/ask under the minute just completed."""
    j = _get(H, f"{DATA}/v1beta1/options/snapshots/SPY", dict(feed="indicative", expiration_date=today,
             strike_price_gte=str(int(spot) - width), strike_price_lte=str(int(spot) + width + 1), limit=1000))
    now = datetime.now(timezone.utc)
    minute = (now - timedelta(minutes=1)).strftime("%H:%M")       # quote at hh:mm:0x ≈ close of minute hh:mm-1
    path = f"data/live/quotes_{today}.json"
    log = json.load(open(path)) if os.path.exists(path) else {}
    for sym, snap in (j.get("snapshots") or {}).items():
        q = snap.get("latestQuote") or {}
        bid, ask = q.get("bp"), q.get("ap")
        if not ask:
            continue
        key = f"{sym[-9]}{int(sym[-8:]) // 1000}"          # SPY261009C00775000 → C775
        log.setdefault(key, {})[minute] = [bid or 0.0, ask]
    json.dump(log, open(path, "w"))
    return len(j.get("snapshots") or {})
