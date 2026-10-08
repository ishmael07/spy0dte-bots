"""Live paper-trading runner on Alpaca data — no Claude, no Mac required (any always-on machine works).

Every minute the market is open: pull SPY 1m bars (IEX, real time), record bid/ask for today's 0DTE strikes,
rerun all 10 bots on everything so far (same code as the backtests), rebuild the leaderboard page.

  python3 alpaca/run.py --env ~/dawn-raid/.env            # run all day (Ctrl-C to stop)
  python3 alpaca/run.py --env ~/dawn-raid/.env --once     # one poll (for testing / cron)
Output: alpaca/runtime/web/paper.html (open in a browser), alpaca/runtime/data/paper_results.pkl
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "alpaca", "runtime")
sys.path.insert(0, ROOT)
from alpaca import feed  # noqa: E402

ET = ZoneInfo("America/New_York")
START = "2026-10-09"


def setup():
    os.makedirs(os.path.join(RUNTIME, "data", "live"), exist_ok=True)
    os.makedirs(os.path.join(RUNTIME, "web"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "data", "validation.json"), os.path.join(RUNTIME, "data", "validation.json"))
    shutil.copy(os.path.join(ROOT, "web", "template4.html"), os.path.join(RUNTIME, "web", "template4.html"))
    replay = os.path.join(ROOT, "data", "replay_page.json")
    if os.path.exists(replay):
        shutil.copy(replay, os.path.join(RUNTIME, "data", "replay_page.json"))
    os.chdir(RUNTIME)


def publish(message):
    """GitHub mode: copy the page to docs/index.html and commit + push the live data and page."""
    docs = os.path.join(ROOT, "docs")
    os.makedirs(docs, exist_ok=True)
    page = os.path.join(RUNTIME, "web", "paper.html")
    if os.path.exists(page):
        shutil.copy(page, os.path.join(docs, "index.html"))
    git = ["git", "-C", ROOT]
    subprocess.run(git + ["add", "docs", "alpaca/runtime/data/live"], check=False)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(git + ["commit", "-q", "-m", message], check=False)
        for _ in range(3):
            if subprocess.run(git + ["push", "-q"]).returncode == 0:
                break
            subprocess.run(git + ["pull", "-q", "--rebase", "-X", "ours"], check=False)


def live_days(start):
    import glob
    return sorted({os.path.basename(f)[7:17] for f in glob.glob("data/live/quotes_*.json")} & {
        os.path.basename(f)[7:17] for f in glob.glob("data/live/spy_1m_*.json")})


def engine(start):
    days = [d for d in live_days(start) if d >= start]
    if not days:
        return "no live days yet"
    env = dict(os.environ, PYTHONPATH=ROOT)
    py = [sys.executable, "-W", "ignore"]
    subprocess.run(py + [os.path.join(ROOT, "paper.py"), *days, "--start", start], check=True, env=env, stdout=subprocess.DEVNULL)
    subprocess.run(py + [os.path.join(ROOT, "export_paper.py")], check=True, env=env, stdout=subprocess.DEVNULL)
    out = subprocess.run(py + [os.path.join(ROOT, "live_poll.py"), "--summary-only"], env=env, capture_output=True, text=True)
    return out.stdout.strip()


def poll(H, today, start):
    bars = feed.refresh_today(H, today)
    if not bars:
        return "no bars yet"
    spot = float(bars[-1]["c"])
    n = feed.record_quotes(H, today, spot)
    t = time.time()
    summary = engine(start)
    return f"SPY {spot:.2f} · {n} quotes · engine {time.time() - t:.1f}s\n{summary}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--max-minutes", type=float, default=None, help="exit after this long (GitHub job limit)")
    ap.add_argument("--publish-every", type=int, default=0, help="commit+push page and data every N polls (GitHub mode)")
    a = ap.parse_args()
    began = time.time()
    polls = 0
    H = feed.load_keys(os.path.expanduser(a.env) if a.env else None)
    setup()
    history_for = None
    while True:
        now = datetime.now(ET)
        today = now.strftime("%Y-%m-%d")
        c = feed.clock(H)
        if history_for != today:
            feed.refresh_history(H, today)
            history_for = today
        if c["is_open"]:
            try:
                print(now.strftime("%H:%M:%S"), poll(H, today, a.start), flush=True)
                polls += 1
                if a.publish_every and polls % a.publish_every == 0:
                    publish(f"paper poll {today} {now.strftime('%H:%M')} ET")
            except Exception as e:  # keep running through transient API errors
                print(now.strftime("%H:%M:%S"), "poll error:", repr(e), flush=True)
        elif a.once:
            print("market closed:", c.get("next_open"))
        no_session_today = not c["is_open"] and c["next_open"][:10] != today and polls == 0
        out_of_time = a.max_minutes and (time.time() - began) / 60 > a.max_minutes
        if no_session_today and not a.once:
            print("no session today; next open", c["next_open"])
            return
        closed_for_day = not c["is_open"] and polls > 0
        if a.once or out_of_time or closed_for_day:
            if a.publish_every:
                publish(f"paper {today} {datetime.now(ET).strftime('%H:%M')} ET (job end)")
            return
        # wake 5 seconds after the next minute boundary (bars for the finished minute are published by then)
        nxt = (datetime.now(timezone.utc) + timedelta(minutes=1)).replace(second=5, microsecond=0)
        if not c["is_open"]:
            nxt = max(nxt, datetime.fromisoformat(c["next_open"]).astimezone(timezone.utc) - timedelta(minutes=20))
        time.sleep(max(1.0, (nxt - datetime.now(timezone.utc)).total_seconds()))


if __name__ == "__main__":
    main()
