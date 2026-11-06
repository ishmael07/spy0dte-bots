"""Claude as Apex's news desk. Two jobs:

  brief(headlines, calendar)  pre-market: regime, scheduled event risk, trend-day odds, directional lean
  read(headline)              each market-relevant Benzinga headline, within seconds: direction / size / urgency for SPY

Nothing here places orders. The rules engine (core.py) decides; the AI can only (a) veto a new entry that fights a
fresh high-urgency headline, and (b) arm the shock ledger (stream.py), which is paper-only and scored separately.
Every AI call is logged with the SPY price at the time, so each call can be graded later (did SPY move that way?).
Without ANTHROPIC_API_KEY the desk is off and Apex trades on rules alone (fail closed).
"""
import json
import os
import re
import time

MODEL = os.environ.get("APEX_AI_MODEL", "claude-opus-5-5")

# headlines worth a model call: macro, Fed, rates, geopolitics, index heavyweights
MACRO = re.compile(r"\b(Fed|FOMC|Powell|Warsh|rate (hike|cut)|CPI|PPI|inflation|payrolls?|jobs report|unemployment|GDP|PCE|"
                   r"ISM|retail sales|jobless|Treasury|yields?|tariff|sanction|Iran|Israel|China|Russia|Ukraine|Hormuz|oil|OPEC|"
                   r"war|strike|missile|attack|shutdown|default|downgrade|recession|S&P 500|stocks?|Nasdaq|Dow|selloff|"
                   r"sell-off|rally|circuit|halt|crash|plunge|surge|data center|AI|chips?|semiconductor|regulat)\b", re.I)
HEAVY = {"SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "AMD", "MU", "INTC", "ORCL", "JPM"}

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean", "description": "Could this move SPY (the whole S&P 500) within the next hour?"},
        "direction": {"type": "string", "enum": ["up", "down", "none"]},
        "magnitude": {"type": "string", "enum": ["none", "small", "medium", "large"],
                      "description": "small <0.2%, medium 0.2-0.6%, large >0.6% SPY move expected"},
        "urgency": {"type": "string", "enum": ["stale", "hours", "minutes"],
                    "description": "minutes = new information the market is reacting to right now"},
        "why": {"type": "string", "description": "one short sentence"},
    },
    "required": ["relevant", "direction", "magnitude", "urgency", "why"],
    "additionalProperties": False,
}

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "regime": {"type": "string", "enum": ["risk_on", "risk_off", "mixed"]},
        "lean": {"type": "string", "enum": ["up", "down", "none"]},
        "trend_day_odds": {"type": "string", "enum": ["low", "normal", "high"],
                           "description": "odds SPY trends one way most of the session vs chops"},
        "events": {"type": "array", "items": {"type": "string"}, "description": "scheduled events today with ET time"},
        "summary": {"type": "string", "description": "3 sentences max"},
    },
    "required": ["regime", "lean", "trend_day_odds", "events", "summary"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the news desk for a SPY 0DTE options bot. You read market headlines the moment they publish and judge "
    "how the S&P 500 (SPY) is likely to react in the next minutes to hours. Be calibrated: most headlines do not move "
    "the index; say relevant=false for single-stock news unless the company is large enough or the theme broad enough "
    "(AI capex, chips, banks, oil) to move the whole index. Old news already priced in is 'stale'. Never invent facts "
    "beyond the headline and context you are given."
)


class Desk:
    def __init__(self, log):
        self.log = log                       # callable(dict) -> persists the record
        self.client = None
        self.on = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.on:
            try:
                import anthropic
                self.client = anthropic.Anthropic(timeout=20.0, max_retries=1)
            except Exception as e:           # SDK missing -> rules only
                self.on = False
                log(dict(kind="ai_error", error=repr(e)))
        self.seen = set()
        self.calls = 0

    def wanted(self, n):
        """cheap prefilter before spending a model call"""
        if n.get("id") in self.seen:
            return False
        syms = set(n.get("symbols") or [])
        return bool(syms & HEAVY) or bool(MACRO.search(n.get("headline", "")))

    def _ask(self, schema, prompt, max_tokens=1500):
        import anthropic
        kw = dict(model=MODEL, max_tokens=max_tokens, system=SYSTEM, output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                  messages=[{"role": "user", "content": prompt}])
        t = time.time()
        try:
            r = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        except (anthropic.BadRequestError, TypeError):
            r = self.client.messages.create(**kw)
        self.calls += 1
        if r.stop_reason == "refusal":
            return None, time.time() - t
        txt = next((b.text for b in r.content if b.type == "text"), "")
        return json.loads(txt), time.time() - t

    def read(self, n, spy, context=""):
        if not self.on or not self.wanted(n):
            return None
        self.seen.add(n.get("id"))
        prompt = (f"Time (ET): {n.get('et')}\nSPY now: {spy:.2f}\n{context}\n\nHeadline: {n.get('headline')}\n"
                  f"Summary: {(n.get('summary') or '')[:600]}\nSymbols: {', '.join(n.get('symbols') or [])}\nSource: {n.get('source')}")
        try:
            out, dt = self._ask(READ_SCHEMA, prompt)
        except Exception as e:
            self.log(dict(kind="ai_error", error=repr(e)[:300]))
            return None
        if out is None:
            return None
        rec = dict(kind="ai_read", id=n.get("id"), et=n.get("et"), headline=n.get("headline"), spy=spy, secs=round(dt, 1), **out)
        self.log(rec)
        return rec

    def brief(self, headlines, spy_prev_close, extra=""):
        if not self.on:
            return None
        lines = "\n".join(f"- [{h.get('et')}] {h.get('headline')}" for h in headlines[-80:])
        prompt = (f"Pre-market brief. SPY prior close {spy_prev_close:.2f}.\n{extra}\nOvernight and pre-market headlines:\n{lines}")
        try:
            out, dt = self._ask(BRIEF_SCHEMA, prompt, max_tokens=3000)
        except Exception as e:
            self.log(dict(kind="ai_error", error=repr(e)[:300]))
            return None
        if out:
            self.log(dict(kind="ai_brief", secs=round(dt, 1), **out))
        return out
