"""Account rules for the paper/compounding simulations.

  none  : sale proceeds reusable immediately, no trade-count limit (how the backtests ran)
  cash  : cash account (Robinhood-style) — option sale proceeds settle next trading day (T+1); you can only buy with
          settled cash. No day-trade limit.
  pdt   : margin account under $25k (every Alpaca account; Robinhood margin) — at most 3 day trades in any 5 trading
          days. Every 0DTE round trip is a day trade. Unsettled funds are usable.
"""
RULES = ("none", "cash", "pdt")


class Account:
    def __init__(self, capital, rule="none", sessions=None):
        self.rule = rule
        self.settled = float(capital)
        self.pending = {}          # day -> proceeds settling the next session (cash rule)
        self.sessions = list(sessions or [])
        self.daytrades = []        # day of each round trip (pdt rule)

    def roll(self, day):
        for d in [d for d in self.pending if d < day]:
            self.settled += self.pending.pop(d)

    def equity(self):
        return self.settled + sum(self.pending.values())

    def buying_power(self, day):
        self.roll(day)
        return self.settled

    def allowed(self, day):
        """May a new round trip be opened on `day`?"""
        if self.rule != "pdt":
            return True
        if day in self.sessions:
            k = self.sessions.index(day)
            window = set(self.sessions[max(0, k - 4): k + 1])
        else:
            window = {day}
        return sum(1 for d in self.daytrades if d in window) < 3

    def budget(self, day, frac_of_equity):
        """Dollars a trade may spend: a fraction of total equity, capped by what the rule lets you use today."""
        self.roll(day)
        return min(frac_of_equity * self.equity(), self.settled)

    def open(self, day, cost):
        self.settled -= cost
        if self.rule == "pdt":
            self.daytrades.append(day)

    def close(self, day, proceeds):
        if self.rule == "cash":
            self.pending[day] = self.pending.get(day, 0.0) + proceeds
        else:
            self.settled += proceeds

    def book(self, day, cost, pnl):
        """Sequential (non-overlapping) trade: open and close on the same day."""
        self.open(day, cost)
        self.close(day, cost + pnl)
