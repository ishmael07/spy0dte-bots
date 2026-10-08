# SPY 0DTE paper bots

Ten SPY 0DTE option-buying bots (Sniper family, Sniper4, Sniped1, CombinedBot, …) paper-trading live on Alpaca market
data. **Paper money only — no real orders.** Results: see the GitHub Pages site for this repo.

- `alpaca/run.py` — every minute the market is open: SPY 1m bars (IEX), bid/ask for today's 0DTE strikes, rerun all bots
  on everything so far, rebuild the leaderboard page (`docs/index.html`).
- `.github/workflows/paper-bots.yml` — runs it each weekday on GitHub Actions. Needs repo secrets
  `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` (Alpaca **paper** keys).
- Fills: mid of the live bid/ask at the decision minute, +$0.02 slippage per side, $0.03/contract fees.
- Strategy specs: `strategies/`.

Not financial advice. 0DTE options can lose 100% of the premium quickly.
