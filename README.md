# SPY 0DTE paper bots

Ten SPY 0DTE option-buying bots (Sniper family, Sniper4, Sniped1, CombinedBot, …) paper-trading live on Alpaca market
data. **Paper money only — no real orders.** Results: see the GitHub Pages site for this repo.

- `alpaca/run.py` — every minute the market is open: SPY 1m bars (IEX), bid/ask for today's 0DTE strikes, rerun all bots
  on everything so far, rebuild the leaderboard page (`docs/index.html`).
- `.github/workflows/paper-bots.yml` — runs it each weekday on GitHub Actions. Needs repo secrets
  `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` (Alpaca **paper** keys).
- Simulated fills: mid of the live bid/ask at the decision minute, +$0.02 slippage per side, $0.03/contract fees.
- `alpaca/orders.py` — every bot's $100 and $500 accounts (50% sizing) also place real orders in the Alpaca **paper**
  account (marketable limit orders, tagged per bot via client_order_id); actual fills and P&L are shown on the page.
  Any 0DTE position still open at 3:50 pm is closed.
- Strategy specs: `strategies/`.

Not financial advice. 0DTE options can lose 100% of the premium quickly.
