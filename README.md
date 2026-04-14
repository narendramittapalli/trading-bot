# Automated ETF Momentum Trading Bot

A production-grade Python trading bot that combines momentum strategy with AI-powered macro analysis and an independent verification layer. Runs on Alpaca paper trading.

## Architecture

The bot has four layers:

1. **Alpaca Connection** — Paper trading, market data, order execution
2. **Momentum Strategy** — 20-day return ranking across SPY, QQQ, IWM, GLD, TLT
3. **News + Claude Reasoning** — Fetches financial headlines, sends them to Claude for macro assessment
4. **Verification Layer** — Independent Claude call that audits the full decision before any trade executes

Every Monday at 9:35 AM ET, the bot ranks the ETF universe by momentum, consults Claude on whether macro conditions support the signals, verifies the decision independently, and then either executes or holds.

## Prerequisites

- Python 3.10+
- An Alpaca paper trading account (free)
- A NewsAPI key (free tier)
- An Anthropic API key

## Getting Credentials

### Alpaca Paper Trading

1. Sign up at [https://alpaca.markets](https://alpaca.markets)
2. Go to the dashboard and select **Paper Trading**
3. Generate API keys under **API Keys**
4. You'll get an `API Key ID` and a `Secret Key`

### NewsAPI

1. Sign up at [https://newsapi.org](https://newsapi.org)
2. The free tier gives 100 requests/day — more than enough
3. Copy your API key from the dashboard

### Anthropic API

1. Sign up at [https://console.anthropic.com](https://console.anthropic.com)
2. Create an API key under **API Keys**
3. The bot uses `claude-sonnet-4-6` by default — typical cost is ~$0.01-0.05 per rebalance cycle

## Setup

### 1. Clone and install dependencies

```bash
cd trading-bot
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export ALPACA_API_KEY="your-alpaca-api-key"
export ALPACA_SECRET_KEY="your-alpaca-secret-key"
export NEWSAPI_KEY="your-newsapi-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

For persistence, add these to your `~/.bashrc`, `~/.zshrc`, or a `.env` file.

### 3. Verify connection

```bash
python main.py status
```

This should print your Alpaca account balance and any open positions.

## Running the Bot

### Start the scheduler

```bash
python main.py run
```

The bot will:
- Pre-fetch news at 9:30 AM ET on Mondays
- Run the full rebalance at 9:35 AM ET on Mondays
- Print every decision to the console
- Log structured JSON to `logs/`

### Other commands

```bash
python main.py status     # Show account and recent activity
python main.py backtest   # Show current momentum rankings (no trades)
python main.py logs       # Tail the last 20 entries from decisions.log
python main.py rebalance  # Run a single rebalance cycle now (for testing)
```

## Configuration

Edit `config.yaml` to customize:

```yaml
universe: [SPY, QQQ, IWM, GLD, TLT]   # ETFs to rank
lookback_days: 20                        # Momentum lookback window
top_n: 2                                 # Number of ETFs to hold
capital: 500                             # Total capital to deploy
rebalance_day: Monday                    # Day of week
rebalance_time: "09:35"                  # Time in ET
```

### Toggle AI layers

```yaml
claude:
  use_claude_layer: true        # Set false to use pure momentum
  use_verification_layer: true  # Set false to skip verification
```

When the Claude layer is off, the bot falls back to pure momentum ranking.

## Reading the Logs

### decisions.log

Structured JSON lines. Each entry has a timestamp and event type:
- `decision` — Full rebalance result with rankings, selections, and verification
- `order` — Individual trade execution
- `status` — Scheduler events, startup, prefetch
- `error` — Any failures

### verification.log

Every verification layer output, including:
- `APPROVED` — Trades executed
- `APPROVED WITH NOTES` — Trades executed, discrepancy logged
- `REJECTED` — Trades blocked, positions held

### predictions.log

Weekly track record: what Claude predicted, what the verification layer said, and what actually happened. Use this to measure reasoning accuracy over time.

## Switching to Live Trading

> **Warning**: Live trading involves real money. Proceed with extreme caution.

1. In `config.yaml`, change:
   ```yaml
   alpaca:
     mode: live
   ```

2. Use your **live** Alpaca API keys (not paper trading keys)

3. Start with a small capital amount and monitor closely

4. The verification layer adds a safety net, but it is not infallible

## Project Structure

```
trading-bot/
├── config.yaml              # All parameters
├── main.py                  # Entry point + CLI + scheduler
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
├── modules/
│   ├── __init__.py
│   ├── alpaca_client.py     # Layer 1: Alpaca connection
│   ├── momentum.py          # Layer 2: Momentum strategy
│   ├── news_ingestion.py    # Layer 3a: News fetching
│   ├── claude_reasoning.py  # Layer 3b: Claude macro analysis
│   ├── verification.py      # Layer 4: Independent verification
│   ├── executor.py          # Orchestrator
│   └── logger.py            # Structured JSON logging
└── logs/
    ├── decisions.log        # All decisions and orders
    ├── predictions.log      # Weekly prediction tracking
    └── verification.log     # Verification audit trail
```
