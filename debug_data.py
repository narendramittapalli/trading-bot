#!/usr/bin/env python3
"""
Quick diagnostic — confirms Alpaca bar data is fetching correctly.
Run from the trading-bot directory: python debug_data.py
"""

import os, sys
from datetime import datetime, timedelta

api_key = os.environ.get("ALPACA_API_KEY", "")
secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

if not api_key or not secret_key:
    print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
    sys.exit(1)

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
crypto_client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)

end = datetime.now()
start = end - timedelta(days=30)

print(f"\nFetching bars: {start.date()} → {end.date()}\n")

# Test equities — use bars[symbol] not bars.get()
for sym in ["SPY", "QQQ", "GLD"]:
    try:
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                               start=start, end=end, feed=DataFeed.IEX)
        bars = stock_client.get_stock_bars(req)
        bar_list = bars[sym]
        print(f"  [OK] {sym}: {len(bar_list)} bars | latest close=${bar_list[-1].close:.2f}")
    except KeyError:
        print(f"  [!!] {sym}: symbol not found in response")
    except Exception as e:
        print(f"  [ERR] {sym}: {e}")

# Test crypto — must use BTC/USD slash format for data API
for slash_sym in ["BTC/USD", "ETH/USD", "SOL/USD"]:
    try:
        req = CryptoBarsRequest(symbol_or_symbols=slash_sym, timeframe=TimeFrame.Day,
                                start=start, end=end)
        bars = crypto_client.get_crypto_bars(req)
        bar_list = bars[slash_sym]
        print(f"  [OK] {slash_sym}: {len(bar_list)} bars | latest close=${bar_list[-1].close:.2f}")
    except KeyError:
        print(f"  [!!] {slash_sym}: symbol not found in response")
    except Exception as e:
        print(f"  [ERR] {slash_sym}: {e}")

print("\nIf all lines show [OK], run: python main.py backtest")
