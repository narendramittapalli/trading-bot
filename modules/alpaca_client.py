"""
Layer 1 — Alpaca paper trading connection.
Handles account info, positions, historical data, and order submission.
Supports both equities and crypto via separate data clients.
Credentials via environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY.
"""

import os
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from modules.momentum import is_crypto


class AlpacaClient:
    """Wrapper around alpaca-py for paper trading — equities and crypto."""

    def __init__(self, config: dict):
        self.config = config
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
            )

        # Paper trading = paper=True
        is_paper = config.get("alpaca", {}).get("mode", "paper") == "paper"

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=is_paper,
        )
        self.stock_data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        self.crypto_data_client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        # Flatten universe for any legacy use
        self._all_symbols = self._flatten_universe(config)

    @staticmethod
    def _to_crypto_slash(symbol: str) -> str:
        """Convert 'BTCUSD' → 'BTC/USD' for Alpaca's data API."""
        # Already has slash
        if "/" in symbol:
            return symbol
        # All crypto pairs end in USD (3 chars)
        if symbol.endswith("USD") and len(symbol) > 3:
            return symbol[:-3] + "/USD"
        return symbol

    @staticmethod
    def _flatten_universe(config: dict) -> list[str]:
        """Extract all instrument symbols from hierarchical universe config."""
        universe_cfg = config.get("universe", {})
        if isinstance(universe_cfg, list):
            return universe_cfg
        symbols = []
        for class_cfg in universe_cfg.values():
            if isinstance(class_cfg, dict) and "instruments" in class_cfg:
                symbols.extend(class_cfg["instruments"])
        return symbols

    def get_account(self) -> dict:
        """Get account info: balance, buying power, equity."""
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
        }

    def get_positions(self) -> list[dict]:
        """Get all current positions."""
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "side": p.side,
            }
            for p in positions
        ]

    def get_historical_bars(self, symbol: str, lookback_days: int) -> list[dict]:
        """
        Get daily OHLCV bars for a symbol.
        Automatically routes to crypto or stock data client.
        """
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 10)  # Extra buffer for weekends/holidays

        if is_crypto(symbol):
            # Alpaca data API requires "BTC/USD" slash format — convert from "BTCUSD"
            slash_symbol = self._to_crypto_slash(symbol)
            request = CryptoBarsRequest(
                symbol_or_symbols=slash_symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = self.crypto_data_client.get_crypto_bars(request)
            try:
                bar_list = bars[slash_symbol]
            except (KeyError, TypeError):
                bar_list = []
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,  # Free tier requires explicit IEX feed
            )
            bars = self.stock_data_client.get_stock_bars(request)
            try:
                bar_list = bars[symbol]
            except (KeyError, TypeError):
                bar_list = []

        return [
            {
                "date": bar.timestamp.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            }
            for bar in bar_list
        ]

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest closing price for a symbol from recent bars."""
        bars = self.get_historical_bars(symbol, lookback_days=5)
        if not bars:
            raise ValueError(f"No price data available for {symbol}")
        return bars[-1]["close"]

    def close_all_positions(self) -> list[dict]:
        """Liquidate all current positions."""
        closed = []
        positions = self.trading_client.get_all_positions()
        for pos in positions:
            try:
                self.trading_client.close_position(pos.symbol)
                closed.append({
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "action": "closed",
                })
            except Exception as e:
                closed.append({
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "action": "error",
                    "error": str(e),
                })
        return closed

    def submit_market_order(self, symbol: str, qty: float, side: str) -> dict:
        """
        Submit a market order.
        side: 'buy' or 'sell'
        qty: number of shares/units (can be fractional)
        Crypto uses GTC time-in-force; equities use DAY.
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
        )
        order = self.trading_client.submit_order(order_request)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.value,
            "status": order.status.value,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "type": order.type.value,
        }

    def is_market_open(self) -> bool:
        """Check if US equity market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception:
            return False

    def print_status(self):
        """Print account balance and current positions to console."""
        account = self.get_account()
        positions = self.get_positions()

        print("\n" + "=" * 60)
        print("ALPACA PAPER TRADING — CONNECTION CONFIRMED")
        print("=" * 60)
        print(f"  Account Status : {account['status']}")
        print(f"  Equity         : ${account['equity']:,.2f}")
        print(f"  Cash           : ${account['cash']:,.2f}")
        print(f"  Buying Power   : ${account['buying_power']:,.2f}")
        print(f"  Portfolio Value : ${account['portfolio_value']:,.2f}")
        print("-" * 60)

        if positions:
            print("  Current Positions:")
            for p in positions:
                print(f"    {p['symbol']:8s}  {p['qty']:>8.4f} units  "
                      f"@ ${p['current_price']:>10.2f}  "
                      f"P/L: ${p['unrealized_pl']:>+8.2f}")
        else:
            print("  No open positions.")
        print("=" * 60 + "\n")
