"""
Layer 2 — Momentum strategy.
Computes 20-day return momentum for individual instruments.
Now supports hierarchical universe (4 asset classes) and crypto symbols.
Also computes rolling volatility for the verification layer.
"""

from __future__ import annotations

import math
from datetime import datetime


# Crypto symbols trade on Alpaca as BTCUSD, ETHUSD, etc.
CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "BCHUSD", "AVAXUSD", "DOTUSD", "LINKUSD"}


def is_crypto(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_SYMBOLS or symbol.upper().endswith("USD") and len(symbol) <= 8


class MomentumStrategy:
    """20-day return momentum ranking — works per-symbol, called by ClassAllocator."""

    def __init__(self, config: dict, alpaca_client):
        self.config = config
        self.alpaca = alpaca_client
        global_cfg = config.get("global", {})
        self.lookback_days = global_cfg.get("lookback_days", config.get("lookback_days", 20))
        self.capital = global_cfg.get("capital", config.get("capital", 500))

        # Flatten universe for backward-compat helpers
        self._all_instruments = self._flatten_universe(config)

    @staticmethod
    def _flatten_universe(config: dict) -> list[str]:
        """Extract all instrument symbols from hierarchical universe config."""
        universe_cfg = config.get("universe", {})
        # Old flat format
        if isinstance(universe_cfg, list):
            return universe_cfg
        # New hierarchical format
        symbols = []
        for class_cfg in universe_cfg.values():
            if isinstance(class_cfg, dict) and "instruments" in class_cfg:
                symbols.extend(class_cfg["instruments"])
        return symbols

    # ── Single-symbol momentum ────────────────────────────

    def compute_momentum(self, symbol: str) -> dict:
        """
        Compute momentum for a single symbol.
        Momentum = (current_close - close_N_days_ago) / close_N_days_ago
        """
        bars = self.alpaca.get_historical_bars(symbol, self.lookback_days)

        if len(bars) < 2:
            return {
                "symbol": symbol,
                "momentum": None,
                "current_price": None,
                "error": f"Insufficient data: {len(bars)} bars",
            }

        relevant_bars = bars[-self.lookback_days:] if len(bars) >= self.lookback_days else bars

        start_price = relevant_bars[0]["close"]
        end_price = relevant_bars[-1]["close"]

        if start_price == 0:
            return {
                "symbol": symbol,
                "momentum": None,
                "current_price": end_price,
                "error": "Start price is zero",
            }

        momentum = (end_price - start_price) / start_price

        return {
            "symbol": symbol,
            "momentum": round(momentum, 6),
            "momentum_pct": round(momentum * 100, 2),
            "current_price": end_price,
            "start_price": start_price,
            "bars_used": len(relevant_bars),
            "start_date": relevant_bars[0]["date"],
            "end_date": relevant_bars[-1]["date"],
            "is_crypto": is_crypto(symbol),
        }

    # ── Volatility computation ────────────────────────────

    def compute_volatility(self, symbol: str, short_window: int = 10, long_window: int = 90) -> dict:
        """
        Compute rolling volatility for a symbol.
        Returns 10-day vol, 90-day average vol, and whether it's flagged (>2σ above mean).
        """
        bars = self.alpaca.get_historical_bars(symbol, long_window + 10)

        if len(bars) < short_window + 1:
            return {
                "symbol": symbol,
                "vol_10d": None,
                "vol_90d_avg": None,
                "vol_90d_std": None,
                "flagged": False,
                "error": f"Insufficient data: {len(bars)} bars",
            }

        # Daily returns
        closes = [b["close"] for b in bars]
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        if len(returns) < short_window:
            return {"symbol": symbol, "vol_10d": None, "flagged": False, "error": "Not enough returns"}

        # 10-day realised volatility (std of daily returns)
        recent_returns = returns[-short_window:]
        vol_10d = self._std(recent_returns)

        # 90-day rolling volatility windows to get mean and std of vol
        vol_windows = []
        available_returns = returns[-long_window:] if len(returns) >= long_window else returns
        for i in range(short_window, len(available_returns) + 1):
            window = available_returns[i - short_window:i]
            vol_windows.append(self._std(window))

        if len(vol_windows) < 2:
            return {
                "symbol": symbol,
                "vol_10d": round(vol_10d, 6),
                "vol_90d_avg": None,
                "vol_90d_std": None,
                "flagged": False,
            }

        vol_mean = sum(vol_windows) / len(vol_windows)
        vol_std = self._std(vol_windows)

        flagged = False
        sigma_above = 0.0
        if vol_std > 0:
            sigma_above = (vol_10d - vol_mean) / vol_std
            flagged = sigma_above > 2.0

        return {
            "symbol": symbol,
            "vol_10d": round(vol_10d, 6),
            "vol_90d_avg": round(vol_mean, 6),
            "vol_90d_std": round(vol_std, 6),
            "sigma_above": round(sigma_above, 2),
            "flagged": flagged,
        }

    @staticmethod
    def _std(values: list[float]) -> float:
        """Population standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    # ── RSI computation ──────────────────────────────────

    def compute_rsi(self, symbol: str, period: int = 14) -> dict:
        """
        Compute 14-period RSI for a symbol.
        Returns rsi value plus overbought / oversold flags.
        Uses Wilder's smoothing (standard definition).
        """
        bars = self.alpaca.get_historical_bars(symbol, period + 30)  # Buffer for weekends

        closes = [b["close"] for b in bars]
        if len(closes) < period + 1:
            return {
                "symbol": symbol,
                "rsi": None,
                "overbought": False,
                "oversold": False,
                "error": f"Insufficient data: {len(closes)} closes",
            }

        # Compute daily changes
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        # Seed: simple average of first `period` values
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder smoothing over remaining bars
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi = round(rsi, 2)
        return {
            "symbol": symbol,
            "rsi": rsi,
            "overbought": rsi >= 70,
            "oversold": rsi <= 30,
        }

    def compute_risk_adjusted_momentum(self, symbol: str) -> dict:
        """
        Risk-adjusted momentum = momentum / annualised_volatility.

        This is a Sharpe-like score that penalises high-volatility returns.
        A stock returning 10% with 5% vol scores 2.0; one returning 10% with
        20% vol scores only 0.5 — even though raw momentum is identical.

        Why this matters: raw momentum chases volatile winners (crypto) and
        gets punished catastrophically during reversals. Risk-adjusted momentum
        naturally down-weights assets that move wildly.
        """
        mom = self.compute_momentum(symbol)
        if mom.get("momentum") is None:
            return {**mom, "risk_adj_momentum": None, "annualised_vol": None}

        bars = self.alpaca.get_historical_bars(symbol, self.lookback_days + 10)
        closes = [b["close"] for b in bars]

        if len(closes) < 5:
            return {**mom, "risk_adj_momentum": mom["momentum"], "annualised_vol": None}

        # Daily returns
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]

        if len(returns) < 4:
            return {**mom, "risk_adj_momentum": mom["momentum"], "annualised_vol": None}

        daily_vol = self._std(returns[-self.lookback_days:])
        annualised_vol = daily_vol * math.sqrt(252)

        if annualised_vol < 0.001:
            # Near-zero vol (e.g., stable bond ETF) — treat raw momentum as score
            risk_adj = mom["momentum"]
        else:
            risk_adj = mom["momentum"] / annualised_vol

        return {
            **mom,
            "risk_adj_momentum": round(risk_adj, 6),
            "annualised_vol": round(annualised_vol, 6),
            "annualised_vol_pct": round(annualised_vol * 100, 2),
        }

    def compute_momentum_with_rsi(self, symbol: str) -> dict:
        """
        Compute momentum + RSI together — used by the adaptive check
        to give Claude richer signal context.
        """
        result = self.compute_momentum(symbol)
        try:
            rsi_data = self.compute_rsi(symbol)
            result["rsi"] = rsi_data.get("rsi")
            result["overbought"] = rsi_data.get("overbought", False)
            result["oversold"] = rsi_data.get("oversold", False)
        except Exception:
            result["rsi"] = None
            result["overbought"] = False
            result["oversold"] = False
        return result

    # ── Rank a flat list of symbols ───────────────────────

    def rank_symbols(self, symbols: list[str]) -> list[dict]:
        """Rank a list of symbols by momentum descending."""
        rankings = []
        for symbol in symbols:
            try:
                result = self.compute_momentum(symbol)
                rankings.append(result)
            except Exception as e:
                rankings.append({"symbol": symbol, "momentum": None, "error": str(e)})

        rankings.sort(
            key=lambda x: x.get("momentum") if x.get("momentum") is not None else float("-inf"),
            reverse=True,
        )
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        return rankings

    # ── Legacy: rank full universe flat (for backtest CLI) ─

    def rank_universe(self) -> list[dict]:
        """Rank all instruments across all classes (flat view)."""
        return self.rank_symbols(self._all_instruments)

    def select_top(self, rankings: list[dict] = None, exclude: list[str] = None,
                   top_n: int = None, capital: float = None) -> list[dict]:
        """Select top N from rankings with equal-weight allocation."""
        if rankings is None:
            rankings = self.rank_universe()
        if top_n is None:
            top_n = 2
        if capital is None:
            capital = self.capital

        exclude = exclude or []
        eligible = [r for r in rankings if r.get("momentum") is not None and r["symbol"] not in exclude]
        selected = eligible[:top_n]

        if not selected:
            return []

        weight = 1.0 / len(selected)
        per_capital = capital * weight
        for s in selected:
            s["weight"] = round(weight, 4)
            s["allocated_capital"] = round(per_capital, 2)
            if s.get("current_price") and s["current_price"] > 0:
                s["target_shares"] = round(per_capital / s["current_price"], 6)
            else:
                s["target_shares"] = 0
        return selected

    # ── Display helpers ───────────────────────────────────

    def print_rankings(self, rankings: list[dict], title: str = "MOMENTUM RANKINGS"):
        """Pretty-print momentum rankings to console."""
        print("\n" + "=" * 75)
        print(f"{title} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"Lookback: {self.lookback_days} days | Capital: ${self.capital}")
        print("=" * 75)
        print(f"{'Rank':<6}{'Symbol':<10}{'Momentum':>10}{'Price':>12}{'Type':>8}")
        print("-" * 75)

        for r in rankings:
            rank = r.get("rank", "?")
            symbol = r.get("symbol", "?")
            mom = r.get("momentum_pct")
            price = r.get("current_price")
            sym_type = "crypto" if r.get("is_crypto") else "equity"

            mom_str = f"{mom:>+8.2f}%" if mom is not None else "   N/A   "
            price_str = f"${price:>10.2f}" if price is not None else "      N/A  "

            print(f"{rank:<6}{symbol:<10}{mom_str}{price_str}{sym_type:>8}")

        print("=" * 75 + "\n")
