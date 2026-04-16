"""
Parameter Optimizer — Monthly self-improvement via backtest sweep.

Runs on the first Sunday of each month at 8:00 AM ET.

How it works:
  1. Downloads 12 months of daily price data via yfinance (free, no API key)
  2. Simulates the momentum strategy across a grid of parameter combinations
  3. Picks the combination that maximises risk-adjusted return (Sharpe ratio)
  4. Requires a meaningful improvement over current params before recommending change
  5. Stores the recommendation in state.db for auto_tuner to apply

What it optimises (safe, mechanical parameters only):
  - lookback_days         : momentum window length
  - top_n                 : instruments to hold per class
  - min_class_momentum_pct: minimum class-level momentum to receive capital

What it NEVER touches:
  - capital, stop_loss_pct, crypto_stop_loss_pct, max_drawdown_pct
  - Claude model, API keys, Alpaca mode
  - Telegram settings

Why pure momentum (no Claude) for backtesting:
  We cannot replay Claude's reasoning on historical data — it would be
  calling the API 1000+ times per sweep and the results wouldn't be
  comparable across runs. The optimizer tunes the MECHANICAL parameters
  and lets Claude layer on top as a qualitative filter.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from itertools import product


# ── Parameter search space ────────────────────────────────────────────────────

SEARCH_SPACE = {
    "lookback_days":              [10, 15, 20, 25, 30],
    "top_n":                      [1, 2, 3],
    "min_class_momentum_pct":     [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    # How negative SPY must go before the regime filter kicks in (forces REDUCE/SKIP).
    # More negative = permissive (lets the bot trade through mild dips).
    # Less negative = defensive (exits early, more cash during downturns).
    "market_regime_threshold_pct": [-1.0, -1.5, -2.0, -2.5, -3.0],
}

# Minimum improvement in Sharpe ratio required to recommend a param change.
# 0.10 = must be at least 10% better Sharpe than current config.
MIN_IMPROVEMENT_RATIO = 0.10

# Instruments used for backtesting — liquid, long history, representative.
# Using broad ETFs keeps the simulation stable and comparable across regimes.
BACKTEST_UNIVERSE = [
    "SPY",   # US large-cap
    "QQQ",   # US tech
    "IWM",   # US small-cap
    "GLD",   # Gold
    "TLT",   # Long-term bonds
    "EFA",   # International developed
]


class ParameterOptimizer:
    """Runs a monthly parameter sweep using pure momentum on historical data."""

    def __init__(self, config: dict, state_manager, logger, telegram=None):
        self.config = config
        self.state = state_manager
        self.logger = logger
        self.telegram = telegram

    # ── Main entry point ──────────────────────────────────

    def run(self) -> dict:
        """
        Download data, sweep parameters, store best result.
        Returns the optimization result dict.
        """
        print("\n" + "=" * 60)
        print("MONTHLY PARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"  Universe : {BACKTEST_UNIVERSE}")
        print(f"  Combinations: {self._total_combinations()} to evaluate")

        # 1. Download 12 months of price data
        print("\n  Downloading 12 months of price data...")
        try:
            prices = self._download_prices()
        except Exception as e:
            msg = f"Parameter optimization failed — data download error: {e}"
            print(f"  [ERROR] {msg}")
            self.logger.log_error(msg)
            return {"error": str(e)}

        if prices is None or len(prices) < 30:
            msg = "Insufficient historical data for optimization (need ≥30 trading days)."
            print(f"  [WARNING] {msg}")
            self.logger.log_status(msg)
            return {"error": msg}

        print(f"  Got {len(prices)} trading days of data.")

        # 2. Compute current config Sharpe as baseline
        current_params = {
            "lookback_days": self.config.get("global", {}).get("lookback_days", 20),
            "top_n": self.config.get("universe", {}).get("us_equities", {}).get("top_n", 2),
            "min_class_momentum_pct": self.config.get("global", {}).get("min_class_momentum_pct", 1.5),
            "market_regime_threshold_pct": self.config.get("global", {}).get("market_regime_threshold_pct", -2.0),
        }
        current_sharpe = self._simulate(prices, current_params)
        print(f"\n  Current params : {current_params}")
        print(f"  Current Sharpe : {current_sharpe:.3f}")

        # 3. Sweep all parameter combinations
        print("\n  Sweeping parameter combinations...")
        results = []
        best_sharpe = current_sharpe
        best_params = current_params

        param_keys = list(SEARCH_SPACE.keys())
        param_values = list(SEARCH_SPACE.values())

        for combo in product(*param_values):
            params = dict(zip(param_keys, combo))
            sharpe = self._simulate(prices, params)
            results.append({"params": params, "sharpe": round(sharpe, 4)})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        # Sort results for reporting
        results.sort(key=lambda x: x["sharpe"], reverse=True)
        top10 = results[:10]

        # 4. Check if improvement is meaningful
        improvement = (best_sharpe - current_sharpe) / abs(current_sharpe) if current_sharpe != 0 else 0
        is_improvement = improvement >= MIN_IMPROVEMENT_RATIO and best_params != current_params

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_params": current_params,
            "current_sharpe": round(current_sharpe, 4),
            "best_params": best_params,
            "best_sharpe": round(best_sharpe, 4),
            "improvement_pct": round(improvement * 100, 2),
            "recommend_change": is_improvement,
            "top10": top10,
            "trading_days_used": len(prices),
        }

        # 5. Print summary
        self._print_summary(result)

        # 6. Store in state.db
        self.state.store_optimization_result(result)
        self.logger.log_status(
            f"Parameter optimization complete. Best Sharpe: {best_sharpe:.3f} "
            f"({'recommend change' if is_improvement else 'keep current'}).",
            {"best_params": best_params, "improvement_pct": result["improvement_pct"]},
        )

        # 7. Telegram
        if self.telegram:
            self._send_telegram(result)

        return result

    # ── Core simulation ───────────────────────────────────

    def _simulate(self, prices: dict[str, list[float]], params: dict) -> float:
        """
        Simulate weekly momentum rebalancing with the given parameters.
        Returns annualised Sharpe ratio (higher is better; 0 if not enough data).

        prices: {symbol: [close_prices chronologically]}
        """
        lookback = params["lookback_days"]
        top_n = params["top_n"]
        min_mom = params["min_class_momentum_pct"] / 100.0
        regime_threshold = params.get("market_regime_threshold_pct", -2.0) / 100.0
        rebalance_freq = 5  # Weekly (5 trading days)

        # Align price series lengths
        min_len = min(len(v) for v in prices.values())
        aligned = {sym: v[-min_len:] for sym, v in prices.items()}
        spy_closes = aligned.get("SPY", [])

        portfolio_returns = []

        i = lookback
        while i + rebalance_freq < min_len:
            # ── Regime filter: check SPY momentum ────────
            # Full cash if SPY < 2x threshold (severe downturn)
            # Halved exposure if SPY < threshold (mild downturn)
            regime_scale = 1.0
            if len(spy_closes) > i and spy_closes[i - lookback] > 0:
                spy_mom = (spy_closes[i] - spy_closes[i - lookback]) / spy_closes[i - lookback]
                if spy_mom < regime_threshold * 2:
                    # Full cash — sit out severe downturns
                    portfolio_returns.append(0.0)
                    i += rebalance_freq
                    continue
                elif spy_mom < regime_threshold:
                    regime_scale = 0.5  # Halved exposure during mild downturns

            # ── Momentum ranking ──────────────────────────
            momentums = {}
            for sym, closes in aligned.items():
                if sym == "SPY":
                    continue  # SPY is regime proxy only, not a position
                start = closes[i - lookback]
                end = closes[i]
                if start > 0:
                    mom = (end - start) / start
                    if mom >= min_mom:
                        momentums[sym] = mom

            if not momentums:
                portfolio_returns.append(0.0)
                i += rebalance_freq
                continue

            # Pick top_n by momentum
            selected = sorted(momentums, key=momentums.get, reverse=True)[:top_n]

            # Equal-weight return over next rebalance period (scaled by regime)
            period_return = 0.0
            valid = 0
            for sym in selected:
                closes = aligned[sym]
                p_start = closes[i]
                p_end = closes[min(i + rebalance_freq, min_len - 1)]
                if p_start > 0:
                    period_return += (p_end - p_start) / p_start
                    valid += 1

            if valid > 0:
                portfolio_returns.append((period_return / valid) * regime_scale)
            else:
                portfolio_returns.append(0.0)

            i += rebalance_freq

        if len(portfolio_returns) < 6:
            return 0.0

        mean = sum(portfolio_returns) / len(portfolio_returns)
        variance = sum((r - mean) ** 2 for r in portfolio_returns) / len(portfolio_returns)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return 0.0

        # Annualise: 52 weekly periods per year
        return (mean / std) * math.sqrt(52)

    # ── Data download ─────────────────────────────────────

    def _download_prices(self) -> dict[str, list[float]] | None:
        """
        Download 12 months of daily close prices via yfinance.
        Returns {symbol: [close_prices]} or None on failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for parameter optimization. Run: pip install yfinance")

        try:
            raw = yf.download(
                tickers=BACKTEST_UNIVERSE,
                period="12mo",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise RuntimeError(f"yfinance download failed: {e}")

        if raw is None or raw.empty:
            return None

        # Extract Close prices
        if hasattr(raw.columns, "levels"):
            # MultiIndex (multiple tickers)
            try:
                close = raw["Close"]
            except KeyError:
                close = raw.xs("Close", level=0, axis=1)
        else:
            close = raw

        prices = {}
        for sym in BACKTEST_UNIVERSE:
            if sym in close.columns:
                series = close[sym].dropna().tolist()
                if len(series) >= 30:
                    prices[sym] = series

        return prices if len(prices) >= 3 else None

    # ── Helpers ───────────────────────────────────────────

    def _total_combinations(self) -> int:
        total = 1
        for v in SEARCH_SPACE.values():
            total *= len(v)
        return total

    def _print_summary(self, result: dict):
        print(f"\n  {'─' * 54}")
        print(f"  {'Parameter':<28} {'Current':>10} {'Best':>10}")
        print(f"  {'─' * 54}")
        current = result["current_params"]
        best = result["best_params"]
        for key in current:
            changed = " ◄" if current[key] != best.get(key) else ""
            print(f"  {key:<28} {str(current[key]):>10} {str(best.get(key, '?')):>10}{changed}")
        print(f"  {'─' * 54}")
        print(f"  Sharpe (current): {result['current_sharpe']:.3f}")
        print(f"  Sharpe (best)   : {result['best_sharpe']:.3f}  ({result['improvement_pct']:+.1f}%)")
        print(f"  Recommendation  : {'✓ APPLY CHANGES' if result['recommend_change'] else '— KEEP CURRENT (improvement < 10%)'}")
        print(f"  Top 3 results:")
        for r in result["top10"][:3]:
            print(f"    Sharpe={r['sharpe']:.3f}  {r['params']}")
        print("=" * 60 + "\n")

    def _send_telegram(self, result: dict):
        try:
            icon = "✅" if result.get("recommend_change") else "➡️"
            msg = (
                f"{icon} *Monthly Optimization Complete*\n"
                f"Current Sharpe: {result['current_sharpe']:.3f}\n"
                f"Best Sharpe: {result['best_sharpe']:.3f} ({result['improvement_pct']:+.1f}%)\n"
                f"Action: {'Applying parameter update' if result.get('recommend_change') else 'Keeping current params'}"
            )
            self.telegram.send_message(msg)
        except Exception as e:
            print(f"  [WARNING] Telegram optimization report failed: {e}")
