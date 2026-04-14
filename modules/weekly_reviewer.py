"""
Weekly Reviewer — Performance feedback loop.

Runs every Friday at 4:15 PM ET (after US market close).

What it measures:
  1. Open position P&L — are current holdings actually working?
  2. Claude confidence calibration — do HIGH-confidence picks outperform MEDIUM?
  3. Per-class win rate — which asset classes are contributing vs dragging?
  4. Regime accuracy — did the market regime filter correctly warn of downturns?

Output:
  - Structured record stored in state.db (feeds the parameter optimizer)
  - Telegram weekly summary message
  - Console report
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from modules.performance_context import PerformanceContext


class WeeklyReviewer:
    """Reads logs and current positions to produce a weekly performance report."""

    def __init__(self, config: dict, alpaca_client, state_manager, logger, telegram=None):
        self.config = config
        self.alpaca = alpaca_client
        self.state = state_manager
        self.logger = logger
        self.telegram = telegram

        # Resolve log path (respects DATA_DIR env var)
        data_dir = os.environ.get("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_cfg = config.get("logging", {})
        self.predictions_path = os.path.join(data_dir, log_cfg.get("predictions", "logs/predictions.log"))
        self.decisions_path = os.path.join(data_dir, log_cfg.get("decisions", "logs/decisions.log"))

    # ── Main entry point ──────────────────────────────────

    def run_review(self) -> dict:
        """Compute weekly metrics and store results. Called by scheduler every Friday."""
        print("\n" + "=" * 60)
        print("WEEKLY PERFORMANCE REVIEW")
        print("=" * 60)

        week_ending = datetime.now(timezone.utc).date().isoformat()

        # 1. Portfolio snapshot from Alpaca
        try:
            account = self.alpaca.get_account()
            portfolio_value = account["portfolio_value"]
            cash = account["cash"]
        except Exception as e:
            print(f"  [ERROR] Could not fetch account: {e}")
            portfolio_value = None
            cash = None

        # 2. Position-level P&L (open positions vs entry prices)
        position_metrics = self._compute_position_pnl()

        # 3. Claude prediction accuracy from predictions.log
        confidence_metrics = self._compute_confidence_accuracy(position_metrics)

        # 4. Class-level aggregation
        class_metrics = self._compute_class_metrics(position_metrics)

        # 5. Drawdown vs peak
        peak = self.state.get_peak()
        drawdown_pct = self.state.get_drawdown_pct(portfolio_value) if portfolio_value and peak else 0

        # 6. Recent decision count (how active has the bot been?)
        recent_decisions = self._count_recent_decisions(days=7)

        # 6b. News impact: how often did news change a decision, and did those changes help?
        news_metrics = self._compute_news_impact(position_metrics)

        # 6c. API cost this week
        weekly_cost = self._compute_weekly_cost()

        # 7. Assemble record
        record = {
            "week_ending": week_ending,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "peak_value": peak,
            "drawdown_pct": drawdown_pct,
            "positions": position_metrics,
            "overall_win_rate": self._win_rate(position_metrics),
            "avg_return_pct": self._avg_return(position_metrics),
            "confidence": confidence_metrics,
            "by_class": class_metrics,
            "decisions_this_week": recent_decisions,
            "news_impact": news_metrics,
            "api_cost_usd": weekly_cost,
        }

        # 8. Print report
        self._print_report(record)

        # 9. Persist
        self.state.store_learning_record(record)
        self.logger.log_status("Weekly review complete.", {"week_ending": week_ending, "win_rate": record["overall_win_rate"]})

        # 10. Regenerate the performance context memo so Claude gets it on Monday
        try:
            memo = PerformanceContext.update(self.state)
            print(f"\n  Performance context memo updated ({len(memo)} chars). "
                  "Claude will use this on the next rebalance.")
        except Exception as e:
            print(f"  [WARNING] Could not update performance context: {e}")

        # 11. Telegram
        if self.telegram:
            self._send_telegram(record)

        return record

    # ── P&L calculation ───────────────────────────────────

    def _compute_position_pnl(self) -> list[dict]:
        """
        For each open position tracked in state.db, compute P&L relative to entry price.
        Falls back to Alpaca's unrealized_pl if no entry price is stored.
        """
        try:
            positions = self.alpaca.get_positions()
        except Exception as e:
            print(f"  [WARNING] Could not fetch positions: {e}")
            return []

        entry_data = self.state.get_entry_prices()
        metrics = []

        for pos in positions:
            sym = pos["symbol"]
            current_price = float(pos.get("current_price", 0))
            qty = float(pos.get("qty", 0))
            market_value = float(pos.get("market_value", 0))
            unrealized_pl = float(pos.get("unrealized_pl", 0))

            if sym in entry_data:
                entry_price = entry_data[sym].get("price", 0)
                entry_ts = entry_data[sym].get("ts", "")
                pct_return = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                days_held = self._days_since(entry_ts)
            else:
                # No entry price stored — use Alpaca's unrealized P&L as fallback
                avg_entry = float(pos.get("avg_entry_price", current_price))
                entry_price = avg_entry
                entry_ts = ""
                pct_return = (current_price - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
                days_held = None

            # Pull confidence from last prediction for this symbol
            confidence = self._last_prediction_confidence(sym)

            metrics.append({
                "symbol": sym,
                "entry_price": round(entry_price, 4),
                "current_price": round(current_price, 4),
                "pct_return": round(pct_return, 2),
                "market_value": round(market_value, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "days_held": days_held,
                "confidence": confidence,
                "winning": pct_return > 0,
            })

        return sorted(metrics, key=lambda x: x["pct_return"], reverse=True)

    # ── Confidence calibration ────────────────────────────

    def _compute_confidence_accuracy(self, position_metrics: list[dict]) -> dict:
        """
        Measure whether HIGH confidence picks actually outperform MEDIUM ones.
        A well-calibrated Claude should have: HIGH win_rate > MEDIUM win_rate.
        """
        buckets: dict[str, list[float]] = {"HIGH": [], "MEDIUM": [], "LOW": []}

        for pos in position_metrics:
            conf = pos.get("confidence", "MEDIUM").upper()
            if conf in buckets:
                buckets[conf].append(pos["pct_return"])

        result = {}
        for level, returns in buckets.items():
            if returns:
                result[level] = {
                    "count": len(returns),
                    "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
                    "avg_return_pct": round(sum(returns) / len(returns), 2),
                }
        return result

    # ── Class-level aggregation ───────────────────────────

    def _compute_class_metrics(self, position_metrics: list[dict]) -> dict:
        """Group position returns by asset class using the universe config."""
        universe_cfg = self.config.get("universe", {})
        class_lookup = {}
        for class_key, class_cfg in universe_cfg.items():
            if isinstance(class_cfg, dict):
                for sym in class_cfg.get("instruments", []):
                    class_lookup[sym] = class_key

        by_class: dict[str, list[float]] = {}
        for pos in position_metrics:
            class_key = class_lookup.get(pos["symbol"], "other")
            by_class.setdefault(class_key, []).append(pos["pct_return"])

        return {
            k: {
                "count": len(v),
                "avg_return_pct": round(sum(v) / len(v), 2),
                "win_rate": round(sum(1 for r in v if r > 0) / len(v) * 100, 1),
            }
            for k, v in by_class.items()
        }

    # ── Helpers ───────────────────────────────────────────

    def _win_rate(self, metrics: list[dict]) -> float | None:
        if not metrics:
            return None
        return round(sum(1 for m in metrics if m["winning"]) / len(metrics) * 100, 1)

    def _avg_return(self, metrics: list[dict]) -> float | None:
        if not metrics:
            return None
        return round(sum(m["pct_return"] for m in metrics) / len(metrics), 2)

    def _days_since(self, ts: str) -> int | None:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts)
            return (datetime.now(timezone.utc) - dt).days
        except ValueError:
            return None

    def _last_prediction_confidence(self, symbol: str) -> str:
        """Read the most recent prediction confidence for a symbol from predictions.log."""
        try:
            if not os.path.exists(self.predictions_path):
                return "MEDIUM"
            with open(self.predictions_path) as f:
                lines = f.readlines()
            # Scan in reverse for this symbol
            for line in reversed(lines[-200:]):  # last 200 lines max
                try:
                    entry = json.loads(line.strip())
                    if entry.get("symbol") == symbol:
                        return entry.get("confidence", "MEDIUM")
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        return "MEDIUM"

    def _compute_news_impact(self, position_metrics: list[dict]) -> dict:
        """
        Read predictions.log and measure whether news-changed decisions led to
        better or worse outcomes than what pure momentum would have chosen.

        Returns a dict summarising:
          - times_news_changed: how many class decisions news altered this week
          - news_changed_avg_return: avg return of positions where news changed the class decision
          - no_change_avg_return: avg return where news agreed with momentum
          - verdict: 'helping' | 'hurting' | 'neutral' | 'insufficient data'
        """
        try:
            if not os.path.exists(self.predictions_path):
                return {"verdict": "no data"}

            cutoff = datetime.now(timezone.utc).timestamp() - 7 * 86400
            changed_returns: list[float] = []
            unchanged_returns: list[float] = []

            # Build a quick lookup of symbol → pct_return from this week's positions
            return_lookup = {p["symbol"]: p["pct_return"] for p in position_metrics}

            with open(self.predictions_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        ts_str = entry.get("timestamp", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str).timestamp()
                            if ts < cutoff:
                                continue

                        symbol = entry.get("symbol")
                        if symbol not in return_lookup:
                            continue

                        ret = return_lookup[symbol]
                        if entry.get("news_changed_decision"):
                            changed_returns.append(ret)
                        else:
                            unchanged_returns.append(ret)
                    except (json.JSONDecodeError, ValueError):
                        continue

            result: dict = {
                "times_news_changed": len(changed_returns),
                "news_changed_avg_return": (
                    round(sum(changed_returns) / len(changed_returns), 2)
                    if changed_returns else None
                ),
                "no_change_avg_return": (
                    round(sum(unchanged_returns) / len(unchanged_returns), 2)
                    if unchanged_returns else None
                ),
            }

            # Verdict
            if not changed_returns:
                result["verdict"] = "no changes this week"
            elif result["news_changed_avg_return"] is None or result["no_change_avg_return"] is None:
                result["verdict"] = "insufficient data"
            else:
                diff = result["news_changed_avg_return"] - result["no_change_avg_return"]
                if diff > 0.3:
                    result["verdict"] = "helping"
                elif diff < -0.3:
                    result["verdict"] = "hurting"
                else:
                    result["verdict"] = "neutral"

            return result

        except Exception as e:
            return {"verdict": f"error: {e}"}

    def _compute_weekly_cost(self) -> float:
        """Sum estimated API costs from decisions.log over the last 7 days."""
        try:
            if not os.path.exists(self.decisions_path):
                return 0.0
            cutoff = datetime.now(timezone.utc).timestamp() - 7 * 86400
            total = 0.0
            with open(self.decisions_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        ts_str = entry.get("timestamp", "")
                        if ts_str and datetime.fromisoformat(ts_str).timestamp() >= cutoff:
                            tokens = entry.get("api_tokens", {})
                            total += tokens.get("estimated_cost_usd", 0.0)
                    except (json.JSONDecodeError, ValueError):
                        continue
            return round(total, 5)
        except Exception:
            return 0.0

    def _count_recent_decisions(self, days: int = 7) -> int:
        """Count how many rebalance decisions were made in the last N days."""
        try:
            if not os.path.exists(self.decisions_path):
                return 0
            cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
            count = 0
            with open(self.decisions_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("event") == "decision":
                            ts_str = entry.get("timestamp", "")
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str).timestamp()
                                if ts >= cutoff:
                                    count += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
            return count
        except Exception:
            return 0

    # ── Console display ───────────────────────────────────

    def _print_report(self, record: dict):
        pv = record.get("portfolio_value")
        peak = record.get("peak_value")
        dd = record.get("drawdown_pct", 0)
        win_rate = record.get("overall_win_rate")
        avg_ret = record.get("avg_return_pct")

        print(f"\n  Week ending : {record['week_ending']}")
        print(f"  Portfolio   : ${pv:,.2f}" if pv else "  Portfolio   : N/A")
        print(f"  Peak        : ${peak:,.2f}" if peak else "  Peak        : N/A")
        print(f"  Drawdown    : {dd:.1f}%")
        print(f"  Win rate    : {win_rate:.1f}%" if win_rate is not None else "  Win rate    : N/A (no positions)")
        print(f"  Avg return  : {avg_ret:+.2f}%" if avg_ret is not None else "  Avg return  : N/A")
        print(f"  Decisions   : {record.get('decisions_this_week', 0)} rebalances this week")

        news = record.get("news_impact", {})
        verdict = news.get("verdict", "no data")
        changed = news.get("times_news_changed", 0)
        news_ret = news.get("news_changed_avg_return")
        base_ret = news.get("no_change_avg_return")
        if changed:
            news_str = (
                f"{changed} change(s) this week — "
                f"avg return {news_ret:+.2f}% (vs {base_ret:+.2f}% no-change) — {verdict}"
                if news_ret is not None and base_ret is not None
                else f"{changed} change(s) — {verdict}"
            )
        else:
            news_str = verdict
        print(f"  News impact : {news_str}")
        cost = record.get("api_cost_usd")
        print(f"  Claude cost : ${cost:.4f} this week" if cost else "  Claude cost : N/A")

        positions = record.get("positions", [])
        if positions:
            print(f"\n  Open positions ({len(positions)}):")
            print(f"  {'Symbol':<10} {'Entry':>8} {'Current':>8} {'Return':>8} {'Confidence'}")
            print("  " + "-" * 52)
            for p in positions:
                sign = "+" if p["pct_return"] >= 0 else ""
                print(f"  {p['symbol']:<10} ${p['entry_price']:>7.2f} ${p['current_price']:>7.2f} "
                      f"{sign}{p['pct_return']:>6.2f}%  {p.get('confidence', 'MEDIUM')}")

        conf = record.get("confidence", {})
        if conf:
            print(f"\n  Claude confidence calibration:")
            for level, stats in conf.items():
                print(f"    {level:<8}: win={stats['win_rate']:.0f}%  avg={stats['avg_return_pct']:+.2f}%  n={stats['count']}")

        by_class = record.get("by_class", {})
        if by_class:
            print(f"\n  Per-class performance:")
            for cls, stats in sorted(by_class.items(), key=lambda x: x[1]["avg_return_pct"], reverse=True):
                label = cls.replace("_", " ").title()
                print(f"    {label:<20}: win={stats['win_rate']:.0f}%  avg={stats['avg_return_pct']:+.2f}%")

        print("=" * 60 + "\n")

    # ── Telegram ──────────────────────────────────────────

    def _send_telegram(self, record: dict):
        """Send a compact weekly summary via Telegram."""
        try:
            pv = record.get("portfolio_value")
            win_rate = record.get("overall_win_rate")
            avg_ret = record.get("avg_return_pct")
            dd = record.get("drawdown_pct", 0)

            news = record.get("news_impact", {})
            news_line = (
                f"News layer: {news['times_news_changed']} change(s) — {news.get('verdict', '?')}"
                if news.get("times_news_changed", 0)
                else "News layer: no decisions changed"
            )
            lines = [
                f"📊 *Weekly Review — {record['week_ending']}*",
                f"Portfolio: ${pv:,.2f}" if pv else "Portfolio: N/A",
                f"Win rate: {win_rate:.1f}%" if win_rate is not None else "Win rate: N/A",
                f"Avg return: {avg_ret:+.2f}%" if avg_ret is not None else "Avg return: N/A",
                f"Drawdown from peak: {dd:.1f}%",
                news_line,
                f"Claude cost: ${record.get('api_cost_usd', 0):.4f} this week",
            ]

            positions = record.get("positions", [])
            if positions:
                lines.append("\nHoldings:")
                for p in positions[:5]:  # Top 5
                    sign = "+" if p["pct_return"] >= 0 else ""
                    lines.append(f"  {p['symbol']}: {sign}{p['pct_return']:.2f}%")

            msg = "\n".join(lines)
            self.telegram.send_message(msg)
        except Exception as e:
            print(f"  [WARNING] Telegram weekly review failed: {e}")
