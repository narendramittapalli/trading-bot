"""
Performance Context — Generates a structured memo from weekly review history.

Runs after every weekly review (Friday 4:15 PM ET) and stores a compact,
human-readable context string in state.db under 'performance_context'.

This memo is injected into Claude's Stage 1 and Stage 2 prompts on every
rebalance day so that Claude reasons with knowledge of its own track record —
not in a vacuum.

What it captures (from the last 8 weekly reviews, newest first):
  - Overall win rate trend (improving / stable / declining)
  - Average position return per week
  - Per-class performance (which classes are working vs dragging)
  - Confidence calibration (is HIGH actually outperforming MEDIUM?)
  - Current drawdown vs peak

Why 8 weeks? Enough to identify a meaningful trend; old enough to include
regime changes but not so much that ancient history drowns recent signals.
"""

from __future__ import annotations


class PerformanceContext:
    """Generates and stores a compact performance memo from weekly review history."""

    STATE_KEY = "performance_context"
    WEEKS = 8

    # ── Public interface ──────────────────────────────────

    @classmethod
    def update(cls, state_manager) -> str:
        """
        Regenerate the memo from the last WEEKS weekly reviews and persist it.
        Called at the end of each weekly review.
        Returns the memo string.
        """
        history = state_manager.get_learning_history(cls.WEEKS)
        if not history:
            memo = (
                "No performance history yet — strategy is in early accumulation phase. "
                "Apply moderate risk weighting and build track record before increasing conviction."
            )
            state_manager.set_kv(cls.STATE_KEY, memo)
            return memo

        memo = cls._build_memo(history)
        state_manager.set_kv(cls.STATE_KEY, memo)
        return memo

    @classmethod
    def get(cls, state_manager) -> str:
        """Retrieve the stored memo. Returns empty string if none exists yet."""
        return state_manager.get_kv(cls.STATE_KEY) or ""

    # ── Memo builder ──────────────────────────────────────

    @classmethod
    def _build_memo(cls, history: list[dict]) -> str:
        """
        Assemble a human-readable performance memo from review history.
        history[0] is the most recent week (DESC order from state.db).
        """
        n = len(history)
        lines = [f"=== YOUR PERFORMANCE TRACK RECORD (last {n} week{'s' if n != 1 else ''}) ==="]
        lines.append("Use this context to inform your decisions today.\n")

        # ── 1. Overall win rate ───────────────────────────
        win_rates = [r["overall_win_rate"] for r in history if r.get("overall_win_rate") is not None]
        avg_returns = [r["avg_return_pct"] for r in history if r.get("avg_return_pct") is not None]

        if win_rates:
            avg_win = sum(win_rates) / len(win_rates)
            recent_win = win_rates[0]
            if len(win_rates) >= 3:
                early_avg = sum(win_rates[-3:]) / 3
                trend = (
                    "improving ↑" if recent_win > early_avg + 5
                    else "declining ↓" if recent_win < early_avg - 5
                    else "stable →"
                )
            else:
                trend = "insufficient data"
            lines.append(f"Win rate:  {avg_win:.0f}% avg over {n} weeks  ({recent_win:.0f}% last week, trend: {trend})")

        if avg_returns:
            avg_ret = sum(avg_returns) / len(avg_returns)
            recent_ret = avg_returns[0]
            lines.append(f"Returns:   {avg_ret:+.2f}% avg per position  ({recent_ret:+.2f}% last week)")

        # ── 2. Per-class track record ─────────────────────
        class_returns: dict[str, list[float]] = {}
        class_win_rates: dict[str, list[float]] = {}

        for record in history:
            for cls_key, stats in record.get("by_class", {}).items():
                class_returns.setdefault(cls_key, []).append(stats.get("avg_return_pct", 0))
                class_win_rates.setdefault(cls_key, []).append(stats.get("win_rate", 50))

        if class_returns:
            lines.append("\nPer-class track record:")
            sorted_classes = sorted(
                class_returns.items(),
                key=lambda kv: sum(kv[1]) / len(kv[1]),
                reverse=True,
            )
            for cls_key, returns in sorted_classes:
                avg_r = sum(returns) / len(returns)
                avg_w = sum(class_win_rates.get(cls_key, [50])) / max(len(class_win_rates.get(cls_key, [50])), 1)
                label = cls_key.replace("_", " ").title()
                verdict = "✓ contributing" if avg_r > 0.5 else "✗ underperforming" if avg_r < -0.5 else "~ neutral"
                lines.append(f"  {label:<22} avg {avg_r:+.2f}%  win {avg_w:.0f}%  {verdict}")

        # ── 3. Confidence calibration ─────────────────────
        conf_returns: dict[str, list[float]] = {}
        conf_wins: dict[str, list[float]] = {}

        for record in history:
            for level, stats in record.get("confidence", {}).items():
                conf_returns.setdefault(level, []).append(stats.get("avg_return_pct", 0))
                conf_wins.setdefault(level, []).append(stats.get("win_rate", 50))

        if len(conf_returns) >= 2:
            lines.append("\nConfidence calibration:")
            for level in ["HIGH", "MEDIUM", "LOW"]:
                if level in conf_returns:
                    avg_r = sum(conf_returns[level]) / len(conf_returns[level])
                    avg_w = sum(conf_wins[level]) / len(conf_wins[level])
                    lines.append(f"  {level:<8} avg {avg_r:+.2f}%  win {avg_w:.0f}%")

            # Calibration verdict — actionable instruction to Claude
            if "HIGH" in conf_returns and "MEDIUM" in conf_returns:
                high_avg = sum(conf_returns["HIGH"]) / len(conf_returns["HIGH"])
                med_avg = sum(conf_returns["MEDIUM"]) / len(conf_returns["MEDIUM"])
                diff = abs(high_avg - med_avg)
                if high_avg < med_avg - 0.5:
                    lines.append(
                        f"\n  ⚠ CALIBRATION WARNING: Your HIGH confidence picks have returned "
                        f"{diff:.2f}% less than MEDIUM on average. "
                        f"You are overconfident — assign HIGH only when there is strong "
                        f"multi-factor evidence, and default to MEDIUM when uncertain."
                    )
                elif high_avg > med_avg + 0.5:
                    lines.append(
                        f"\n  ✓ Calibration healthy: HIGH confidence outperforms MEDIUM by {diff:.2f}% avg. "
                        f"Continue current confidence assessment approach."
                    )
                else:
                    lines.append(
                        f"\n  ~ Confidence levels performing similarly ({diff:.2f}% gap). "
                        f"Ensure HIGH is reserved for your strongest convictions only."
                    )

        # ── 4. Drawdown / portfolio health ────────────────
        most_recent = history[0]
        dd = most_recent.get("drawdown_pct", 0)
        pv = most_recent.get("portfolio_value")
        week = most_recent.get("week_ending", "?")

        lines.append(f"\nPortfolio status (as of {week}):")
        if pv:
            lines.append(f"  Value: ${pv:,.2f}")
        if dd > 5:
            lines.append(
                f"  ⚠ Drawdown: {dd:.1f}% below peak — portfolio under stress. "
                f"Favour SKIP or REDUCE over ACTIVE where signals are ambiguous."
            )
        elif dd > 2:
            lines.append(f"  Drawdown: {dd:.1f}% — moderate. Apply normal risk weighting.")
        else:
            lines.append(f"  Drawdown: {dd:.1f}% — portfolio near peak, healthy.")

        lines.append("\n=== END TRACK RECORD ===")
        return "\n".join(lines)
