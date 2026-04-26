"""
Live-Readiness Evaluator — answers "is it time to trade real money?"

Runs automatically every Friday after the weekly review, and on-demand
via: python main.py check-readiness

How it works:
  Reads the last N weekly review records from state.db and scores 7 criteria.
  Each criterion gets PASS / WARN / FAIL. The verdict is:
    READY        — all PASS, or at most 1 WARN
    APPROACHING  — 2–3 criteria WARN, no FAIL
    NOT READY    — any FAIL, or 4+ WARNs

Criteria scored:
  1. Run time      — ≥ MIN_DAYS_PAPER days of paper trading
  2. Win rate      — ≥ WIN_RATE_PCT% across recent positions
  3. Avg return    — ≥ AVG_RETURN_PCT% per rebalance cycle
  4. Drawdown      — peak drawdown never exceeded MAX_DRAWDOWN_PCT%
  5. Consistency   — profitable in ≥ CONSISTENCY_WEEKS of last 6 weekly reviews
  6. Calibration   — HIGH-confidence picks outperform MEDIUM on avg
  7. Stability     — no week with avg return < WORST_WEEK_PCT%

Thresholds live in config.yaml under 'live_readiness' (with sensible defaults).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple


# ── Criterion result ───────────────────────────────────────────────────────────

class Criterion(NamedTuple):
    name: str
    status: str          # "PASS" | "WARN" | "FAIL"
    detail: str          # human-readable explanation
    value: float | None  # the measured value (for display)
    threshold: float     # the target threshold


# ── Verdict constants ──────────────────────────────────────────────────────────

READY       = "READY"
APPROACHING = "APPROACHING"
NOT_READY   = "NOT READY"


class LiveReadinessEvaluator:
    """Evaluates paper-trading history against go-live thresholds."""

    def __init__(self, config: dict, state_manager, telegram=None):
        self.state = state_manager
        self.telegram = telegram

        cfg = config.get("live_readiness", {})
        self.min_days_paper      = cfg.get("min_days_paper", 30)
        self.min_weeks_data      = cfg.get("min_weeks_data", 4)
        self.win_rate_pct        = cfg.get("win_rate_pct", 60.0)
        self.avg_return_pct      = cfg.get("avg_return_pct", 0.20)
        self.max_drawdown_pct    = cfg.get("max_drawdown_pct", 5.0)
        self.consistency_weeks   = cfg.get("consistency_weeks", 4)   # out of last 6
        self.worst_week_pct      = cfg.get("worst_week_pct", -3.0)   # single-week floor
        self.notify_on_ready     = cfg.get("notify_on_ready", True)
        self.notify_weekly       = cfg.get("notify_weekly", True)

    # ── Main entry point ───────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Run the full readiness evaluation. Returns a dict with:
          verdict, criteria, summary, weeks_evaluated, first_record_date
        """
        history = self.state.get_learning_history(n=12)  # up to 12 weeks

        if not history:
            return self._no_data_result()

        # Sort oldest → newest (get_learning_history returns newest first)
        history = list(reversed(history))
        weeks_evaluated = len(history)

        # Estimate run duration from first record
        first_date = history[0].get("week_ending", "")
        last_date  = history[-1].get("week_ending", "")
        days_running = self._days_between(first_date, last_date) if first_date and last_date else 0

        criteria = [
            self._check_run_time(days_running),
            self._check_win_rate(history),
            self._check_avg_return(history),
            self._check_drawdown(history),
            self._check_consistency(history),
            self._check_calibration(history),
            self._check_stability(history),
        ]

        verdict = self._compute_verdict(criteria, weeks_evaluated)

        result = {
            "verdict": verdict,
            "criteria": [c._asdict() for c in criteria],
            "weeks_evaluated": weeks_evaluated,
            "days_running": days_running,
            "first_record_date": first_date,
            "last_record_date": last_date,
            "summary": self._build_summary(verdict, criteria, days_running),
        }

        self._print_report(result)

        if self.telegram and self.notify_weekly:
            self._send_telegram(result)

        return result

    # ── Individual criteria ────────────────────────────────────────────────────

    def _check_run_time(self, days_running: int) -> Criterion:
        threshold = self.min_days_paper
        if days_running >= threshold:
            return Criterion("Run time", "PASS",
                             f"{days_running} days of paper trading (need ≥{threshold})",
                             days_running, threshold)
        elif days_running >= threshold * 0.6:
            return Criterion("Run time", "WARN",
                             f"Only {days_running} days so far — need ≥{threshold} before going live",
                             days_running, threshold)
        else:
            return Criterion("Run time", "FAIL",
                             f"Only {days_running} days — far too early to consider live trading",
                             days_running, threshold)

    def _check_win_rate(self, history: list[dict]) -> Criterion:
        threshold = self.win_rate_pct
        rates = [r["overall_win_rate"] for r in history if r.get("overall_win_rate") is not None]
        if not rates:
            return Criterion("Win rate", "WARN", "No win rate data yet", None, threshold)
        avg_win_rate = sum(rates) / len(rates)
        if avg_win_rate >= threshold:
            return Criterion("Win rate", "PASS",
                             f"Avg {avg_win_rate:.1f}% across {len(rates)} weeks (need ≥{threshold}%)",
                             avg_win_rate, threshold)
        elif avg_win_rate >= threshold * 0.85:
            return Criterion("Win rate", "WARN",
                             f"Avg {avg_win_rate:.1f}% — close but below {threshold}% target",
                             avg_win_rate, threshold)
        else:
            return Criterion("Win rate", "FAIL",
                             f"Avg {avg_win_rate:.1f}% — well below the {threshold}% minimum",
                             avg_win_rate, threshold)

    def _check_avg_return(self, history: list[dict]) -> Criterion:
        threshold = self.avg_return_pct
        returns = [r["avg_return_pct"] for r in history if r.get("avg_return_pct") is not None]
        if not returns:
            return Criterion("Avg return", "WARN", "No return data yet", None, threshold)
        avg_ret = sum(returns) / len(returns)
        if avg_ret >= threshold:
            return Criterion("Avg return", "PASS",
                             f"Avg {avg_ret:+.2f}% per cycle (need ≥{threshold:+.2f}%)",
                             avg_ret, threshold)
        elif avg_ret >= 0:
            return Criterion("Avg return", "WARN",
                             f"Avg {avg_ret:+.2f}% — positive but below {threshold:+.2f}% target",
                             avg_ret, threshold)
        else:
            return Criterion("Avg return", "FAIL",
                             f"Avg {avg_ret:+.2f}% — negative average return",
                             avg_ret, threshold)

    def _check_drawdown(self, history: list[dict]) -> Criterion:
        threshold = self.max_drawdown_pct
        drawdowns = [r.get("drawdown_pct", 0.0) for r in history]
        peak_drawdown = max(drawdowns) if drawdowns else 0.0
        if peak_drawdown <= threshold:
            return Criterion("Max drawdown", "PASS",
                             f"Peak drawdown {peak_drawdown:.1f}% (limit: {threshold}%)",
                             peak_drawdown, threshold)
        elif peak_drawdown <= threshold * 1.5:
            return Criterion("Max drawdown", "WARN",
                             f"Peak drawdown {peak_drawdown:.1f}% exceeded {threshold}% limit",
                             peak_drawdown, threshold)
        else:
            return Criterion("Max drawdown", "FAIL",
                             f"Peak drawdown {peak_drawdown:.1f}% — significantly beyond {threshold}% limit",
                             peak_drawdown, threshold)

    def _check_consistency(self, history: list[dict]) -> Criterion:
        required = self.consistency_weeks
        # Look at last 6 weeks
        recent = history[-6:] if len(history) >= 6 else history
        profitable_weeks = sum(
            1 for r in recent
            if r.get("avg_return_pct") is not None and r["avg_return_pct"] > 0
        )
        total = len(recent)
        label = f"{profitable_weeks}/{total} profitable weeks"
        if profitable_weeks >= required:
            return Criterion("Consistency", "PASS",
                             f"{label} (need ≥{required}/{min(6, total)})",
                             profitable_weeks, required)
        elif profitable_weeks >= required - 1:
            return Criterion("Consistency", "WARN",
                             f"{label} — one short of the {required}/{min(6, total)} target",
                             profitable_weeks, required)
        else:
            return Criterion("Consistency", "FAIL",
                             f"{label} — too many losing weeks to go live",
                             profitable_weeks, required)

    def _check_calibration(self, history: list[dict]) -> Criterion:
        """HIGH-confidence picks should outperform MEDIUM ones on average."""
        high_returns, medium_returns = [], []
        for r in history:
            conf = r.get("confidence", {})
            if "HIGH" in conf and conf["HIGH"].get("avg_return_pct") is not None:
                high_returns.append(conf["HIGH"]["avg_return_pct"])
            if "MEDIUM" in conf and conf["MEDIUM"].get("avg_return_pct") is not None:
                medium_returns.append(conf["MEDIUM"]["avg_return_pct"])

        if not high_returns or not medium_returns:
            return Criterion("Confidence calibration", "WARN",
                             "Insufficient data to assess calibration",
                             None, 0.0)

        avg_high   = sum(high_returns) / len(high_returns)
        avg_medium = sum(medium_returns) / len(medium_returns)
        gap = avg_high - avg_medium

        if gap >= 0.1:
            return Criterion("Confidence calibration", "PASS",
                             f"HIGH ({avg_high:+.2f}%) outperforms MEDIUM ({avg_medium:+.2f}%) — calibration working",
                             gap, 0.1)
        elif gap >= -0.1:
            return Criterion("Confidence calibration", "WARN",
                             f"HIGH ({avg_high:+.2f}%) ≈ MEDIUM ({avg_medium:+.2f}%) — calibration unclear",
                             gap, 0.1)
        else:
            return Criterion("Confidence calibration", "FAIL",
                             f"HIGH ({avg_high:+.2f}%) underperforms MEDIUM ({avg_medium:+.2f}%) — confidence scoring unreliable",
                             gap, 0.1)

    def _check_stability(self, history: list[dict]) -> Criterion:
        """No single week should be a catastrophic outlier."""
        threshold = self.worst_week_pct
        returns = [r["avg_return_pct"] for r in history if r.get("avg_return_pct") is not None]
        if not returns:
            return Criterion("Stability", "WARN", "No return data yet", None, threshold)
        worst = min(returns)
        if worst >= threshold:
            return Criterion("Stability", "PASS",
                             f"Worst week: {worst:+.2f}% (floor: {threshold:+.2f}%)",
                             worst, threshold)
        elif worst >= threshold * 1.5:
            return Criterion("Stability", "WARN",
                             f"Worst week {worst:+.2f}% dipped below {threshold:+.2f}% floor",
                             worst, threshold)
        else:
            return Criterion("Stability", "FAIL",
                             f"Worst week {worst:+.2f}% — a severe outlier that needs explaining",
                             worst, threshold)

    # ── Verdict ────────────────────────────────────────────────────────────────

    def _compute_verdict(self, criteria: list[Criterion], weeks: int) -> str:
        if weeks < self.min_weeks_data:
            return NOT_READY
        fails = sum(1 for c in criteria if c.status == "FAIL")
        warns = sum(1 for c in criteria if c.status == "WARN")
        if fails == 0 and warns <= 1:
            return READY
        elif fails == 0 and warns <= 3:
            return APPROACHING
        else:
            return NOT_READY

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _days_between(self, date_a: str, date_b: str) -> int:
        try:
            fmt = "%Y-%m-%d"
            d1 = datetime.strptime(date_a[:10], fmt)
            d2 = datetime.strptime(date_b[:10], fmt)
            return abs((d2 - d1).days) + 7   # +7 for the current week
        except ValueError:
            return 0

    def _build_summary(self, verdict: str, criteria: list[Criterion], days: int) -> str:
        fails  = [c for c in criteria if c.status == "FAIL"]
        warns  = [c for c in criteria if c.status == "WARN"]
        passes = [c for c in criteria if c.status == "PASS"]

        if verdict == READY:
            return (
                f"Bot has been paper trading for ~{days} days and meets all go-live criteria. "
                f"All {len(passes)} criteria passed. "
                f"Consider starting with a small real-money allocation and scaling up over 4 weeks."
            )
        elif verdict == APPROACHING:
            warn_names = ", ".join(c.name for c in warns)
            return (
                f"Almost there — {len(passes)} criteria pass, but {len(warns)} still need improvement: "
                f"{warn_names}. Continue paper trading and re-assess in 2–4 weeks."
            )
        else:
            if fails:
                fail_names = ", ".join(c.name for c in fails)
                return (
                    f"Not ready — {len(fails)} criteria failed: {fail_names}. "
                    f"Do not trade real money until these are resolved."
                )
            return (
                f"Not ready — only {days} days of paper trading data available. "
                f"Need at least {self.min_days_paper} days and {self.min_weeks_data} weekly reviews."
            )

    def _no_data_result(self) -> dict:
        return {
            "verdict": NOT_READY,
            "criteria": [],
            "weeks_evaluated": 0,
            "days_running": 0,
            "first_record_date": None,
            "last_record_date": None,
            "summary": (
                "No weekly review records found in state.db. "
                "The bot needs to run for at least several weeks before a readiness assessment is possible."
            ),
        }

    # ── Console report ─────────────────────────────────────────────────────────

    def _print_report(self, result: dict):
        verdict = result["verdict"]
        emoji = {"READY": "🟢", "APPROACHING": "🟡", "NOT READY": "🔴"}.get(verdict, "⚪")

        print("\n" + "=" * 65)
        print(f"  LIVE-READINESS ASSESSMENT  {emoji} {verdict}")
        print("=" * 65)
        print(f"  Paper trading duration : ~{result['days_running']} days")
        print(f"  Weekly records used    : {result['weeks_evaluated']}")
        print(f"  Date range             : {result['first_record_date']} → {result['last_record_date']}")
        print()

        status_icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}
        for c in result.get("criteria", []):
            icon = status_icon.get(c["status"], "  ")
            print(f"  {icon} {c['name']:<26} {c['detail']}")

        print()
        print(f"  Verdict: {result['summary']}")
        print("=" * 65 + "\n")

    # ── Telegram notification ──────────────────────────────────────────────────

    def _send_telegram(self, result: dict):
        verdict  = result["verdict"]
        days     = result["days_running"]
        weeks    = result["weeks_evaluated"]
        emoji    = {"READY": "🟢", "APPROACHING": "🟡", "NOT READY": "🔴"}.get(verdict, "⚪")

        status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}

        lines = [
            f"{emoji} <b>LIVE-READINESS REPORT</b>",
            f"<i>~{days} days paper trading | {weeks} weekly reviews</i>",
            "",
            f"<b>Verdict: {verdict}</b>",
            "",
            "<b>— Criteria —</b>",
        ]

        for c in result.get("criteria", []):
            icon = status_icon.get(c["status"], "•")
            lines.append(f"  {icon} <b>{c['name']}</b>: {c['detail']}")

        lines += [
            "",
            f"<b>Summary:</b> {result['summary']}",
        ]

        if verdict == READY:
            lines += [
                "",
                "💡 <b>Suggested next step:</b>",
                "Start with 10–20% of your intended live capital for 2 weeks, "
                "then scale up if results remain consistent.",
            ]

        try:
            self.telegram.send_message("\n".join(lines))
        except Exception as e:
            print(f"  [READINESS] Telegram send failed: {e}")
