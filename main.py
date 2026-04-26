#!/usr/bin/env python3
"""
Automated Multi-Asset Momentum Trading Bot
============================================
Paper trading bot with two-level allocation across 4 asset classes:
  Level 1: Class allocation (ACTIVE / REDUCE / SKIP per class)
  Level 2: Instrument selection within each active class

Layers:
  1. Alpaca connection & market data (equities + crypto)
  2. Momentum strategy (20-day return + RSI) with volatility flagging
  3. News + Claude reasoning (two-stage: class → instrument)
  4. Verification layer (independent audit with volatility checks)

Adaptive daily scheduling:
  - Checks every market day (Mon–Fri) at 9:35 AM ET
  - Only rebalances when momentum leaders change or signals shift meaningfully
  - Drawdown circuit breaker halts trading if portfolio drops too far from peak

CLI Commands:
  python main.py run              Start the daily adaptive scheduler
  python main.py status           Show current positions, state, and last rebalance
  python main.py backtest         Run two-level momentum ranking on recent data
  python main.py logs             Tail the last 20 lines of decisions.log
  python main.py rebalance        Run a single rebalance cycle now (for testing)
  python main.py deposit <amount> Add monthly petty cash and update capital
"""

import json
import os
import re
import sys

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from modules.alpaca_client import AlpacaClient
from modules.momentum import MomentumStrategy
from modules.news_ingestion import NewsIngestion
from modules.claude_reasoning import ClaudeReasoning
from modules.verification import VerificationLayer
from modules.class_allocator import ClassAllocator
from modules.executor import Executor
from modules.telegram_alerts import TelegramAlerts
from modules.logger import BotLogger
from modules.state_manager import StateManager
from modules.weekly_reviewer import WeeklyReviewer
from modules.parameter_optimizer import ParameterOptimizer
from modules.auto_tuner import AutoTuner
from modules.live_readiness import LiveReadinessEvaluator


# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load config from YAML file. Searches in script directory by default."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_components(config: dict):
    """Instantiate all bot components from config."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    logger = BotLogger(config, base_dir=base_dir)
    alpaca = AlpacaClient(config)
    momentum = MomentumStrategy(config, alpaca)
    news = NewsIngestion(config)
    claude = ClaudeReasoning(config)
    allocator = ClassAllocator(config, momentum)
    verification = VerificationLayer(config, momentum_strategy=momentum)
    telegram = TelegramAlerts(config)

    executor = Executor(
        config=config,
        alpaca_client=alpaca,
        momentum=momentum,
        news=news,
        claude_reasoning=claude,
        verification=verification,
        class_allocator=allocator,
        logger=logger,
        telegram_alerts=telegram,
    )

    return {
        "logger": logger,
        "alpaca": alpaca,
        "momentum": momentum,
        "news": news,
        "claude": claude,
        "allocator": allocator,
        "verification": verification,
        "telegram": telegram,
        "executor": executor,
    }


# ─────────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────────

def cmd_run(config: dict):
    """
    Start the adaptive daily scheduler.

    Runs every market day (Mon–Fri):
      - 9:30 AM ET: pre-fetch news + Level 1 momentum
      - 9:35 AM ET: adaptive check (rebalance only if signals have changed)

    The bot will run continuously and only place trades when momentum leaders
    change or signal strength shifts meaningfully — preventing unnecessary churn.
    """
    components = build_components(config)
    executor = components["executor"]
    logger = components["logger"]

    print("\n" + "=" * 60)
    print("TRADING BOT — STARTING UP (Adaptive Daily, Multi-Asset)")
    print("=" * 60)
    components["alpaca"].print_status()

    # Print state summary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    state = StateManager(base_dir)
    state.print_summary()

    global_cfg = config.get("global", {})
    adaptive_cfg = config.get("adaptive", {})
    tz_str = global_cfg.get("timezone", "America/New_York")
    tz = pytz.timezone(tz_str)
    check_time = global_cfg.get("rebalance_time", "09:35")
    hour, minute = check_time.split(":")

    scheduler = BlockingScheduler(timezone=tz)

    # Pre-fetch job: every market day at 9:30 AM ET (5 min before check)
    prefetch_minute = max(0, int(minute) - 5)
    scheduler.add_job(
        executor.prefetch,
        CronTrigger(day_of_week="mon-fri", hour=int(hour), minute=prefetch_minute, timezone=tz),
        id="prefetch",
        name="Daily pre-fetch: news + Level 1 momentum",
        misfire_grace_time=300,
    )

    # Adaptive check job: every market day at 9:35 AM ET
    scheduler.add_job(
        executor.adaptive_check,
        CronTrigger(day_of_week="mon-fri", hour=int(hour), minute=int(minute), timezone=tz),
        id="adaptive_check",
        name="Daily adaptive check (rebalance only if signals changed)",
        misfire_grace_time=300,
    )

    # ── Self-improvement loop ─────────────────────────────

    # Weekly review: every Friday at 4:15 PM ET (after US market close)
    reviewer = WeeklyReviewer(
        config=config,
        alpaca_client=components["alpaca"],
        state_manager=components["executor"].state,
        logger=logger,
        telegram=components["telegram"],
    )
    scheduler.add_job(
        reviewer.run_review,
        CronTrigger(day_of_week="fri", hour=16, minute=15, timezone=tz),
        id="weekly_review",
        name="Weekly performance review + learning record",
        misfire_grace_time=3600,
    )

    # Monthly optimizer + auto-tuner: first Sunday of each month at 8:00 AM ET
    optimizer = ParameterOptimizer(
        config=config,
        state_manager=components["executor"].state,
        logger=logger,
        telegram=components["telegram"],
    )
    tuner = AutoTuner(
        config=config,
        state_manager=components["executor"].state,
        logger=logger,
        telegram=components["telegram"],
    )

    def run_monthly_improvement():
        """Run optimizer then immediately apply any safe improvements."""
        result = optimizer.run()
        if result.get("recommend_change"):
            tuner.apply()

    scheduler.add_job(
        run_monthly_improvement,
        CronTrigger(day="1-7", day_of_week="sun", hour=8, minute=0, timezone=tz),
        id="monthly_optimize",
        name="Monthly parameter optimization + auto-tune",
        misfire_grace_time=3600,
    )

    universe_cfg = config.get("universe", {})
    class_count = len([k for k in universe_cfg if isinstance(universe_cfg.get(k), dict)])
    capital = global_cfg.get("capital", 500)
    momentum_threshold = adaptive_cfg.get("momentum_shift_threshold", 0.03)
    max_drawdown = adaptive_cfg.get("max_drawdown_pct", 10.0)

    logger.log_status(
        f"Adaptive scheduler started. Daily check Mon-Fri at {check_time} {tz_str}",
        {
            "classes": class_count,
            "capital": capital,
            "momentum_threshold_pct": momentum_threshold * 100,
            "max_drawdown_pct": max_drawdown,
        },
    )

    print(f"\nAdaptive scheduler running — checking every market day at {check_time} ET")
    print(f"  Asset classes  : {class_count}")
    print(f"  Capital        : ${capital}")
    print(f"  Rebalance when : holdings change OR momentum shifts >{momentum_threshold*100:.0f}%")
    print(f"  Circuit breaker: hold cash if drawdown >{max_drawdown:.0f}% from peak")
    print(f"\n  Self-improvement loop:")
    print(f"    Weekly review  : every Friday at 4:15 PM ET")
    print(f"    Monthly tune   : first Sunday of month at 8:00 AM ET")
    print("\nPress Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler stopped.")
        logger.log_status("Scheduler stopped by user.")


def cmd_status(config: dict):
    """Show current account status, positions, bot state, and stop-loss levels."""
    components = build_components(config)
    components["alpaca"].print_status()

    # Show persistent bot state
    base_dir = os.path.dirname(os.path.abspath(__file__))
    state = StateManager(base_dir)
    state.print_summary()

    # Show position health (stop-loss proximity)
    from modules.risk_manager import RiskManager
    risk = RiskManager(config, state)
    positions = components["alpaca"].get_positions()
    if positions:
        health = risk.get_position_health(positions)
        if health:
            print("Position Stop-Loss Status:")
            print("-" * 50)
            for h in health:
                flag = "  STOP HIT" if h["stop_triggered"] else ""
                print(
                    f"  {h['symbol']:<8} entry=${h['entry_price']:.4f}  "
                    f"now=${h['current_price']:.4f}  "
                    f"P/L={h['pct_change']:>+.1f}%  "
                    f"stop@${h['stop_loss_level']:.4f}{flag}"
                )
            print()

    # Show last decision from log
    logger = components["logger"]
    lines = logger.tail(5)
    if lines and lines[0].strip() != "No decisions log found.":
        print("\nRecent log entries:")
        print("-" * 60)
        for line in lines:
            try:
                entry = json.loads(line.strip())
                ts = entry.get("timestamp", "?")[:19]
                event = entry.get("event", "?")
                msg = entry.get("message", "")
                action = entry.get("action", "")
                print(f"  [{ts}] {event}: {msg or action}")
            except json.JSONDecodeError:
                print(f"  {line.strip()[:100]}")


def cmd_backtest(config: dict):
    """Run two-level momentum ranking on recent data. No trades placed."""
    components = build_components(config)
    allocator = components["allocator"]
    momentum = components["momentum"]

    print("\n" + "=" * 60)
    print("BACKTEST — Two-Level Momentum Rankings (current snapshot)")
    print("=" * 60)

    # Level 1: Class momentum
    class_rankings = allocator.compute_class_momentum()
    allocator.print_class_rankings(class_rankings)

    # Level 2: Top instruments per class (pure momentum, no Claude)
    print("Per-class instrument rankings (pure momentum, no Claude layer):")
    print("-" * 60)
    for cr in class_rankings:
        key = cr["class_key"]
        label = cr["class_label"]
        instruments = cr.get("instruments", [])
        valid = [i for i in instruments if i.get("momentum") is not None]
        valid.sort(key=lambda x: x["momentum"], reverse=True)

        top_n = allocator.get_top_n(key)
        selected = valid[:top_n]

        if not selected:
            print(f"  {label}: no valid instruments")
            continue

        for i, inst in enumerate(selected, 1):
            print(
                f"  {label} #{i}: {inst['symbol']:<8} "
                f"mom={inst.get('momentum_pct', 0):>+7.2f}% "
                f"@ ${inst.get('current_price', 0):>10.2f}"
            )

    # Volatility check
    print("\nVolatility check (all instruments):")
    print("-" * 60)
    all_instruments = []
    for cr in class_rankings:
        for inst in cr.get("instruments", []):
            if inst.get("momentum") is not None:
                all_instruments.append(inst)

    verification = components["verification"]
    vol_flags = verification.compute_volatility_flags(all_instruments)
    if vol_flags:
        for vf in vol_flags:
            print(f"  [FLAG] {vf['message']}")
    else:
        print("  No volatility flags — all instruments within normal range.")

    print(f"\nNote: This shows current snapshot rankings. Full rebalance includes Claude reasoning + verification.")
    print("Run 'python main.py rebalance' to execute the full pipeline.\n")


def cmd_logs(config: dict):
    """Tail the last 20 lines of decisions.log."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logger = BotLogger(config, base_dir=base_dir)
    lines = logger.tail(20)

    print("\n" + "=" * 60)
    print("DECISIONS LOG — Last 20 entries")
    print("=" * 60)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp", "?")[:19]
            event = entry.get("event", "?")
            # Compact display
            if event == "decision":
                action = entry.get("action", "?")
                instruments = entry.get("instruments", [])
                symbols = [s.get("symbol", "?") for s in instruments]
                verdict = entry.get("verification", {}).get("verdict", "?")
                print(f"  [{ts}] DECISION: action={action}, instruments={symbols}, verdict={verdict}")
            elif event == "order":
                symbol = entry.get("symbol", "?")
                side = entry.get("side", "?")
                qty = entry.get("qty", "?")
                print(f"  [{ts}] ORDER: {side} {qty} {symbol}")
            elif event == "verification":
                verdict = entry.get("verdict", "?")
                notes = (entry.get("notes") or "")[:80]
                print(f"  [{ts}] VERIFY: {verdict} — {notes}")
            elif event == "status":
                msg = entry.get("message", "")
                print(f"  [{ts}] STATUS: {msg}")
            elif event == "error":
                msg = entry.get("message", "")
                print(f"  [{ts}] ERROR: {msg}")
            else:
                print(f"  [{ts}] {event}: {json.dumps(entry)[:100]}")
        except json.JSONDecodeError:
            print(f"  {line[:120]}")

    print("=" * 60 + "\n")


def cmd_rebalance(config: dict):
    """Run a single rebalance cycle immediately (for testing)."""
    components = build_components(config)
    components["alpaca"].print_status()

    print("\n[MANUAL REBALANCE] Running full two-level cycle now...\n")
    result = components["executor"].run_rebalance()

    print("\n" + "=" * 60)
    print("REBALANCE COMPLETE")
    print("=" * 60)
    action = result.get("action", "unknown")
    print(f"  Action: {action}")

    class_allocs = result.get("class_allocations", [])
    if class_allocs:
        print("  Class Allocations:")
        for a in class_allocs:
            print(f"    {a['class_label']:<18}: {a['decision']:<8} weight={a['weight']:.0%}  ${a['allocated_capital']:.2f}")

    inst_sels = result.get("instrument_selections", [])
    if inst_sels:
        print(f"  Instruments: {', '.join(s['symbol'] for s in inst_sels)}")

    verdict = result.get("verification", {}).get("verdict", "N/A")
    print(f"  Verification: {verdict}")
    print("=" * 60 + "\n")


def cmd_backtest_historical(config: dict):
    """
    Run the historical backtest simulation.
    Uses Yahoo Finance (free) — no Alpaca connection needed.

    Usage: python main.py backtest-historical [years]
    Example: python main.py backtest-historical 2
    """
    from backtesting.backtest import run_backtest

    years = 2
    if len(sys.argv) >= 3:
        try:
            years = int(sys.argv[2])
        except ValueError:
            pass

    run_backtest(config, period_years=years)


def cmd_review(config: dict):
    """Run the weekly performance review immediately (for testing or on-demand)."""
    components = build_components(config)
    reviewer = WeeklyReviewer(
        config=config,
        alpaca_client=components["alpaca"],
        state_manager=components["executor"].state,
        logger=components["logger"],
        telegram=components["telegram"],
    )
    reviewer.run_review()


def cmd_optimize(config: dict):
    """Run the monthly parameter optimizer immediately and apply results."""
    components = build_components(config)
    state = components["executor"].state
    logger = components["logger"]
    telegram = components["telegram"]

    optimizer = ParameterOptimizer(config=config, state_manager=state, logger=logger, telegram=telegram)
    result = optimizer.run()

    if result.get("recommend_change"):
        tuner = AutoTuner(config=config, state_manager=state, logger=logger, telegram=telegram)
        tuner.apply()
    else:
        print("Optimization complete — current parameters are already optimal (or within 10% of best).")


def cmd_restore_params(config: dict):
    """Undo the last auto-tuner change and restore previous config.yaml parameters."""
    components = build_components(config)
    state = components["executor"].state
    logger = components["logger"]
    tuner = AutoTuner(config=config, state_manager=state, logger=logger)

    restored = tuner.restore()
    if restored:
        print("Parameters restored. Restart the bot for changes to take effect.")
    else:
        print("Nothing to restore — no previous parameter history found.")


def cmd_check_readiness(config: dict):
    """
    Evaluate whether the bot is ready to trade with real money.

    Reads the weekly review history from state.db and scores 7 criteria:
      run time, win rate, avg return, max drawdown, consistency,
      confidence calibration, and stability.

    Sends the report to Telegram and prints it to the console.
    """
    components = build_components(config)
    state = components["executor"].state
    telegram = components["telegram"]

    evaluator = LiveReadinessEvaluator(
        config=config,
        state_manager=state,
        telegram=telegram,
    )
    evaluator.evaluate()


def cmd_deposit(config: dict):
    """
    Record a monthly petty cash deposit and update the bot's capital.

    Usage: python main.py deposit <amount>
    Example: python main.py deposit 300

    This updates config.yaml and logs the deposit to state.json so the bot
    uses the new total capital on the next rebalance cycle.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py deposit <amount>")
        print("Example: python main.py deposit 300")
        sys.exit(1)

    try:
        amount = float(sys.argv[2])
    except ValueError:
        print(f"Error: '{sys.argv[2]}' is not a valid number.")
        sys.exit(1)

    if amount <= 0:
        print("Error: deposit amount must be positive.")
        sys.exit(1)

    # Load current capital from config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    global_cfg = config.get("global", {})
    old_capital = float(global_cfg.get("capital", 500))
    new_capital = round(old_capital + amount, 2)

    # Update config.yaml (replace the capital value in the global section)
    with open(config_path, "r") as f:
        config_text = f.read()

    # Replace 'capital: <number>' under [global] section
    config_text = re.sub(
        r"(capital:\s*)\d+(\.\d+)?",
        f"capital: {new_capital}",
        config_text,
    )

    with open(config_path, "w") as f:
        f.write(config_text)

    # Record deposit in state manager
    state = StateManager(script_dir)
    state.record_deposit(amount, new_capital)

    print("\n" + "=" * 50)
    print("DEPOSIT RECORDED")
    print("=" * 50)
    print(f"  Deposited      : +${amount:.2f}")
    print(f"  Previous capital: ${old_capital:.2f}")
    print(f"  New capital    : ${new_capital:.2f}")
    print(f"  Total deposited: ${state.get_total_deposited():.2f}")
    print("\nConfig updated. The next rebalance will deploy the new capital.")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────

COMMANDS = {
    "run": cmd_run,
    "status": cmd_status,
    "backtest": cmd_backtest,
    "backtest-historical": cmd_backtest_historical,
    "logs": cmd_logs,
    "rebalance": cmd_rebalance,
    "deposit": cmd_deposit,
    "review": cmd_review,
    "optimize": cmd_optimize,
    "restore_params": cmd_restore_params,
    "check-readiness": cmd_check_readiness,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py <command>")
        print()
        print("Commands:")
        print("  run               Start the adaptive daily scheduler (Mon-Fri)")
        print("  status            Show account, positions, and bot state")
        print("  backtest          Run momentum ranking snapshot, no trades")
        print("  backtest-historical [years]  Run historical simulation (default: 2yr)")
        print("  logs              Tail the last 20 lines of decisions.log")
        print("  rebalance         Run a single full rebalance cycle now (testing)")
        print("  deposit <amount>  Record a monthly deposit and update capital")
        print()
        print("  Self-improvement loop:")
        print("  review            Run the weekly performance review now")
        print("  optimize          Run monthly parameter sweep + auto-tune now")
        print("  restore_params    Undo last auto-tune (revert config.yaml params)")
        print("  check-readiness   Evaluate whether the bot is ready for real money")
        print()
        print("Examples:")
        print("  python main.py run")
        print("  python main.py deposit 300")
        print("  python main.py review")
        print("  python main.py optimize")
        sys.exit(1)

    command = sys.argv[1]
    config = load_config()
    COMMANDS[command](config)


if __name__ == "__main__":
    main()
