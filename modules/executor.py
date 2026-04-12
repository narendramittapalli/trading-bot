"""
Executor — Orchestrates the full two-level rebalance pipeline.
Level 1: Class allocation (which asset classes to invest in)
Level 2: Instrument selection (which instruments within each active class)
Then: verification, execution, logging.

Also provides adaptive_check() for daily smart scheduling:
- Skips rebalance if holdings and signals are unchanged
- Triggers drawdown circuit breaker if portfolio falls too far
"""

import time
from datetime import datetime, timezone

from modules.momentum import is_crypto
from modules.state_manager import StateManager
from modules.risk_manager import RiskManager


class Executor:
    """Runs the full two-level rebalance cycle."""

    def __init__(
        self,
        config: dict,
        alpaca_client,
        momentum,
        news,
        claude_reasoning,
        verification,
        class_allocator,
        logger,
        telegram_alerts=None,
    ):
        self.config = config
        self.alpaca = alpaca_client
        self.momentum = momentum
        self.news = news
        self.claude = claude_reasoning
        self.verification = verification
        self.allocator = class_allocator
        self.logger = logger
        self.telegram = telegram_alerts
        global_cfg = config.get("global", {})
        self.capital = global_cfg.get("capital", config.get("capital", 500))

        # State manager (persists across sessions)
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)  # trading-bot/
        self.state = StateManager(parent_dir)
        self.risk = RiskManager(config, self.state)

    def run_rebalance(self) -> dict:
        """
        Execute the full two-level rebalance pipeline:
        1. Compute class-level momentum (Level 1)
        2. Fetch and filter news
        3. Claude Stage 1 — class allocation decisions
        4. Apply class allocations and compute capital distribution
        5. Claude Stage 2 — instrument selection per active class
        6. Build final instrument selections and proposed trades
        7. Compute volatility flags
        8. Run verification layer
        9. Execute or hold based on verdict
        """
        result = {"timestamp": datetime.now(timezone.utc).isoformat()}

        # --- Pre-trade snapshot ---
        try:
            pre_account = self.alpaca.get_account()
            result["pre_portfolio_value"] = pre_account["portfolio_value"]
        except Exception as e:
            self.logger.log_error(f"Failed to get pre-trade account info: {e}")
            result["pre_portfolio_value"] = None

        # --- Step 1: Class-level momentum ---
        print("\n[STEP 1] Computing class-level momentum rankings...")
        try:
            class_rankings = self.allocator.compute_class_momentum()
            self.allocator.print_class_rankings(class_rankings)
            result["class_rankings"] = [
                {
                    "class_key": cr["class_key"],
                    "class_label": cr["class_label"],
                    "avg_momentum_pct": cr["avg_momentum_pct"],
                    "rank": cr["rank"],
                }
                for cr in class_rankings
            ]
        except Exception as e:
            self.logger.log_error(f"Class momentum ranking failed: {e}")
            return {"error": str(e), "stage": "class_momentum"}

        # --- Step 2: News ingestion ---
        print("[STEP 2] Fetching news digest...")
        try:
            news_digest = self.news.get_headline_digest()
            self.news.print_digest(news_digest)
            result["news_summary"] = {
                "total": news_digest["total_fetched"],
                "relevant": news_digest["relevant_count"],
            }
        except Exception as e:
            self.logger.log_error(f"News ingestion failed: {e}")
            news_digest = {"digest_text": "News unavailable.", "relevant_count": 0, "headlines": []}

        # --- Step 3: Claude Stage 1 — class allocation ---
        print("[STEP 3] Running Claude Stage 1 — class allocation...")
        try:
            class_assessment = self.claude.assess_classes(class_rankings, news_digest)
            self.claude.print_class_assessment(class_assessment)
            claude_decisions = class_assessment.get("decisions", {})
            result["class_assessment"] = {
                "decisions": claude_decisions,
                "macro_summary": class_assessment.get("macro_summary", ""),
                "fallback": class_assessment.get("fallback", False),
            }
        except Exception as e:
            self.logger.log_error(f"Claude Stage 1 failed: {e}")
            claude_decisions = {}
            class_assessment = {"decisions": {}, "macro_summary": "", "fallback": True}
            for cr in class_rankings:
                claude_decisions[cr["class_key"]] = {"decision": "ACTIVE", "reason": "Fallback — Claude unavailable."}

        # --- Step 3.5: Market regime check ---
        print("[STEP 3.5] Checking market regime (SPY momentum)...")
        try:
            market_regime = self.allocator.get_market_regime()
            regime = market_regime["regime"]
            spy_mom = market_regime["spy_momentum_pct"]
            scale = market_regime["scale_factor"]
            print(f"  Regime: {regime.upper()}  |  SPY 20d momentum: {spy_mom:+.2f}%  |  Allocation scale: {scale:.0%}")
            result["market_regime"] = market_regime
        except Exception as e:
            self.logger.log_error(f"Regime check failed: {e}")
            market_regime = {"regime": "neutral", "scale_factor": 1.0, "spy_momentum_pct": 0}

        # --- Step 4: Apply class allocations ---
        print("[STEP 4] Applying class allocation decisions...")
        class_allocations, cash_weight, cash_capital = self.allocator.apply_class_decisions(
            class_rankings, claude_decisions, market_regime=market_regime
        )
        self.allocator.print_allocations(class_allocations, cash_weight, cash_capital)
        result["class_allocations"] = class_allocations
        result["cash"] = {"weight": cash_weight, "capital": cash_capital}

        # Check if everything is SKIP (full cash hold)
        active_classes = [a for a in class_allocations if a["decision"] != "SKIP" and a["allocated_capital"] > 0]
        if not active_classes:
            print("[INFO] All classes set to SKIP. Holding full cash position.")
            result["action"] = "hold_cash"
            result["reason"] = "All asset classes set to SKIP by Claude reasoning."
            self.logger.log_decision(result)
            if self.telegram:
                self.telegram.send_cash_hold_alert(result)
            return result

        # --- Step 5: Claude Stage 2 — instrument selection per class ---
        print("[STEP 5] Running Claude Stage 2 — instrument selection...")
        # First get momentum-ranked instruments per class
        instrument_selections = []
        class_instrument_rankings = {}
        for cr in class_rankings:
            class_instrument_rankings[cr["class_key"]] = cr.get("instruments", [])

        for alloc in class_allocations:
            key = alloc["class_key"]
            if alloc["decision"] == "SKIP" or alloc["allocated_capital"] <= 0:
                continue

            instrument_rankings = class_instrument_rankings.get(key, [])
            top_n = self.allocator.get_top_n(key)

            try:
                stage2_result = self.claude.assess_instruments(
                    class_key=key,
                    class_label=alloc["class_label"],
                    decision=alloc["decision"],
                    instrument_rankings=instrument_rankings,
                    news_digest=news_digest,
                    top_n=top_n,
                )
                self.claude.print_instrument_assessment(alloc["class_label"], stage2_result)

                # Map Claude's selections to instrument details
                selected_symbols = {s["symbol"] for s in stage2_result.get("selected", [])}
                claude_notes = {s["symbol"]: s for s in stage2_result.get("selected", [])}

            except Exception as e:
                self.logger.log_error(f"Claude Stage 2 failed for {key}: {e}")
                selected_symbols = set()
                claude_notes = {}

            # Fall back to pure momentum if Claude didn't select
            valid_instruments = [i for i in instrument_rankings if i.get("momentum") is not None]
            valid_instruments.sort(key=lambda x: x["momentum"], reverse=True)

            if not selected_symbols:
                # Pure momentum fallback
                chosen = valid_instruments[:top_n]
            else:
                # Use Claude's selection, ordered by momentum
                chosen = [i for i in valid_instruments if i["symbol"] in selected_symbols][:top_n]
                # If Claude picked instruments not in valid list, fall back
                if not chosen:
                    chosen = valid_instruments[:top_n]

            # Compute per-instrument capital allocation
            class_capital = alloc["allocated_capital"]
            max_instrument_capital = self.capital * self.allocator.max_single_instrument

            if chosen:
                per_instrument = class_capital / len(chosen)
                for inst in chosen:
                    capped_capital = min(per_instrument, max_instrument_capital)
                    price = inst.get("current_price", 0)
                    shares = round(capped_capital / price, 6) if price and price > 0 else 0

                    note_info = claude_notes.get(inst["symbol"], {})
                    instrument_selections.append({
                        "symbol": inst["symbol"],
                        "class_key": key,
                        "class_label": alloc["class_label"],
                        "allocated_capital": round(capped_capital, 2),
                        "allocation_pct": round(capped_capital / self.capital, 4) if self.capital > 0 else 0,
                        "target_shares": shares,
                        "current_price": price,
                        "momentum_pct": inst.get("momentum_pct", 0),
                        "momentum": inst.get("momentum", 0),
                        "class_decision": alloc["decision"],
                        "confidence": note_info.get("confidence", "MEDIUM"),
                        "claude_note": note_info.get("note", ""),
                    })

        self.allocator.print_instrument_selections(instrument_selections)
        result["instrument_selections"] = instrument_selections

        if not instrument_selections:
            print("[INFO] No instruments selected. Holding cash.")
            result["action"] = "hold_cash"
            result["reason"] = "No instruments selected after two-level analysis."
            self.logger.log_decision(result)
            if self.telegram:
                self.telegram.send_cash_hold_alert(result)
            return result

        # --- Step 6: Build proposed trades ---
        print("[STEP 6] Building proposed trades...")
        proposed_trades = []
        for sel in instrument_selections:
            proposed_trades.append({
                "symbol": sel["symbol"],
                "side": "buy",
                "qty": sel["target_shares"],
                "estimated_price": sel["current_price"],
                "allocated_capital": sel["allocated_capital"],
                "momentum_pct": sel["momentum_pct"],
                "class_key": sel["class_key"],
                "class_label": sel["class_label"],
                "is_crypto": is_crypto(sel["symbol"]),
            })
        result["proposed_trades"] = proposed_trades

        # --- Step 7: Volatility flags ---
        print("[STEP 7] Computing volatility flags...")
        try:
            volatility_flags = self.verification.compute_volatility_flags(instrument_selections)
            result["volatility_flags"] = volatility_flags
            if volatility_flags:
                print(f"  {len(volatility_flags)} instrument(s) flagged for elevated volatility:")
                for vf in volatility_flags:
                    print(f"    {vf['message']}")
            else:
                print("  No volatility flags — all instruments within normal range.")
        except Exception as e:
            self.logger.log_error(f"Volatility flagging failed: {e}")
            volatility_flags = []

        # --- Step 8: Verification ---
        print("[STEP 8] Running verification layer...")
        try:
            verification_result = self.verification.verify(
                class_allocation=class_assessment,
                instrument_selections=instrument_selections,
                proposed_trades=proposed_trades,
                volatility_flags=volatility_flags,
                news_digest=news_digest,
                capital=self.capital,
            )
            result["verification"] = {
                "verdict": verification_result.get("verdict"),
                "notes": verification_result.get("notes"),
                "rejection_reason": verification_result.get("rejection_reason"),
                "checks": verification_result.get("checks"),
                "volatility_flags": volatility_flags,
            }
            self.logger.log_verification(verification_result)
        except Exception as e:
            self.logger.log_error(f"Verification failed: {e}")
            verification_result = {
                "verdict": "REJECTED",
                "rejection_reason": f"Verification error: {e}",
                "notes": str(e),
            }
            result["verification"] = verification_result

        # --- Step 9: Execute or Hold ---
        verdict = verification_result.get("verdict", "REJECTED")

        if verdict in ("APPROVED", "APPROVED WITH NOTES"):
            print(f"\n[STEP 9] Verification: {verdict} — Executing trades...")
            result["action"] = "execute"
            execution_results = self._execute_trades(proposed_trades)
            result["execution"] = execution_results
        else:
            reason = verification_result.get("rejection_reason", "Unknown")
            print(f"\n[STEP 9] Verification: REJECTED — Holding positions.")
            print(f"  Reason: {reason}")
            result["action"] = "hold"

        # --- Post-trade snapshot ---
        try:
            post_account = self.alpaca.get_account()
            result["post_portfolio_value"] = post_account["portfolio_value"]
        except Exception:
            result["post_portfolio_value"] = None

        # --- Log everything ---
        self.logger.log_decision(result)

        # --- Log predictions for tracking ---
        for sel in instrument_selections:
            self.logger.log_prediction({
                "symbol": sel["symbol"],
                "class_key": sel["class_key"],
                "class_label": sel["class_label"],
                "momentum_pct": sel.get("momentum_pct"),
                "confidence": sel.get("confidence", "MEDIUM"),
                "class_decision": sel.get("class_decision"),
                "verification_verdict": verdict,
                "action_taken": result["action"],
            })

        # --- Email alert ---
        if self.telegram:
            self.telegram.send_rebalance_alert(result)

        return result

    def _wait_for_fill(self, order_id: str, symbol: str, timeout: int = 30) -> dict:
        """
        Poll Alpaca for order status until filled, cancelled, or timeout.
        Returns final order status dict.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                order = self.alpaca.trading_client.get_order_by_id(order_id)
                status = order.status.value if hasattr(order.status, "value") else str(order.status)
                if status in ("filled", "partially_filled", "canceled", "expired", "rejected"):
                    filled_qty = float(order.filled_qty or 0)
                    filled_price = float(order.filled_avg_price or 0)
                    return {
                        "order_id": order_id,
                        "symbol": symbol,
                        "status": status,
                        "filled_qty": filled_qty,
                        "filled_avg_price": filled_price,
                    }
            except Exception as e:
                self.logger.log_error(f"Fill check error for {order_id}: {e}")
            time.sleep(2)

        return {"order_id": order_id, "symbol": symbol, "status": "timeout", "filled_qty": 0}

    def _reconcile_positions(self, expected_symbols: list[str]) -> list[dict]:
        """
        After execution, compare expected symbols vs actual Alpaca positions.
        Logs any discrepancies for visibility.
        """
        issues = []
        try:
            actual_positions = {p["symbol"] for p in self.alpaca.get_positions()}
            missing = set(expected_symbols) - actual_positions
            unexpected = actual_positions - set(expected_symbols)

            if missing:
                msg = f"Reconciliation: expected but missing positions: {sorted(missing)}"
                self.logger.log_error(msg)
                issues.append({"type": "missing", "symbols": sorted(missing)})

            if unexpected:
                msg = f"Reconciliation: unexpected open positions: {sorted(unexpected)}"
                self.logger.log_status(msg)
                issues.append({"type": "unexpected", "symbols": sorted(unexpected)})

            if not issues:
                self.logger.log_status(
                    f"Reconciliation OK — {len(actual_positions)} positions match expected."
                )
        except Exception as e:
            self.logger.log_error(f"Reconciliation failed: {e}")

        return issues

    def _execute_trades(self, proposed_trades: list[dict]) -> list[dict]:
        """
        Execute the proposed trades:
        1. Close all existing positions
        2. Buy the new selections (respecting market hours for equities)
        3. Wait for fill confirmation on each order
        4. Reconcile expected vs actual positions
        5. Record entry prices for position-level stop tracking
        """
        results = []

        # Liquidate existing positions and clear old entry price tracking
        print("  Closing existing positions...")
        try:
            closed = self.alpaca.close_all_positions()
            closed_symbols = [c["symbol"] for c in closed if c.get("action") != "error"]
            self.state.clear_entry_prices(closed_symbols)
            for c in closed:
                self.logger.log_order({
                    "symbol": c["symbol"],
                    "side": "sell",
                    "qty": c["qty"],
                    "action": c["action"],
                })
                results.append({"type": "close", **c})
        except Exception as e:
            self.logger.log_error(f"Error closing positions: {e}")
            results.append({"type": "close_error", "error": str(e)})

        # Check if equity market is open (crypto trades 24/7)
        market_open = self.alpaca.is_market_open()

        # Enter new positions
        print("  Entering new positions...")
        entry_prices = {}
        bought_symbols = []

        for trade in proposed_trades:
            symbol = trade["symbol"]
            qty = trade.get("qty", 0)
            trade_is_crypto = trade.get("is_crypto", is_crypto(symbol))

            if qty <= 0:
                self.logger.log_error(f"Invalid quantity for {symbol}: {qty}")
                results.append({"type": "skip", "symbol": symbol, "reason": "zero quantity"})
                continue

            # Skip equity orders if market is closed
            if not trade_is_crypto and not market_open:
                self.logger.log_error(f"Market closed — skipping equity order for {symbol}")
                results.append({"type": "skip", "symbol": symbol, "reason": "market closed"})
                continue

            try:
                order = self.alpaca.submit_market_order(symbol, qty, "buy")
                order_id = order.get("id")

                # Wait for fill confirmation
                if order_id:
                    fill = self._wait_for_fill(order_id, symbol, timeout=30)
                    fill_status = fill.get("status", "unknown")
                    fill_qty = fill.get("filled_qty", qty)
                    fill_price = fill.get("filled_avg_price", trade.get("estimated_price", 0))

                    print(
                        f"    {symbol}: {fill_status} "
                        f"{fill_qty:.6f} {'units' if trade_is_crypto else 'shares'} "
                        f"@ ${fill_price:.4f}"
                    )

                    if fill_status in ("filled", "partially_filled") and fill_price > 0:
                        entry_prices[symbol] = fill_price
                        bought_symbols.append(symbol)
                else:
                    # No order ID — assume submitted; use estimated price
                    entry_prices[symbol] = trade.get("estimated_price", 0)
                    bought_symbols.append(symbol)
                    print(f"    Submitted {qty:.6f} {'units' if trade_is_crypto else 'shares'} of {symbol}")

                self.logger.log_order({
                    "symbol": symbol,
                    "side": "buy",
                    "qty": qty,
                    "order_id": order_id,
                    "status": order.get("status"),
                    "estimated_price": trade.get("estimated_price"),
                    "allocated_capital": trade.get("allocated_capital"),
                    "is_crypto": trade_is_crypto,
                })
                results.append({"type": "buy", **order})

            except Exception as e:
                self.logger.log_error(f"Order failed for {symbol}: {e}")
                results.append({"type": "order_error", "symbol": symbol, "error": str(e)})

        # Record entry prices for stop-loss tracking
        if entry_prices:
            self.state.record_entry_prices(entry_prices)

        # Reconcile expected vs actual positions
        if bought_symbols:
            reconciliation = self._reconcile_positions(bought_symbols)
            if reconciliation:
                results.append({"type": "reconciliation", "issues": reconciliation})

        return results

    def adaptive_check(self) -> dict:
        """
        Daily smart check: only run a full rebalance if something meaningful has changed.
        Prevents unnecessary churn on a small account.

        Triggers a full rebalance when:
          - The set of momentum leaders has changed (different symbols)
          - Any momentum score has shifted by more than the threshold
          - A drawdown circuit breaker condition is detected

        Skips when:
          - Leaders are unchanged AND signals are stable
        """
        adaptive_cfg = self.config.get("adaptive", {})
        momentum_threshold = adaptive_cfg.get("momentum_shift_threshold", 0.03)
        max_drawdown_pct = adaptive_cfg.get("max_drawdown_pct", 10.0)

        print("\n" + "=" * 60)
        print("ADAPTIVE DAILY CHECK")
        print("=" * 60)

        # ── 0. Position-level stop-loss check ────────────
        try:
            current_positions = self.alpaca.get_positions()
            stops_hit = self.risk.check_position_stops(current_positions)
            if stops_hit:
                print(f"\n  ⚠️  STOP-LOSS TRIGGERED for: {[s['symbol'] for s in stops_hit]}")
                for stop in stops_hit:
                    print(
                        f"    {stop['symbol']}: entry ${stop['entry_price']:.4f} → "
                        f"now ${stop['current_price']:.4f} "
                        f"({stop['pct_change']:+.2f}%)"
                    )
                # Close only the stop-hit positions (not the whole portfolio)
                for stop in stops_hit:
                    try:
                        self.alpaca.trading_client.close_position(stop["symbol"])
                        self.state.clear_entry_prices([stop["symbol"]])
                        self.logger.log_order({
                            "symbol": stop["symbol"],
                            "side": "sell",
                            "qty": stop.get("qty", 0),
                            "action": "stop_loss_triggered",
                            "pct_change": stop["pct_change"],
                        })
                        print(f"    Closed {stop['symbol']} (stop-loss).")
                    except Exception as e:
                        self.logger.log_error(f"Failed to close stop-loss position {stop['symbol']}: {e}")
        except Exception as e:
            print(f"  [WARNING] Stop-loss check failed: {e}")

        # ── 1. Drawdown circuit breaker ───────────────────
        try:
            account = self.alpaca.get_account()
            portfolio_value = account["portfolio_value"]
            self.state.update_peak(portfolio_value)

            drawdown_pct = self.state.get_drawdown_pct(portfolio_value)
            peak = self.state.get_peak()
            print(f"  Portfolio: ${portfolio_value:,.2f}  |  Peak: ${peak:,.2f}  |  Drawdown: {drawdown_pct:.1f}%")

            if self.state.check_drawdown(portfolio_value, max_drawdown_pct):
                print(f"\n  ⚠️  DRAWDOWN CIRCUIT BREAKER — {drawdown_pct:.1f}% below peak (threshold: {max_drawdown_pct}%)")
                print("  Holding cash until portfolio recovers.")
                result = {
                    "action": "hold_drawdown",
                    "drawdown_pct": drawdown_pct,
                    "peak_value": peak,
                    "portfolio_value": portfolio_value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.logger.log_status(
                    f"Drawdown circuit breaker: {drawdown_pct:.1f}% below peak — holding cash.",
                    result,
                )
                if self.telegram:
                    self.telegram.send_cash_hold_alert(result)
                return result

        except Exception as e:
            print(f"  [WARNING] Could not check drawdown: {e}")

        # ── 2. Compute current momentum leaders (no Claude, fast) ─
        print("\n  Computing momentum leaders...")
        try:
            class_rankings = self.allocator.compute_class_momentum()
        except Exception as e:
            print(f"  [WARNING] Momentum computation failed: {e}. Running full rebalance to be safe.")
            return self.run_rebalance()

        # Collect top-N instruments per class with their scores
        proposed_symbols = set()
        current_scores: dict[str, float] = {}

        for cr in class_rankings:
            class_key = cr["class_key"]
            instruments = cr.get("instruments", [])
            valid = sorted(
                [i for i in instruments if i.get("momentum") is not None],
                key=lambda x: x["momentum"],
                reverse=True,
            )
            top_n = self.allocator.get_top_n(class_key)
            for inst in valid[:top_n]:
                proposed_symbols.add(inst["symbol"])
                current_scores[inst["symbol"]] = inst["momentum"]

        # ── 3. Compare with last state ────────────────────
        last_holdings = set(self.state.get_last_holdings())
        last_scores = self.state.get_last_momentum_scores()

        holdings_changed = proposed_symbols != last_holdings

        signal_shifted = False
        shift_details = []
        for sym, score in current_scores.items():
            if sym in last_scores:
                shift = abs(score - last_scores[sym])
                if shift > momentum_threshold:
                    signal_shifted = True
                    shift_details.append(f"{sym}: {shift * 100:.1f}% shift")

        print(f"  Proposed leaders  : {sorted(proposed_symbols)}")
        print(f"  Last holdings     : {sorted(last_holdings) if last_holdings else 'None (first run)'}")
        print(f"  Holdings changed  : {'YES' if holdings_changed else 'No'}")
        print(f"  Signal shifted    : {'YES — ' + ', '.join(shift_details) if signal_shifted else 'No'}")

        # ── 4. Decide ─────────────────────────────────────
        if not holdings_changed and not signal_shifted and last_holdings:
            print("\n  ✓ No significant change detected — skipping rebalance for today.")
            result = {
                "action": "skip",
                "reason": "Holdings and signals unchanged since last rebalance.",
                "current_symbols": sorted(proposed_symbols),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.logger.log_status("Adaptive: no rebalance needed today.", result)
            return result

        reason = []
        if not last_holdings:
            reason.append("First run — initialising")
        if holdings_changed:
            reason.append(f"Holdings changed: {sorted(last_holdings)} → {sorted(proposed_symbols)}")
        if signal_shifted:
            reason.append(f"Signal shift: {', '.join(shift_details)}")

        print(f"\n  ⟳ Change detected ({'; '.join(reason)}) — running full rebalance.")

        # ── 5. Full rebalance ─────────────────────────────
        result = self.run_rebalance()

        # ── 6. Save new state ─────────────────────────────
        self.state.update_holdings(list(proposed_symbols), current_scores)

        return result

    def prefetch(self) -> dict:
        """
        Pre-fetch news and run Level 1 analysis before market open.
        Called by the secondary scheduler job at 9:30 AM ET.
        """
        print("\n[PREFETCH] Running pre-market data collection...")

        class_rankings = self.allocator.compute_class_momentum()
        self.allocator.print_class_rankings(class_rankings)

        news_digest = self.news.get_headline_digest()
        self.news.print_digest(news_digest)

        class_assessment = self.claude.assess_classes(class_rankings, news_digest)
        self.claude.print_class_assessment(class_assessment)

        self.logger.log_status("Prefetch complete. Data ready for rebalance.", {
            "class_rankings_count": len(class_rankings),
            "news_relevant": news_digest.get("relevant_count", 0),
            "claude_fallback": class_assessment.get("fallback", False),
        })

        return {
            "class_rankings": class_rankings,
            "news_digest": news_digest,
            "class_assessment": class_assessment,
        }
