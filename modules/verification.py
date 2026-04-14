"""
Layer 4 — Verification layer.
Independent Claude API call that audits the two-level decision package.
Now includes volatility flagging for selected instruments.
"""

from __future__ import annotations

import json

import anthropic


VERIFICATION_SYSTEM_PROMPT = """You are an independent risk auditor for a multi-asset-class momentum trading system.
You receive a COMPLETE two-level decision package and must verify it before execution.

Verify:
1. Do class allocation decisions follow logically from class momentum and news?
2. Do instrument selections follow from the class decisions?
3. Are all allocation percentages within the stated constraints?
4. Do volatility flags change your assessment?
5. Is there anything in the news that the reasoning layer may have missed or misjudged?

You MUST respond with valid JSON matching this exact structure:
{
  "verdict": "APPROVED" | "APPROVED_WITH_NOTES" | "REJECTED",
  "notes": "Any discrepancies or concerns. Empty string if none.",
  "rejection_reason": "Reason for rejection, or null if approved.",
  "checks": {
    "class_logic": true | false,
    "instrument_logic": true | false,
    "allocation_constraints": true | false,
    "volatility_assessment": "Your assessment of flagged instruments or 'No flags'",
    "missed_risks": "Description of any missed risks, or 'None identified'"
  }
}

Rules:
- APPROVED: Everything checks out. Proceed to execution.
- APPROVED_WITH_NOTES: Mostly fine, but log a discrepancy or concern.
- REJECTED: A clear error, contradiction, or risk that should halt execution.

Default to APPROVED unless you find a genuine issue.
Be specific — cite actual data or headlines.
Do NOT reject based on general market uncertainty; only reject on clear logical errors,
constraint violations, or contradictions between the news and the decisions."""


class VerificationLayer:
    """
    Independent audit with two layers:
      1. Rule-based hard checks (no LLM — deterministic, fast, zero hallucination risk)
      2. Claude soft audit (only reached if hard checks pass)

    Hard checks catch: allocation constraint violations, negative-momentum selections,
    position-size cap breaches, and zero-quantity trades.
    Claude then audits the logic and news interpretation.
    """

    def __init__(self, config: dict, momentum_strategy=None):
        self.config = config
        claude_cfg = config.get("claude", {})
        # Use a dedicated verification model if configured — ideally a different
        # (more capable) model than the reasoning layer to avoid groupthink.
        self.model = claude_cfg.get("verification_model") or claude_cfg.get("model", "claude-sonnet-4-6")
        self.enabled = claude_cfg.get("use_verification_layer", True)
        self.client = anthropic.Anthropic()
        self.momentum = momentum_strategy
        self.global_cfg = config.get("global", {})
        self.universe_cfg = config.get("universe", {})
        print(f"[VERIFY] Auditor model: {self.model}")

    # ── Volatility flagging ───────────────────────────────

    def compute_volatility_flags(self, selections: list[dict]) -> list[dict]:
        """
        For each selected instrument, compute 10-day rolling vol and flag
        if it is >2 std above the 90-day average.
        """
        if self.momentum is None:
            return []

        flags = []
        for sel in selections:
            symbol = sel["symbol"]
            try:
                vol = self.momentum.compute_volatility(symbol)
                if vol.get("flagged"):
                    flags.append({
                        "symbol": symbol,
                        "vol_10d": vol.get("vol_10d"),
                        "vol_90d_avg": vol.get("vol_90d_avg"),
                        "sigma_above": vol.get("sigma_above"),
                        "message": (
                            f"{symbol}: elevated — {vol.get('sigma_above', 0):.1f} std above 90-day average"
                        ),
                    })
            except Exception as e:
                print(f"[VERIFY] Volatility computation error for {symbol}: {e}")

        return flags

    # ── Rule-based hard checks (primary, LLM-free) ────────

    def _run_hard_checks(
        self,
        class_allocation: dict,
        instrument_selections: list[dict],
        proposed_trades: list[dict],
        capital: float,
    ) -> dict:
        """
        Deterministic rule checks. Returns a result dict with verdict and failures list.
        These never hallucinate and run in <1ms.
        """
        failures = []
        warnings = []

        max_single = self.global_cfg.get("max_single_instrument", 0.25)

        # 1. Total allocation must not exceed 100%
        total_alloc = sum(s.get("allocation_pct", 0) for s in instrument_selections)
        if total_alloc > 1.05:  # 5% tolerance for rounding
            failures.append(
                f"Total allocation {total_alloc:.1%} exceeds 100% — possible double-counting."
            )

        # 2. No single instrument can exceed max_single_instrument
        for sel in instrument_selections:
            pct = sel.get("allocation_pct", 0)
            if pct > max_single + 0.02:  # 2% tolerance
                failures.append(
                    f"{sel['symbol']}: allocation {pct:.1%} exceeds max single-instrument "
                    f"cap of {max_single:.0%}."
                )

        # 3. Class allocations must not exceed their configured max_allocation
        decisions = class_allocation.get("decisions", {})
        for sel in instrument_selections:
            class_key = sel.get("class_key", "")
            class_cfg = self.universe_cfg.get(class_key, {})
            max_class = class_cfg.get("max_allocation", 1.0)
            # Sum allocations for this class
            class_total = sum(
                s.get("allocation_pct", 0)
                for s in instrument_selections
                if s.get("class_key") == class_key
            )
            if class_total > max_class + 0.05:
                failures.append(
                    f"{class_key}: total class allocation {class_total:.1%} exceeds "
                    f"configured max of {max_class:.0%}."
                )

        # 4. No zero-quantity trades
        for trade in proposed_trades:
            if trade.get("qty", 0) <= 0:
                failures.append(
                    f"{trade['symbol']}: proposed quantity is zero or negative — "
                    f"likely a price/capital calculation error."
                )

        # 5. Negative-momentum instruments get a warning (not hard failure —
        #    REDUCE decisions may intentionally include weak-momentum instruments)
        for sel in instrument_selections:
            mom = sel.get("momentum", None)
            class_decision = sel.get("class_decision", "ACTIVE")
            if mom is not None and mom < -0.05 and class_decision == "ACTIVE":
                warnings.append(
                    f"{sel['symbol']}: momentum {sel.get('momentum_pct', 0):+.2f}% is "
                    f"significantly negative but class is ACTIVE."
                )

        # 6. Capital sanity check — total allocated capital must not exceed configured capital
        total_capital = sum(s.get("allocated_capital", 0) for s in instrument_selections)
        if total_capital > capital * 1.05:
            failures.append(
                f"Total allocated capital ${total_capital:.2f} exceeds bot capital "
                f"${capital:.2f} by more than 5%."
            )

        if failures:
            return {
                "verdict": "REJECTED",
                "notes": "; ".join(warnings) if warnings else "",
                "rejection_reason": "Hard rule violations: " + " | ".join(failures),
                "checks": {
                    "class_logic": True,
                    "instrument_logic": False,
                    "allocation_constraints": False,
                    "volatility_assessment": "Skipped — hard check failed",
                    "missed_risks": "Hard rule failures: " + " | ".join(failures),
                },
                "rule_failures": failures,
                "rule_warnings": warnings,
                "hard_check": True,
            }

        return {
            "verdict": "PASS",
            "rule_warnings": warnings,
            "hard_check": True,
        }

    # ── Verification ──────────────────────────────────────

    def verify(
        self,
        class_allocation: dict,
        instrument_selections: list[dict],
        proposed_trades: list[dict],
        volatility_flags: list[dict],
        news_digest: dict,
        capital: float,
    ) -> dict:
        """
        Verify the full two-level decision package.
        Returns verification result with verdict, notes, and checks.
        """
        # ── Layer 1: Deterministic hard checks (always run) ──
        hard_result = self._run_hard_checks(
            class_allocation, instrument_selections, proposed_trades, capital
        )
        if hard_result["verdict"] == "REJECTED":
            hard_result["volatility_flags"] = volatility_flags
            return hard_result

        rule_warnings = hard_result.get("rule_warnings", [])

        if not self.enabled:
            notes = "; ".join(rule_warnings) if rule_warnings else ""
            verdict = "APPROVED WITH NOTES" if rule_warnings else "APPROVED"
            return {
                "verdict": verdict,
                "notes": notes,
                "rejection_reason": None,
                "checks": {
                    "class_logic": True,
                    "instrument_logic": True,
                    "allocation_constraints": True,
                    "volatility_assessment": "Claude verification disabled",
                    "missed_risks": "Claude verification disabled",
                },
                "volatility_flags": volatility_flags,
                "skipped": True,
                "rule_warnings": rule_warnings,
            }

        # ── Layer 2: Claude soft audit ────────────────────
        package = self._format_package(
            class_allocation, instrument_selections, proposed_trades,
            volatility_flags, news_digest, capital,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0,
                system=VERIFICATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": package}],
            )

            raw = response.content[0].text.strip()
            if "```json" in raw:
                raw_json = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw_json = raw.split("```")[1].split("```")[0].strip()
            else:
                raw_json = raw

            result = json.loads(raw_json)

            if "verdict" not in result:
                raise ValueError("Missing 'verdict' in verification response")

            # Normalise verdict
            verdict = result["verdict"].upper().strip().replace("_", " ")
            # Accept both "APPROVED_WITH_NOTES" and "APPROVED WITH NOTES"
            if verdict in ("APPROVED WITH NOTES", "APPROVED_WITH_NOTES"):
                verdict = "APPROVED WITH NOTES"
            elif verdict not in ("APPROVED", "REJECTED"):
                verdict = "APPROVED WITH NOTES"

            result["verdict"] = verdict
            result["raw_response"] = raw
            result["volatility_flags"] = volatility_flags
            result["model_used"] = self.model
            result["input_tokens"] = response.usage.input_tokens
            result["output_tokens"] = response.usage.output_tokens

            return result

        except json.JSONDecodeError as e:
            print(f"[VERIFY] JSON parse error: {e}")
            return self._error_result(f"JSON parse error: {e}", volatility_flags)
        except anthropic.APIError as e:
            print(f"[VERIFY] API error: {e}")
            return self._error_result(f"API error: {e}", volatility_flags)
        except Exception as e:
            print(f"[VERIFY] Unexpected error: {e}")
            return self._error_result(str(e), volatility_flags)

    # ── Package formatter ─────────────────────────────────

    def _format_package(
        self,
        class_allocation: dict,
        instrument_selections: list[dict],
        proposed_trades: list[dict],
        volatility_flags: list[dict],
        news_digest: dict,
        capital: float,
    ) -> str:
        lines = [
            "COMPLETE DECISION PACKAGE FOR VERIFICATION",
            "=" * 55, "",
        ]

        # Level 1
        lines.append("LEVEL 1 — CLASS ALLOCATION DECISIONS:")
        decisions = class_allocation.get("decisions", {})
        for key, info in decisions.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {info.get('decision', '?')} — {info.get('reason', '')}")
        lines.append(f"  Macro Summary: {class_allocation.get('macro_summary', 'N/A')}")
        lines.append("")

        # Level 2
        lines.append("LEVEL 2 — INSTRUMENT SELECTIONS:")
        for sel in instrument_selections:
            lines.append(
                f"  {sel.get('class_label', '?')}: {sel['symbol']} "
                f"alloc={sel.get('allocation_pct', 0):.0%} "
                f"${sel.get('allocated_capital', 0):.2f} "
                f"mom={sel.get('momentum_pct', 0):+.2f}%"
            )
        lines.append("")

        # Proposed trades
        lines.append(f"PROPOSED TRADES (capital: ${capital}):")
        for t in proposed_trades:
            lines.append(
                f"  {t.get('side', '?').upper()} {t.get('symbol', '?')}: "
                f"{t.get('qty', 0):.6f} units @ ~${t.get('estimated_price', 0):.2f}, "
                f"allocated=${t.get('allocated_capital', 0):.2f}"
            )
        lines.append("")

        # Volatility flags
        lines.append("VOLATILITY FLAGS:")
        if volatility_flags:
            for vf in volatility_flags:
                lines.append(f"  {vf['message']}")
        else:
            lines.append("  None — all instruments within normal volatility range.")
        lines.append("")

        # Constraints
        global_cfg = self.config.get("global", {})
        universe_cfg = self.config.get("universe", {})
        lines.append("PORTFOLIO CONSTRAINTS:")
        lines.append(f"  Max single instrument: {global_cfg.get('max_single_instrument', 0.25):.0%}")
        for key in ["us_equities", "crypto", "commodities", "international"]:
            cfg = universe_cfg.get(key, {})
            label = key.replace("_", " ").title()
            lines.append(f"  Max {label}: {cfg.get('max_allocation', 0):.0%}")
        lines.append(f"  Total capital: ${capital}")
        lines.append("")

        # News digest (truncated)
        lines.append(f"NEWS DIGEST ({news_digest.get('relevant_count', 0)} relevant headlines):")
        lines.append(news_digest.get("digest_text", "No headlines.")[:1500])

        return "\n".join(lines)

    # ── Error fallback ────────────────────────────────────

    def _error_result(self, error: str, volatility_flags: list[dict] = None) -> dict:
        return {
            "verdict": "REJECTED",
            "notes": error,
            "rejection_reason": f"Verification layer error: {error}. Holding positions for safety.",
            "checks": {
                "class_logic": False,
                "instrument_logic": False,
                "allocation_constraints": False,
                "volatility_assessment": "Verification failed — cannot confirm.",
                "missed_risks": "Verification failed — cannot confirm.",
            },
            "volatility_flags": volatility_flags or [],
            "error": True,
        }
