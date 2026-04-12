"""
Layer 3 — Claude reasoning (two-stage).
Stage 1: Class-level allocation (ACTIVE / REDUCE / SKIP per asset class).
Stage 2: Instrument selection within each active class.
"""

from __future__ import annotations

import json

import anthropic


# ── Stage 1 system prompt ─────────────────────────────

STAGE1_SYSTEM_PROMPT = """You are a portfolio manager running a moderate risk strategy with $500 capital.

You will receive:
1. Asset class momentum rankings (20-day return average per class)
2. A recent macro news digest (last 48 hours)

For each asset class, output ACTIVE, REDUCE, or SKIP with one-line reasoning.
- ACTIVE: allocate capital to this class at full weight
- REDUCE: allocate at half the normal weight (positive but uncertain signal)
- SKIP: do not invest in this class this week

Consider: macro regime, risk-off vs risk-on signals, cross-asset correlations.
Moderate risk profile means: preserve capital in uncertain regimes, participate in clear trends.

You MUST respond with valid JSON matching this exact structure:
{
  "us_equities": {"decision": "ACTIVE", "reason": "one-line reasoning"},
  "crypto": {"decision": "REDUCE", "reason": "one-line reasoning"},
  "commodities": {"decision": "SKIP", "reason": "one-line reasoning"},
  "international": {"decision": "ACTIVE", "reason": "one-line reasoning"},
  "macro_summary": "one paragraph overall macro summary"
}

Be specific. Reference actual data or headlines.
Default to ACTIVE unless there is a clear reason to reduce or skip.
Do not hallucinate headlines — only reference information from the digest provided."""


# ── Stage 2 system prompt ─────────────────────────────

STAGE2_SYSTEM_PROMPT = """You are a portfolio manager selecting instruments within a single asset class.

You will receive:
1. The asset class name and its allocation decision (ACTIVE or REDUCE)
2. Instruments ranked by 20-day momentum
3. Relevant news headlines for this class

Select the top instruments for this class. Apply a moderate risk profile.
Flag any concerns per instrument.

You MUST respond with valid JSON matching this exact structure:
{
  "selected": [
    {"symbol": "SPY", "confidence": "HIGH", "note": "one-line note"}
  ]
}

confidence: HIGH = strong conviction, MEDIUM = acceptable, LOW = proceed with caution.
Only select instruments with positive or near-positive momentum unless the class is ACTIVE with strong macro support.
Keep notes concise — one sentence max."""


class ClaudeReasoning:
    """Two-stage Claude reasoning: class allocation then instrument selection."""

    def __init__(self, config: dict):
        self.config = config
        claude_cfg = config.get("claude", {})
        self.model = claude_cfg.get("model", "claude-sonnet-4-0")
        self.enabled = claude_cfg.get("use_claude_layer", True)
        self.client = anthropic.Anthropic()

    # ── JSON parsing helper ───────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse JSON from Claude response, handling markdown fences."""
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        return json.loads(cleaned)

    # ── Stage 1: class allocation ─────────────────────────

    def assess_classes(self, class_rankings: list[dict], news_digest: dict) -> dict:
        """
        Stage 1: ask Claude to decide ACTIVE/REDUCE/SKIP for each asset class.
        Returns dict keyed by class_key with decision and reason.
        """
        if not self.enabled:
            return self._default_class_response(class_rankings)

        # Format class rankings
        rankings_text = "Asset class momentum rankings (20-day return average):\n"
        for cr in class_rankings:
            rankings_text += (
                f"  #{cr['rank']} {cr['class_label']}: "
                f"avg_momentum={cr['avg_momentum_pct']:+.2f}%, "
                f"instruments={cr['valid_count']}/{cr['instrument_count']}\n"
            )

        digest_text = news_digest.get("digest_text", "No headlines available.")

        user_message = (
            f"Analyse the following and provide your class allocation decisions as JSON.\n\n"
            f"{rankings_text}\n"
            f"Recent macro news digest (last 48 hours, "
            f"{news_digest.get('relevant_count', 0)} relevant headlines):\n"
            f"{digest_text}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0,
                system=STAGE1_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw = response.content[0].text
            result = self._parse_json(raw)

            # Extract macro_summary separately
            macro_summary = result.pop("macro_summary", "")

            # Normalise into consistent format
            decisions = {}
            for key in ["us_equities", "crypto", "commodities", "international"]:
                entry = result.get(key, {})
                if isinstance(entry, dict):
                    decisions[key] = {
                        "decision": entry.get("decision", "ACTIVE").upper(),
                        "reason": entry.get("reason", ""),
                    }
                else:
                    decisions[key] = {"decision": "ACTIVE", "reason": "Defaulted — unexpected format."}

            return {
                "decisions": decisions,
                "macro_summary": macro_summary,
                "raw_response": raw,
                "model_used": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "fallback": False,
            }

        except json.JSONDecodeError as e:
            print(f"[CLAUDE-S1] JSON parse error: {e}")
            return self._default_class_response(class_rankings, error=str(e))
        except anthropic.APIError as e:
            print(f"[CLAUDE-S1] API error: {e}")
            return self._default_class_response(class_rankings, error=str(e))
        except Exception as e:
            print(f"[CLAUDE-S1] Unexpected error: {e}")
            return self._default_class_response(class_rankings, error=str(e))

    # ── Stage 2: instrument selection ─────────────────────

    def assess_instruments(
        self,
        class_key: str,
        class_label: str,
        decision: str,
        instrument_rankings: list[dict],
        news_digest: dict,
        top_n: int,
    ) -> dict:
        """
        Stage 2: for one active/reduced class, ask Claude to pick top instruments.
        Returns list of selected instruments with confidence and notes.
        """
        if not self.enabled:
            return self._default_instrument_response(instrument_rankings, top_n)

        rankings_text = f"Instruments ranked by 20-day momentum (class: {class_label}, decision: {decision}):\n"
        for inst in instrument_rankings:
            if inst.get("momentum") is not None:
                rankings_text += (
                    f"  #{inst.get('rank', '?')} {inst['symbol']}: "
                    f"momentum={inst.get('momentum_pct', 'N/A')}%, "
                    f"price=${inst.get('current_price', 'N/A')}\n"
                )

        digest_text = news_digest.get("digest_text", "No headlines available.")

        user_message = (
            f"Asset class: {class_label}\n"
            f"Class decision: {decision}\n"
            f"Select top {top_n} instruments.\n\n"
            f"{rankings_text}\n"
            f"News relevant to this class:\n{digest_text[:1500]}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                system=STAGE2_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw = response.content[0].text
            result = self._parse_json(raw)

            return {
                "selected": result.get("selected", []),
                "raw_response": raw,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        except Exception as e:
            print(f"[CLAUDE-S2] Error for {class_label}: {e}")
            return self._default_instrument_response(instrument_rankings, top_n)

    # ── Defaults / fallbacks ──────────────────────────────

    def _default_class_response(self, class_rankings: list[dict], error: str = None) -> dict:
        """
        Fallback when Claude is unavailable.
        Uses pure momentum ranking instead of defaulting everything to ACTIVE
        (which would blindly invest in all classes regardless of market conditions).

        Logic:
          - Top classes with positive avg momentum → ACTIVE
          - Classes with near-zero or negative avg momentum → SKIP
          - Everything else → REDUCE
        """
        decisions = {}
        sorted_classes = sorted(
            class_rankings,
            key=lambda x: x.get("avg_momentum", 0),
            reverse=True,
        )
        n = len(sorted_classes)

        for i, cr in enumerate(sorted_classes):
            key = cr["class_key"]
            avg_mom = cr.get("avg_momentum", 0)
            rank = i + 1

            if avg_mom <= 0:
                decision = "SKIP"
                reason = (
                    f"Momentum fallback — negative avg momentum "
                    f"({avg_mom * 100:+.2f}%); avoiding this class."
                )
            elif rank <= max(1, n // 2):
                # Top half with positive momentum
                decision = "ACTIVE"
                reason = (
                    f"Momentum fallback — top-{rank} class, "
                    f"avg momentum {avg_mom * 100:+.2f}%."
                )
            else:
                # Bottom half with positive momentum — reduce, don't skip
                decision = "REDUCE"
                reason = (
                    f"Momentum fallback — rank #{rank}, "
                    f"positive but weaker avg momentum {avg_mom * 100:+.2f}%."
                )

            decisions[key] = {"decision": decision, "reason": reason}

        result = {
            "decisions": decisions,
            "macro_summary": (
                "Claude layer unavailable — pure momentum fallback applied. "
                "Classes ranked by 20-day average momentum; negative-momentum classes skipped."
            ),
            "fallback": True,
        }
        if error:
            result["error"] = error
        return result

    def _default_instrument_response(self, instrument_rankings: list[dict], top_n: int) -> dict:
        valid = [i for i in instrument_rankings if i.get("momentum") is not None]
        valid.sort(key=lambda x: x["momentum"], reverse=True)
        selected = [
            {"symbol": i["symbol"], "confidence": "MEDIUM", "note": "Fallback — pure momentum selection."}
            for i in valid[:top_n]
        ]
        return {"selected": selected, "fallback": True}

    # ── Display ───────────────────────────────────────────

    def print_class_assessment(self, assessment: dict):
        """Print Stage 1 results."""
        print("\n" + "=" * 70)
        print("CLAUDE STAGE 1 — CLASS ALLOCATION ASSESSMENT")
        print("=" * 70)

        if assessment.get("fallback"):
            print("  [FALLBACK] All classes set to ACTIVE — Claude layer inactive.")
            return

        decisions = assessment.get("decisions", {})
        for key, info in decisions.items():
            marker = {"ACTIVE": "+", "REDUCE": "~", "SKIP": "-"}.get(info["decision"], "?")
            label = key.replace("_", " ").title()
            print(f"  [{marker}] {label:<18}: {info['decision']:<8} — {info['reason']}")

        summary = assessment.get("macro_summary", "")
        if summary:
            print(f"\n  Macro Summary: {summary[:400]}")

        tokens_in = assessment.get("input_tokens", "?")
        tokens_out = assessment.get("output_tokens", "?")
        print(f"\n  Tokens: {tokens_in} in / {tokens_out} out")
        print("=" * 70 + "\n")

    def print_instrument_assessment(self, class_label: str, result: dict):
        """Print Stage 2 results for one class."""
        selected = result.get("selected", [])
        if not selected:
            print(f"  {class_label}: no instruments selected")
            return
        for s in selected:
            conf = s.get("confidence", "?")
            note = s.get("note", "")
            print(f"  {class_label}: {s['symbol']:<8} confidence={conf:<6} — {note}")
