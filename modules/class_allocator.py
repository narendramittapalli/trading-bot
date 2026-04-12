"""
Level 1 — Asset class allocation.
Decides which of the 4 asset classes are worth investing in this week.
Ranks classes by average momentum, then consults Claude for ACTIVE/REDUCE/SKIP.
Distributes capital across active classes proportionally within max_allocation caps.
"""

from __future__ import annotations


# Human-readable labels for config keys
CLASS_LABELS = {
    "us_equities": "US Equities",
    "crypto": "Crypto",
    "commodities": "Commodities",
    "international": "International",
}


class ClassAllocator:
    """Two-level allocator: first decides class weights, then defers to momentum for instruments."""

    def __init__(self, config: dict, momentum_strategy):
        self.config = config
        self.momentum = momentum_strategy
        self.universe_cfg = config.get("universe", {})
        self.global_cfg = config.get("global", {})
        self.capital = self.global_cfg.get("capital", 500)
        self.max_single_instrument = self.global_cfg.get("max_single_instrument", 0.20)
        self.min_class_momentum_pct = self.global_cfg.get("min_class_momentum_pct", 1.5)
        self.regime_threshold_pct = self.global_cfg.get("market_regime_threshold_pct", -2.0)
        self.regime_symbol = self.global_cfg.get("market_regime_symbol", "SPY")

    # ── helpers ───────────────────────────────────────────

    def get_class_names(self) -> list[str]:
        """Return list of asset-class config keys (e.g. 'us_equities')."""
        return [k for k in self.universe_cfg if isinstance(self.universe_cfg[k], dict)]

    def get_instruments(self, class_key: str) -> list[str]:
        return self.universe_cfg.get(class_key, {}).get("instruments", [])

    def get_max_allocation(self, class_key: str) -> float:
        return self.universe_cfg.get(class_key, {}).get("max_allocation", 0.25)

    def get_top_n(self, class_key: str) -> int:
        return self.universe_cfg.get(class_key, {}).get("top_n", 1)

    # ── Market regime ─────────────────────────────────────

    def get_market_regime(self) -> dict:
        """
        Check SPY momentum as a proxy for overall market health.

        Returns:
          regime: "bull" | "neutral" | "bear" | "severe_bear"
          spy_momentum_pct: float
          scale_factor: multiplier applied to all class allocations (0.0–1.0)
        """
        try:
            result = self.momentum.compute_momentum(self.regime_symbol)
            spy_mom_pct = result.get("momentum_pct", 0) or 0
        except Exception as e:
            print(f"[REGIME] Could not compute SPY momentum: {e} — assuming neutral")
            return {"regime": "neutral", "spy_momentum_pct": 0, "scale_factor": 1.0}

        threshold = self.regime_threshold_pct  # e.g., -2.0

        if spy_mom_pct >= 0:
            regime = "bull"
            scale = 1.0
        elif spy_mom_pct >= threshold:
            regime = "neutral"
            scale = 1.0
        elif spy_mom_pct >= threshold * 2:
            regime = "bear"
            scale = 0.5          # Half allocations — keep 50% cash
        else:
            regime = "severe_bear"
            scale = 0.0          # Full cash — do not invest

        return {
            "regime": regime,
            "spy_momentum_pct": round(spy_mom_pct, 2),
            "scale_factor": scale,
        }

    # ── Level 1: class-level momentum ─────────────────────

    def compute_class_momentum(self) -> list[dict]:
        """
        For each asset class, compute average 20-day momentum across its instruments.
        Returns list sorted by avg momentum descending.
        """
        results = []
        for class_key in self.get_class_names():
            instruments = self.get_instruments(class_key)
            momentums = []
            instrument_details = []

            for symbol in instruments:
                try:
                    m = self.momentum.compute_momentum(symbol)
                    instrument_details.append(m)
                    if m.get("momentum") is not None:
                        momentums.append(m["momentum"])
                except Exception as e:
                    instrument_details.append({"symbol": symbol, "momentum": None, "error": str(e)})

            avg_mom = sum(momentums) / len(momentums) if momentums else 0.0

            results.append({
                "class_key": class_key,
                "class_label": CLASS_LABELS.get(class_key, class_key),
                "avg_momentum": round(avg_mom, 6),
                "avg_momentum_pct": round(avg_mom * 100, 2),
                "instrument_count": len(instruments),
                "valid_count": len(momentums),
                "instruments": instrument_details,
            })

        results.sort(key=lambda x: x["avg_momentum"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    # ── Level 1: apply Claude class decisions ─────────────

    def apply_class_decisions(self, class_rankings: list[dict], claude_decisions: dict,
                              market_regime: dict = None) -> list[dict]:
        """
        Take Claude's ACTIVE/REDUCE/SKIP per class and compute actual capital allocation.

        claude_decisions format:
        {
          "us_equities": {"decision": "ACTIVE", "reason": "..."},
          "crypto": {"decision": "REDUCE", "reason": "..."},
          ...
        }

        Returns list of class allocations with capital amounts.
        """
        regime = market_regime or {"regime": "neutral", "scale_factor": 1.0, "spy_momentum_pct": 0}
        regime_scale = regime.get("scale_factor", 1.0)
        regime_name = regime.get("regime", "neutral")

        if regime_name == "severe_bear":
            print(f"  [REGIME] Severe bear market detected — forcing full cash hold.")

        # First pass: determine raw weights
        raw_weights = {}
        for cr in class_rankings:
            key = cr["class_key"]
            decision_info = claude_decisions.get(key, {})
            decision = decision_info.get("decision", "ACTIVE").upper()
            max_alloc = self.get_max_allocation(key)
            avg_mom_pct = cr.get("avg_momentum_pct", 0) or 0

            # Hard guard 1: severe bear → full cash
            if regime_name == "severe_bear":
                raw_weights[key] = 0.0
                continue

            # Hard guard 2: class momentum below minimum threshold → force SKIP
            if avg_mom_pct < self.min_class_momentum_pct and decision != "SKIP":
                decision = "SKIP"
                print(
                    f"  [THRESHOLD] {key}: avg momentum {avg_mom_pct:+.2f}% < "
                    f"{self.min_class_momentum_pct:.1f}% minimum — forcing SKIP"
                )

            if decision == "SKIP":
                raw_weights[key] = 0.0
            elif decision == "REDUCE":
                raw_weights[key] = max_alloc * 0.5
            else:  # ACTIVE
                raw_weights[key] = max_alloc

            # Apply regime scale (bear market → all allocations halved)
            raw_weights[key] *= regime_scale

        # Normalise so total <= 1.0 (if sum exceeds 1.0, scale down proportionally)
        total_weight = sum(raw_weights.values())
        if total_weight > 1.0:
            scale = 1.0 / total_weight
            for k in raw_weights:
                raw_weights[k] *= scale

        # Build allocation list
        allocations = []
        for cr in class_rankings:
            key = cr["class_key"]
            decision_info = claude_decisions.get(key, {})
            decision = decision_info.get("decision", "ACTIVE").upper()
            reason = decision_info.get("reason", "")
            weight = raw_weights.get(key, 0.0)
            avg_mom_pct = cr.get("avg_momentum_pct", 0) or 0

            # Reflect actual applied decision
            if regime_name == "severe_bear":
                decision = "SKIP"
                reason = f"Severe bear market: SPY momentum {regime['spy_momentum_pct']:+.2f}% — full cash."
            elif avg_mom_pct < self.min_class_momentum_pct and weight == 0.0:
                decision = "SKIP"
                reason = f"Momentum {avg_mom_pct:+.2f}% below minimum threshold {self.min_class_momentum_pct:.1f}%."
            elif regime_scale < 1.0 and weight > 0.0:
                reason = f"{reason} [Bear scale {regime_scale:.0%} applied — SPY {regime['spy_momentum_pct']:+.2f}%]"

            allocations.append({
                "class_key": key,
                "class_label": CLASS_LABELS.get(key, key),
                "decision": decision,
                "reason": reason,
                "weight": round(weight, 4),
                "allocated_capital": round(weight * self.capital, 2),
                "max_allocation": self.get_max_allocation(key),
                "avg_momentum_pct": avg_mom_pct,
                "rank": cr.get("rank", 0),
                "regime": regime_name,
            })

        # Calculate cash position
        invested_weight = sum(a["weight"] for a in allocations)
        cash_weight = round(max(0.0, 1.0 - invested_weight), 4)
        cash_capital = round(cash_weight * self.capital, 2)

        return allocations, cash_weight, cash_capital

    # ── Level 2: instrument selection within each class ───

    def select_instruments(self, class_allocations: list[dict], class_rankings: list[dict]) -> list[dict]:
        """
        For each active class, rank instruments by momentum and select top_n.
        Applies max_single_instrument cap.
        Returns flat list of instrument allocations.
        """
        # Build lookup for instrument details
        class_instruments = {}
        for cr in class_rankings:
            class_instruments[cr["class_key"]] = cr.get("instruments", [])

        all_selections = []

        for alloc in class_allocations:
            key = alloc["class_key"]
            if alloc["decision"] == "SKIP" or alloc["allocated_capital"] <= 0:
                continue

            instruments = class_instruments.get(key, [])
            top_n = self.get_top_n(key)
            class_capital = alloc["allocated_capital"]

            # Sort instruments by momentum descending
            valid = [i for i in instruments if i.get("momentum") is not None]
            valid.sort(key=lambda x: x["momentum"], reverse=True)

            selected = valid[:top_n]
            if not selected:
                continue

            # Equal weight within class, then apply single-instrument cap
            max_instrument_capital = self.capital * self.max_single_instrument
            per_instrument = class_capital / len(selected)

            for inst in selected:
                capped_capital = min(per_instrument, max_instrument_capital)
                price = inst.get("current_price", 0)
                shares = round(capped_capital / price, 6) if price and price > 0 else 0

                all_selections.append({
                    "symbol": inst["symbol"],
                    "class_key": key,
                    "class_label": CLASS_LABELS.get(key, key),
                    "allocated_capital": round(capped_capital, 2),
                    "allocation_pct": round(capped_capital / self.capital, 4) if self.capital > 0 else 0,
                    "target_shares": shares,
                    "current_price": price,
                    "momentum_pct": inst.get("momentum_pct", 0),
                    "momentum": inst.get("momentum", 0),
                    "class_decision": alloc["decision"],
                })

        return all_selections

    # ── Console display ───────────────────────────────────

    def print_class_rankings(self, class_rankings: list[dict]):
        """Pretty-print class-level momentum rankings."""
        print("\n" + "=" * 70)
        print("LEVEL 1 — ASSET CLASS MOMENTUM RANKINGS")
        print("=" * 70)
        print(f"{'Rank':<6}{'Class':<18}{'Avg Momentum':>14}{'Instruments':>14}")
        print("-" * 70)
        for cr in class_rankings:
            print(
                f"  {cr['rank']:<4}{cr['class_label']:<18}"
                f"{cr['avg_momentum_pct']:>+10.2f}%"
                f"{cr['valid_count']:>8}/{cr['instrument_count']}"
            )
        print("=" * 70 + "\n")

    def print_allocations(self, allocations: list[dict], cash_weight: float, cash_capital: float):
        """Pretty-print class allocation decisions."""
        print("\n" + "=" * 70)
        print("LEVEL 1 — CLASS ALLOCATION DECISIONS")
        print("=" * 70)
        for a in allocations:
            marker = {"ACTIVE": "+", "REDUCE": "~", "SKIP": "-"}.get(a["decision"], "?")
            print(
                f"  [{marker}] {a['class_label']:<18}: {a['decision']:<8} "
                f"weight={a['weight']:.0%}  ${a['allocated_capital']:>7.2f}  "
                f"— {a['reason']}"
            )
        if cash_weight > 0:
            print(f"  [$] {'Cash':<18}: {'HOLD':<8} weight={cash_weight:.0%}  ${cash_capital:>7.2f}")
        print("=" * 70 + "\n")

    def print_instrument_selections(self, selections: list[dict]):
        """Pretty-print Level 2 instrument selections."""
        print("\n" + "=" * 70)
        print("LEVEL 2 — INSTRUMENT SELECTIONS")
        print("=" * 70)
        for s in selections:
            print(
                f"  {s['class_label']:<18}: {s['symbol']:<8} "
                f"alloc={s['allocation_pct']:.0%}  ${s['allocated_capital']:>7.2f}  "
                f"mom={s['momentum_pct']:>+6.2f}%  "
                f"shares={s['target_shares']:.4f} @ ${s.get('current_price', 0):.2f}"
            )
        print("=" * 70 + "\n")
