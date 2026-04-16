"""
Auto Tuner — Applies parameter optimizer recommendations to config.yaml.

Runs immediately after the parameter optimizer completes (first Sunday monthly).

Safety rules (hard-coded, cannot be overridden):
  SAFE to auto-change  : lookback_days, top_n, min_class_momentum_pct
  NEVER auto-change    : capital, stop_loss_pct, crypto_stop_loss_pct,
                         max_drawdown_pct, alpaca mode, API keys

Change guard:
  - Only applies changes if optimizer found ≥10% Sharpe improvement
  - Only applies changes present in the SAFE list above
  - Logs every change with before/after values
  - Sends Telegram notification listing what changed
  - Creates a changelog entry in state.db

All config changes are reversible: previous values are stored in state.db
under 'param_history' and can be restored with 'python main.py restore_params'.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone


# Params the auto-tuner is allowed to modify in config.yaml
SAFE_PARAMS = {
    "lookback_days",
    "top_n",
    "min_class_momentum_pct",
    "market_regime_threshold_pct",   # How negative SPY must go to trigger regime filter
}

# Params that must NEVER be auto-changed, regardless of any recommendation
PROTECTED_PARAMS = {
    "capital",
    "stop_loss_pct",
    "crypto_stop_loss_pct",
    "max_drawdown_pct",
    "max_single_instrument",
    "mode",  # alpaca mode (paper/live)
}


class AutoTuner:
    """Reads optimization recommendations and safely applies them to config.yaml."""

    def __init__(self, config: dict, state_manager, logger, telegram=None):
        self.config = config
        self.state = state_manager
        self.logger = logger
        self.telegram = telegram

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(os.path.dirname(script_dir), "config.yaml")

    # ── Main entry point ──────────────────────────────────

    def apply(self) -> list[dict]:
        """
        Read the latest optimization result and apply safe changes to config.yaml.
        Returns list of changes made (empty if nothing changed).
        """
        print("\n" + "=" * 60)
        print("AUTO TUNER")
        print("=" * 60)

        result = self.state.get_latest_optimization()
        if not result:
            print("  No optimization result found. Run 'python main.py optimize' first.")
            return []

        if not result.get("recommend_change"):
            print(f"  Optimizer did not recommend changes "
                  f"(improvement {result.get('improvement_pct', 0):.1f}% < 10% threshold).")
            print("  Keeping current parameters.")
            return []

        current_params = result.get("current_params", {})
        best_params = result.get("best_params", {})
        best_sharpe = result.get("best_sharpe", 0)
        current_sharpe = result.get("current_sharpe", 0)

        # Determine what actually needs changing
        changes = []
        for key, new_value in best_params.items():
            if key in PROTECTED_PARAMS:
                continue
            if key not in SAFE_PARAMS:
                continue
            old_value = current_params.get(key)
            if old_value != new_value:
                changes.append({"param": key, "old": old_value, "new": new_value})

        if not changes:
            print("  Best params are identical to current config — nothing to apply.")
            return []

        print(f"  Improvement: {current_sharpe:.3f} → {best_sharpe:.3f} "
              f"({result.get('improvement_pct', 0):+.1f}% Sharpe)")
        print(f"  Applying {len(changes)} change(s) to config.yaml:")
        for c in changes:
            print(f"    {c['param']}: {c['old']} → {c['new']}")

        # Save old values to state.db before overwriting
        self._save_param_history(changes, current_sharpe, best_sharpe)

        # Apply each change to config.yaml
        applied = []
        with open(self.config_path, "r") as f:
            config_text = f.read()

        for change in changes:
            param = change["param"]
            new_val = change["new"]

            # Format value appropriately
            if isinstance(new_val, float):
                new_str = f"{new_val}"
            else:
                new_str = str(new_val)

            # Replace the param value in YAML (handles int, float, and negative float)
            pattern = rf"(\b{re.escape(param)}:\s*)-?\d+(\.\d+)?"
            new_config_text = re.sub(pattern, f"{param}: {new_str}", config_text)

            if new_config_text != config_text:
                config_text = new_config_text
                applied.append(change)
            else:
                print(f"  [WARNING] Could not find '{param}' in config.yaml — skipping.")

        if applied:
            with open(self.config_path, "w") as f:
                f.write(config_text)
            print(f"\n  ✓ config.yaml updated with {len(applied)} change(s).")
        else:
            print("  [WARNING] No changes could be applied to config.yaml.")

        # Log
        self.logger.log_status(
            f"Auto-tuner applied {len(applied)} parameter change(s).",
            {"changes": applied, "new_sharpe": best_sharpe},
        )

        # Telegram
        if self.telegram and applied:
            self._send_telegram(applied, current_sharpe, best_sharpe)

        print("=" * 60 + "\n")
        return applied

    def restore(self) -> bool:
        """
        Restore the previous set of parameters (undo last auto-tune).
        Called by 'python main.py restore_params'.
        """
        history = self._load_param_history()
        if not history:
            print("No previous parameters to restore.")
            return False

        last = history[-1]
        print(f"\nRestoring parameters from {last.get('applied_at', '?')}:")

        with open(self.config_path, "r") as f:
            config_text = f.read()

        restored = []
        for change in last.get("changes", []):
            param = change["param"]
            old_val = change["old"]
            new_str = str(old_val)
            pattern = rf"(\b{re.escape(param)}:\s*)-?\d+(\.\d+)?"
            new_config_text = re.sub(pattern, f"{param}: {new_str}", config_text)
            if new_config_text != config_text:
                config_text = new_config_text
                restored.append(change)
                print(f"  Restored {param}: {change['new']} → {change['old']}")

        if restored:
            with open(self.config_path, "w") as f:
                f.write(config_text)
            self.logger.log_status("Parameters restored to previous values.", {"restored": restored})
            return True
        return False

    # ── Param history ─────────────────────────────────────

    def _save_param_history(self, changes: list[dict], old_sharpe: float, new_sharpe: float):
        history = self._load_param_history()
        history.append({
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "changes": changes,
            "sharpe_before": old_sharpe,
            "sharpe_after": new_sharpe,
        })
        # Keep last 12 entries (1 year of monthly changes)
        history = history[-12:]
        self.state.set_kv("param_history", json.dumps(history))

    def _load_param_history(self) -> list[dict]:
        val = self.state.get_kv("param_history")
        if not val:
            return []
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []

    # ── Telegram ──────────────────────────────────────────

    def _send_telegram(self, changes: list[dict], old_sharpe: float, new_sharpe: float):
        try:
            lines = [
                f"🔧 *Auto-Tuner Applied {len(changes)} Change(s)*",
                f"Sharpe: {old_sharpe:.3f} → {new_sharpe:.3f}",
                "",
                "Changes:",
            ]
            for c in changes:
                lines.append(f"  • {c['param']}: {c['old']} → {c['new']}")
            lines.append("\nConfig updated. Takes effect on next market day.")
            self.telegram.send_message("\n".join(lines))
        except Exception as e:
            print(f"  [WARNING] Telegram auto-tune notification failed: {e}")
