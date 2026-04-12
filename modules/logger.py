"""
Structured JSON logging for the trading bot.
Writes decisions, predictions, and verification results to separate log files.
Supports two-level class allocation and instrument selection logging.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BotLogger:
    """Structured JSON-line logger for trading bot events."""

    def __init__(self, config: dict, base_dir: str = "."):
        log_cfg = config.get("logging", {})
        # On Railway, DATA_DIR points to the mounted persistent volume (/data).
        # Logs go there so they survive redeploys. Locally falls back to base_dir.
        data_dir = os.environ.get("DATA_DIR", base_dir)
        self.decisions_path = os.path.join(data_dir, log_cfg.get("decisions", "logs/decisions.log"))
        self.predictions_path = os.path.join(data_dir, log_cfg.get("predictions", "logs/predictions.log"))
        self.verification_path = os.path.join(data_dir, log_cfg.get("verification", "logs/verification.log"))

        # Ensure log directories exist
        for path in [self.decisions_path, self.predictions_path, self.verification_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, path: str, data: dict):
        """Write a JSON line to the specified log file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        # Sanitise: remove raw_response and very large fields to keep logs manageable
        for key in ["raw_response"]:
            entry.pop(key, None)
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_decision(self, data: dict):
        """Log a trading decision to decisions.log (includes class allocation + instruments)."""
        # Build a compact summary for the log
        summary = {
            "event": "decision",
            "action": data.get("action", "unknown"),
        }

        # Class allocation summary
        class_allocs = data.get("class_allocations", [])
        if class_allocs:
            summary["class_allocations"] = [
                {
                    "class": a["class_label"],
                    "decision": a["decision"],
                    "weight": a["weight"],
                    "capital": a["allocated_capital"],
                }
                for a in class_allocs
            ]

        # Cash position
        cash = data.get("cash", {})
        if cash:
            summary["cash"] = cash

        # Instrument selections summary
        inst_sels = data.get("instrument_selections", [])
        if inst_sels:
            summary["instruments"] = [
                {
                    "symbol": s["symbol"],
                    "class": s["class_label"],
                    "capital": s["allocated_capital"],
                    "momentum_pct": s.get("momentum_pct"),
                    "confidence": s.get("confidence", "N/A"),
                }
                for s in inst_sels
            ]

        # Verification summary
        ver = data.get("verification", {})
        if ver:
            summary["verification"] = {
                "verdict": ver.get("verdict"),
                "notes": (ver.get("notes") or "")[:200],
            }

        # Volatility flags
        vol_flags = data.get("volatility_flags", [])
        if vol_flags:
            summary["volatility_flags"] = [vf.get("message", "") for vf in vol_flags]

        # Portfolio values
        summary["pre_portfolio_value"] = data.get("pre_portfolio_value")
        summary["post_portfolio_value"] = data.get("post_portfolio_value")

        self._write(self.decisions_path, summary)

        # Console output — compact
        action = data.get("action", "?")
        verdict = ver.get("verdict", "N/A") if ver else "N/A"
        n_instruments = len(inst_sels)
        symbols = ", ".join(s["symbol"] for s in inst_sels) if inst_sels else "none"
        print(f"\n[DECISION] action={action} | verdict={verdict} | instruments={n_instruments} ({symbols})")

    def log_prediction(self, data: dict):
        """Log a prediction/confidence call to predictions.log."""
        self._write(self.predictions_path, {"event": "prediction", **data})

    def log_verification(self, data: dict):
        """Log verification layer output to verification.log."""
        # Strip raw_response to keep log size reasonable
        log_data = {k: v for k, v in data.items() if k != "raw_response"}
        self._write(self.verification_path, {"event": "verification", **log_data})

        # Print alerts for rejections
        verdict = data.get("verdict", "")
        if verdict == "REJECTED":
            reason = data.get("rejection_reason", data.get("reason", "Unknown"))
            print(f"\n[ALERT] Trade rejected by verification layer. Reason: {reason}. Holding current positions.\n")
        elif verdict == "APPROVED WITH NOTES":
            notes = data.get("notes", "")
            print(f"[VERIFY] Approved with notes: {notes[:200]}")
        else:
            print(f"[VERIFY] {verdict}")

    def log_order(self, data: dict):
        """Log an executed order."""
        self._write(self.decisions_path, {"event": "order", **data})
        symbol = data.get("symbol", "?")
        side = data.get("side", "?")
        qty = data.get("qty", "?")
        is_crypto = data.get("is_crypto", False)
        unit = "units" if is_crypto else "shares"
        print(f"[ORDER] {side} {qty} {unit} of {symbol}")

    def log_error(self, message: str, details: dict = None):
        """Log an error event."""
        entry = {"event": "error", "message": message}
        if details:
            entry.update(details)
        self._write(self.decisions_path, entry)
        print(f"[ERROR] {message}")

    def log_status(self, message: str, data: dict = None):
        """Log a status event."""
        entry = {"event": "status", "message": message}
        if data:
            entry.update(data)
        self._write(self.decisions_path, entry)
        print(f"[STATUS] {message}")

    def tail(self, n: int = 20) -> list[str]:
        """Return last n lines from decisions.log."""
        try:
            with open(self.decisions_path, "r") as f:
                lines = f.readlines()
            return lines[-n:]
        except FileNotFoundError:
            return ["No decisions log found."]
