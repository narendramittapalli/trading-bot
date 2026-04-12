"""
Risk Manager — Position-level stop-loss tracking.

Tracks entry prices for each open position (stored in SQLite via StateManager).
At the start of each adaptive_check, compares current prices against entry prices
and flags any position that has fallen beyond the configured stop-loss threshold.

Per-asset-class stops:
  - Equities: stop_loss_pct (default 8%)
  - Crypto:   crypto_stop_loss_pct (default 5%) — tighter because crypto
              moves 3-5x faster and crashes harder than equities

Config:
  risk:
    stop_loss_pct:        8.0   # Equities stop
    crypto_stop_loss_pct: 5.0   # Crypto stop (tighter)
    min_hold_days:        1     # Avoid noise on day 1
"""

from __future__ import annotations

from datetime import datetime, timezone

from modules.momentum import is_crypto


class RiskManager:
    """Position-level stop-loss checker with per-asset-class thresholds."""

    def __init__(self, config: dict, state_manager):
        risk_cfg = config.get("risk", {})
        self.stop_loss_pct = risk_cfg.get("stop_loss_pct", 8.0)
        self.crypto_stop_loss_pct = risk_cfg.get("crypto_stop_loss_pct", 5.0)
        self.min_hold_days = risk_cfg.get("min_hold_days", 1)
        self.state = state_manager

    def _stop_threshold(self, symbol: str) -> float:
        """Return the appropriate stop-loss % for a symbol."""
        if is_crypto(symbol):
            return self.crypto_stop_loss_pct
        return self.stop_loss_pct

    def check_position_stops(self, current_positions: list[dict]) -> list[dict]:
        """
        Compare current positions against recorded entry prices.
        Returns list of positions that have breached the stop-loss threshold.
        """
        entry_data = self.state.get_entry_prices()
        if not entry_data:
            return []

        current = {p["symbol"]: p for p in current_positions}
        triggered = []
        now = datetime.now(timezone.utc)

        for symbol, entry_info in entry_data.items():
            if symbol not in current:
                self.state.clear_entry_prices([symbol])
                continue

            entry_price = entry_info.get("price", 0)
            entry_ts = entry_info.get("ts", "")
            if entry_price <= 0:
                continue

            # Respect min hold period
            if entry_ts and self.min_hold_days > 0:
                try:
                    entry_dt = datetime.fromisoformat(entry_ts)
                    held_days = (now - entry_dt).days
                    if held_days < self.min_hold_days:
                        continue
                except ValueError:
                    pass

            pos = current[symbol]
            current_price = float(pos.get("current_price", 0))
            if current_price <= 0:
                continue

            pct_change = (current_price - entry_price) / entry_price * 100
            threshold = self._stop_threshold(symbol)

            if pct_change <= -threshold:
                triggered.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pct_change": round(pct_change, 2),
                    "qty": float(pos.get("qty", 0)),
                    "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                    "stop_threshold": threshold,
                    "is_crypto": is_crypto(symbol),
                })

        return triggered

    def get_position_health(self, current_positions: list[dict]) -> list[dict]:
        """Returns health summary for all tracked positions (for status display)."""
        entry_data = self.state.get_entry_prices()
        current = {p["symbol"]: p for p in current_positions}
        health = []

        for symbol, entry_info in entry_data.items():
            entry_price = entry_info.get("price", 0)
            if symbol not in current or entry_price <= 0:
                continue

            pos = current[symbol]
            current_price = float(pos.get("current_price", 0))
            pct_change = (current_price - entry_price) / entry_price * 100 if current_price > 0 else 0
            threshold = self._stop_threshold(symbol)

            health.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "current_price": current_price,
                "pct_change": round(pct_change, 2),
                "stop_loss_level": round(entry_price * (1 - threshold / 100), 4),
                "stop_threshold_pct": threshold,
                "stop_triggered": pct_change <= -threshold,
                "is_crypto": is_crypto(symbol),
            })

        return sorted(health, key=lambda x: x["pct_change"])
