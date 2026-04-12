"""
State Manager — Persists bot state across sessions using SQLite.

Replaces the fragile JSON file with a proper embedded database.
Uses Python stdlib sqlite3 — no extra dependencies.

Tables:
  kv       — key/value for scalar state (peak, last_rebalance, etc.)
  holdings — last rebalance target symbols + momentum scores
  deposits — deposit history

SQLite file location: trading-bot/state.db
Backward-compatible interface — same public methods as the JSON version.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone


class StateManager:
    """Reads/writes persistent bot state to a SQLite database."""

    def __init__(self, base_dir: str):
        # On Railway (or any cloud deployment), DATA_DIR env var points to the
        # mounted persistent volume (e.g. /data). Locally falls back to base_dir.
        data_dir = os.environ.get("DATA_DIR", base_dir)
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "state.db")
        self._conn = self._connect()
        self._create_tables()
        self._migrate_from_json(base_dir)

    # ── Connection & schema ───────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL mode: safer concurrent reads, better for Railway volumes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS holdings (
                symbol         TEXT PRIMARY KEY,
                momentum_score REAL NOT NULL,
                updated_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS deposits (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                date             TEXT NOT NULL,
                amount           REAL NOT NULL,
                capital_after    REAL NOT NULL
            );
        """)
        self._conn.commit()

    def _migrate_from_json(self, base_dir: str):
        """One-time migration from state.json → state.db, then renames the old file."""
        json_path = os.path.join(base_dir, "state.json")
        if not os.path.exists(json_path):
            return
        # Only migrate if db is empty (first run after upgrade)
        row = self._conn.execute("SELECT value FROM kv WHERE key='migrated'").fetchone()
        if row:
            return
        try:
            with open(json_path) as f:
                data = json.load(f)

            peak = data.get("peak_portfolio_value")
            if peak is not None:
                self._set_kv("peak_portfolio_value", str(peak))

            last_rebalance = data.get("last_rebalance")
            if last_rebalance:
                self._set_kv("last_rebalance", last_rebalance)

            total_deposited = data.get("total_deposited", 0.0)
            self._set_kv("total_deposited", str(total_deposited))

            # Holdings
            last_holdings = data.get("last_holdings", [])
            last_scores = data.get("last_momentum_scores", {})
            now = datetime.now(timezone.utc).isoformat()
            for sym in last_holdings:
                score = last_scores.get(sym, 0.0)
                self._conn.execute(
                    "INSERT OR REPLACE INTO holdings (symbol, momentum_score, updated_at) VALUES (?,?,?)",
                    (sym, score, now),
                )

            # Deposits
            for d in data.get("deposits", []):
                self._conn.execute(
                    "INSERT INTO deposits (date, amount, capital_after) VALUES (?,?,?)",
                    (d["date"], d["amount"], d["capital_after"]),
                )

            self._set_kv("migrated", "1")
            self._conn.commit()

            # Rename old file so we don't migrate again
            os.rename(json_path, json_path + ".migrated")
            print(f"[STATE] Migrated state.json → state.db")

        except Exception as e:
            print(f"[STATE] Migration warning: {e}")

    # ── KV helpers ────────────────────────────────────────

    def _set_kv(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)", (key, value)
        )

    def _get_kv(self, key: str, default=None):
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    # ── Peak / Drawdown ───────────────────────────────────

    def update_peak(self, portfolio_value: float):
        """Update peak if current value is a new high."""
        current_peak = self.get_peak()
        if current_peak is None or portfolio_value > current_peak:
            self._set_kv("peak_portfolio_value", str(round(portfolio_value, 2)))
            self._conn.commit()

    def get_peak(self) -> float | None:
        val = self._get_kv("peak_portfolio_value")
        return float(val) if val is not None else None

    def get_drawdown_pct(self, current_value: float) -> float:
        peak = self.get_peak()
        if peak is None or peak == 0:
            return 0.0
        return round((peak - current_value) / peak * 100, 2)

    def check_drawdown(self, current_value: float, max_drawdown_pct: float) -> bool:
        return self.get_drawdown_pct(current_value) >= max_drawdown_pct

    # ── Holdings / Signal tracking ────────────────────────

    def get_last_holdings(self) -> list[str]:
        rows = self._conn.execute("SELECT symbol FROM holdings").fetchall()
        return sorted(r["symbol"] for r in rows)

    def get_last_momentum_scores(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT symbol, momentum_score FROM holdings"
        ).fetchall()
        return {r["symbol"]: r["momentum_score"] for r in rows}

    def update_holdings(self, symbols: list[str], momentum_scores: dict[str, float]):
        """Replace holdings with new rebalance targets."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("DELETE FROM holdings")
        for sym in symbols:
            score = momentum_scores.get(sym, 0.0)
            self._conn.execute(
                "INSERT INTO holdings (symbol, momentum_score, updated_at) VALUES (?,?,?)",
                (sym, score, now),
            )
        self._set_kv("last_rebalance", now)
        self._conn.commit()

    def get_last_rebalance(self) -> str | None:
        return self._get_kv("last_rebalance")

    # ── Entry price tracking (for position-level stops) ───

    def record_entry_prices(self, entries: dict[str, float]):
        """
        Save entry prices for newly bought positions.
        entries: {symbol: entry_price}
        """
        now = datetime.now(timezone.utc).isoformat()
        for symbol, price in entries.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)",
                (f"entry_price:{symbol}", json.dumps({"price": price, "ts": now})),
            )
        self._conn.commit()

    def get_entry_prices(self) -> dict[str, dict]:
        """Return all tracked entry prices: {symbol: {price, ts}}"""
        rows = self._conn.execute(
            "SELECT key, value FROM kv WHERE key LIKE 'entry_price:%'"
        ).fetchall()
        result = {}
        for row in rows:
            symbol = row["key"].replace("entry_price:", "")
            result[symbol] = json.loads(row["value"])
        return result

    def clear_entry_prices(self, symbols: list[str] = None):
        """Clear entry prices for sold positions (or all if symbols is None)."""
        if symbols is None:
            self._conn.execute("DELETE FROM kv WHERE key LIKE 'entry_price:%'")
        else:
            for sym in symbols:
                self._conn.execute(
                    "DELETE FROM kv WHERE key=?", (f"entry_price:{sym}",)
                )
        self._conn.commit()

    # ── Public KV helpers (for external modules) ─────────

    def set_kv(self, key: str, value: str):
        """Public wrapper — store an arbitrary string value under key."""
        self._set_kv(key, value)

    def get_kv(self, key: str) -> str | None:
        """Public wrapper — retrieve a stored string value."""
        return self._get_kv(key)

    # ── Learning records (weekly review) ─────────────────

    def store_learning_record(self, record: dict):
        """Append a weekly performance review record."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                week_ending TEXT NOT NULL,
                data        TEXT NOT NULL
            );
        """)
        self._conn.execute(
            "INSERT INTO learning_records (week_ending, data) VALUES (?,?)",
            (record.get("week_ending", ""), json.dumps(record, default=str)),
        )
        self._conn.commit()

    def get_learning_history(self, n: int = 12) -> list[dict]:
        """Return the last n weekly review records."""
        try:
            rows = self._conn.execute(
                "SELECT data FROM learning_records ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            return [json.loads(r["data"]) for r in rows]
        except Exception:
            return []

    # ── Optimization results (parameter optimizer) ────────

    def store_optimization_result(self, result: dict):
        """Store the latest parameter optimization result."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at    TEXT NOT NULL,
                data      TEXT NOT NULL
            );
        """)
        self._conn.execute(
            "INSERT INTO optimization_results (run_at, data) VALUES (?,?)",
            (result.get("timestamp", ""), json.dumps(result, default=str)),
        )
        self._set_kv("latest_optimization", json.dumps(result, default=str))
        self._conn.commit()

    def get_latest_optimization(self) -> dict | None:
        """Return the most recent optimization result, or None."""
        val = self._get_kv("latest_optimization")
        if val:
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
        return None

    # ── Deposits ──────────────────────────────────────────

    def record_deposit(self, amount: float, new_total_capital: float):
        today = datetime.now(timezone.utc).date().isoformat()
        self._conn.execute(
            "INSERT INTO deposits (date, amount, capital_after) VALUES (?,?,?)",
            (today, round(amount, 2), round(new_total_capital, 2)),
        )
        total = self.get_total_deposited() + amount
        self._set_kv("total_deposited", str(round(total, 2)))
        self._conn.commit()

    def get_total_deposited(self) -> float:
        val = self._get_kv("total_deposited")
        return float(val) if val else 0.0

    def get_deposits(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT date, amount, capital_after FROM deposits ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Summary ───────────────────────────────────────────

    def print_summary(self):
        peak = self.get_peak()
        last_rebalance = self.get_last_rebalance()
        total_deposited = self.get_total_deposited()
        deposits = self.get_deposits()
        holdings = self.get_last_holdings()

        print("\n" + "=" * 55)
        print("BOT STATE SUMMARY")
        print("=" * 55)
        if peak:
            print(f"  Peak portfolio value : ${peak:,.2f}")
        else:
            print("  Peak portfolio value : N/A (no history yet)")
        print(f"  Last rebalance       : {last_rebalance[:19] if last_rebalance else 'Never'}")
        print(f"  Last target holdings : {holdings or 'None'}")
        print(f"  Total deposited      : ${total_deposited:,.2f}")
        if deposits:
            print(f"  Deposit history ({len(deposits)}):")
            for d in deposits[-5:]:
                print(f"    {d['date']}  +${d['amount']:.2f}  → capital: ${d['capital_after']:.2f}")
        print("=" * 55 + "\n")
