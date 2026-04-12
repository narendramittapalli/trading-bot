"""
Telegram alerts for the two-level trading bot.
Sends rebalance summaries and cash-hold notifications via Telegram Bot API.

Setup:
  1. Message @BotFather on Telegram → /newbot → get your BOT_TOKEN
  2. Message your new bot, then visit:
     https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
     to find your chat_id
  3. Set in config.yaml or env vars:
     - TELEGRAM_BOT_TOKEN
     - telegram.chat_id in config (or TELEGRAM_CHAT_ID env var)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests


class TelegramAlerts:
    """Telegram Bot API alerting for rebalance events."""

    def __init__(self, config: dict):
        tg_cfg = config.get("telegram", {})
        self.enabled = tg_cfg.get("enabled", False)
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", tg_cfg.get("bot_token", ""))
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", str(tg_cfg.get("chat_id", "")))

        global_cfg = config.get("global", {})
        self.capital = global_cfg.get("capital", 500)

    def _send(self, text: str, parse_mode: str = "HTML"):
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            print("[TELEGRAM] Alerts disabled — skipping.")
            return
        if not self.bot_token or not self.chat_id:
            print("[TELEGRAM] Missing bot_token or chat_id — skipping.")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        # Telegram message limit is 4096 chars — split if needed
        chunks = self._split_message(text, max_len=4000)

        for chunk in chunks:
            try:
                resp = requests.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": chunk,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    print(f"[TELEGRAM] Alert sent to chat {self.chat_id}")
                else:
                    error = resp.json().get("description", resp.text[:200])
                    print(f"[TELEGRAM] API error {resp.status_code}: {error}")
            except Exception as e:
                print(f"[TELEGRAM] Send failed: {e}")

    @staticmethod
    def _split_message(text: str, max_len: int = 4000) -> list[str]:
        """Split a long message into chunks that fit Telegram's limit."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        lines = text.split("\n")
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > max_len:
                if current:
                    chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)
        return chunks

    def send_rebalance_alert(self, result: dict):
        """Send a two-level rebalance summary via Telegram."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        action = result.get("action", "unknown")
        verdict = result.get("verification", {}).get("verdict", "N/A")

        # Decision emoji
        emoji = {"execute": "\u2705", "hold": "\u26a0\ufe0f", "hold_cash": "\U0001f4b0"}.get(action, "\u2753")

        lines = [
            f"{emoji} <b>REBALANCE — {action.upper()}</b>",
            f"<i>{ts}</i>",
            f"Verdict: <b>{verdict}</b> | Capital: <b>${self.capital}</b>",
            "",
        ]

        # Level 1 — Class Allocation
        class_allocs = result.get("class_allocations", [])
        if class_allocs:
            lines.append("<b>\u2014 Level 1: Class Allocation \u2014</b>")
            for a in class_allocs:
                dec = a["decision"]
                icon = {"ACTIVE": "\U0001f7e2", "REDUCE": "\U0001f7e1", "SKIP": "\u26d4"}.get(dec, "\u2022")
                weight_pct = f"{a['weight']:.0%}"
                lines.append(
                    f"{icon} {a['class_label']}: <b>{dec}</b>  "
                    f"{weight_pct}  ${a['allocated_capital']:.2f}"
                )
            lines.append("")

        # Cash position
        cash_info = result.get("cash", {})
        if cash_info.get("capital", 0) > 0:
            lines.append(f"\U0001f4b5 Cash: ${cash_info['capital']:.2f} ({cash_info['weight']:.0%})")
            lines.append("")

        # Level 2 — Instrument Selections
        inst_sels = result.get("instrument_selections", [])
        if inst_sels:
            lines.append("<b>\u2014 Level 2: Instruments \u2014</b>")
            for s in inst_sels:
                mom = s.get("momentum_pct", 0)
                arrow = "\u2191" if mom >= 0 else "\u2193"
                conf = s.get("confidence", "N/A")
                lines.append(
                    f"  <b>{s['symbol']}</b> ({s['class_label']})\n"
                    f"    ${s['allocated_capital']:.2f} | "
                    f"{arrow}{mom:+.2f}% | "
                    f"${s.get('current_price', 0):.2f} | "
                    f"{s.get('target_shares', 0):.4f} shr | "
                    f"conf={conf}"
                )
            lines.append("")

        # Volatility flags
        vol_flags = result.get("volatility_flags", [])
        if vol_flags:
            lines.append("<b>\u26a1 Volatility Flags</b>")
            for vf in vol_flags:
                lines.append(f"  \u26a0\ufe0f {vf['message']}")
            lines.append("")

        # Verification
        ver = result.get("verification", {})
        if ver:
            notes = ver.get("notes", "")
            if notes:
                lines.append(f"<b>Verification notes:</b> {notes[:300]}")

        # Execution
        execution = result.get("execution", [])
        if execution:
            lines.append("<b>\u2014 Execution \u2014</b>")
            for ex in execution:
                lines.append(f"  {ex.get('type', '?')}: {ex.get('symbol', '?')} — {ex.get('status', '')}")
            lines.append("")

        # Portfolio
        pre = result.get("pre_portfolio_value")
        post = result.get("post_portfolio_value")
        if pre is not None or post is not None:
            lines.append(f"Portfolio: ${pre or '?'} \u2192 ${post or '?'}")

        self._send("\n".join(lines))

    def send_cash_hold_alert(self, result: dict):
        """Send a cash-hold notification when all classes are skipped."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        reason = result.get("reason", "All asset classes set to SKIP.")

        msg = (
            f"\U0001f4b0 <b>CASH HOLD</b>\n"
            f"<i>{ts}</i>\n\n"
            f"All asset classes set to SKIP.\n"
            f"<b>Reason:</b> {reason}\n\n"
            f"Capital: <b>${self.capital}</b> (100% cash)\n"
            f"No trades executed this cycle."
        )

        self._send(msg)
