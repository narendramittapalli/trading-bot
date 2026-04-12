"""
Email alerts for the two-level trading bot.
Sends rebalance summaries and cash-hold notifications via SMTP.
"""

from __future__ import annotations

import smtplib
import os
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailAlerts:
    """SMTP email alerting for rebalance events."""

    def __init__(self, config: dict):
        email_cfg = config.get("email", {})
        self.enabled = email_cfg.get("enabled", False)
        self.smtp_host = email_cfg.get("smtp_host", "smtp.gmail.com")
        self.smtp_port = email_cfg.get("smtp_port", 587)
        self.sender = email_cfg.get("sender", "")
        self.recipient = email_cfg.get("recipient", "")
        self.password = os.environ.get("EMAIL_APP_PASSWORD", "")

        global_cfg = config.get("global", {})
        self.capital = global_cfg.get("capital", 500)

    def _send(self, subject: str, body_html: str):
        """Send an email via SMTP with TLS."""
        if not self.enabled:
            print("[EMAIL] Alerts disabled — skipping.")
            return
        if not self.sender or not self.recipient or not self.password:
            print("[EMAIL] Missing sender, recipient, or EMAIL_APP_PASSWORD — skipping.")
            return

        msg = MIMEMultipart("alternative")
        msg["From"] = self.sender
        msg["To"] = self.recipient
        msg["Subject"] = subject

        # Plain-text fallback
        plain = body_html.replace("<br>", "\n").replace("</td><td>", " | ")
        # Strip remaining tags (simple approach)
        import re
        plain = re.sub(r"<[^>]+>", "", plain)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipient, msg.as_string())
            print(f"[EMAIL] Alert sent to {self.recipient}")
        except Exception as e:
            print(f"[EMAIL] Send failed: {e}")

    def send_rebalance_alert(self, result: dict):
        """Send a two-level rebalance summary email."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        action = result.get("action", "unknown")
        verdict = result.get("verification", {}).get("verdict", "N/A")

        subject = f"Trading Bot — {action.upper()} — {ts}"

        # Build HTML body
        lines = [
            "<html><body style='font-family: monospace; font-size: 13px;'>",
            f"<h2>Rebalance Report — {ts}</h2>",
            f"<p><b>Action:</b> {action.upper()} | <b>Verdict:</b> {verdict} | <b>Capital:</b> ${self.capital}</p>",
        ]

        # Class allocations
        class_allocs = result.get("class_allocations", [])
        if class_allocs:
            lines.append("<h3>Level 1 — Class Allocation</h3>")
            lines.append("<table border='1' cellpadding='4' cellspacing='0'>")
            lines.append("<tr><th>Class</th><th>Decision</th><th>Weight</th><th>Capital</th><th>Reason</th></tr>")
            for a in class_allocs:
                color = {"ACTIVE": "#2d7d2d", "REDUCE": "#c77d00", "SKIP": "#999"}.get(a["decision"], "#333")
                lines.append(
                    f"<tr><td>{a['class_label']}</td>"
                    f"<td style='color:{color};font-weight:bold'>{a['decision']}</td>"
                    f"<td>{a['weight']:.0%}</td>"
                    f"<td>${a['allocated_capital']:.2f}</td>"
                    f"<td>{a.get('reason', '')}</td></tr>"
                )
            lines.append("</table>")

        cash_info = result.get("cash", {})
        if cash_info.get("capital", 0) > 0:
            lines.append(f"<p><b>Cash held:</b> ${cash_info['capital']:.2f} ({cash_info['weight']:.0%})</p>")

        # Instrument selections
        inst_sels = result.get("instrument_selections", [])
        if inst_sels:
            lines.append("<h3>Level 2 — Instrument Selections</h3>")
            lines.append("<table border='1' cellpadding='4' cellspacing='0'>")
            lines.append(
                "<tr><th>Symbol</th><th>Class</th><th>Alloc</th>"
                "<th>Momentum</th><th>Price</th><th>Shares</th><th>Confidence</th></tr>"
            )
            for s in inst_sels:
                lines.append(
                    f"<tr><td><b>{s['symbol']}</b></td>"
                    f"<td>{s['class_label']}</td>"
                    f"<td>${s['allocated_capital']:.2f}</td>"
                    f"<td>{s.get('momentum_pct', 0):+.2f}%</td>"
                    f"<td>${s.get('current_price', 0):.2f}</td>"
                    f"<td>{s.get('target_shares', 0):.6f}</td>"
                    f"<td>{s.get('confidence', 'N/A')}</td></tr>"
                )
            lines.append("</table>")

        # Volatility flags
        vol_flags = result.get("volatility_flags", [])
        if vol_flags:
            lines.append("<h3>Volatility Flags</h3><ul>")
            for vf in vol_flags:
                lines.append(f"<li style='color:#c00'>{vf['message']}</li>")
            lines.append("</ul>")

        # Verification
        ver = result.get("verification", {})
        if ver:
            notes = ver.get("notes", "")
            lines.append(f"<h3>Verification</h3>")
            lines.append(f"<p><b>Verdict:</b> {ver.get('verdict', 'N/A')}</p>")
            if notes:
                lines.append(f"<p><b>Notes:</b> {notes[:500]}</p>")

        # Execution
        execution = result.get("execution", [])
        if execution:
            lines.append("<h3>Execution</h3><ul>")
            for ex in execution:
                lines.append(f"<li>{ex.get('type', '?')}: {ex.get('symbol', '?')} — {ex.get('status', '')}</li>")
            lines.append("</ul>")

        # Portfolio values
        pre = result.get("pre_portfolio_value")
        post = result.get("post_portfolio_value")
        if pre is not None or post is not None:
            lines.append(f"<p><b>Portfolio:</b> ${pre or '?'} → ${post or '?'}</p>")

        lines.append("</body></html>")

        self._send(subject, "\n".join(lines))

    def send_cash_hold_alert(self, result: dict):
        """Send a cash-hold notification when all classes are skipped."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        reason = result.get("reason", "All asset classes set to SKIP.")

        subject = f"Trading Bot — CASH HOLD — {ts}"

        body = (
            f"<html><body style='font-family: monospace; font-size: 13px;'>"
            f"<h2>Cash Hold Alert — {ts}</h2>"
            f"<p><b>Action:</b> HOLD CASH</p>"
            f"<p><b>Reason:</b> {reason}</p>"
            f"<p><b>Capital:</b> ${self.capital} (100% cash)</p>"
            f"<p>No trades were executed this cycle. All capital remains in cash.</p>"
            f"</body></html>"
        )

        self._send(subject, body)
