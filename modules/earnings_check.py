"""
Earnings Check — Flag instruments with earnings reports in the next N days.

Runs as part of instrument selection in the executor, before verification.
Uses yfinance (already a dependency via parameter_optimizer) to pull each
symbol's upcoming earnings date.

Why this matters:
  An earnings surprise can move a stock 10–20% overnight. If the bot enters
  a position the day before earnings, it's taking unquantified binary risk
  that has nothing to do with momentum. This module flags those instruments
  so the verification layer and Claude can make an informed decision.

Behaviour:
  - WARN  (1–3 days): confidence downgraded to LOW, flagged in verification package
  - CAUTION (4–7 days): noted in Claude note, confidence stays as-is
  - CLEAR (>7 days or unknown): no change

Crypto symbols (ending in USD) are skipped — they don't report earnings.
ETFs are skipped — they don't report earnings either.
"""

from __future__ import annotations

from datetime import date, timedelta

# Known ETFs in the universe — skip earnings checks for these
ETF_SYMBOLS = {
    "SPY", "QQQ", "IWM", "VTI", "ARKK", "XLK", "XLE", "XLF",
    "GLD", "SLV", "USO", "PDBC",
    "EFA", "EEM", "EWJ", "EWA", "INDA",
    "TLT",
}


def _is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith("USD")


def _is_etf(symbol: str) -> bool:
    return symbol.upper() in ETF_SYMBOLS


def get_earnings_flags(symbols: list[str], warn_days: int = 3, caution_days: int = 7) -> dict[str, dict]:
    """
    Check each symbol for upcoming earnings dates.

    Returns a dict keyed by symbol:
      {
        "symbol": "NVDA",
        "earnings_date": date(2026, 4, 16),
        "days_until": 2,
        "level": "WARN" | "CAUTION" | "CLEAR",
        "message": "NVDA reports earnings in 2 days (2026-04-16) — binary risk."
      }

    Symbols with no upcoming earnings data return level="CLEAR".
    """
    results: dict[str, dict] = {}
    today = date.today()

    # Lazy import — yfinance is only needed here, not at bot startup
    try:
        import yfinance as yf
    except ImportError:
        print("[EARNINGS] yfinance not installed — skipping earnings check.")
        return {}

    for symbol in symbols:
        if _is_crypto(symbol) or _is_etf(symbol):
            results[symbol] = {"symbol": symbol, "level": "CLEAR", "message": "ETF/crypto — no earnings."}
            continue

        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar

            earnings_date = None

            # yfinance returns a dict or DataFrame depending on version
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date") or cal.get("earnings_date")
                if raw:
                    # May be a list of dates
                    if isinstance(raw, list):
                        raw = raw[0]
                    if hasattr(raw, "date"):
                        earnings_date = raw.date()
                    elif isinstance(raw, date):
                        earnings_date = raw
            elif cal is not None and hasattr(cal, "loc"):
                # DataFrame — try to get the value
                try:
                    val = cal.loc["Earnings Date"].iloc[0]
                    if hasattr(val, "date"):
                        earnings_date = val.date()
                except Exception:
                    pass

        except Exception as e:
            print(f"[EARNINGS] Could not fetch calendar for {symbol}: {e}")
            results[symbol] = {"symbol": symbol, "level": "CLEAR", "message": f"Calendar unavailable: {e}"}
            continue

        if earnings_date is None or earnings_date < today:
            results[symbol] = {"symbol": symbol, "level": "CLEAR", "message": "No upcoming earnings found."}
            continue

        days_until = (earnings_date - today).days

        if days_until <= warn_days:
            level = "WARN"
            message = (
                f"{symbol} reports earnings in {days_until} day{'s' if days_until != 1 else ''} "
                f"({earnings_date}) — binary risk, confidence downgraded to LOW."
            )
        elif days_until <= caution_days:
            level = "CAUTION"
            message = (
                f"{symbol} reports earnings in {days_until} days ({earnings_date}) — "
                f"monitor closely."
            )
        else:
            level = "CLEAR"
            message = f"{symbol} earnings in {days_until} days ({earnings_date}) — clear."

        results[symbol] = {
            "symbol": symbol,
            "earnings_date": earnings_date.isoformat(),
            "days_until": days_until,
            "level": level,
            "message": message,
        }

    return results


def apply_earnings_flags(instrument_selections: list[dict], flags: dict[str, dict]) -> list[dict]:
    """
    Apply earnings flags to instrument selections in-place:
      - WARN  → downgrade confidence to LOW, append earnings note
      - CAUTION → append note, keep confidence

    Returns the modified list.
    """
    for sel in instrument_selections:
        symbol = sel["symbol"]
        flag = flags.get(symbol, {})
        level = flag.get("level", "CLEAR")

        if level == "WARN":
            sel["confidence"] = "LOW"
            existing_note = sel.get("claude_note", "")
            sel["claude_note"] = (
                f"⚠ EARNINGS IN {flag.get('days_until', '?')} DAYS ({flag.get('earnings_date', '?')}) — "
                f"binary risk. {existing_note}"
            ).strip()
            sel["earnings_flag"] = flag

        elif level == "CAUTION":
            existing_note = sel.get("claude_note", "")
            sel["claude_note"] = (
                f"Earnings in {flag.get('days_until', '?')} days ({flag.get('earnings_date', '?')}). "
                f"{existing_note}"
            ).strip()
            sel["earnings_flag"] = flag

    return instrument_selections
