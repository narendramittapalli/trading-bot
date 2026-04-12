"""
Historical Backtest — Two-Level Momentum Strategy (Fixed + Enhanced)
=====================================================================
Simulates the momentum strategy on historical data using yfinance.

Fixes from v1:
  - Missing price data no longer silently destroys capital.
    If open price is unavailable, we use the last known close.
    If no data at all for a symbol on liquidation, we carry the
    last known value rather than zeroing it.
  - Portfolio value tracking is consistent across all iterations.

Strategy improvements vs original live bot:
  - Risk-adjusted momentum (return / annualised_vol) ranks instruments.
    A 10% return with 5% vol beats a 10% return with 30% vol.
  - Minimum class momentum threshold: classes with avg momentum below
    min_class_momentum_pct are forced SKIP (not just de-weighted).
  - Market regime filter: if SPY 20-day momentum < regime_threshold,
    all ACTIVE allocations are halved (bear) or zeroed (severe bear).
  - Rebalances monthly (first trading day of each month).

Metrics reported:
  Total return, CAGR, Sharpe, Sortino, Max drawdown, Calmar,
  Win rate, Alpha vs SPY, year-by-year, last-12-month detail.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance: pip install yfinance")


# ── Constants ──────────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.045       # 4.5% annual
TRADING_DAYS_PER_YEAR = 252


# ── Price helpers ──────────────────────────────────────────────────────────────

CRYPTO_YF_MAP = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "LTCUSD": "LTC-USD",
    "BCHUSD": "BCH-USD",
}


def _to_yf(symbol: str) -> str:
    return CRYPTO_YF_MAP.get(symbol, symbol)


def _from_yf(yf_sym: str, orig_map: dict) -> str:
    return orig_map.get(yf_sym, yf_sym)


def _is_crypto(symbol: str) -> bool:
    return symbol in CRYPTO_YF_MAP or symbol.endswith("USD")


def fetch_price_data(symbols: list[str], period_years: int = 2) -> dict[str, list[dict]]:
    """
    Download daily OHLCV from Yahoo Finance.
    Returns {orig_symbol: [{date, open, close}, ...]} ascending.

    If a symbol fails to download, it is excluded with a warning
    rather than silently dropping its capital later.
    """
    yf_syms = [_to_yf(s) for s in symbols]
    orig_map = {_to_yf(s): s for s in symbols}

    end = date.today()
    start = end.replace(year=end.year - period_years)
    # Extra buffer so we have lookback data from day 1 of the sim
    start_buffered = start.replace(year=start.year - 1)

    print(f"  Downloading {len(yf_syms)} symbols {start_buffered} → {end}...")

    raw = yf.download(
        yf_syms,
        start=str(start_buffered),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    data: dict[str, list[dict]] = {}

    def _extract(yf_sym: str) -> list[dict]:
        orig = _from_yf(yf_sym, orig_map)
        bars = []
        try:
            if len(yf_syms) == 1:
                opens  = raw["Open"]
                closes = raw["Close"]
            else:
                opens  = raw["Open"][yf_sym]
                closes = raw["Close"][yf_sym]

            for idx in closes.index:
                o = float(opens[idx])
                c = float(closes[idx])
                if math.isnan(o) or math.isnan(c) or o <= 0 or c <= 0:
                    continue
                bars.append({"date": idx.date(), "open": o, "close": c})
        except (KeyError, TypeError, Exception) as e:
            print(f"  Warning: could not extract {yf_sym}: {e}")
        bars.sort(key=lambda x: x["date"])
        return bars

    for yf_sym in yf_syms:
        orig = _from_yf(yf_sym, orig_map)
        bars = _extract(yf_sym)
        if bars:
            data[orig] = bars
        else:
            print(f"  Warning: no usable data for {orig} ({yf_sym}) — excluded from backtest.")

    return data


# ── Price lookups (never lose capital on missing data) ─────────────────────────

def get_price_on_or_after(series: list[dict], on_date: date) -> float | None:
    """First available open on or after on_date."""
    for b in series:
        if b["date"] >= on_date:
            return b["open"]
    return None


def get_price_on_or_before(series: list[dict], on_date: date) -> float | None:
    """Last available close on or before on_date."""
    result = None
    for b in series:
        if b["date"] <= on_date:
            result = b["close"]
        else:
            break
    return result


def get_best_price(series: list[dict], on_date: date) -> float | None:
    """
    Best-effort price for on_date:
    1. Try open on that date
    2. Fall back to last close before that date
    3. Fall back to first open after that date
    Returns None only if series is completely empty.
    """
    # Exact or next open
    for b in series:
        if b["date"] == on_date:
            return b["open"]

    # Last close before
    last_close = get_price_on_or_before(series, on_date)
    if last_close:
        return last_close

    # First open after
    return get_price_on_or_after(series, on_date)


# ── Momentum + risk-adjusted momentum ─────────────────────────────────────────

def compute_momentum(series: list[dict], as_of: date, lookback: int = 20) -> float | None:
    avail = [b for b in series if b["date"] <= as_of]
    if len(avail) < lookback + 1:
        return None
    start = avail[-lookback]["close"]
    end   = avail[-1]["close"]
    return (end - start) / start if start > 0 else None


def compute_risk_adjusted_momentum(series: list[dict], as_of: date, lookback: int = 20) -> float | None:
    """
    Return momentum / annualised_volatility.
    Penalises high-vol assets even when raw return looks attractive.
    Falls back to raw momentum if vol can't be computed.
    """
    mom = compute_momentum(series, as_of, lookback)
    if mom is None:
        return None

    avail = [b for b in series if b["date"] <= as_of]
    if len(avail) < lookback + 2:
        return mom  # not enough for vol

    closes = [b["close"] for b in avail[-lookback-1:]]
    rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]

    if len(rets) < 4:
        return mom

    mean = sum(rets) / len(rets)
    var  = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    daily_vol = math.sqrt(var) if var > 0 else 0

    ann_vol = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    if ann_vol < 0.01:
        return mom

    return mom / ann_vol


# ── Rebalance calendar ─────────────────────────────────────────────────────────

def get_rebalance_dates(price_data: dict, sim_start: date, sim_end: date) -> list[date]:
    """First trading day of each calendar month within [sim_start, sim_end]."""
    all_trading = sorted({b["date"] for bars in price_data.values() for b in bars
                          if sim_start <= b["date"] <= sim_end})
    seen, result = set(), []
    for d in all_trading:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


# ── Strategy simulation ────────────────────────────────────────────────────────

def run_strategy(
    price_data: dict[str, list[dict]],
    spy_data: list[dict],
    universe_cfg: dict,
    global_cfg: dict,
    rebalance_dates: list[date],
) -> list[dict]:
    """
    Simulate the two-level momentum strategy.
    Returns list of monthly snapshots.
    """
    capital           = float(global_cfg.get("capital", 500))
    max_single        = float(global_cfg.get("max_single_instrument", 0.20))
    lookback          = int(global_cfg.get("lookback_days", 20))
    min_mom_pct       = float(global_cfg.get("min_class_momentum_pct", 1.5))
    regime_thresh_pct = float(global_cfg.get("market_regime_threshold_pct", -2.0))

    class_keys = [k for k in universe_cfg if isinstance(universe_cfg[k], dict)]

    # {symbol: shares_held}
    holdings: dict[str, float] = {}
    # {symbol: last_known_value}  ← prevents capital disappearing on missing data
    last_known_values: dict[str, float] = {}
    # Cash not deployed into positions — carried forward across rebalances
    cash_held: float = 0.0

    portfolio_value = capital
    snapshots = []

    for i, rebal_date in enumerate(rebalance_dates):

        # ── Market regime check (SPY) ──────────────────────────────────────────
        spy_mom = compute_momentum(spy_data, rebal_date, lookback)
        spy_mom_pct = (spy_mom * 100) if spy_mom is not None else 0.0

        if spy_mom_pct < regime_thresh_pct * 2:
            regime = "severe_bear"
            regime_scale = 0.0
        elif spy_mom_pct < regime_thresh_pct:
            regime = "bear"
            regime_scale = 0.5
        elif spy_mom_pct >= 0:
            regime = "bull"
            regime_scale = 1.0
        else:
            regime = "neutral"
            regime_scale = 1.0

        # ── Liquidate existing holdings at this rebalance date's open ──────────
        if holdings:
            # Start liquidation value with cash that was not deployed last period
            liquidation_value = cash_held
            cash_held = 0.0
            for sym, shares in holdings.items():
                if shares <= 0:
                    continue
                price = get_best_price(price_data.get(sym, []), rebal_date)
                if price:
                    val = shares * price
                    last_known_values[sym] = val
                    liquidation_value += val
                else:
                    # No price at all — carry last known value (don't destroy capital)
                    val = last_known_values.get(sym, 0)
                    liquidation_value += val
                    if val == 0:
                        print(f"  Warning: no price for {sym} on {rebal_date} and no last-known value.")
            portfolio_value = liquidation_value

        # ── Level 1: Compute class momentum + apply filters ────────────────────
        class_scores: list[dict] = []
        for class_key in class_keys:
            syms = universe_cfg[class_key].get("instruments", [])
            momenta, risk_adj = [], []
            for sym in syms:
                if sym not in price_data:
                    continue
                m = compute_momentum(price_data[sym], rebal_date, lookback)
                ra = compute_risk_adjusted_momentum(price_data[sym], rebal_date, lookback)
                if m is not None:
                    momenta.append(m)
                if ra is not None:
                    risk_adj.append(ra)

            avg_mom     = (sum(momenta)   / len(momenta))   if momenta   else None
            avg_risk_adj = (sum(risk_adj) / len(risk_adj)) if risk_adj else None

            class_scores.append({
                "class_key":       class_key,
                "avg_momentum":    avg_mom,
                "avg_momentum_pct": (avg_mom * 100) if avg_mom is not None else None,
                "avg_risk_adj":    avg_risk_adj,
                "max_allocation":  universe_cfg[class_key].get("max_allocation", 0.25),
                "top_n":           universe_cfg[class_key].get("top_n", 1),
                "symbols":         syms,
            })

        # Sort by risk-adjusted momentum (falls back to raw)
        class_scores.sort(
            key=lambda x: (x["avg_risk_adj"] or x["avg_momentum"] or -999),
            reverse=True,
        )

        # ── Level 1 decisions ──────────────────────────────────────────────────
        active_classes = []

        if regime == "severe_bear":
            # Full cash — no investments
            pass
        else:
            n = len(class_scores)
            for rank, cs in enumerate(class_scores):
                avg_mom_pct = cs.get("avg_momentum_pct")

                # Minimum momentum threshold — must have genuine positive trend
                if avg_mom_pct is None or avg_mom_pct < min_mom_pct:
                    cs["decision"] = "SKIP"
                    continue

                # Top half of remaining classes → ACTIVE; bottom half → REDUCE
                if rank < max(1, n // 2):
                    cs["decision"] = "ACTIVE"
                    cs["effective_alloc"] = cs["max_allocation"] * regime_scale
                else:
                    cs["decision"] = "REDUCE"
                    cs["effective_alloc"] = cs["max_allocation"] * 0.5 * regime_scale

                if cs["effective_alloc"] > 0:
                    active_classes.append(cs)

        # ── Cash hold if nothing qualifies ────────────────────────────────────
        if not active_classes:
            holdings = {}
            snapshots.append({
                "date":            rebal_date,
                "portfolio_value": round(portfolio_value, 2),
                "end_value":       round(portfolio_value, 2),
                "return_pct":      0.0,
                "holdings":        {},
                "action":          "hold_cash",
                "n_positions":     0,
                "regime":          regime,
                "spy_momentum_pct": round(spy_mom_pct, 2),
            })
            continue

        # Normalise class weights
        raw_weights = {cs["class_key"]: cs["effective_alloc"] for cs in active_classes}
        total_w = sum(raw_weights.values())
        if total_w > 1.0:
            raw_weights = {k: v / total_w for k, v in raw_weights.items()}

        # ── Level 2: Instrument selection (risk-adjusted ranking) ──────────────
        new_targets: dict[str, float] = {}

        for cs in active_classes:
            key        = cs["class_key"]
            class_cap  = raw_weights[key] * portfolio_value
            top_n      = cs["top_n"]
            syms       = cs["symbols"]

            # Rank instruments by risk-adjusted momentum (higher = better)
            scored = []
            for sym in syms:
                if sym not in price_data:
                    continue
                m = compute_momentum(price_data[sym], rebal_date, lookback)
                ra = compute_risk_adjusted_momentum(price_data[sym], rebal_date, lookback)
                if m is not None and m > 0:   # Only positive-momentum instruments
                    scored.append((sym, ra if ra is not None else m))

            scored.sort(key=lambda x: x[1], reverse=True)
            selected = [s for s, _ in scored[:top_n]]

            if not selected:
                continue

            per_inst = class_cap / len(selected)
            max_inst = portfolio_value * max_single

            for sym in selected:
                capped = min(per_inst, max_inst)
                new_targets[sym] = new_targets.get(sym, 0) + capped

        if not new_targets:
            holdings = {}
            snapshots.append({
                "date":             rebal_date,
                "portfolio_value":  round(portfolio_value, 2),
                "end_value":        round(portfolio_value, 2),
                "return_pct":       0.0,
                "holdings":         {},
                "action":           "hold_cash_no_instruments",
                "n_positions":      0,
                "regime":           regime,
                "spy_momentum_pct": round(spy_mom_pct, 2),
            })
            continue

        # ── Buy new positions at open ──────────────────────────────────────────
        new_holdings: dict[str, float] = {}
        total_allocated = 0.0

        for sym, target_cap in new_targets.items():
            price = get_price_on_or_after(price_data.get(sym, []), rebal_date)
            if not price or price <= 0:
                # Can't buy — keep as unallocated cash
                continue
            shares = target_cap / price
            new_holdings[sym] = shares
            last_known_values[sym] = target_cap  # initial value
            total_allocated += target_cap

        holdings = new_holdings
        cash_held = max(0, portfolio_value - total_allocated)

        # ── Compute period return (open this month → open next month) ──────────
        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None

        if next_rebal:
            end_value = cash_held
            for sym, shares in holdings.items():
                price = get_best_price(price_data.get(sym, []), next_rebal)
                if price:
                    val = shares * price
                    last_known_values[sym] = val
                    end_value += val
                else:
                    # Carry last known value — don't zero out capital
                    end_value += last_known_values.get(sym, 0)
        else:
            # Last period — value at latest available prices
            end_value = cash_held
            for sym, shares in holdings.items():
                price = get_price_on_or_before(price_data.get(sym, []),
                                               rebal_date + timedelta(days=45))
                if price:
                    val = shares * price
                    last_known_values[sym] = val
                    end_value += val
                else:
                    end_value += last_known_values.get(sym, 0)

        period_return_pct = (
            (end_value - portfolio_value) / portfolio_value * 100
            if portfolio_value > 0 else 0.0
        )

        snapshots.append({
            "date":             rebal_date,
            "portfolio_value":  round(portfolio_value, 2),
            "end_value":        round(end_value, 2),
            "return_pct":       round(period_return_pct, 4),
            "holdings":         {s: round(sh, 6) for s, sh in holdings.items()},
            "action":           "rebalance",
            "n_positions":      len(holdings),
            "regime":           regime,
            "spy_momentum_pct": round(spy_mom_pct, 2),
        })

        if next_rebal:
            portfolio_value = end_value

    return snapshots


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(
    snapshots: list[dict],
    initial_capital: float,
    spy_data: list[dict],
) -> dict:
    if not snapshots:
        return {}

    monthly_returns = [s["return_pct"] / 100 for s in snapshots]
    start_value     = initial_capital
    end_value       = snapshots[-1]["end_value"]

    total_return = (end_value - start_value) / start_value * 100
    n_years      = len(snapshots) / 12

    cagr = ((end_value / start_value) ** (1 / n_years) - 1) * 100 if n_years > 0 and end_value > 0 else 0

    rf_monthly = (1 + RISK_FREE_RATE) ** (1 / 12) - 1
    excess     = [r - rf_monthly for r in monthly_returns]

    # Sharpe
    if len(excess) > 1:
        mean_e = sum(excess) / len(excess)
        std_e  = math.sqrt(sum((x - mean_e) ** 2 for x in excess) / (len(excess) - 1))
        sharpe = mean_e / std_e * math.sqrt(12) if std_e > 0 else 0
    else:
        sharpe = 0

    # Sortino (only penalises downside deviation)
    downside = [r for r in excess if r < 0]
    if len(downside) > 1:
        down_std = math.sqrt(sum(x ** 2 for x in downside) / len(downside))
        mean_e2  = sum(excess) / len(excess)
        sortino  = mean_e2 / down_std * math.sqrt(12) if down_std > 0 else 0
    else:
        sortino = 0

    # Max drawdown using end_value per period
    values  = [start_value] + [s["end_value"] for s in snapshots]
    peak    = values[0]
    max_dd  = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Win rate
    wins     = sum(1 for r in monthly_returns if r > 0)
    win_rate = wins / len(monthly_returns) * 100 if monthly_returns else 0

    calmar = cagr / max_dd if max_dd > 0 else float("inf")

    # SPY benchmark
    spy_return = _spy_total_return(spy_data, snapshots[0]["date"], snapshots[-1]["date"])

    # Cash-hold months
    cash_months = sum(1 for s in snapshots if s["action"] != "rebalance")

    # Per-year
    yearly: dict[int, list[float]] = defaultdict(list)
    for s in snapshots:
        yearly[s["date"].year].append(s["return_pct"] / 100)
    yearly_returns = {}
    for year, rets in sorted(yearly.items()):
        compound = 1.0
        for r in rets:
            compound *= (1 + r)
        yearly_returns[year] = round((compound - 1) * 100, 2)

    return {
        "total_return_pct":      round(total_return, 2),
        "cagr_pct":              round(cagr, 2),
        "sharpe_ratio":          round(sharpe, 3),
        "sortino_ratio":         round(sortino, 3),
        "max_drawdown_pct":      round(max_dd, 2),
        "calmar_ratio":          round(calmar, 3) if calmar != float("inf") else "∞",
        "win_rate_pct":          round(win_rate, 1),
        "n_months":              len(snapshots),
        "cash_months":           cash_months,
        "start_value":           start_value,
        "end_value":             round(end_value, 2),
        "spy_total_return_pct":  round(spy_return, 2),
        "alpha_pct":             round(total_return - spy_return, 2),
        "yearly_returns":        yearly_returns,
    }


def _spy_total_return(spy_data: list[dict], sim_start: date, sim_end: date) -> float:
    avail = [b for b in spy_data if sim_start <= b["date"] <= sim_end]
    if len(avail) < 2:
        return 0.0
    return (avail[-1]["close"] - avail[0]["close"]) / avail[0]["close"] * 100


# ── CLI entry point ────────────────────────────────────────────────────────────

def run_backtest(config: dict, period_years: int = 2):
    universe_cfg = config.get("universe", {})
    global_cfg   = config.get("global", {})
    capital      = float(global_cfg.get("capital", 500))
    min_mom_pct  = float(global_cfg.get("min_class_momentum_pct", 1.5))
    regime_thr   = float(global_cfg.get("market_regime_threshold_pct", -2.0))

    all_symbols = []
    for class_cfg in universe_cfg.values():
        if isinstance(class_cfg, dict):
            all_symbols.extend(class_cfg.get("instruments", []))
    all_symbols = list(dict.fromkeys(all_symbols))

    print("\n" + "=" * 68)
    print("HISTORICAL BACKTEST — Two-Level Momentum Strategy (Enhanced)")
    print(f"Period: {period_years} year(s) | Capital: ${capital} | Monthly rebalance")
    print(f"Min class momentum: {min_mom_pct}% | Regime threshold: {regime_thr}%")
    print("=" * 68)

    # Download (include SPY for regime + benchmark)
    price_data = fetch_price_data(all_symbols + ["SPY"], period_years + 1)

    spy_data  = price_data.pop("SPY", [])
    available = set(price_data.keys())
    missing   = [s for s in all_symbols if s not in available]
    if missing:
        print(f"  Excluded (no data): {missing}")

    # Simulation date range
    all_dates = [b["date"] for bars in price_data.values() for b in bars]
    sim_start = date.today().replace(year=date.today().year - period_years)
    sim_end   = date.today()

    rebalance_dates = get_rebalance_dates(price_data, sim_start, sim_end)
    print(f"  Sim: {sim_start} → {sim_end} | {len(rebalance_dates)} rebalance dates\n")

    if len(rebalance_dates) < 3:
        print("  ERROR: Not enough trading dates — check your data download.")
        return

    # Run
    snapshots = run_strategy(price_data, spy_data, universe_cfg, global_cfg, rebalance_dates)

    if not snapshots:
        print("  ERROR: No simulation snapshots generated.")
        return

    metrics = compute_metrics(snapshots, capital, spy_data)

    # ── Report ────────────────────────────────────────────────────────────────
    alpha = metrics["alpha_pct"]
    alpha_marker = "✓ BEAT" if alpha >= 0 else "✗ LAGGED"

    print("\n" + "=" * 68)
    print("BACKTEST RESULTS")
    print("=" * 68)
    print(f"  Portfolio return  : {metrics['total_return_pct']:>+8.2f}%")
    print(f"  SPY benchmark     : {metrics['spy_total_return_pct']:>+8.2f}%")
    print(f"  Alpha vs SPY      : {alpha:>+8.2f}%  {alpha_marker}")
    print(f"  CAGR              : {metrics['cagr_pct']:>+8.2f}%")
    print(f"  Sharpe ratio      : {metrics['sharpe_ratio']:>8.3f}  (>1 = good)")
    print(f"  Sortino ratio     : {metrics['sortino_ratio']:>8.3f}  (>1.5 = good)")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Calmar ratio      : {metrics['calmar_ratio']}")
    print(f"  Win rate          : {metrics['win_rate_pct']:>8.1f}%")
    print(f"  Months simulated  : {metrics['n_months']} ({metrics['cash_months']} cash-hold)")
    print(f"  ${capital:.0f} → ${metrics['end_value']:.2f}")

    print("\n  Year-by-year returns:")
    for year, ret in metrics["yearly_returns"].items():
        bar   = ("+" * min(int(abs(ret) // 2), 30)) if ret >= 0 else ("-" * min(int(abs(ret) // 2), 30))
        color = "▲" if ret >= 0 else "▼"
        print(f"    {year}: {color} {ret:>+7.2f}%  {bar}")

    print("\n  Monthly detail (last 12):")
    for s in snapshots[-12:]:
        if s["action"] != "rebalance":
            syms_str = "CASH"
        else:
            syms_str = ",".join(list(s["holdings"].keys())[:4])
            if len(s["holdings"]) > 4:
                syms_str += "…"
        ret = s["return_pct"]
        arrow = "▲" if ret >= 0 else "▼"
        regime_tag = f"[{s.get('regime','?')[:4].upper()}]"
        print(
            f"    {s['date']}  {arrow} {ret:>+6.2f}%  "
            f"${s['portfolio_value']:>8.2f}→${s['end_value']:>8.2f}  "
            f"{regime_tag}  {syms_str}"
        )

    print("\n" + "=" * 68)
    print("INTERPRETATION GUIDE")
    print("=" * 68)
    _print_interpretation(metrics)
    print()

    return metrics


def _print_interpretation(m: dict):
    issues = []
    strengths = []

    if m["total_return_pct"] > m["spy_total_return_pct"]:
        strengths.append(f"Beats SPY by {m['alpha_pct']:+.1f}%")
    else:
        issues.append(f"Underperforms SPY by {abs(m['alpha_pct']):.1f}%")

    if m["sharpe_ratio"] >= 1.0:
        strengths.append(f"Sharpe {m['sharpe_ratio']:.2f} — good risk-adjusted return")
    elif m["sharpe_ratio"] >= 0.5:
        issues.append(f"Sharpe {m['sharpe_ratio']:.2f} — acceptable but below 1.0 target")
    else:
        issues.append(f"Sharpe {m['sharpe_ratio']:.2f} — poor risk-adjusted return")

    if m["max_drawdown_pct"] <= 15:
        strengths.append(f"Max drawdown {m['max_drawdown_pct']:.1f}% — well-controlled")
    elif m["max_drawdown_pct"] <= 30:
        issues.append(f"Max drawdown {m['max_drawdown_pct']:.1f}% — significant; consider tighter stops")
    else:
        issues.append(f"Max drawdown {m['max_drawdown_pct']:.1f}% — too high for real money deployment")

    if m["win_rate_pct"] >= 55:
        strengths.append(f"Win rate {m['win_rate_pct']:.0f}% — strategy direction is correct more than not")
    else:
        issues.append(f"Win rate {m['win_rate_pct']:.0f}% — near coin-flip")

    cash_ratio = m["cash_months"] / m["n_months"] * 100
    if cash_ratio > 30:
        strengths.append(f"Cash-hold {cash_ratio:.0f}% of months — regime filter is protecting capital")
    elif cash_ratio > 0:
        strengths.append(f"Regime filter triggered {m['cash_months']} month(s) — working as intended")

    if strengths:
        print("  Strengths:")
        for s in strengths:
            print(f"    ✓ {s}")
    if issues:
        print("  Issues:")
        for s in issues:
            print(f"    ✗ {s}")

    # Deployment verdict
    print()
    if m["max_drawdown_pct"] > 30 or m["total_return_pct"] < -20:
        print("  VERDICT: ⛔ Not ready for real money. Fix max drawdown first.")
    elif m["sharpe_ratio"] < 0.5 or m["alpha_pct"] < -10:
        print("  VERDICT: ⚠️  Paper-trade for 3 more months before going live.")
    elif m["total_return_pct"] > 0 and m["sharpe_ratio"] >= 0.8:
        print("  VERDICT: ✅ Strategy looks viable. Start with minimum capital and monitor.")
    else:
        print("  VERDICT: ⚠️  Borderline. Paper-trade for 1-2 more months to confirm.")
