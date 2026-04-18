"""
backtest/engine.py
──────────────────
Vectorized backtester.
Takes a list of TradeSignal objects and a price DataFrame, simulates each
trade bar-by-bar (no future look-ahead), and returns a detailed trade log.

Trade resolution
────────────────
For each signal we scan forward from the entry bar, checking whether the
HIGH or LOW of each subsequent bar touches the TP or SL first.
  • If the bar's range crosses both TP and SL, we assume worst-case: SL hit.
  • We continue until TP, SL, or end-of-data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from signals.confluence import TradeSignal


# ─── Trade result ─────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    symbol:    str
    entry_ts:  pd.Timestamp
    exit_ts:   pd.Timestamp | None
    direction: int
    entry:     float
    exit_price: float | None
    sl:        float
    tp:        float
    outcome:   str          # "tp" | "sl" | "open"
    pnl_pct:   float        # % gain/loss on the trade (post-commission)
    score:     float
    duration_bars: int


@dataclass
class BacktestResult:
    symbol:     str
    trades:     list[TradeResult] = field(default_factory=list)
    equity:     pd.Series         = field(default_factory=pd.Series)


# ─── Core engine ──────────────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    signals: list[TradeSignal],
    price_df: pd.DataFrame,
    initial_capital: float = cfg.INITIAL_CAPITAL,
    risk_per_trade: float  = cfg.RISK_PER_TRADE,
    commission: float      = cfg.COMMISSION,
    max_open:   int        = cfg.MAX_OPEN_TRADES,
) -> BacktestResult:
    """
    Simulate all signals on price_df.

    Parameters
    ----------
    signals    : from confluence.generate_signals()
    price_df   : OHLCV at the same timeframe as the signals

    Returns
    -------
    BacktestResult with trade log and equity curve.
    """
    result   = BacktestResult(symbol=symbol)
    capital  = initial_capital
    equity   = {price_df.index[0]: capital}
    open_positions = 0

    highs  = price_df["high"]
    lows   = price_df["low"]
    closes = price_df["close"]

    for sig in signals:
        if sig.bar_time not in price_df.index:
            continue
        if open_positions >= max_open:
            continue

        entry_idx = price_df.index.get_loc(sig.bar_time)
        entry     = sig.entry
        sl, tp    = sig.sl, sig.tp
        direction = sig.direction

        exit_price: float | None = None
        exit_ts: pd.Timestamp | None = None
        outcome = "open"

        # Scan forward bar by bar
        for j in range(entry_idx + 1, len(price_df)):
            bar_high = highs.iloc[j]
            bar_low  = lows.iloc[j]

            if direction == 1:  # long
                tp_hit = bar_high >= tp
                sl_hit = bar_low  <= sl
            else:               # short
                tp_hit = bar_low  <= tp
                sl_hit = bar_high >= sl

            if sl_hit and tp_hit:
                # Worst-case: SL hit first
                outcome    = "sl"
                exit_price = sl
                exit_ts    = price_df.index[j]
                break
            elif tp_hit:
                outcome    = "tp"
                exit_price = tp
                exit_ts    = price_df.index[j]
                break
            elif sl_hit:
                outcome    = "sl"
                exit_price = sl
                exit_ts    = price_df.index[j]
                break

        if outcome == "open":
            exit_price = closes.iloc[-1]
            exit_ts    = price_df.index[-1]

        # P&L calculation
        gross_pnl_pct = direction * (exit_price - entry) / entry
        net_pnl_pct   = gross_pnl_pct - 2 * commission  # entry + exit

        # Position size based on risk
        risk_amount = capital * risk_per_trade
        sl_dist_pct = abs(entry - sl) / entry
        position_size_usd = risk_amount / sl_dist_pct if sl_dist_pct > 0 else 0

        trade_pnl = position_size_usd * net_pnl_pct
        capital  += trade_pnl

        open_positions = max(0, open_positions + (1 if outcome == "open" else 0))
        equity[exit_ts] = capital

        result.trades.append(TradeResult(
            symbol     = symbol,
            entry_ts   = sig.bar_time,
            exit_ts    = exit_ts,
            direction  = direction,
            entry      = entry,
            exit_price = exit_price,
            sl         = sl,
            tp         = tp,
            outcome    = outcome,
            pnl_pct    = net_pnl_pct * 100,
            score      = sig.score,
            duration_bars = (price_df.index.get_loc(exit_ts) - entry_idx) if exit_ts else 0,
        ))

    # Build equity curve on full index
    eq_series = pd.Series(equity).sort_index()
    eq_series = eq_series.reindex(price_df.index, method="ffill").fillna(initial_capital)
    result.equity = eq_series

    return result
