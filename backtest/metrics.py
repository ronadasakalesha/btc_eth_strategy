"""
backtest/metrics.py
───────────────────
Performance analytics: win rate, Sharpe, max drawdown, profit factor,
average trade metrics, and a formatted report.
"""

import pandas as pd
import numpy as np
from backtest.engine import BacktestResult, TradeResult
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


def compute_metrics(result: BacktestResult) -> dict:
    trades = result.trades
    if not trades:
        return {"error": "no trades"}

    closed  = [t for t in trades if t.outcome != "open"]
    winners = [t for t in closed if t.pnl_pct > 0]
    losers  = [t for t in closed if t.pnl_pct <= 0]

    win_rate    = len(winners) / len(closed) if closed else 0
    avg_win     = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss    = np.mean([t.pnl_pct for t in losers])  if losers  else 0
    profit_factor = (
        abs(sum(t.pnl_pct for t in winners)) / abs(sum(t.pnl_pct for t in losers))
        if losers else float("inf")
    )

    # Sharpe (annualised, assuming 15m bars → 96 bars/day → 35040/year)
    pnl_series = pd.Series([t.pnl_pct for t in closed])
    sharpe = (
        pnl_series.mean() / pnl_series.std() * np.sqrt(len(pnl_series))
        if pnl_series.std() > 0 else 0
    )

    # Max drawdown on equity curve
    eq = result.equity
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    max_dd = dd.min()

    # Expectancy
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Long vs short breakdown
    longs  = [t for t in closed if t.direction ==  1]
    shorts = [t for t in closed if t.direction == -1]
    long_wr  = len([t for t in longs  if t.pnl_pct > 0]) / len(longs)  if longs  else 0
    short_wr = len([t for t in shorts if t.pnl_pct > 0]) / len(shorts) if shorts else 0

    # Score vs win rate correlation
    if len(closed) >= 10:
        scores  = pd.Series([t.score   for t in closed])
        outcomes = pd.Series([1 if t.pnl_pct > 0 else 0 for t in closed])
        score_corr = float(scores.corr(outcomes))
    else:
        score_corr = float("nan")

    initial_capital = cfg.INITIAL_CAPITAL
    final_capital   = eq.iloc[-1] if not eq.empty else initial_capital
    total_return_pct = (final_capital - initial_capital) / initial_capital * 100

    return {
        "symbol":          result.symbol,
        "total_trades":    len(closed),
        "win_rate":        round(win_rate * 100, 2),
        "long_win_rate":   round(long_wr  * 100, 2),
        "short_win_rate":  round(short_wr * 100, 2),
        "avg_win_pct":     round(avg_win,  3),
        "avg_loss_pct":    round(avg_loss, 3),
        "profit_factor":   round(profit_factor, 2),
        "expectancy_pct":  round(expectancy, 3),
        "sharpe_ratio":    round(sharpe, 2),
        "max_drawdown_pct":round(max_dd, 2),
        "total_return_pct":round(total_return_pct, 2),
        "score_corr":      round(score_corr, 3),
        "avg_duration_bars": round(np.mean([t.duration_bars for t in closed]), 1),
        "initial_capital": initial_capital,
        "final_capital":   round(final_capital, 2),
    }


def print_report(metrics: dict) -> None:
    """Pretty-print the strategy performance report."""
    sep = "─" * 44
    print(f"\n{'═'*44}")
    print(f"  Strategy Report — {metrics.get('symbol', 'N/A')}")
    print(f"{'═'*44}")
    print(f"  Total trades      : {metrics['total_trades']}")
    print(f"  Win rate          : {metrics['win_rate']}%   ← target ≥60%")
    print(f"    Long win rate   : {metrics['long_win_rate']}%")
    print(f"    Short win rate  : {metrics['short_win_rate']}%")
    print(sep)
    print(f"  Avg win  (+%)     : +{metrics['avg_win_pct']}%")
    print(f"  Avg loss (%)      :  {metrics['avg_loss_pct']}%")
    print(f"  Profit factor     : {metrics['profit_factor']}")
    print(f"  Expectancy        : {metrics['expectancy_pct']}%")
    print(sep)
    print(f"  Sharpe ratio      : {metrics['sharpe_ratio']}")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']}%")
    print(f"  Total return      : {metrics['total_return_pct']}%")
    print(f"  Final capital     : ${metrics['final_capital']:,.2f}")
    print(sep)
    print(f"  Score correlation : {metrics['score_corr']}")
    print(f"  Avg trade length  : {metrics['avg_duration_bars']} bars")
    print(f"{'═'*44}\n")


def trades_to_df(result: BacktestResult) -> pd.DataFrame:
    """Convert trade list to a DataFrame for further analysis or CSV export."""
    if not result.trades:
        return pd.DataFrame()
    rows = []
    for t in result.trades:
        rows.append({
            "entry_time":    t.entry_ts,
            "exit_time":     t.exit_ts,
            "direction":     "LONG" if t.direction == 1 else "SHORT",
            "entry_price":   t.entry,
            "exit_price":    t.exit_price,
            "sl":            t.sl,
            "tp":            t.tp,
            "outcome":       t.outcome,
            "pnl_pct":       t.pnl_pct,
            "score":         t.score,
            "duration_bars": t.duration_bars,
        })
    return pd.DataFrame(rows)
