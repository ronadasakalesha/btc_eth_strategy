"""
main.py
───────
Orchestrator: wire all modules together.

Usage
─────
  python main.py                      # full backtest both symbols
  python main.py --symbol BTC/USDT    # single symbol
  python main.py --live               # print latest signal (no backtest)
  python main.py --export trades.csv  # export trade log to CSV

This module is the ONLY file that calls external APIs.
All feature modules operate on DataFrames and are testable offline.
"""

import argparse
import logging
import sys
import os

import pandas as pd

# ── Path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from data.fetcher          import build_master_df, fetch_ohlcv
from signals.confluence    import build_features, generate_signals, score_dataframe
from backtest.engine       import run_backtest
from backtest.metrics      import compute_metrics, print_report, trades_to_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Per-symbol pipeline ──────────────────────────────────────────────────────

def run_symbol(symbol: str, export_csv: str | None = None) -> dict:
    """Full pipeline for one symbol. Returns metrics dict."""
    is_eth = symbol.startswith("ETH")
    logger.info("▶ Processing %s", symbol)

    # 1. Data
    logger.info("  Fetching data …")
    data = build_master_df(symbol, cfg.TIMEFRAME, cfg.HTF, cfg.LOOKBACK_DAYS)
    primary_df = data["primary"]
    htf_df     = data["htf"]

    # Cross-asset data (best-effort)
    btcd_df   = None
    ethbtc_df = None
    try:
        btcd_df = fetch_ohlcv("BTC/USDT", cfg.HTF, cfg.LOOKBACK_DAYS)
    except Exception as exc:
        logger.warning("BTC/USDT proxy fetch failed: %s", exc)

    if is_eth:
        try:
            ethbtc_df = fetch_ohlcv("ETH/BTC", cfg.HTF, cfg.LOOKBACK_DAYS)
        except Exception as exc:
            logger.warning("ETH/BTC fetch failed: %s", exc)

    # 2. Features
    logger.info("  Computing features …")
    feats = build_features(primary_df, htf_df, btcd_df, ethbtc_df, is_eth=is_eth)
    logger.info("  Feature matrix: %d bars × %d features", *feats.shape)

    # 3. Signals
    logger.info("  Generating signals …")
    signals = generate_signals(feats, primary_df)
    logger.info("  Signals generated: %d", len(signals))

    if not signals:
        logger.warning("  No signals generated — check thresholds or data quality")
        return {}

    # 4. Backtest
    logger.info("  Running backtest …")
    bt_result = run_backtest(symbol, signals, primary_df)

    # 5. Metrics
    metrics = compute_metrics(bt_result)
    print_report(metrics)

    # 6. Export
    if export_csv:
        fname = f"{symbol.replace('/', '_')}_{export_csv}"
        trades_to_df(bt_result).to_csv(fname, index=False)
        logger.info("  Trade log exported → %s", fname)

    return metrics


def print_live_signal(symbol: str) -> None:
    """Print the most recent confluence score + signal for live trading."""
    logger.info("Live signal check for %s …", symbol)
    is_eth = symbol.startswith("ETH")

    data = build_master_df(symbol, cfg.TIMEFRAME, cfg.HTF, since_days=5)
    primary_df = data["primary"]
    htf_df     = data["htf"]

    btcd_df   = None
    ethbtc_df = None
    try:
        btcd_df = fetch_ohlcv("BTC/USDT", cfg.HTF, since_days=5)
    except Exception:
        pass
    if is_eth:
        try:
            ethbtc_df = fetch_ohlcv("ETH/BTC", cfg.HTF, since_days=5)
        except Exception:
            pass

    feats  = build_features(primary_df, htf_df, btcd_df, ethbtc_df, is_eth=is_eth)
    scores = score_dataframe(feats)

    last_bar   = feats.iloc[-1]
    last_score = scores.iloc[-1]
    last_price = primary_df["close"].iloc[-1]

    print(f"\n{'═'*40}")
    print(f"  Live Signal: {symbol}")
    print(f"  Bar time   : {feats.index[-1]}")
    print(f"  Price      : {last_price:,.2f}")
    print(f"  Bull score : {last_score['bull']:.3f}  (threshold {cfg.LONG_THRESHOLD})")
    print(f"  Bear score : {last_score['bear']:.3f}  (threshold {cfg.SHORT_THRESHOLD})")
    print(f"  Net score  : {last_score['net']:+.3f}")
    if last_score["bull"] >= cfg.LONG_THRESHOLD:
        print("  ▲ LONG signal active")
    elif last_score["bear"] >= cfg.SHORT_THRESHOLD:
        print("  ▼ SHORT signal active")
    else:
        print("  ─ No signal (FLAT)")
    print(f"{'═'*40}\n")

    # Print active feature values
    print("Active feature snapshot:")
    for col in feats.columns:
        val = last_bar[col]
        print(f"  {col:<22} {val:>6.3f}" if not pd.isna(val) else f"  {col:<22}    NaN")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC/ETH Confluence Strategy")
    parser.add_argument("--symbol",  type=str, default=None,
                        help="Single symbol, e.g. BTC/USDT")
    parser.add_argument("--live",    action="store_true",
                        help="Print latest signal without running backtest")
    parser.add_argument("--export",  type=str, default=None,
                        help="Export trade log to CSV filename suffix")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else cfg.SYMBOLS

    if args.live:
        for sym in symbols:
            print_live_signal(sym)
        return

    all_metrics = {}
    for sym in symbols:
        m = run_symbol(sym, export_csv=args.export)
        if m:
            all_metrics[sym] = m

    # Combined summary
    if len(all_metrics) > 1:
        print("\n" + "═" * 44)
        print("  COMBINED SUMMARY")
        print("═" * 44)
        for sym, m in all_metrics.items():
            print(f"  {sym:<12}  WR: {m['win_rate']}%   Return: {m['total_return_pct']}%")
        avg_wr = sum(m["win_rate"] for m in all_metrics.values()) / len(all_metrics)
        print(f"\n  Avg win rate  : {avg_wr:.1f}%")
        print("═" * 44 + "\n")


if __name__ == "__main__":
    main()
