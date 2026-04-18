"""
data/fetcher.py
Fetches OHLCV, open interest, and funding rate data from Binance via ccxt.
Returns clean pandas DataFrames ready for feature engineering.
"""

import time
import logging
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

logger = logging.getLogger(__name__)


def _make_exchange() -> ccxt.Exchange:
    """Instantiate a rate-limit-aware Binance exchange object."""
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},  # perp futures for OI + funding
    })
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_days: int = cfg.LOOKBACK_DAYS,
    exchange: ccxt.Exchange | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for *symbol* on *timeframe*.

    Returns
    -------
    pd.DataFrame  columns: open, high, low, close, volume
                  index:   UTC datetime
    """
    if exchange is None:
        exchange = _make_exchange()

    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000
    )

    all_bars = []
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        since_ms = bars[-1][0] + 1
        if len(bars) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info("Fetched %d bars for %s %s", len(df), symbol, timeframe)
    return df


def fetch_funding_rate(
    symbol: str,
    since_days: int = cfg.LOOKBACK_DAYS,
    exchange: ccxt.Exchange | None = None,
) -> pd.Series:
    """
    Fetch 8-hour funding rate history for a perp contract.

    Returns
    -------
    pd.Series  index: UTC datetime, values: funding rate (float, e.g. 0.0001)
    """
    if exchange is None:
        exchange = _make_exchange()

    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000
    )

    try:
        rows = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
    except Exception as exc:
        logger.warning("Funding rate fetch failed for %s: %s", symbol, exc)
        return pd.Series(dtype=float)

    if not rows:
        return pd.Series(dtype=float)

    sr = pd.Series(
        {pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): r["fundingRate"] for r in rows}
    ).sort_index()
    return sr


def fetch_open_interest(
    symbol: str,
    timeframe: str = "1h",
    since_days: int = cfg.LOOKBACK_DAYS,
    exchange: ccxt.Exchange | None = None,
) -> pd.Series:
    """
    Fetch open interest history (USD notional).

    Returns
    -------
    pd.Series  index: UTC datetime, values: OI in USD
    """
    if exchange is None:
        exchange = _make_exchange()

    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000
    )

    try:
        rows = exchange.fetch_open_interest_history(
            symbol, timeframe=timeframe, since=since_ms, limit=500
        )
    except Exception as exc:
        logger.warning("OI fetch failed for %s: %s", symbol, exc)
        return pd.Series(dtype=float)

    if not rows:
        return pd.Series(dtype=float)

    sr = pd.Series(
        {pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): r["openInterestValue"] for r in rows}
    ).sort_index()
    return sr


def build_master_df(
    symbol: str,
    primary_tf: str = cfg.TIMEFRAME,
    htf: str = cfg.HTF,
    since_days: int = cfg.LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all required data for one symbol and return a dict of DataFrames.

    Keys
    ----
    "primary"  — OHLCV at primary timeframe (e.g. 15m)
    "htf"      — OHLCV at higher timeframe (e.g. 1h)
    "daily"    — OHLCV at daily
    "funding"  — funding rate series (reindexed to primary)
    "oi"       — open interest series (reindexed to primary)
    """
    exchange = _make_exchange()

    primary = fetch_ohlcv(symbol, primary_tf, since_days, exchange)
    htf_df  = fetch_ohlcv(symbol, htf,        since_days, exchange)
    daily   = fetch_ohlcv(symbol, "1d",       since_days, exchange)
    funding = fetch_funding_rate(symbol, since_days, exchange)
    oi      = fetch_open_interest(symbol, "1h", since_days, exchange)

    # Forward-fill auxiliary series onto the primary index
    def _align(sr: pd.Series, ref: pd.DataFrame) -> pd.Series:
        if sr.empty:
            return pd.Series(np.nan, index=ref.index)
        return sr.reindex(ref.index, method="ffill")

    primary["funding_rate"] = _align(funding, primary)
    primary["open_interest"] = _align(oi, primary)

    return {
        "primary": primary,
        "htf":     htf_df,
        "daily":   daily,
    }


# ─── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = build_master_df("BTC/USDT", since_days=7)
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(f"{k}: {len(v)} rows, cols={list(v.columns)}")
        else:
            print(f"{k}: {len(v)} rows")
