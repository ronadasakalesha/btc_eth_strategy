"""
features/volatility.py
──────────────────────
Volatility features:
  1. ATR percentile    — Is current ATR in a tradeable vol range?
  2. BB width state    — Squeeze (await breakout) or expansion (trade)?
  3. Funding rate bias — Extreme funding = contrarian signal
  4. Realized vol ratio — 1H vol vs 24H vol, flags high-vol sessions

All return Series with values in {-1, 0, 1} or scalar scores.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ─── 1. ATR Percentile ────────────────────────────────────────────────────────

def atr_percentile(
    df: pd.DataFrame,
    period: int = cfg.ATR_PERIOD,
    lookback: int = cfg.ATR_PERCENTILE_LOOKBACK,
    min_pct: float = cfg.ATR_MIN_PERCENTILE,
) -> pd.Series:
    """
    Returns the rolling percentile rank of the current ATR.
    Signal:
      +1 / -1  when ATR rank >= min_pct  (enough volatility to trade)
       0       when ATR rank <  min_pct  (too quiet — skip entry)

    NOTE: The sign is inherited from the upstream trend signal; this feature
    acts as a GATE (0 = no trade regardless of other signals).
    Stored as float 0–100 (percentile rank) for the scoring engine to use.
    """
    atr_vals = _atr(df, period)
    rank = atr_vals.rolling(lookback).rank(pct=True) * 100
    # Return as float; scoring engine decides how to weight
    return rank.rename("atr_percentile")


def atr_gate(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Binary gate: 1 = vol is high enough, 0 = skip."""
    pct = atr_percentile(df, **kwargs)
    return (pct >= cfg.ATR_MIN_PERCENTILE).astype(int).rename("atr_gate")


# ─── 2. Bollinger Band Width ──────────────────────────────────────────────────

def bb_state(
    df: pd.DataFrame,
    period: int = cfg.BB_PERIOD,
    std: float = cfg.BB_STD,
    squeeze_threshold: float = cfg.BB_SQUEEZE_THRESHOLD,
) -> pd.Series:
    """
    Returns:
      +1 = bands expanding after squeeze (breakout setup)
       0 = bands are in squeeze (no edge — wait)
      -1 = bands contracting (losing momentum)
    """
    sma   = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    upper = sma + std * sigma
    lower = sma - std * sigma

    width = (upper - lower) / sma            # normalised width
    width_lag = width.shift(1)
    squeeze = width < squeeze_threshold      # currently in squeeze
    expanding = width > width_lag            # width increasing vs last bar

    signal = pd.Series(0, index=df.index)
    signal[~squeeze & expanding] = 1        # post-squeeze expansion
    signal[~squeeze & ~expanding] = -1      # contracting
    # squeeze bars remain 0 — flag to wait
    return signal.rename("bb_state")


def bb_width(df: pd.DataFrame, period: int = cfg.BB_PERIOD, std: float = cfg.BB_STD) -> pd.Series:
    """Raw normalised BB width — useful for position sizing."""
    sma   = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    return ((sma + std*sigma - (sma - std*sigma)) / sma).rename("bb_width")


# ─── 3. Funding Rate Bias ─────────────────────────────────────────────────────

def funding_bias(
    df: pd.DataFrame,
    extreme_threshold: float = 0.001,   # 0.1% per 8h is extreme
) -> pd.Series:
    """
    Reads the 'funding_rate' column (float) from df.
    Extreme positive funding → crowded longs → contrarian SHORT bias.
    Extreme negative funding → crowded shorts → contrarian LONG bias.

    Returns:
      +1 = extreme negative funding (buy contrarian)
      -1 = extreme positive funding (sell contrarian)
       0 = neutral funding (no opinion)
    """
    if "funding_rate" not in df.columns:
        return pd.Series(0, index=df.index, name="funding_bias")

    fr = df["funding_rate"].fillna(0)
    signal = pd.Series(0, index=df.index)
    signal[fr < -extreme_threshold] = 1
    signal[fr >  extreme_threshold] = -1
    return signal.rename("funding_bias")


# ─── 4. Realized Vol Ratio ────────────────────────────────────────────────────

def realized_vol_ratio(
    df: pd.DataFrame,
    short_bars: int = 4,    # 4 × 15m = 1H
    long_bars: int = 96,    # 96 × 15m = 24H
    high_ratio: float = 1.5,
) -> pd.Series:
    """
    Compares recent short-window realized vol against the longer baseline.
    Returns +1 when current session is high-vol (good for momentum trades),
             0 otherwise.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    short_vol = log_ret.rolling(short_bars).std()
    long_vol  = log_ret.rolling(long_bars).std()

    ratio = short_vol / long_vol.replace(0, np.nan)
    signal = (ratio >= high_ratio).astype(int)
    return signal.rename("rvol_ratio")


# ─── Combined ─────────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        atr_percentile(df),
        atr_gate(df),
        bb_state(df),
        bb_width(df),
        funding_bias(df),
        realized_vol_ratio(df),
    ], axis=1)
