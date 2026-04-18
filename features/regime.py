"""
features/regime.py
──────────────────
Regime & cross-asset features:
  1. Market regime    — HMM-lite (trending bull/bear vs range) using ADX + BB width
  2. Session filter   — UTC trading session gate
  3. Weekend filter   — skip low-liquidity weekend bars
  4. BTC dominance    — ETH trade quality gate
  5. ETH/BTC ratio    — relative strength leading indicator for ETH
  6. OI confirmation  — open interest expanding with price = trend confirm
"""

import pandas as pd
import numpy as np
from datetime import time as dtime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─── 1. Market Regime ─────────────────────────────────────────────────────────

def market_regime(
    df: pd.DataFrame,
    adx_period: int = cfg.ADX_PERIOD,
    adx_threshold: float = cfg.ADX_THRESHOLD,
    bb_period: int = cfg.BB_PERIOD,
    bb_std: float = cfg.BB_STD,
    squeeze_threshold: float = cfg.BB_SQUEEZE_THRESHOLD,
) -> pd.Series:
    """
    Three-state regime classifier:
      2 = strong trend (ADX > threshold + BB expanding)
      1 = weak trend   (ADX > threshold OR BB expanding, not both)
      0 = range / chop (ADX low AND BB squeeze)

    Entry is only taken in regime 2; position size is reduced in regime 1.
    """
    # ADX
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr        = tr.ewm(alpha=1/adx_period, adjust=False).mean()
    high_diff  = df["high"].diff()
    low_diff   = df["low"].diff().mul(-1)
    plus_dm    = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm   = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    plus_di    = 100 * plus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / atr
    minus_di   = 100 * minus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / atr
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx        = dx.ewm(alpha=1/adx_period, adjust=False).mean()
    trending   = adx > adx_threshold

    # BB width
    sma   = df["close"].rolling(bb_period).mean()
    sigma = df["close"].rolling(bb_period).std()
    width = ((sma + bb_std*sigma) - (sma - bb_std*sigma)) / sma
    expanding = width > squeeze_threshold

    regime = pd.Series(0, index=df.index)
    regime[trending & expanding]    = 2
    regime[trending | expanding]    = regime[trending | expanding].clip(lower=1)
    # Note: above sets 1 for either condition, then overwrites with 2 if both.
    # Recalculate cleanly:
    regime = pd.Series(0, index=df.index)
    regime[trending | expanding]    = 1
    regime[trending & expanding]    = 2

    return regime.rename("market_regime")


# ─── 2. Session Filter ────────────────────────────────────────────────────────

def session_gate(
    df: pd.DataFrame,
    sessions: list[tuple[int, int]] = cfg.ACTIVE_SESSIONS,
) -> pd.Series:
    """
    Returns 1 if the bar falls within an active trading session, 0 otherwise.
    Sessions are defined as (start_hour_utc, end_hour_utc) tuples.
    """
    hour = df.index.hour
    in_session = pd.Series(False, index=df.index)
    for start, end in sessions:
        in_session |= (hour >= start) & (hour < end)
    return in_session.astype(int).rename("session_gate")


# ─── 3. Weekend Filter ────────────────────────────────────────────────────────

def weekend_gate(df: pd.DataFrame) -> pd.Series:
    """
    Returns 1 on weekdays (Mon–Fri), 0 on weekends.
    Used to skip Saturday/Sunday low-liquidity entries.
    """
    is_weekday = (df.index.dayofweek < 5).astype(int)
    return pd.Series(is_weekday, index=df.index, name="weekend_gate")


# ─── 4. BTC Dominance Gate ────────────────────────────────────────────────────

def btcd_gate(
    btcd_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    ema_period: int = 20,
    is_eth: bool = False,
) -> pd.Series:
    """
    When BTC.D is rising (BTC gaining dominance), ETH long signals are penalised.
    When BTC.D is falling, ETH longs are confirmed.

    Parameters
    ----------
    btcd_df    : DataFrame with 'close' = BTC.D index (or BTC/USDT as proxy)
    primary_df : Primary timeframe df to align to
    is_eth     : If True, invert logic (dominance rise = bad for ETH longs)

    Returns 1 = favourable, -1 = unfavourable, 0 = neutral
    """
    if btcd_df is None or btcd_df.empty:
        return pd.Series(0, index=primary_df.index, name="btcd_gate")

    btcd_ema = btcd_df["close"].ewm(span=ema_period, adjust=False).mean()
    btcd_rising = (btcd_df["close"] > btcd_ema).astype(int).replace(0, -1)
    aligned = btcd_rising.reindex(primary_df.index, method="ffill").fillna(0).astype(int)

    if is_eth:
        # Rising dominance = bad for ETH
        aligned = aligned * -1

    return aligned.rename("btcd_gate")


# ─── 5. ETH/BTC Ratio Momentum ────────────────────────────────────────────────

def ethbtc_momentum(
    ethbtc_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    ETH/BTC ratio above its SMA = ETH outperforming BTC → bullish ETH bias.
    ETH/BTC below SMA = underperforming → bearish ETH bias.

    Returns +1, -1, or 0 (when series unavailable).
    """
    if ethbtc_df is None or ethbtc_df.empty:
        return pd.Series(0, index=primary_df.index, name="ethbtc_momentum")

    sma = ethbtc_df["close"].rolling(period).mean()
    signal_raw = pd.Series(
        np.where(ethbtc_df["close"] > sma, 1, -1),
        index=ethbtc_df.index,
    )
    aligned = signal_raw.reindex(primary_df.index, method="ffill").fillna(0).astype(int)
    return aligned.rename("ethbtc_momentum")


# ─── 6. Open Interest Confirmation ────────────────────────────────────────────

def oi_confirm(
    df: pd.DataFrame,
    period: int = cfg.OI_CHANGE_PERIOD,
) -> pd.Series:
    """
    Open interest confirmation:
      Price up + OI up   = fresh money entering longs  → +1 (confirm bull)
      Price down + OI up = fresh money entering shorts → -1 (confirm bear)
      Price up + OI down = short covering rally        →  0 (weaker signal)
      Price down + OI down = long liquidation          →  0 (weaker signal)

    Requires 'open_interest' column in df.
    """
    if "open_interest" not in df.columns:
        return pd.Series(0, index=df.index, name="oi_confirm")

    price_delta = df["close"].diff(period)
    oi_delta    = df["open_interest"].diff(period)

    signal = pd.Series(0, index=df.index)
    signal[(price_delta > 0) & (oi_delta > 0)] = 1
    signal[(price_delta < 0) & (oi_delta > 0)] = -1
    return signal.rename("oi_confirm")


# ─── Combined ─────────────────────────────────────────────────────────────────

def compute_all(
    primary_df: pd.DataFrame,
    btcd_df: pd.DataFrame | None = None,
    ethbtc_df: pd.DataFrame | None = None,
    is_eth: bool = False,
) -> pd.DataFrame:
    return pd.concat([
        market_regime(primary_df),
        session_gate(primary_df),
        weekend_gate(primary_df),
        btcd_gate(btcd_df, primary_df, is_eth=is_eth),
        ethbtc_momentum(ethbtc_df, primary_df),
        oi_confirm(primary_df),
    ], axis=1)
