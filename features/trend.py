"""
features/trend.py
─────────────────
Trend features:
  1. EMA stack   — are 9/21/55/200 EMAs in bull or bear order?
  2. ADX slope   — is trend strength above threshold and rising?
  3. HTF bias    — is price above the 1H EMA50 on the higher timeframe?
  4. Supertrend  — ATR-based dynamic support/resistance direction

Each function receives a DataFrame (OHLCV) and returns a Series or DataFrame
with values in {-1, 0, 1}:
  +1 = bullish signal
   0 = neutral / no signal
  -1 = bearish signal
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─── 1. EMA Stack ─────────────────────────────────────────────────────────────

def ema_stack(df: pd.DataFrame, periods: list[int] = cfg.EMA_PERIODS) -> pd.Series:
    """
    Returns +1 when EMAs are in perfect bull order (9 > 21 > 55 > 200),
    -1 for perfect bear order, 0 otherwise.

    Parameters
    ----------
    df      : DataFrame with 'close' column
    periods : EMA periods, default [9, 21, 55, 200]
    """
    emas = {p: df["close"].ewm(span=p, adjust=False).mean() for p in sorted(periods)}
    ema_df = pd.DataFrame(emas)

    bull = (ema_df.diff(axis=1).iloc[:, 1:] < 0).all(axis=1)  # each col < previous
    bear = (ema_df.diff(axis=1).iloc[:, 1:] > 0).all(axis=1)

    signal = pd.Series(0, index=df.index)
    signal[bull] = 1
    signal[bear] = -1
    return signal.rename("ema_stack")


# ─── 2. ADX Slope ─────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def adx_signal(
    df: pd.DataFrame,
    period: int = cfg.ADX_PERIOD,
    threshold: float = cfg.ADX_THRESHOLD,
) -> pd.Series:
    """
    Returns +1 when ADX > threshold AND +DI > -DI (bull trend),
    -1 when ADX > threshold AND -DI > +DI (bear trend),
    0 when ADX <= threshold (ranging, avoid trading).
    """
    tr  = _true_range(df)
    high_diff = df["high"].diff()
    low_diff  = df["low"].diff().mul(-1)

    plus_dm  = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    atr   = tr.ewm(alpha=1/period, adjust=False).mean()
    plus  = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di  = 100 * plus  / atr
    minus_di = 100 * minus / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    signal = pd.Series(0, index=df.index)
    strong_bull = (adx > threshold) & (plus_di > minus_di)
    strong_bear = (adx > threshold) & (minus_di > plus_di)
    signal[strong_bull] = 1
    signal[strong_bear] = -1
    return signal.rename("adx_signal")


# ─── 3. HTF Bias ──────────────────────────────────────────────────────────────

def htf_bias(
    htf_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    ema_period: int = 50,
) -> pd.Series:
    """
    Compute EMA-50 on the higher-timeframe close, then forward-fill to the
    primary timeframe. Returns +1 when price > HTF EMA50, else -1.
    """
    htf_ema = htf_df["close"].ewm(span=ema_period, adjust=False).mean()
    htf_bias_raw = pd.Series(
        np.where(htf_df["close"] > htf_ema, 1, -1),
        index=htf_df.index,
    )
    # Align to primary index by forward-filling
    aligned = htf_bias_raw.reindex(primary_df.index, method="ffill").fillna(0).astype(int)
    return aligned.rename("htf_bias")


# ─── 4. Supertrend ────────────────────────────────────────────────────────────

def supertrend(
    df: pd.DataFrame,
    period: int = cfg.SUPERTREND_PERIOD,
    multiplier: float = cfg.SUPERTREND_MULTIPLIER,
) -> pd.Series:
    """
    Classic Supertrend indicator.
    Returns +1 when price is above Supertrend line (uptrend),
    -1 when price is below (downtrend).
    """
    tr  = _true_range(df)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    hl2 = (df["high"] + df["low"]) / 2

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    # Iterative calculation — required to track band adjustments
    n = len(df)
    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, n):
        fu_prev = final_upper.iloc[i - 1]
        fl_prev = final_lower.iloc[i - 1]
        c_prev  = df["close"].iloc[i - 1]
        c_curr  = df["close"].iloc[i]

        final_upper.iloc[i] = (
            upper.iloc[i] if upper.iloc[i] < fu_prev or c_prev > fu_prev else fu_prev
        )
        final_lower.iloc[i] = (
            lower.iloc[i] if lower.iloc[i] > fl_prev or c_prev < fl_prev else fl_prev
        )

        if trend.iloc[i - 1] == -1 and c_curr > final_upper.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and c_curr < final_lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend.rename("supertrend")


# ─── Combined Trend Score ─────────────────────────────────────────────────────

def compute_all(
    primary_df: pd.DataFrame,
    htf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all trend features and return a DataFrame with columns:
    ema_stack, adx_signal, htf_bias, supertrend
    """
    return pd.concat([
        ema_stack(primary_df),
        adx_signal(primary_df),
        htf_bias(htf_df, primary_df),
        supertrend(primary_df),
    ], axis=1)
