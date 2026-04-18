"""
features/momentum.py
────────────────────
Momentum features:
  1. RSI regime + divergence  — overbought/oversold + hidden/regular divergence
  2. MACD momentum            — histogram slope + zero-line cross
  3. Rate of change (ROC)     — z-scored multi-period price momentum
  4. CVD divergence           — cumulative volume delta vs price divergence
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _swing_highs(series: pd.Series, window: int) -> pd.Series:
    """Boolean mask: True at local maxima."""
    return series == series.rolling(2 * window + 1, center=True).max()


def _swing_lows(series: pd.Series, window: int) -> pd.Series:
    """Boolean mask: True at local minima."""
    return series == series.rolling(2 * window + 1, center=True).min()


# ─── 1. RSI ───────────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = cfg.RSI_PERIOD) -> pd.Series:
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def rsi_regime(
    df: pd.DataFrame,
    ob: float = cfg.RSI_OVERBOUGHT,
    os_: float = cfg.RSI_OVERSOLD,
) -> pd.Series:
    """
    +1 = RSI in bull zone (40–70, momentum without OB exhaustion)
    -1 = RSI in bear zone (30–60)
     0 = RSI extreme or middle (no directional edge)
    """
    r = rsi(df)
    signal = pd.Series(0, index=df.index)
    signal[(r > 50) & (r < ob)] = 1
    signal[(r < 50) & (r > os_)] = -1
    return signal.rename("rsi_regime")


def rsi_divergence(
    df: pd.DataFrame,
    period: int = cfg.RSI_PERIOD,
    swing_bars: int = cfg.STRUCTURE_SWING_BARS,
) -> pd.Series:
    """
    Detect regular bullish/bearish RSI divergence on swing pivots.
    Regular bullish:  price lower low, RSI higher low  → +1
    Regular bearish:  price higher high, RSI lower high → -1
    """
    r     = rsi(df)
    price = df["close"]

    sh_price = _swing_highs(price, swing_bars)
    sl_price = _swing_lows(price,  swing_bars)
    sh_rsi   = _swing_highs(r,     swing_bars)
    sl_rsi   = _swing_lows(r,      swing_bars)

    signal = pd.Series(0, index=df.index)

    # Find pivot pairs (current vs most recent previous pivot)
    ph_idx = price.index[sh_price].tolist()
    pl_idx = price.index[sl_price].tolist()

    for i in range(1, len(pl_idx)):
        curr, prev = pl_idx[i], pl_idx[i - 1]
        if price[curr] < price[prev] and r[curr] > r[prev]:
            signal[curr] = 1   # regular bullish div

    for i in range(1, len(ph_idx)):
        curr, prev = ph_idx[i], ph_idx[i - 1]
        if price[curr] > price[prev] and r[curr] < r[prev]:
            signal[curr] = -1  # regular bearish div

    # Propagate signal forward for N bars (so it's visible to the engine)
    signal = signal.replace(0, np.nan).ffill(limit=swing_bars).fillna(0).astype(int)
    return signal.rename("rsi_divergence")


# ─── 2. MACD ──────────────────────────────────────────────────────────────────

def macd_signal(
    df: pd.DataFrame,
    fast: int = cfg.MACD_FAST,
    slow: int = cfg.MACD_SLOW,
    signal_period: int = cfg.MACD_SIGNAL,
) -> pd.Series:
    """
    +1 when MACD histogram > 0 and increasing (strengthening bull momentum)
    -1 when MACD histogram < 0 and decreasing (strengthening bear momentum)
     0 otherwise
    """
    ema_fast = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist      = macd_line - sig_line

    hist_slope = hist.diff()
    signal = pd.Series(0, index=df.index)
    signal[(hist > 0) & (hist_slope > 0)] = 1
    signal[(hist < 0) & (hist_slope < 0)] = -1
    return signal.rename("macd_momentum")


# ─── 3. Rate of Change (z-scored) ─────────────────────────────────────────────

def roc_zscore(
    df: pd.DataFrame,
    periods: list[int] = cfg.ROC_PERIODS,
    zscore_window: int = 100,
) -> pd.Series:
    """
    Compute ROC for multiple periods, z-score each, average them.
    Returns +1 if average z-score > 0.5, -1 if < -0.5, else 0.
    """
    z_scores = []
    for p in periods:
        roc = df["close"].pct_change(p)
        mu  = roc.rolling(zscore_window).mean()
        sig = roc.rolling(zscore_window).std()
        z   = (roc - mu) / sig.replace(0, np.nan)
        z_scores.append(z)

    avg_z = pd.concat(z_scores, axis=1).mean(axis=1)
    signal = pd.Series(0, index=df.index)
    signal[avg_z >  0.5] = 1
    signal[avg_z < -0.5] = -1
    return signal.rename("roc_momentum")


# ─── 4. CVD Divergence ────────────────────────────────────────────────────────

def cvd_divergence(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Cumulative Volume Delta (CVD) proxy using close position within the bar.

    Bar delta heuristic:
      If close > open → positive delta = +volume * (close-open)/(high-low)
      If close < open → negative delta = -volume * (close-open)/(high-low)

    Returns:
      +1  price falling but CVD rising  → bullish absorption / accumulation
      -1  price rising but CVD falling  → bearish distribution
       0  no divergence
    """
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    delta = df["volume"] * (df["close"] - df["open"]) / bar_range
    cvd   = delta.cumsum()

    price_slope = df["close"].rolling(window).mean().diff(window)
    cvd_slope   = cvd.rolling(window).mean().diff(window)

    signal = pd.Series(0, index=df.index)
    signal[(price_slope < 0) & (cvd_slope > 0)] = 1   # bullish absorption
    signal[(price_slope > 0) & (cvd_slope < 0)] = -1  # bearish distribution
    return signal.rename("cvd_divergence")


# ─── Combined ─────────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        rsi(df),
        rsi_regime(df),
        rsi_divergence(df),
        macd_signal(df),
        roc_zscore(df),
        cvd_divergence(df),
    ], axis=1)
