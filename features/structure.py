"""
features/structure.py
─────────────────────
Market structure features:
  1. VWAP bands    — session VWAP ± 1σ / 2σ; mean-reversion + trend signals
  2. Order blocks  — origin candle of a strong impulsive move (institutional)
  3. FVG           — fair value gap / imbalance detection
  4. CHoCH / BOS   — change of character / break of structure (SMC)
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─── 1. VWAP + Bands ──────────────────────────────────────────────────────────

def _session_groups(df: pd.DataFrame) -> pd.Series:
    """Assign a session ID to each bar (resets at midnight UTC)."""
    return df.index.normalize().astype(np.int64)   # daily bucket as int


def vwap_bands(
    df: pd.DataFrame,
    band_mults: list[float] = cfg.VWAP_BAND_MULT,
) -> pd.DataFrame:
    """
    Compute session-anchored VWAP and ±1σ / ±2σ bands.

    Returns DataFrame with columns:
      vwap, upper_1, lower_1, upper_2, lower_2, vwap_signal
    vwap_signal:
      +1 = price between vwap and upper_1 (bull trend confirmation)
      -1 = price between vwap and lower_1 (bear trend confirmation)
      +2 = price at lower_2 band (mean-revert long setup)
      -2 = price at upper_2 band (mean-revert short setup)
       0 = no clear signal
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tpv     = typical * df["volume"]

    session = pd.Series(_session_groups(df), index=df.index)
    cum_tpv = tpv.groupby(session).cumsum()
    cum_vol = df["volume"].groupby(session).cumsum()
    vwap_s  = cum_tpv / cum_vol

    # Rolling std of typical price deviation from VWAP within session
    dev = (typical - vwap_s) ** 2
    cum_dev = dev.groupby(session).cumsum()
    vol_g   = df["volume"].groupby(session).cumsum()
    vwap_std = (cum_dev / vol_g).apply(np.sqrt)

    out = pd.DataFrame(index=df.index)
    out["vwap"] = vwap_s
    for m in band_mults:
        out[f"upper_{int(m)}"] = vwap_s + m * vwap_std
        out[f"lower_{int(m)}"] = vwap_s - m * vwap_std

    c = df["close"]
    signal = pd.Series(0, index=df.index)
    signal[(c > out["vwap"]) & (c < out["upper_1"])]  = 1
    signal[(c < out["vwap"]) & (c > out["lower_1"])]  = -1
    signal[c >= out["upper_2"]]                        = -2   # overextended
    signal[c <= out["lower_2"]]                        = 2    # overextended
    out["vwap_signal"] = signal
    return out


def vwap_signal_only(df: pd.DataFrame) -> pd.Series:
    """Convenience wrapper returning just the vwap_signal column."""
    return vwap_bands(df)["vwap_signal"].rename("vwap_signal")


# ─── 2. Order Blocks ──────────────────────────────────────────────────────────

def order_blocks(
    df: pd.DataFrame,
    lookback: int = cfg.OB_LOOKBACK,
    min_move_atr_mult: float = 1.5,
) -> pd.Series:
    """
    An order block is the origin candle of an impulsive move that is at least
    1.5 × ATR in size and leaves behind an unmitigated zone.

    Returns:
      +1 at bars where price is retesting a bullish OB (buy zone)
      -1 at bars where price is retesting a bearish OB (sell zone)
       0 otherwise
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    signal = pd.Series(0, index=df.index)
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    opens   = df["open"].values
    atr_v   = atr.values

    for i in range(lookback, len(df) - lookback):
        # Strong bullish impulse ahead → origin candle is bearish OB (selling)
        future_move = closes[i + lookback] - closes[i]
        if future_move > min_move_atr_mult * atr_v[i]:
            ob_high = highs[i]
            ob_low  = lows[i]
            # Flag any later bar that re-enters this zone
            for j in range(i + 1, min(i + lookback * 3, len(df))):
                if lows[j] <= ob_high and closes[j] >= ob_low:
                    signal.iloc[j] = 1
                    break

        if future_move < -min_move_atr_mult * atr_v[i]:
            ob_high = highs[i]
            ob_low  = lows[i]
            for j in range(i + 1, min(i + lookback * 3, len(df))):
                if highs[j] >= ob_low and closes[j] <= ob_high:
                    signal.iloc[j] = -1
                    break

    return signal.rename("order_block")


# ─── 3. Fair Value Gap ────────────────────────────────────────────────────────

def fvg(
    df: pd.DataFrame,
    min_body_ratio: float = cfg.FVG_MIN_BODY_RATIO,
) -> pd.Series:
    """
    A fair value gap (imbalance) occurs when:
      candle[i-1].high < candle[i+1].low  → bullish FVG (gap up)
      candle[i-1].low  > candle[i+1].high → bearish FVG (gap down)

    Returns +1 when price re-enters a bullish FVG (fill = support),
            -1 when price re-enters a bearish FVG (fill = resistance).

    Uses a lookahead-free detection by tagging the next bar after the gap.
    """
    signal = pd.Series(0, index=df.index)
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    bar_range = h - l
    body = (df["close"] - df["open"]).abs().values

    for i in range(1, len(df) - 1):
        if bar_range[i] == 0:
            continue
        if body[i] / bar_range[i] < min_body_ratio:
            continue

        # Bullish FVG: gap between [i-1].high and [i+1].low
        if h[i - 1] < l[i + 1]:
            gap_top = l[i + 1]
            gap_bot = h[i - 1]
            for j in range(i + 2, min(i + 20, len(df))):
                if l[j] <= gap_top and c[j] >= gap_bot:
                    signal.iloc[j] = 1
                    break

        # Bearish FVG
        elif l[i - 1] > h[i + 1]:
            gap_top = l[i - 1]
            gap_bot = h[i + 1]
            for j in range(i + 2, min(i + 20, len(df))):
                if h[j] >= gap_bot and c[j] <= gap_top:
                    signal.iloc[j] = -1
                    break

    return signal.rename("fvg")


# ─── 4. CHoCH / BOS ───────────────────────────────────────────────────────────

def choch_bos(
    df: pd.DataFrame,
    swing_bars: int = cfg.STRUCTURE_SWING_BARS,
) -> pd.Series:
    """
    Change of Character (CHoCH) and Break of Structure (BOS) detection.

    BOS:  price breaks the most recent swing high/low WITH trend confirmation
    CHoCH: price breaks the most recent swing high/low AGAINST prior trend

    Returns:
      +1 = bullish BOS or bullish CHoCH (trend flip to bull)
      -1 = bearish BOS or bearish CHoCH (trend flip to bear)
       0 = no confirmed break
    """
    highs  = df["high"]
    lows   = df["low"]
    closes = df["close"]

    sh = (highs == highs.rolling(2 * swing_bars + 1, center=True).max())
    sl = (lows  == lows.rolling(2 * swing_bars + 1, center=True).min())

    swing_h_vals = highs[sh].reindex(df.index, method="ffill")
    swing_l_vals = lows[sl].reindex(df.index, method="ffill")

    signal = pd.Series(0, index=df.index)
    signal[closes > swing_h_vals.shift(1)] = 1   # break above prior swing high
    signal[closes < swing_l_vals.shift(1)] = -1  # break below prior swing low
    return signal.rename("choch_bos")


# ─── Combined ─────────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        vwap_signal_only(df),
        order_blocks(df),
        fvg(df),
        choch_bos(df),
    ], axis=1)
