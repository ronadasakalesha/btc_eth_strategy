"""
signals/confluence.py
─────────────────────
Weighted confluence engine.
Takes all feature DataFrames, computes a normalised directional score,
and emits final LONG / SHORT / FLAT signals with entry, SL, and TP levels.

Score design
────────────
Each feature contributes a weighted vote in {-1, 0, +1}.
  bull_score = sum(w_i * max(f_i, 0))
  bear_score = sum(w_i * max(-f_i, 0))

A trade is triggered when bull_score / total_weight >= LONG_THRESHOLD.
Gating features (session, weekend, ATR, regime) hard-block entries when 0.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

import features.trend      as trend_f
import features.volatility as vol_f
import features.momentum   as mom_f
import features.structure  as struct_f
import features.regime     as regime_f


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    bar_time:  pd.Timestamp
    direction: int          # +1 long, -1 short
    entry:     float
    sl:        float
    tp:        float
    score:     float        # confluence score 0–1
    atr:       float
    features:  dict         # snapshot of all feature values


# ─── Weights + Normalisation ──────────────────────────────────────────────────

def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


WEIGHTS = _normalize_weights(cfg.FEATURE_WEIGHTS)

# Hard-gate features — if any equals 0, no trade regardless of score
HARD_GATES = {"session_gate", "weekend_gate", "atr_gate", "market_regime"}


# ─── Feature Pipeline ─────────────────────────────────────────────────────────

def build_features(
    primary_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    btcd_df: pd.DataFrame | None = None,
    ethbtc_df: pd.DataFrame | None = None,
    is_eth: bool = False,
) -> pd.DataFrame:
    """
    Run all feature modules and return a combined DataFrame aligned to
    the primary timeframe index.
    """
    trend_feats  = trend_f.compute_all(primary_df, htf_df)
    vol_feats    = vol_f.compute_all(primary_df)
    mom_feats    = mom_f.compute_all(primary_df)
    struct_feats = struct_f.compute_all(primary_df)
    regime_feats = regime_f.compute_all(primary_df, btcd_df, ethbtc_df, is_eth)

    all_feats = pd.concat(
        [trend_feats, vol_feats, mom_feats, struct_feats, regime_feats],
        axis=1,
    )
    return all_feats


# ─── Scoring Engine ───────────────────────────────────────────────────────────

def score_bar(row: pd.Series) -> tuple[float, float]:
    """
    Compute (bull_score, bear_score) for a single bar's feature row.
    Scores are in [0, 1].
    """
    bull = 0.0
    bear = 0.0
    for feat, w in WEIGHTS.items():
        if feat not in row.index:
            continue
        v = row[feat]
        if pd.isna(v):
            continue
        if v > 0:
            bull += w * v
        elif v < 0:
            bear += w * abs(v)
    return bull, bear


def _hard_gate_open(row: pd.Series) -> bool:
    """Returns True if all hard gates are open (non-zero)."""
    for g in HARD_GATES:
        if g in row.index and row[g] == 0:
            return False
    return True


def generate_signals(
    features_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    atr_mult_sl: float = cfg.ATR_SL_MULT,
    atr_mult_tp: float = cfg.ATR_TP_MULT,
    long_thresh: float = cfg.LONG_THRESHOLD,
    short_thresh: float = cfg.SHORT_THRESHOLD,
) -> list[TradeSignal]:
    """
    Walk through every bar, compute the confluence score, and emit
    TradeSignal objects when thresholds are met.

    ATR is computed here for SL/TP placement.
    """
    prev_close = primary_df["close"].shift(1)
    tr = pd.concat([
        primary_df["high"] - primary_df["low"],
        (primary_df["high"] - prev_close).abs(),
        (primary_df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/cfg.ATR_PERIOD, adjust=False).mean()

    signals: list[TradeSignal] = []
    skip_until: pd.Timestamp | None = None

    for ts, row in features_df.iterrows():
        # Skip warm-up period
        if ts < features_df.index[cfg.WARM_UP_BARS]:
            continue
        # Skip if we're in an open trade (simple: skip until next bar)
        if skip_until and ts <= skip_until:
            continue

        if not _hard_gate_open(row):
            continue

        bull, bear = score_bar(row)
        entry = primary_df.loc[ts, "close"]
        atr   = atr_series.loc[ts]

        if bull >= long_thresh:
            sl = entry - atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr
            signals.append(TradeSignal(
                bar_time=ts, direction=1,
                entry=entry, sl=sl, tp=tp,
                score=bull, atr=atr,
                features=row.to_dict(),
            ))
            skip_until = ts  # prevent re-entry same bar

        elif bear >= short_thresh:
            sl = entry + atr_mult_sl * atr
            tp = entry - atr_mult_tp * atr
            signals.append(TradeSignal(
                bar_time=ts, direction=-1,
                entry=entry, sl=sl, tp=tp,
                score=bear, atr=atr,
                features=row.to_dict(),
            ))
            skip_until = ts

    return signals


# ─── Convenience: score DataFrame ─────────────────────────────────────────────

def score_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: bull_score, bear_score, net_score.
    Useful for plotting / analysis without the full signal generation.
    """
    scores = features_df.apply(lambda row: pd.Series(score_bar(row), index=["bull", "bear"]), axis=1)
    scores["net"] = scores["bull"] - scores["bear"]
    return scores
