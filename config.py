"""
config.py — Central configuration for BTC/ETH Confluence Strategy
All tuneable parameters live here. Edit this file, not the logic modules.
"""

# ─── Assets ───────────────────────────────────────────────────────────────────
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
EXCHANGE = "binance"          # ccxt exchange id
TIMEFRAME = "15m"             # Primary signal timeframe
HTF = "1h"                    # Higher timeframe for bias
DAILY_TF = "1d"               # Daily for macro bias

# ─── Data ─────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 90            # How many days of history to fetch
WARM_UP_BARS = 200            # Bars needed before signals are valid

# ─── Trend Features ───────────────────────────────────────────────────────────
EMA_PERIODS = [9, 21, 55, 200]
ADX_PERIOD = 14
ADX_THRESHOLD = 20            # Min ADX for "trending" regime
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# ─── Volatility Features ──────────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_PERCENTILE_LOOKBACK = 100  # Bars to compute ATR percentile over
ATR_MIN_PERCENTILE = 30        # Skip entries below this vol percentile
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.02    # BB width / price below this = squeeze

# ─── Momentum Features ────────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ROC_PERIODS = [5, 10, 20]

# ─── Market Structure Features ────────────────────────────────────────────────
VWAP_BAND_MULT = [1.0, 2.0]   # σ multipliers for VWAP bands
OB_LOOKBACK = 10               # Bars back to identify order block origin
FVG_MIN_BODY_RATIO = 0.6       # Min body/range ratio for FVG candle
STRUCTURE_SWING_BARS = 5       # Bars each side for swing high/low detection

# ─── Cross-Asset Features ─────────────────────────────────────────────────────
BTCD_SYMBOL = "BTC.D"          # BTC dominance — fetched separately
ETHBTC_SYMBOL = "ETH/BTC"
OI_CHANGE_PERIOD = 4           # Bars for OI momentum (4 × 15m = 1H)

# ─── Regime Features ──────────────────────────────────────────────────────────
# Session windows in UTC hours (start, end)
ACTIVE_SESSIONS = [
    (7, 12),    # London open
    (13, 17),   # NY open
]
SKIP_WEEKEND = True            # Skip Saturday/Sunday entries

# ─── Signal Confluence ────────────────────────────────────────────────────────
# Minimum score (out of 1.0) to trigger a trade.
# ASYMMETRIC: longs require higher confluence (longs at 52% WR — filter harder).
#             shorts demonstrated 68% WR so a lower bar gets us more of them.
LONG_THRESHOLD  = 0.63   # was 0.58 — filters weaker long setups
SHORT_THRESHOLD = 0.58   # restored — 0.54 let in weak shorts (50% WR vs 68% at 0.58)

# Minimum bars between consecutive signals (prevents over-trading)
# 8 bars x 15m = 2 hours minimum between entries
MIN_SIGNAL_GAP_BARS = 4   # 4×15m = 1-hour minimum gap; targets 35-40 trades/90d

# RSI guard rails applied ONLY to long entries (not shorts).
# Longs on overbought RSI (>65) or pre-momentum RSI (<45) are the
# primary driver of the 52% long win rate.
LONG_RSI_MIN = 45    # RSI must be above this — confirms momentum exists
LONG_RSI_MAX = 65    # RSI must be below this — avoids overbought exhaustion


# Feature weights — must sum to 1.0 (weights are auto-normalised).
# ONLY include features that return values in {-1, 0, +1}.
# DO NOT include raw-value features like atr_percentile (0–100) or rsi (0–100)
# — those are used as gates/inputs inside their own modules, not scored here.
FEATURE_WEIGHTS = {
    # Trend layer (34%)
    "ema_stack":       0.15,
    "adx_signal":      0.08,   # FIX: was "adx_trending" (wrong key — column is "adx_signal")
    "htf_bias":        0.11,

    # Volatility layer (8%)
    "bb_state":        0.05,
    "funding_bias":    0.03,

    # Momentum layer (22%)
    "rsi_regime":      0.08,
    "rsi_divergence":  0.05,
    "macd_momentum":   0.05,
    "cvd_divergence":  0.04,

    # Structure layer (28%)
    "vwap_signal":      0.13,  # +0.01 from cross-asset redistribution
    "order_block":      0.10,  # +0.01 from cross-asset redistribution
    "choch_bos":       0.07,

    # Cross-asset / OI (8%)
    "oi_confirm":      0.04,
    "btcd_gate":        0.01,  # reduced — over-weighting ETH score
    "ethbtc_momentum":  0.01,  # reduced — causing ETH negative corr
}

# ─── Trade Management ─────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.01          # 1% of capital per trade
ATR_SL_MULT = 1.5              # SL = entry ± ATR * multiplier
ATR_TP_MULT = 2.5              # TP = entry ± ATR * multiplier (RR = 1.67)
MAX_OPEN_TRADES = 2            # Per symbol
COMMISSION = 0.001             # 0.1% per side (taker fee)

# ─── Backtest ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000       # USD
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2024-12-31"