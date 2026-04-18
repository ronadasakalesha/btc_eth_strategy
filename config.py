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
# Minimum score (out of 1.0) to trigger a trade
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.60

# Feature weights — must sum to 1.0 (or will be normalised)
FEATURE_WEIGHTS = {
    "ema_stack":        0.15,
    "adx_trending":     0.08,
    "htf_bias":         0.12,
    "supertrend":       0.05,
    "atr_percentile":   0.06,
    "bb_state":         0.05,
    "rsi_regime":       0.08,
    "macd_momentum":    0.06,
    "cvd_divergence":   0.08,
    "vwap_position":    0.10,
    "order_block":      0.07,
    "choch_bos":        0.06,
    "oi_confirm":       0.04,
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
