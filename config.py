# TSX Stock Scanner Configuration
import os

# =============================================================================
# API KEYS
# =============================================================================
# Twelve Data API key (get free key at https://twelvedata.com/ - 800 calls/day)
TWELVE_DATA_API_KEY = os.environ.get('TWELVE_DATA_API_KEY', '')

# =============================================================================
# STOCK UNIVERSE
# =============================================================================
# SCAN_MODE options:
#   "full"    - Scan entire TSX market (~250+ stocks)
#   "default" - Scan default list of ~70 major stocks
#   "custom"  - Use CUSTOM_SYMBOLS list below
SCAN_MODE = "full"

# Custom symbols list (used when SCAN_MODE = "custom")
# Add symbols without .TO suffix
CUSTOM_SYMBOLS = None

# Default list of major TSX stocks (largest by market cap)
DEFAULT_TSX_SYMBOLS = [
    "RY", "TD", "ENB", "CNR", "BN", "BMO", "CP", "BNS", "CNQ", "TRI",
    "CSU", "ATD", "MFC", "SU", "ABX", "NTR", "BCE", "TRP", "CM", "SLF",
    "FTS", "QSR", "IFC", "WCN", "SAP", "GIB.A", "L", "FFH", "DOL", "EMA",
    "T", "MG", "IMO", "CVE", "NA", "POW", "AEM", "WPM", "RCI.B", "CCL.B",
    "GWO", "CAR.UN", "WN", "CTC.A", "K", "X", "FSV", "CCO", "FM", "PKI",
    "CAE", "TIH", "TFII", "WSP", "BAM", "SHOP", "OTEX", "DSG", "LUN", "BTO",
    "H", "AC", "EQB", "CLS", "ONEX", "GFL", "BB", "MRE", "LSPD", "REAL",
]

# =============================================================================
# PRICE & VOLUME FILTERS
# =============================================================================
MIN_PRICE = 5.0          # Minimum stock price in CAD
MIN_VOLUME = 500000      # Minimum average daily volume

# =============================================================================
# RSI SETTINGS
# =============================================================================
RSI_PERIOD = 14          # Standard RSI period
RSI_OVERBOUGHT = 80      # RSI overbought level (sell when crosses back down)
RSI_OVERSOLD = 20        # RSI oversold level (buy when crosses back up)
RSI_EXTREME_OVERSOLD = 10  # Extreme oversold (strong buy signal)

# =============================================================================
# MACD SETTINGS
# =============================================================================
MACD_FAST = 12           # Fast EMA period
MACD_SLOW = 26           # Slow EMA period
MACD_SIGNAL = 9          # Signal line period

# =============================================================================
# SLOW STOCHASTIC SETTINGS
# =============================================================================
STOCH_K_PERIOD = 5       # %K period
STOCH_D_PERIOD = 1       # %D period (smoothing)
STOCH_SMOOTH = 3         # Additional smoothing for slow stochastic
STOCH_OVERBOUGHT = 80    # Overbought level
STOCH_OVERSOLD = 20      # Oversold level

# =============================================================================
# POINT & FIGURE SETTINGS (Traditional)
# =============================================================================
PF_BOX_SIZE = 1.0        # Box size in dollars (traditional)
PF_REVERSAL = 3          # Number of boxes for reversal

# =============================================================================
# BOLLINGER BANDS SETTINGS
# =============================================================================
BB_PERIOD = 20           # Middle band SMA period
BB_STD_DEV = 2           # Standard deviation multiplier for bands

# =============================================================================
# OBV (ON-BALANCE VOLUME) SETTINGS
# =============================================================================
OBV_SMA_PERIOD = 20      # Period for OBV trend comparison

# =============================================================================
# ADX (AVERAGE DIRECTIONAL INDEX) SETTINGS
# =============================================================================
ADX_PERIOD = 14          # ADX calculation period
ADX_STRONG_TREND = 25    # ADX threshold for strong trend

# =============================================================================
# WILLIAMS %R SETTINGS
# =============================================================================
WILLIAMS_R_PERIOD = 14   # Williams %R period
WILLIAMS_R_OVERSOLD = -80   # Oversold level (typical -80)
WILLIAMS_R_OVERBOUGHT = -20 # Overbought level (typical -20)

# =============================================================================
# SIGNAL TIER SETTINGS
# =============================================================================
# Diamond Standard: Minimum aligned signals (all BUY or all SELL) - highest tier
DIAMOND_STANDARD_MIN_SIGNALS = 4

# Gold Standard: Minimum aligned signals (all BUY or all SELL)
GOLD_STANDARD_MIN_SIGNALS = 3

# Silver Standard: Minimum aligned signals for secondary tier
SILVER_STANDARD_MIN_SIGNALS = 2

# =============================================================================
# DATA SETTINGS
# =============================================================================
LOOKBACK_DAYS = 365      # Days of historical data to fetch

# =============================================================================
# INDICATOR SELECTION
# =============================================================================
# Default indicators enabled when no selection is provided
DEFAULT_INDICATORS = ['RSI', 'MACD', 'SlowSto', 'P&F']

# All available indicators
ALL_INDICATORS = ['RSI', 'MACD', 'SlowSto', 'P&F', 'BB', 'OBV', 'ADX', 'WilliamsR']
