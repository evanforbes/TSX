"""
Technical Indicators Module
Calculates RSI, MACD, Slow Stochastic, and Point & Figure
"""

import pandas as pd
import numpy as np
from config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_EXTREME_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_SMOOTH, STOCH_OVERBOUGHT, STOCH_OVERSOLD,
    PF_BOX_SIZE, PF_REVERSAL,
    BB_PERIOD, BB_STD_DEV,
    OBV_SMA_PERIOD,
    ADX_PERIOD, ADX_STRONG_TREND,
    WILLIAMS_R_PERIOD, WILLIAMS_R_OVERSOLD, WILLIAMS_R_OVERBOUGHT
)


def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series,
                   fast: int = MACD_FAST,
                   slow: int = MACD_SLOW,
                   signal: int = MACD_SIGNAL) -> tuple:
    """
    Calculate MACD, Signal line, and Histogram
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_slow_stochastic(high: pd.Series,
                               low: pd.Series,
                               close: pd.Series,
                               k_period: int = STOCH_K_PERIOD,
                               d_period: int = STOCH_D_PERIOD,
                               smooth: int = STOCH_SMOOTH) -> tuple:
    """
    Calculate Slow Stochastic (%K and %D)
    Returns: (slow_k, slow_d)
    """
    # Fast %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Slow %K (smoothed fast %K)
    slow_k = fast_k.rolling(window=smooth).mean()

    # Slow %D (smoothed slow %K)
    slow_d = slow_k.rolling(window=d_period).mean()

    return slow_k, slow_d


def calculate_point_and_figure(prices: pd.Series,
                                box_size: float = None,
                                reversal: int = PF_REVERSAL) -> dict:
    """
    Calculate Point & Figure chart data and detect triple top breakout / triple bottom breakdown

    Returns dict with:
        - columns: list of (direction, start_price, end_price, count)
        - triple_top_breakout: bool (True if recent triple top breakout detected)
        - triple_bottom_breakdown: bool (True if recent triple bottom breakdown detected)
        - current_direction: 'X' or 'O'
    """
    prices = prices.dropna().values
    if len(prices) < 2:
        return {'columns': [], 'triple_top_breakout': False, 'triple_bottom_breakdown': False, 'current_direction': None}

    # Use percentage-based box size if not specified (2% of average price)
    if box_size is None:
        avg_price = prices.mean()
        box_size = max(0.50, avg_price * 0.02)  # 2% of price, minimum $0.50

    # Initialize
    columns = []
    current_direction = None  # 'X' for up, 'O' for down
    current_start = prices[0]
    current_end = prices[0]
    box_count = 0

    for price in prices[1:]:
        if current_direction is None:
            # Determine initial direction
            diff = price - current_start
            boxes_moved = abs(diff) / box_size

            if boxes_moved >= 1:
                if diff > 0:
                    current_direction = 'X'
                    current_end = current_start + (int(boxes_moved) * box_size)
                    box_count = int(boxes_moved)
                else:
                    current_direction = 'O'
                    current_end = current_start - (int(boxes_moved) * box_size)
                    box_count = int(boxes_moved)
        else:
            if current_direction == 'X':
                # Currently in X column (going up)
                if price > current_end + box_size:
                    # Continue up
                    new_boxes = int((price - current_end) / box_size)
                    current_end += new_boxes * box_size
                    box_count += new_boxes
                elif price < current_end - (reversal * box_size):
                    # Reversal down
                    columns.append({
                        'direction': 'X',
                        'start': current_start,
                        'end': current_end,
                        'boxes': box_count
                    })
                    current_direction = 'O'
                    current_start = current_end - box_size
                    new_boxes = int((current_start - price) / box_size)
                    current_end = current_start - (new_boxes * box_size)
                    box_count = new_boxes + 1
            else:
                # Currently in O column (going down)
                if price < current_end - box_size:
                    # Continue down
                    new_boxes = int((current_end - price) / box_size)
                    current_end -= new_boxes * box_size
                    box_count += new_boxes
                elif price > current_end + (reversal * box_size):
                    # Reversal up
                    columns.append({
                        'direction': 'O',
                        'start': current_start,
                        'end': current_end,
                        'boxes': box_count
                    })
                    current_direction = 'X'
                    current_start = current_end + box_size
                    new_boxes = int((price - current_start) / box_size)
                    current_end = current_start + (new_boxes * box_size)
                    box_count = new_boxes + 1

    # Add current column
    if current_direction is not None:
        columns.append({
            'direction': current_direction,
            'start': current_start,
            'end': current_end,
            'boxes': box_count
        })

    # Detect Triple Top Breakout and Triple Bottom Breakdown
    triple_top_breakout = detect_triple_top_breakout(columns, box_size)
    triple_bottom_breakdown = detect_triple_bottom_breakdown(columns, box_size)

    return {
        'columns': columns,
        'triple_top_breakout': triple_top_breakout,
        'triple_bottom_breakdown': triple_bottom_breakdown,
        'current_direction': current_direction,
        'box_size': box_size
    }


def detect_triple_top_breakout(columns: list, box_size: float) -> bool:
    """
    Detect RECENT Triple Top Breakout pattern in Point & Figure columns

    Triple Top Breakout: Three X columns reaching similar highs,
    with the most recent X column breaking above the previous two highs.
    Only signals if the breakout is in the current or most recent X column.
    """
    if len(columns) < 5:
        return False

    # Get X columns with their indices
    x_columns_with_idx = [(i, c) for i, c in enumerate(columns) if c['direction'] == 'X']

    if len(x_columns_with_idx) < 3:
        return False

    # Check last 3 X columns
    recent_x = x_columns_with_idx[-3:]

    # Get the highs (end prices for X columns)
    high1 = recent_x[0][1]['end']
    high2 = recent_x[1][1]['end']
    high3 = recent_x[2][1]['end']

    # Triple top breakout: third high breaks above first two
    # Allow some tolerance (within 1.5 boxes)
    tolerance = box_size * 1.5

    # First two highs should be roughly equal (resistance level)
    if abs(high1 - high2) <= tolerance:
        # Third high should break above both
        if high3 > max(high1, high2):
            # Check if this breakout is RECENT (last X column is current or second-to-last column)
            last_x_idx = recent_x[2][0]
            if last_x_idx >= len(columns) - 2:  # Current or previous column
                return True

    return False


def detect_triple_bottom_breakdown(columns: list, box_size: float) -> bool:
    """
    Detect RECENT Triple Bottom Breakdown pattern in Point & Figure columns

    Triple Bottom Breakdown: Three O columns reaching similar lows,
    with the most recent O column breaking below the previous two lows.
    Only signals if the breakdown is in the current or most recent O column.
    """
    if len(columns) < 5:
        return False

    # Get O columns with their indices
    o_columns_with_idx = [(i, c) for i, c in enumerate(columns) if c['direction'] == 'O']

    if len(o_columns_with_idx) < 3:
        return False

    # Check last 3 O columns
    recent_o = o_columns_with_idx[-3:]

    # Get the lows (end prices for O columns)
    low1 = recent_o[0][1]['end']
    low2 = recent_o[1][1]['end']
    low3 = recent_o[2][1]['end']

    # Triple bottom breakdown: third low breaks below first two
    # Allow some tolerance (within 1.5 boxes)
    tolerance = box_size * 1.5

    # First two lows should be roughly equal (support level)
    if abs(low1 - low2) <= tolerance:
        # Third low should break below both
        if low3 < min(low1, low2):
            # Check if this breakdown is RECENT (last O column is current or second-to-last column)
            last_o_idx = recent_o[2][0]
            if last_o_idx >= len(columns) - 2:  # Current or previous column
                return True

    return False


def get_rsi_signal(rsi_series: pd.Series) -> dict:
    """
    Determine RSI signal based on crossovers

    Returns dict with:
        - signal: 'BUY', 'SELL', 'STRONG_BUY', or None
        - rsi_value: current RSI
        - description: explanation of signal
    """
    if len(rsi_series) < 2:
        return {'signal': None, 'rsi_value': None, 'description': 'Insufficient data'}

    current_rsi = rsi_series.iloc[-1]
    prev_rsi = rsi_series.iloc[-2]

    signal = None
    description = f"RSI: {current_rsi:.1f}"

    # Extreme oversold - strong buy
    if current_rsi <= RSI_EXTREME_OVERSOLD:
        signal = 'STRONG_BUY'
        description = f"RSI extremely oversold at {current_rsi:.1f}"
    # Crossing back up through oversold level
    elif prev_rsi < RSI_OVERSOLD and current_rsi >= RSI_OVERSOLD:
        signal = 'BUY'
        description = f"RSI crossed above {RSI_OVERSOLD} (was {prev_rsi:.1f}, now {current_rsi:.1f})"
    # Currently oversold and turning up
    elif current_rsi < RSI_OVERSOLD and current_rsi > prev_rsi:
        signal = 'BUY'
        description = f"RSI oversold at {current_rsi:.1f} and rising"
    # Crossing back down through overbought level
    elif prev_rsi > RSI_OVERBOUGHT and current_rsi <= RSI_OVERBOUGHT:
        signal = 'SELL'
        description = f"RSI crossed below {RSI_OVERBOUGHT} (was {prev_rsi:.1f}, now {current_rsi:.1f})"

    return {
        'signal': signal,
        'rsi_value': current_rsi,
        'description': description
    }


def get_macd_signal(histogram: pd.Series) -> dict:
    """
    Determine MACD signal based on histogram direction

    Returns dict with:
        - signal: 'BUY', 'SELL', or None
        - histogram_value: current histogram value
        - description: explanation
    """
    if len(histogram) < 2:
        return {'signal': None, 'histogram_value': None, 'description': 'Insufficient data'}

    current_hist = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]

    signal = None
    description = f"MACD Histogram: {current_hist:.3f}"

    # Histogram turned positive
    if prev_hist <= 0 and current_hist > 0:
        signal = 'BUY'
        description = f"MACD histogram turned positive ({current_hist:.3f})"
    # Histogram turned negative
    elif prev_hist >= 0 and current_hist < 0:
        signal = 'SELL'
        description = f"MACD histogram turned negative ({current_hist:.3f})"

    return {
        'signal': signal,
        'histogram_value': current_hist,
        'description': description
    }


def get_stochastic_signal(slow_k: pd.Series, slow_d: pd.Series) -> dict:
    """
    Determine Slow Stochastic signal

    Returns dict with:
        - signal: 'BUY', 'SELL', or None
        - k_value, d_value: current values
        - description: explanation
    """
    if len(slow_k) < 2:
        return {'signal': None, 'k_value': None, 'd_value': None, 'description': 'Insufficient data'}

    current_k = slow_k.iloc[-1]
    current_d = slow_d.iloc[-1]
    prev_k = slow_k.iloc[-2]

    signal = None
    description = f"SlowSto %K: {current_k:.1f}, %D: {current_d:.1f}"

    # Oversold and turning up (or K crossing above D in oversold)
    if current_k < STOCH_OVERSOLD:
        if current_k > prev_k:
            signal = 'BUY'
            description = f"SlowSto oversold at {current_k:.1f} and rising"
    # Crossing up through oversold level
    elif prev_k < STOCH_OVERSOLD and current_k >= STOCH_OVERSOLD:
        signal = 'BUY'
        description = f"SlowSto crossed above {STOCH_OVERSOLD}"
    # Overbought and turning down
    elif current_k > STOCH_OVERBOUGHT:
        if current_k < prev_k:
            signal = 'SELL'
            description = f"SlowSto overbought at {current_k:.1f} and falling"
    # Crossing down through overbought level
    elif prev_k > STOCH_OVERBOUGHT and current_k <= STOCH_OVERBOUGHT:
        signal = 'SELL'
        description = f"SlowSto crossed below {STOCH_OVERBOUGHT}"

    return {
        'signal': signal,
        'k_value': current_k,
        'd_value': current_d,
        'description': description
    }


# =============================================================================
# BOLLINGER BANDS
# =============================================================================

def calculate_bollinger_bands(prices: pd.Series,
                               period: int = BB_PERIOD,
                               std_dev: float = BB_STD_DEV) -> tuple:
    """
    Calculate Bollinger Bands
    Returns: (middle_band, upper_band, lower_band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower


def get_bollinger_signal(prices: pd.Series,
                          lower: pd.Series,
                          upper: pd.Series) -> dict:
    """
    Determine Bollinger Bands signal

    Returns dict with:
        - signal: 'BUY', 'SELL', or None
        - value: current price position relative to bands
        - description: explanation
    """
    if len(prices) < 2:
        return {'signal': None, 'value': None, 'description': 'Insufficient data'}

    current_price = prices.iloc[-1]
    current_lower = lower.iloc[-1]
    current_upper = upper.iloc[-1]

    if pd.isna(current_lower) or pd.isna(current_upper):
        return {'signal': None, 'value': current_price, 'description': 'Insufficient data for bands'}

    signal = None
    description = f"BB: Price ${current_price:.2f}"

    if current_price < current_lower:
        signal = 'BUY'
        description = f"Price ${current_price:.2f} below lower band ${current_lower:.2f}"
    elif current_price > current_upper:
        signal = 'SELL'
        description = f"Price ${current_price:.2f} above upper band ${current_upper:.2f}"

    return {
        'signal': signal,
        'value': current_price,
        'lower_band': current_lower,
        'upper_band': current_upper,
        'description': description
    }


# =============================================================================
# OBV (ON-BALANCE VOLUME)
# =============================================================================

def calculate_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume
    OBV = cumulative sum of volume * sign(price_change)
    """
    price_change = prices.diff()
    sign = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (sign * volume).cumsum()
    return pd.Series(obv, index=prices.index)


def get_obv_signal(obv: pd.Series, prices: pd.Series,
                    period: int = OBV_SMA_PERIOD) -> dict:
    """
    Determine OBV signal based on OBV trend vs price trend

    BUY: OBV rising while price rising (confirmation)
    SELL: OBV falling while price falling (confirmation)
    """
    if len(obv) < period:
        return {'signal': None, 'obv_value': None, 'description': 'Insufficient data'}

    current_obv = obv.iloc[-1]

    obv_trend = 'up' if obv.iloc[-1] > obv.iloc[-period] else 'down'
    price_trend = 'up' if prices.iloc[-1] > prices.iloc[-period] else 'down'

    signal = None
    description = f"OBV: {current_obv:,.0f}"

    if obv_trend == 'up' and price_trend == 'up':
        signal = 'BUY'
        description = f"OBV rising with price - bullish confirmation"
    elif obv_trend == 'down' and price_trend == 'down':
        signal = 'SELL'
        description = f"OBV falling with price - bearish confirmation"

    return {
        'signal': signal,
        'obv_value': current_obv,
        'description': description
    }


# =============================================================================
# ADX (AVERAGE DIRECTIONAL INDEX)
# =============================================================================

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = ADX_PERIOD) -> tuple:
    """
    Calculate ADX, +DI, and -DI
    Returns: (adx, plus_di, minus_di)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed TR and DM
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    smoothed_plus_dm = pd.Series(plus_dm, index=high.index).ewm(span=period, adjust=False).mean()
    smoothed_minus_dm = pd.Series(minus_dm, index=high.index).ewm(span=period, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * smoothed_plus_dm / atr
    minus_di = 100 * smoothed_minus_dm / atr

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx, plus_di, minus_di


def get_adx_signal(adx: pd.Series, plus_di: pd.Series,
                    minus_di: pd.Series) -> dict:
    """
    Determine ADX signal based on trend strength and DI crossover

    BUY: Strong trend (ADX > 25) + +DI crosses above -DI
    SELL: Strong trend (ADX > 25) + -DI crosses above +DI
    """
    if len(adx) < 2:
        return {'signal': None, 'adx_value': None, 'description': 'Insufficient data'}

    current_adx = adx.iloc[-1]
    prev_plus_di = plus_di.iloc[-2]
    prev_minus_di = minus_di.iloc[-2]
    curr_plus_di = plus_di.iloc[-1]
    curr_minus_di = minus_di.iloc[-1]

    if pd.isna(current_adx):
        return {'signal': None, 'adx_value': None, 'description': 'Insufficient data'}

    signal = None
    description = f"ADX: {current_adx:.1f}"

    strong_trend = current_adx >= ADX_STRONG_TREND

    # +DI crosses above -DI
    if strong_trend and prev_plus_di < prev_minus_di and curr_plus_di > curr_minus_di:
        signal = 'BUY'
        description = f"ADX {current_adx:.1f} - Strong uptrend, +DI crossed above -DI"
    # -DI crosses above +DI
    elif strong_trend and prev_minus_di < prev_plus_di and curr_minus_di > curr_plus_di:
        signal = 'SELL'
        description = f"ADX {current_adx:.1f} - Strong downtrend, -DI crossed above +DI"

    return {
        'signal': signal,
        'adx_value': current_adx,
        'plus_di': curr_plus_di,
        'minus_di': curr_minus_di,
        'description': description
    }


# =============================================================================
# WILLIAMS %R
# =============================================================================

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                          period: int = WILLIAMS_R_PERIOD) -> pd.Series:
    """
    Calculate Williams %R
    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    return williams_r


def get_williams_r_signal(williams_r: pd.Series) -> dict:
    """
    Determine Williams %R signal

    BUY: Crosses above -80 (leaving oversold)
    SELL: Crosses below -20 (leaving overbought)
    """
    if len(williams_r) < 2:
        return {'signal': None, 'value': None, 'description': 'Insufficient data'}

    current = williams_r.iloc[-1]
    prev = williams_r.iloc[-2]

    if pd.isna(current) or pd.isna(prev):
        return {'signal': None, 'value': None, 'description': 'Insufficient data'}

    signal = None
    description = f"Williams %R: {current:.1f}"

    # Crossing above oversold level
    if prev < WILLIAMS_R_OVERSOLD and current >= WILLIAMS_R_OVERSOLD:
        signal = 'BUY'
        description = f"Williams %R crossed above {WILLIAMS_R_OVERSOLD} (was {prev:.1f}, now {current:.1f})"
    # Crossing below overbought level
    elif prev > WILLIAMS_R_OVERBOUGHT and current <= WILLIAMS_R_OVERBOUGHT:
        signal = 'SELL'
        description = f"Williams %R crossed below {WILLIAMS_R_OVERBOUGHT} (was {prev:.1f}, now {current:.1f})"

    return {
        'signal': signal,
        'value': current,
        'description': description
    }
