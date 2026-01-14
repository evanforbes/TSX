"""
Backtesting module for TSX Stock Scanner
Validates historical indicator performance by detecting signal transitions
and measuring forward returns at 5/10/20/30/60/90/180/365 day intervals.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable
import json

from config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_EXTREME_OVERSOLD,
    STOCH_OVERBOUGHT, STOCH_OVERSOLD,
    BB_PERIOD, BB_STD_DEV,
    OBV_SMA_PERIOD,
    ADX_PERIOD, ADX_STRONG_TREND,
    WILLIAMS_R_PERIOD, WILLIAMS_R_OVERSOLD, WILLIAMS_R_OVERBOUGHT
)
from indicators import (
    calculate_rsi, calculate_macd, calculate_slow_stochastic,
    calculate_point_and_figure,
    calculate_bollinger_bands, calculate_obv, calculate_adx, calculate_williams_r
)

HOLDING_PERIODS = [5, 10, 20, 30, 60, 90, 180, 365]


@dataclass
class SignalEvent:
    """A single historical signal occurrence"""
    date: str
    indicator: str           # 'RSI', 'MACD', 'SlowSto', 'P&F'
    signal_type: str         # 'BUY', 'SELL', 'STRONG_BUY'
    indicator_value: float
    entry_price: float
    symbol: str = ''

    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    return_30d: Optional[float] = None
    return_60d: Optional[float] = None
    return_90d: Optional[float] = None
    return_180d: Optional[float] = None
    return_365d: Optional[float] = None


@dataclass
class PeriodStats:
    """Statistics for a single holding period"""
    win_rate: Optional[float]
    avg_return: Optional[float]
    median_return: Optional[float]
    profit_factor: Optional[float]
    count: int


@dataclass
class IndicatorStats:
    """Aggregated statistics for one indicator"""
    indicator: str
    signal_type: str
    total_signals: int
    stats_5d: Dict
    stats_10d: Dict
    stats_20d: Dict
    stats_30d: Dict
    best_trade: Optional[Dict]
    worst_trade: Optional[Dict]


def detect_rsi_signals(rsi_series: pd.Series, prices: pd.Series) -> List[SignalEvent]:
    """
    Scan RSI series for all historical buy/sell transitions.

    Buy signals:
    - RSI crosses back UP through oversold (20)
    - RSI extreme oversold (<= 10) -> STRONG_BUY

    Sell signals:
    - RSI crosses back DOWN through overbought (80)
    """
    signals = []

    for i in range(1, len(rsi_series)):
        current = rsi_series.iloc[i]
        prev = rsi_series.iloc[i - 1]

        if pd.isna(current) or pd.isna(prev):
            continue

        signal_date = rsi_series.index[i]
        entry_price = prices.iloc[i]
        signal_type = None

        # Extreme oversold
        if current <= RSI_EXTREME_OVERSOLD and prev > RSI_EXTREME_OVERSOLD:
            signal_type = 'STRONG_BUY'
        # Crossing up through oversold
        elif prev < RSI_OVERSOLD and current >= RSI_OVERSOLD:
            signal_type = 'BUY'
        # Crossing down through overbought
        elif prev > RSI_OVERBOUGHT and current <= RSI_OVERBOUGHT:
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='RSI',
                signal_type=signal_type,
                indicator_value=round(current, 2),
                entry_price=round(entry_price, 2)
            ))

    return signals


def detect_macd_signals(histogram: pd.Series, prices: pd.Series) -> List[SignalEvent]:
    """
    Scan MACD histogram for zero-line crossovers.

    Buy: histogram crosses from negative to positive
    Sell: histogram crosses from positive to negative
    """
    signals = []

    for i in range(1, len(histogram)):
        current = histogram.iloc[i]
        prev = histogram.iloc[i - 1]

        if pd.isna(current) or pd.isna(prev):
            continue

        signal_date = histogram.index[i]
        entry_price = prices.iloc[i]
        signal_type = None

        if prev <= 0 and current > 0:
            signal_type = 'BUY'
        elif prev >= 0 and current < 0:
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='MACD',
                signal_type=signal_type,
                indicator_value=round(current, 4),
                entry_price=round(entry_price, 2)
            ))

    return signals


def detect_stochastic_signals(slow_k: pd.Series, prices: pd.Series) -> List[SignalEvent]:
    """
    Scan Slow Stochastic for oversold/overbought transitions.

    Buy: Crossing up through 20 (oversold)
    Sell: Crossing down through 80 (overbought)
    """
    signals = []

    for i in range(1, len(slow_k)):
        current_k = slow_k.iloc[i]
        prev_k = slow_k.iloc[i - 1]

        if pd.isna(current_k) or pd.isna(prev_k):
            continue

        signal_date = slow_k.index[i]
        entry_price = prices.iloc[i]
        signal_type = None

        # Crossing up through oversold
        if prev_k < STOCH_OVERSOLD and current_k >= STOCH_OVERSOLD:
            signal_type = 'BUY'
        # Crossing down through overbought
        elif prev_k > STOCH_OVERBOUGHT and current_k <= STOCH_OVERBOUGHT:
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='SlowSto',
                signal_type=signal_type,
                indicator_value=round(current_k, 2),
                entry_price=round(entry_price, 2)
            ))

    return signals


def detect_pf_signals(prices: pd.Series) -> List[SignalEvent]:
    """
    Detect Point & Figure triple top breakouts and triple bottom breakdowns.
    Track when breakout/breakdown state changes from False to True.
    """
    signals = []
    prev_breakout = False
    prev_breakdown = False

    # Need at least 60 days of data for meaningful P&F
    MIN_PF_DAYS = 60

    if len(prices) < MIN_PF_DAYS:
        return signals

    for i in range(MIN_PF_DAYS, len(prices)):
        # Calculate P&F on data up to this point
        try:
            pf_result = calculate_point_and_figure(prices.iloc[:i+1])
            current_breakout = pf_result.get('triple_top_breakout', False)
            current_breakdown = pf_result.get('triple_bottom_breakdown', False)

            signal_date = prices.index[i]
            date_str = str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date)

            # Detect transition to breakout (BUY)
            if current_breakout and not prev_breakout:
                signals.append(SignalEvent(
                    date=date_str,
                    indicator='P&F',
                    signal_type='BUY',
                    indicator_value=1.0,
                    entry_price=round(prices.iloc[i], 2)
                ))

            # Detect transition to breakdown (SELL)
            if current_breakdown and not prev_breakdown:
                signals.append(SignalEvent(
                    date=date_str,
                    indicator='P&F',
                    signal_type='SELL',
                    indicator_value=-1.0,
                    entry_price=round(prices.iloc[i], 2)
                ))

            prev_breakout = current_breakout
            prev_breakdown = current_breakdown
        except Exception:
            continue

    return signals


def detect_bollinger_signals(prices: pd.Series, lower: pd.Series,
                              upper: pd.Series) -> List[SignalEvent]:
    """Detect Bollinger Bands signals - price crossing bands"""
    signals = []

    for i in range(1, len(prices)):
        if pd.isna(lower.iloc[i]) or pd.isna(upper.iloc[i]):
            continue

        current_price = prices.iloc[i]
        current_lower = lower.iloc[i]
        current_upper = upper.iloc[i]
        signal_date = prices.index[i]
        signal_type = None

        if current_price < current_lower:
            signal_type = 'BUY'
        elif current_price > current_upper:
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='BB',
                signal_type=signal_type,
                indicator_value=round(current_price, 2),
                entry_price=round(current_price, 2)
            ))

    return signals


def detect_obv_signals(obv: pd.Series, prices: pd.Series,
                        period: int = OBV_SMA_PERIOD) -> List[SignalEvent]:
    """Detect OBV trend confirmation signals"""
    signals = []

    for i in range(period, len(obv)):
        signal_date = obv.index[i]

        obv_trend = 'up' if obv.iloc[i] > obv.iloc[i-period] else 'down'
        price_trend = 'up' if prices.iloc[i] > prices.iloc[i-period] else 'down'

        signal_type = None
        if obv_trend == 'up' and price_trend == 'up':
            signal_type = 'BUY'
        elif obv_trend == 'down' and price_trend == 'down':
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='OBV',
                signal_type=signal_type,
                indicator_value=round(obv.iloc[i], 0),
                entry_price=round(prices.iloc[i], 2)
            ))

    return signals


def detect_adx_signals(adx: pd.Series, plus_di: pd.Series,
                        minus_di: pd.Series, prices: pd.Series) -> List[SignalEvent]:
    """Detect ADX DI crossover signals"""
    signals = []

    for i in range(1, len(adx)):
        if pd.isna(adx.iloc[i]) or adx.iloc[i] < ADX_STRONG_TREND:
            continue

        signal_date = adx.index[i]
        prev_plus = plus_di.iloc[i-1]
        prev_minus = minus_di.iloc[i-1]
        curr_plus = plus_di.iloc[i]
        curr_minus = minus_di.iloc[i]

        signal_type = None
        if prev_plus < prev_minus and curr_plus > curr_minus:
            signal_type = 'BUY'
        elif prev_minus < prev_plus and curr_minus > curr_plus:
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='ADX',
                signal_type=signal_type,
                indicator_value=round(adx.iloc[i], 2),
                entry_price=round(prices.iloc[i], 2)
            ))

    return signals


def detect_williams_r_signals(williams_r: pd.Series,
                               prices: pd.Series) -> List[SignalEvent]:
    """Detect Williams %R crossover signals"""
    signals = []

    for i in range(1, len(williams_r)):
        current = williams_r.iloc[i]
        prev = williams_r.iloc[i-1]

        if pd.isna(current) or pd.isna(prev):
            continue

        signal_date = williams_r.index[i]
        signal_type = None

        if prev < WILLIAMS_R_OVERSOLD and current >= WILLIAMS_R_OVERSOLD:
            signal_type = 'BUY'
        elif prev > WILLIAMS_R_OVERBOUGHT and current <= WILLIAMS_R_OVERBOUGHT:
            signal_type = 'SELL'

        if signal_type:
            signals.append(SignalEvent(
                date=str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
                indicator='WilliamsR',
                signal_type=signal_type,
                indicator_value=round(current, 2),
                entry_price=round(prices.iloc[i], 2)
            ))

    return signals


def calculate_forward_returns(signals: List[SignalEvent], prices: pd.Series) -> List[SignalEvent]:
    """
    For each signal, calculate returns at all holding periods forward.
    """
    price_values = prices.values

    # Create date string -> index mapping
    date_to_idx = {}
    for i, idx in enumerate(prices.index):
        date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)
        date_to_idx[date_str] = i

    for signal in signals:
        entry_idx = date_to_idx.get(signal.date)
        if entry_idx is None:
            continue

        entry_price = signal.entry_price

        for period in HOLDING_PERIODS:
            exit_idx = entry_idx + period
            if exit_idx < len(price_values):
                exit_price = price_values[exit_idx]
                return_pct = ((exit_price - entry_price) / entry_price) * 100
                setattr(signal, f'return_{period}d', round(return_pct, 2))

    return signals


def calculate_period_stats(signals: List[SignalEvent], period: int, signal_type: str) -> Dict:
    """Calculate stats for one holding period"""
    attr_name = f'return_{period}d'
    returns = [getattr(s, attr_name) for s in signals
               if getattr(s, attr_name) is not None]

    if not returns:
        return {'win_rate': None, 'avg_return': None,
                'median_return': None, 'profit_factor': None, 'count': 0}

    # For BUY signals, positive return = win
    # For SELL signals, negative return = win (avoided loss / correct sell)
    if signal_type in ('BUY', 'STRONG_BUY'):
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
    else:  # SELL
        wins = [r for r in returns if r < 0]
        losses = [r for r in returns if r >= 0]

    win_rate = len(wins) / len(returns) * 100 if returns else 0
    avg_return = sum(returns) / len(returns) if returns else 0

    sorted_returns = sorted(returns)
    median_return = sorted_returns[len(sorted_returns)//2] if returns else 0

    # Profit factor = gross profits / gross losses
    gross_profit = sum([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
    gross_loss = abs(sum([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

    return {
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 2),
        'median_return': round(median_return, 2),
        'profit_factor': profit_factor,
        'count': len(returns)
    }


def calculate_indicator_stats(signals: List[SignalEvent],
                               indicator: str,
                               signal_type: str) -> Optional[Dict]:
    """
    Calculate aggregated statistics for one indicator/signal_type combination.
    """
    # Filter signals
    if signal_type == 'BUY':
        filtered = [s for s in signals
                    if s.indicator == indicator and s.signal_type in ('BUY', 'STRONG_BUY')]
    else:
        filtered = [s for s in signals
                    if s.indicator == indicator and s.signal_type == signal_type]

    if not filtered:
        return None

    # Find best/worst trades (using 20-day return)
    all_returns = [(s, s.return_20d) for s in filtered
                   if s.return_20d is not None]

    best = max(all_returns, key=lambda x: x[1]) if all_returns else (None, None)
    worst = min(all_returns, key=lambda x: x[1]) if all_returns else (None, None)

    return {
        'indicator': indicator,
        'signal_type': signal_type,
        'total_signals': len(filtered),
        'stats_5d': calculate_period_stats(filtered, 5, signal_type),
        'stats_10d': calculate_period_stats(filtered, 10, signal_type),
        'stats_20d': calculate_period_stats(filtered, 20, signal_type),
        'stats_30d': calculate_period_stats(filtered, 30, signal_type),
        'best_trade': {
            'symbol': best[0].symbol if best[0] else None,
            'date': best[0].date if best[0] else None,
            'return': best[1]
        } if best[0] else None,
        'worst_trade': {
            'symbol': worst[0].symbol if worst[0] else None,
            'date': worst[0].date if worst[0] else None,
            'return': worst[1]
        } if worst[0] else None
    }


def backtest_symbol(symbol: str, df: pd.DataFrame) -> Dict:
    """
    Run complete backtest for a single symbol.
    """
    prices = df['Close']

    # Calculate all indicators
    rsi = calculate_rsi(prices)
    macd_line, signal_line, histogram = calculate_macd(prices)
    slow_k, slow_d = calculate_slow_stochastic(df['High'], df['Low'], prices)

    # Detect all signal transitions
    all_signals = []
    all_signals.extend(detect_rsi_signals(rsi, prices))
    all_signals.extend(detect_macd_signals(histogram, prices))
    all_signals.extend(detect_stochastic_signals(slow_k, prices))
    # P&F is expensive, skip for per-symbol to speed up
    # all_signals.extend(detect_pf_signals(prices))

    # Calculate forward returns for each signal
    all_signals = calculate_forward_returns(all_signals, prices)

    # Tag with symbol
    for sig in all_signals:
        sig.symbol = symbol

    return {
        'symbol': symbol,
        'start_date': str(df.index[0].date()) if hasattr(df.index[0], 'date') else str(df.index[0]),
        'end_date': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
        'total_trading_days': len(df),
        'total_signals': len(all_signals),
        'signals': [asdict(s) for s in all_signals]
    }


@dataclass
class TierSignal:
    """A multi-indicator signal (Silver or Gold tier)"""
    date: str
    symbol: str
    tier: str                # 'GOLD', 'SILVER', 'SINGLE'
    signal_type: str         # 'BUY' or 'SELL'
    indicators: List[str]    # Which indicators triggered
    entry_price: float

    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    return_30d: Optional[float] = None
    return_60d: Optional[float] = None
    return_90d: Optional[float] = None
    return_180d: Optional[float] = None
    return_365d: Optional[float] = None


def group_signals_by_date(signals: List[SignalEvent]) -> Dict[str, Dict[str, List[SignalEvent]]]:
    """
    Group signals by date and signal type.
    Returns: {date: {'BUY': [signals], 'SELL': [signals]}}
    """
    grouped = {}
    for sig in signals:
        if sig.date not in grouped:
            grouped[sig.date] = {'BUY': [], 'SELL': []}

        # Normalize signal type
        sig_type = 'BUY' if sig.signal_type in ('BUY', 'STRONG_BUY') else 'SELL'
        grouped[sig.date][sig_type].append(sig)

    return grouped


def create_tier_signals(signals: List[SignalEvent], symbol: str) -> List[TierSignal]:
    """
    Convert individual signals into tier-based signals.
    Groups same-day signals and classifies as Gold (3+), Silver (2), or Single (1).
    """
    grouped = group_signals_by_date(signals)
    tier_signals = []

    for date, by_type in grouped.items():
        for sig_type, sigs in by_type.items():
            if not sigs:
                continue

            num_indicators = len(sigs)
            indicators = list(set(s.indicator for s in sigs))

            # Determine tier
            if num_indicators >= 4:
                tier = 'DIAMOND'
            elif num_indicators == 3:
                tier = 'GOLD'
            elif num_indicators == 2:
                tier = 'SILVER'
            else:
                tier = 'SINGLE'

            # Use first signal for price/returns (they should be same since same day)
            first_sig = sigs[0]

            tier_signals.append(TierSignal(
                date=date,
                symbol=symbol,
                tier=tier,
                signal_type=sig_type,
                indicators=indicators,
                entry_price=first_sig.entry_price,
                return_5d=first_sig.return_5d,
                return_10d=first_sig.return_10d,
                return_20d=first_sig.return_20d,
                return_30d=first_sig.return_30d,
                return_60d=first_sig.return_60d,
                return_90d=first_sig.return_90d,
                return_180d=first_sig.return_180d,
                return_365d=first_sig.return_365d
            ))

    return tier_signals


def calculate_tier_stats(tier_signals: List[TierSignal], tier: str, signal_type: str) -> Optional[Dict]:
    """
    Calculate statistics for a specific tier (GOLD, SILVER, SINGLE) and signal type.
    """
    filtered = [s for s in tier_signals if s.tier == tier and s.signal_type == signal_type]

    if not filtered:
        return None

    def calc_period_stats(signals: List[TierSignal], period: int) -> Dict:
        attr_name = f'return_{period}d'
        returns = [getattr(s, attr_name) for s in signals if getattr(s, attr_name) is not None]

        if not returns:
            return {'win_rate': None, 'avg_return': None, 'median_return': None,
                    'profit_factor': None, 'count': 0}

        # For BUY signals, positive return = win
        # For SELL signals, negative return = win
        if signal_type == 'BUY':
            wins = [r for r in returns if r > 0]
        else:
            wins = [r for r in returns if r < 0]

        win_rate = len(wins) / len(returns) * 100 if returns else 0
        avg_return = sum(returns) / len(returns) if returns else 0
        sorted_returns = sorted(returns)
        median_return = sorted_returns[len(sorted_returns)//2] if returns else 0

        gross_profit = sum([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        gross_loss = abs(sum([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

        return {
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 2),
            'median_return': round(median_return, 2),
            'profit_factor': profit_factor,
            'count': len(returns)
        }

    # Find best/worst
    all_returns = [(s, s.return_20d) for s in filtered if s.return_20d is not None]
    best = max(all_returns, key=lambda x: x[1]) if all_returns else (None, None)
    worst = min(all_returns, key=lambda x: x[1]) if all_returns else (None, None)

    return {
        'tier': tier,
        'signal_type': signal_type,
        'total_signals': len(filtered),
        'stats_5d': calc_period_stats(filtered, 5),
        'stats_10d': calc_period_stats(filtered, 10),
        'stats_20d': calc_period_stats(filtered, 20),
        'stats_30d': calc_period_stats(filtered, 30),
        'stats_60d': calc_period_stats(filtered, 60),
        'stats_90d': calc_period_stats(filtered, 90),
        'stats_180d': calc_period_stats(filtered, 180),
        'stats_365d': calc_period_stats(filtered, 365),
        'best_trade': {
            'symbol': best[0].symbol if best[0] else None,
            'date': best[0].date if best[0] else None,
            'indicators': best[0].indicators if best[0] else [],
            'return': best[1]
        } if best[0] else None,
        'worst_trade': {
            'symbol': worst[0].symbol if worst[0] else None,
            'date': worst[0].date if worst[0] else None,
            'indicators': worst[0].indicators if worst[0] else [],
            'return': worst[1]
        } if worst[0] else None
    }


def run_market_backtest(progress_callback: Optional[Callable] = None, indicators: List[str] = None, stop_check: Optional[Callable] = None) -> Dict:
    """
    Run backtest across all TSX symbols.
    Returns aggregated results focused on tier performance (Gold, Silver, Single).

    Args:
        progress_callback: Optional callback for progress updates
        indicators: List of indicators to include (default: all indicators)
        stop_check: Optional callback that returns True if stop requested
    """
    from scanner import get_tsx_symbols, fetch_stock_data, passes_filters

    # Default to all indicators if none specified
    if indicators is None:
        indicators = ['RSI', 'MACD', 'SlowSto', 'P&F', 'BB', 'OBV', 'ADX', 'WilliamsR']

    symbols = get_tsx_symbols()
    all_tier_signals = []
    symbols_analyzed = 0

    for i, symbol in enumerate(symbols):
        # Check if stop was requested
        if stop_check and stop_check():
            break

        if progress_callback:
            progress_callback(i + 1, len(symbols))

        try:
            df = fetch_stock_data(symbol)
            if df is None or len(df) < 50:
                continue

            passes, _, _, _ = passes_filters(df)
            if not passes:
                continue

            prices = df['Close']
            symbol_signals = []

            # Only calculate and detect signals for selected indicators
            if 'RSI' in indicators:
                rsi = calculate_rsi(prices)
                symbol_signals.extend(detect_rsi_signals(rsi, prices))

            if 'MACD' in indicators:
                macd_line, signal_line, histogram = calculate_macd(prices)
                symbol_signals.extend(detect_macd_signals(histogram, prices))

            if 'SlowSto' in indicators:
                slow_k, slow_d = calculate_slow_stochastic(df['High'], df['Low'], prices)
                symbol_signals.extend(detect_stochastic_signals(slow_k, prices))

            if 'BB' in indicators:
                middle, upper, lower = calculate_bollinger_bands(prices)
                symbol_signals.extend(detect_bollinger_signals(prices, lower, upper))

            if 'OBV' in indicators:
                obv = calculate_obv(prices, df['Volume'])
                symbol_signals.extend(detect_obv_signals(obv, prices))

            if 'ADX' in indicators:
                adx, plus_di, minus_di = calculate_adx(df['High'], df['Low'], prices)
                symbol_signals.extend(detect_adx_signals(adx, plus_di, minus_di, prices))

            if 'WilliamsR' in indicators:
                williams_r = calculate_williams_r(df['High'], df['Low'], prices)
                symbol_signals.extend(detect_williams_r_signals(williams_r, prices))

            if 'P&F' in indicators:
                symbol_signals.extend(detect_pf_signals(prices))

            # Calculate forward returns
            symbol_signals = calculate_forward_returns(symbol_signals, prices)

            # Convert to tier signals
            tier_signals = create_tier_signals(symbol_signals, symbol)
            all_tier_signals.extend(tier_signals)
            symbols_analyzed += 1

        except Exception as e:
            continue

    # Calculate stats by tier
    tier_stats = {}
    for tier in ['DIAMOND', 'GOLD', 'SILVER', 'SINGLE']:
        for signal_type in ['BUY', 'SELL']:
            key = f"{tier}_{signal_type}"
            stats = calculate_tier_stats(all_tier_signals, tier, signal_type)
            if stats:
                tier_stats[key] = stats

    # Find top/worst trades (prioritize Diamond/Gold/Silver) with ALL return periods
    top_tiers = [s for s in all_tier_signals if s.tier in ('DIAMOND', 'GOLD', 'SILVER') and s.return_20d is not None]

    if top_tiers:
        trades_with_returns = [(s, s.return_20d) for s in top_tiers]
    else:
        trades_with_returns = [(s, s.return_20d) for s in all_tier_signals if s.return_20d is not None]

    trades_with_returns.sort(key=lambda x: x[1], reverse=True)

    # Include all return periods for each trade so UI can switch between them
    top_trades = [{
        'symbol': t[0].symbol,
        'tier': t[0].tier,
        'indicators': t[0].indicators,
        'signal_type': t[0].signal_type,
        'date': t[0].date,
        'return': t[1],
        'return_5d': t[0].return_5d,
        'return_10d': t[0].return_10d,
        'return_20d': t[0].return_20d,
        'return_30d': t[0].return_30d,
        'return_60d': t[0].return_60d,
        'return_90d': t[0].return_90d,
        'return_180d': t[0].return_180d,
        'return_365d': t[0].return_365d
    } for t in trades_with_returns[:10]]

    worst_trades = [{
        'symbol': t[0].symbol,
        'tier': t[0].tier,
        'indicators': t[0].indicators,
        'signal_type': t[0].signal_type,
        'date': t[0].date,
        'return': t[1],
        'return_5d': t[0].return_5d,
        'return_10d': t[0].return_10d,
        'return_20d': t[0].return_20d,
        'return_30d': t[0].return_30d,
        'return_60d': t[0].return_60d,
        'return_90d': t[0].return_90d,
        'return_180d': t[0].return_180d,
        'return_365d': t[0].return_365d
    } for t in trades_with_returns[-10:]]

    # Count totals by tier
    diamond_count = len([s for s in all_tier_signals if s.tier == 'DIAMOND'])
    gold_count = len([s for s in all_tier_signals if s.tier == 'GOLD'])
    silver_count = len([s for s in all_tier_signals if s.tier == 'SILVER'])
    single_count = len([s for s in all_tier_signals if s.tier == 'SINGLE'])

    # Store individual gold/silver signals with dates for period filtering
    gold_signals_list = []
    silver_signals_list = []
    for s in all_tier_signals:
        signal_data = {
            'symbol': s.symbol,
            'date': s.date,
            'signal_type': s.signal_type,
            'indicators': s.indicators,
            'return_5d': s.return_5d,
            'return_10d': s.return_10d,
            'return_20d': s.return_20d,
            'return_30d': s.return_30d,
            'return_60d': s.return_60d,
            'return_90d': s.return_90d,
            'return_180d': s.return_180d,
            'return_365d': s.return_365d
        }
        if s.tier == 'GOLD':
            gold_signals_list.append(signal_data)
        elif s.tier == 'SILVER':
            silver_signals_list.append(signal_data)

    # Also create aggregated stock lists (for backward compatibility)
    gold_stocks = {}
    silver_stocks = {}
    for s in all_tier_signals:
        if s.tier == 'GOLD':
            if s.symbol not in gold_stocks:
                gold_stocks[s.symbol] = {'symbol': s.symbol, 'buy_signals': 0, 'sell_signals': 0}
            if s.signal_type == 'BUY':
                gold_stocks[s.symbol]['buy_signals'] += 1
            else:
                gold_stocks[s.symbol]['sell_signals'] += 1
        elif s.tier == 'SILVER':
            if s.symbol not in silver_stocks:
                silver_stocks[s.symbol] = {'symbol': s.symbol, 'buy_signals': 0, 'sell_signals': 0}
            if s.signal_type == 'BUY':
                silver_stocks[s.symbol]['buy_signals'] += 1
            else:
                silver_stocks[s.symbol]['sell_signals'] += 1

    return {
        'run_timestamp': datetime.now().isoformat(),
        'symbols_analyzed': symbols_analyzed,
        'total_signals': len(all_tier_signals),
        'diamond_signals': diamond_count,
        'gold_signals': gold_count,
        'silver_signals': silver_count,
        'single_signals': single_count,
        'tier_stats': tier_stats,
        'top_trades': top_trades,
        'worst_trades': list(reversed(worst_trades)),
        'gold_stocks': list(gold_stocks.values()),
        'silver_stocks': list(silver_stocks.values()),
        'gold_signals_list': gold_signals_list,
        'silver_signals_list': silver_signals_list
    }


if __name__ == '__main__':
    # Test with a single symbol
    from scanner import fetch_stock_data

    print("Testing backtest on RY (Royal Bank)...")
    df = fetch_stock_data('RY')
    if df is not None:
        result = backtest_symbol('RY', df)
        print(f"Symbol: {result['symbol']}")
        print(f"Period: {result['start_date']} to {result['end_date']}")
        print(f"Trading days: {result['total_trading_days']}")
        print(f"Total signals: {result['total_signals']}")

        # Count by indicator
        signals = result['signals']
        for ind in ['RSI', 'MACD', 'SlowSto']:
            buy_count = len([s for s in signals if s['indicator'] == ind and s['signal_type'] in ('BUY', 'STRONG_BUY')])
            sell_count = len([s for s in signals if s['indicator'] == ind and s['signal_type'] == 'SELL'])
            print(f"  {ind}: {buy_count} buys, {sell_count} sells")
