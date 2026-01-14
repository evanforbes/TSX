#!/usr/bin/env python3
"""
TSX Stock Scanner
Scans Toronto Stock Exchange stocks for technical signals

Signals detected:
- RSI overbought/oversold crossovers
- MACD histogram direction changes
- Slow Stochastic overbought/oversold
- Point & Figure triple top breakouts
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import sys
import requests

from config import (
    SCAN_MODE, CUSTOM_SYMBOLS, DEFAULT_TSX_SYMBOLS,
    MIN_PRICE, MIN_VOLUME, LOOKBACK_DAYS,
    DIAMOND_STANDARD_MIN_SIGNALS, GOLD_STANDARD_MIN_SIGNALS, SILVER_STANDARD_MIN_SIGNALS,
    DEFAULT_INDICATORS, ALL_INDICATORS
)
from indicators import (
    calculate_rsi, calculate_macd, calculate_slow_stochastic,
    calculate_point_and_figure,
    get_rsi_signal, get_macd_signal, get_stochastic_signal,
    calculate_bollinger_bands, get_bollinger_signal,
    calculate_obv, get_obv_signal,
    calculate_adx, get_adx_signal,
    calculate_williams_r, get_williams_r_signal
)
from tsx_symbols import get_all_tsx_symbols


def get_tsx_symbols() -> list:
    """Get list of TSX symbols to scan based on SCAN_MODE"""
    if SCAN_MODE == "custom" and CUSTOM_SYMBOLS:
        return CUSTOM_SYMBOLS
    elif SCAN_MODE == "full":
        return get_all_tsx_symbols()
    else:
        return DEFAULT_TSX_SYMBOLS


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for a TSX stock

    Args:
        symbol: Stock symbol without .TO suffix
        days: Number of days of history to fetch

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    # Handle TSX symbol formatting
    tsx_symbol = f"{symbol}.TO"

    try:
        # Use yf.download() which is more reliable on cloud platforms
        df = yf.download(
            tsx_symbol,
            period=f"{days}d",
            progress=False,
            timeout=10
        )

        if df.empty:
            print(f"[DEBUG] Empty data for {tsx_symbol}")
            return None

        return df
    except Exception as e:
        print(f"[DEBUG] Error fetching {tsx_symbol}: {e}")
        return None


def fetch_stock_info(symbol: str) -> dict:
    """
    Fetch additional stock info (52-week, sector, dividend, earnings)
    """
    tsx_symbol = f"{symbol}.TO"
    info = {
        'week_52_high': None,
        'week_52_low': None,
        'week_52_pct': None,  # % from 52-week low
        'sector': None,
        'industry': None,
        'market_cap': None,
        'dividend_yield': None,
        'earnings_date': None,
        'volume_avg': None,
        'volume_ratio': None  # today's volume vs avg
    }

    try:
        ticker = yf.Ticker(tsx_symbol)
        ticker_info = ticker.info

        # 52-week high/low
        info['week_52_high'] = ticker_info.get('fiftyTwoWeekHigh')
        info['week_52_low'] = ticker_info.get('fiftyTwoWeekLow')

        # Calculate % from 52-week low
        if info['week_52_high'] and info['week_52_low']:
            current = ticker_info.get('currentPrice') or ticker_info.get('regularMarketPrice')
            if current and info['week_52_low'] > 0:
                range_size = info['week_52_high'] - info['week_52_low']
                if range_size > 0:
                    info['week_52_pct'] = ((current - info['week_52_low']) / range_size) * 100

        # Sector info
        info['sector'] = ticker_info.get('sector')
        info['industry'] = ticker_info.get('industry')
        info['market_cap'] = ticker_info.get('marketCap')

        # Dividend
        dividend = ticker_info.get('dividendYield')
        if dividend:
            info['dividend_yield'] = dividend * 100  # Convert to percentage

        # Volume analysis
        info['volume_avg'] = ticker_info.get('averageVolume')
        current_vol = ticker_info.get('volume') or ticker_info.get('regularMarketVolume')
        if info['volume_avg'] and current_vol and info['volume_avg'] > 0:
            info['volume_ratio'] = current_vol / info['volume_avg']

        # Earnings date
        try:
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    earnings = calendar.loc['Earnings Date']
                    if hasattr(earnings, 'iloc') and len(earnings) > 0:
                        info['earnings_date'] = str(earnings.iloc[0])[:10]
                    elif earnings is not None:
                        info['earnings_date'] = str(earnings)[:10]
        except Exception:
            pass

    except Exception:
        pass

    return info


def passes_filters(df: pd.DataFrame) -> tuple:
    """
    Check if stock passes price and volume filters

    Returns: (passes: bool, reason: str, price: float, volume: float)
    """
    if df.empty or len(df) < 20:
        return False, "Insufficient data", 0, 0

    current_price = df['Close'].iloc[-1]
    avg_volume = df['Volume'].tail(20).mean()

    if current_price < MIN_PRICE:
        return False, f"Price ${current_price:.2f} below ${MIN_PRICE}", current_price, avg_volume

    if avg_volume < MIN_VOLUME:
        return False, f"Volume {avg_volume:,.0f} below {MIN_VOLUME:,}", current_price, avg_volume

    return True, "OK", current_price, avg_volume


def analyze_stock(symbol: str, df: pd.DataFrame, indicators: list = None) -> dict:
    """
    Run technical analysis on a stock with selectable indicators

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data
        indicators: List of indicator names to run.
                    None or empty = use DEFAULT_INDICATORS

    Returns dict with all signals and indicator values
    """
    # Use defaults if not specified
    if not indicators:
        indicators = DEFAULT_INDICATORS

    result = {
        'symbol': symbol,
        'price': df['Close'].iloc[-1],
        'volume': df['Volume'].tail(20).mean(),
        'signals': [],
        'indicators_used': indicators,
        'rsi': None,
        'macd': None,
        'stochastic': None,
        'pf': None,
        'bollinger': None,
        'obv': None,
        'adx': None,
        'williams_r': None
    }

    # RSI
    if 'RSI' in indicators:
        rsi = calculate_rsi(df['Close'])
        rsi_signal = get_rsi_signal(rsi)
        result['rsi'] = rsi_signal
        if rsi_signal['signal']:
            result['signals'].append(('RSI', rsi_signal['signal'], rsi_signal['description']))

    # MACD
    if 'MACD' in indicators:
        macd_line, signal_line, histogram = calculate_macd(df['Close'])
        macd_signal = get_macd_signal(histogram)
        result['macd'] = macd_signal
        if macd_signal['signal']:
            result['signals'].append(('MACD', macd_signal['signal'], macd_signal['description']))

    # Slow Stochastic
    if 'SlowSto' in indicators:
        slow_k, slow_d = calculate_slow_stochastic(df['High'], df['Low'], df['Close'])
        stoch_signal = get_stochastic_signal(slow_k, slow_d)
        result['stochastic'] = stoch_signal
        if stoch_signal['signal']:
            result['signals'].append(('SlowSto', stoch_signal['signal'], stoch_signal['description']))

    # Point & Figure
    if 'P&F' in indicators:
        pf_result = calculate_point_and_figure(df['Close'])
        result['pf'] = pf_result
        if pf_result['triple_top_breakout']:
            result['signals'].append(('P&F', 'BUY', 'Triple Top Breakout detected'))
        elif pf_result.get('triple_bottom_breakdown'):
            result['signals'].append(('P&F', 'SELL', 'Triple Bottom Breakdown detected'))

    # Bollinger Bands
    if 'BB' in indicators:
        middle, upper, lower = calculate_bollinger_bands(df['Close'])
        bb_signal = get_bollinger_signal(df['Close'], lower, upper)
        result['bollinger'] = bb_signal
        if bb_signal['signal']:
            result['signals'].append(('BB', bb_signal['signal'], bb_signal['description']))

    # OBV (On-Balance Volume)
    if 'OBV' in indicators:
        obv = calculate_obv(df['Close'], df['Volume'])
        obv_signal = get_obv_signal(obv, df['Close'])
        result['obv'] = obv_signal
        if obv_signal['signal']:
            result['signals'].append(('OBV', obv_signal['signal'], obv_signal['description']))

    # ADX (Average Directional Index)
    if 'ADX' in indicators:
        adx, plus_di, minus_di = calculate_adx(df['High'], df['Low'], df['Close'])
        adx_signal = get_adx_signal(adx, plus_di, minus_di)
        result['adx'] = adx_signal
        if adx_signal['signal']:
            result['signals'].append(('ADX', adx_signal['signal'], adx_signal['description']))

    # Williams %R
    if 'WilliamsR' in indicators:
        williams_r = calculate_williams_r(df['High'], df['Low'], df['Close'])
        wr_signal = get_williams_r_signal(williams_r)
        result['williams_r'] = wr_signal
        if wr_signal['signal']:
            result['signals'].append(('WilliamsR', wr_signal['signal'], wr_signal['description']))

    return result


def classify_signal_tier(result: dict) -> dict:
    """
    Classify stock into Diamond (4+), Gold (3+), Silver (2+), or regular tier

    Returns dict with:
        - tier_buy: 'DIAMOND', 'GOLD', 'SILVER', or None
        - tier_sell: 'DIAMOND', 'GOLD', 'SILVER', or None
        - buy_count: number of buy signals
        - sell_count: number of sell signals
        - buy_indicators: list of indicators with buy signals
        - sell_indicators: list of indicators with sell signals
    """
    buy_signals = []
    sell_signals = []

    for indicator, signal, desc in result['signals']:
        if signal in ('BUY', 'STRONG_BUY'):
            buy_signals.append((indicator, desc))
        elif signal == 'SELL':
            sell_signals.append((indicator, desc))

    # Determine buy tier
    tier_buy = None
    if len(buy_signals) >= DIAMOND_STANDARD_MIN_SIGNALS:
        tier_buy = 'DIAMOND'
    elif len(buy_signals) >= GOLD_STANDARD_MIN_SIGNALS:
        tier_buy = 'GOLD'
    elif len(buy_signals) >= SILVER_STANDARD_MIN_SIGNALS:
        tier_buy = 'SILVER'

    # Determine sell tier
    tier_sell = None
    if len(sell_signals) >= DIAMOND_STANDARD_MIN_SIGNALS:
        tier_sell = 'DIAMOND'
    elif len(sell_signals) >= GOLD_STANDARD_MIN_SIGNALS:
        tier_sell = 'GOLD'
    elif len(sell_signals) >= SILVER_STANDARD_MIN_SIGNALS:
        tier_sell = 'SILVER'

    return {
        'tier_buy': tier_buy,
        'tier_sell': tier_sell,
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'buy_indicators': buy_signals,
        'sell_indicators': sell_signals
    }


def print_header():
    """Print scanner header"""
    print("=" * 80)
    print("TSX STOCK SCANNER")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Filters: Price > ${MIN_PRICE:.2f}, Volume > {MIN_VOLUME:,}")
    print("=" * 80)
    print()


def print_results(results: list):
    """Print scan results"""
    # Classify all stocks into tiers
    diamond_buys = []
    diamond_sells = []
    gold_buys = []
    gold_sells = []
    silver_buys = []
    silver_sells = []
    regular_buys = []
    regular_sells = []
    strong_buys = []

    for r in results:
        classification = classify_signal_tier(r)

        # Check for Diamond/Gold/Silver tiers
        if classification['tier_buy'] == 'DIAMOND':
            diamond_buys.append((r, classification))
        elif classification['tier_sell'] == 'DIAMOND':
            diamond_sells.append((r, classification))
        elif classification['tier_buy'] == 'GOLD':
            gold_buys.append((r, classification))
        elif classification['tier_sell'] == 'GOLD':
            gold_sells.append((r, classification))
        elif classification['tier_buy'] == 'SILVER':
            silver_buys.append((r, classification))
        elif classification['tier_sell'] == 'SILVER':
            silver_sells.append((r, classification))
        else:
            # Regular signals (single indicator)
            for indicator, signal, desc in r['signals']:
                if signal == 'STRONG_BUY':
                    strong_buys.append((r, indicator, desc))
                elif signal == 'BUY':
                    regular_buys.append((r, indicator, desc))
                elif signal == 'SELL':
                    regular_sells.append((r, indicator, desc))

    # Print Diamond Standard Buys (HIGHEST PRIORITY)
    if diamond_buys:
        print("\n" + "=" * 80)
        print("💎 DIAMOND STANDARD BUY 💎 (4+ Aligned Buy Signals)")
        print("=" * 80)
        for r, classification in diamond_buys:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}  [{classification['buy_count']} BUY signals]")
            for indicator, desc in classification['buy_indicators']:
                print(f"  [BUY] {indicator}: {desc}")

    # Print Diamond Standard Sells
    if diamond_sells:
        print("\n" + "=" * 80)
        print("💎 DIAMOND STANDARD SELL 💎 (4+ Aligned Sell Signals)")
        print("=" * 80)
        for r, classification in diamond_sells:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}  [{classification['sell_count']} SELL signals]")
            for indicator, desc in classification['sell_indicators']:
                print(f"  [SELL] {indicator}: {desc}")

    # Print Gold Standard Buys
    if gold_buys:
        print("\n" + "=" * 80)
        print("*** GOLD STANDARD BUY *** (3 Aligned Buy Signals)")
        print("=" * 80)
        for r, classification in gold_buys:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}  [{classification['buy_count']} BUY signals]")
            for indicator, desc in classification['buy_indicators']:
                print(f"  [BUY] {indicator}: {desc}")

    # Print Gold Standard Sells
    if gold_sells:
        print("\n" + "=" * 80)
        print("*** GOLD STANDARD SELL *** (3 Aligned Sell Signals)")
        print("=" * 80)
        for r, classification in gold_sells:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}  [{classification['sell_count']} SELL signals]")
            for indicator, desc in classification['sell_indicators']:
                print(f"  [SELL] {indicator}: {desc}")

    # Print Silver Standard Buys
    if silver_buys:
        print("\n" + "=" * 80)
        print("** SILVER STANDARD BUY ** (2 Aligned Buy Signals)")
        print("=" * 80)
        for r, classification in silver_buys:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}  [{classification['buy_count']} BUY signals]")
            for indicator, desc in classification['buy_indicators']:
                print(f"  [BUY] {indicator}: {desc}")

    # Print Silver Standard Sells
    if silver_sells:
        print("\n" + "=" * 80)
        print("** SILVER STANDARD SELL ** (2 Aligned Sell Signals)")
        print("=" * 80)
        for r, classification in silver_sells:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}  [{classification['sell_count']} SELL signals]")
            for indicator, desc in classification['sell_indicators']:
                print(f"  [SELL] {indicator}: {desc}")

    # Print Strong Buys (RSI extreme oversold)
    if strong_buys:
        print("\n" + "=" * 80)
        print("STRONG BUY SIGNALS (RSI Extreme Oversold)")
        print("=" * 80)
        for r, indicator, desc in strong_buys:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}")
            print(f"  [{indicator}] {desc}")

    # Print Regular Buys
    if regular_buys:
        print("\n" + "=" * 80)
        print("BUY SIGNALS (Single Indicator)")
        print("=" * 80)
        for r, indicator, desc in regular_buys:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}")
            print(f"  [{indicator}] {desc}")

    # Print Regular Sells
    if regular_sells:
        print("\n" + "=" * 80)
        print("SELL SIGNALS (Single Indicator)")
        print("=" * 80)
        for r, indicator, desc in regular_sells:
            print(f"\n{r['symbol']}.TO - ${r['price']:.2f}")
            print(f"  [{indicator}] {desc}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"💎 DIAMOND STANDARD BUYS:  {len(diamond_buys)} 💎")
    print(f"💎 DIAMOND STANDARD SELLS: {len(diamond_sells)} 💎")
    print(f"*** GOLD STANDARD BUYS:    {len(gold_buys)} ***")
    print(f"*** GOLD STANDARD SELLS:   {len(gold_sells)} ***")
    print(f"**  SILVER STANDARD BUYS:  {len(silver_buys)} **")
    print(f"**  SILVER STANDARD SELLS: {len(silver_sells)} **")
    print(f"Strong Buys: {len(strong_buys)}")
    print(f"Regular Buy Signals: {len(regular_buys)}")
    print(f"Regular Sell Signals: {len(regular_sells)}")
    print(f"Total Stocks with Signals: {len(results)}")


def run_scanner(verbose: bool = False):
    """Main scanner function"""
    print_header()

    symbols = get_tsx_symbols()
    total = len(symbols)
    print(f"Scanning {total} TSX stocks...\n")

    results = []
    skipped = 0
    signals_found = []

    # Progress display interval (every N stocks)
    progress_interval = 25 if total > 100 else 10

    for i, symbol in enumerate(symbols, 1):
        # Show progress
        if i % progress_interval == 0 or i == total:
            pct = (i / total) * 100
            print(f"Progress: {i}/{total} ({pct:.0f}%) - {len(signals_found)} signals found so far...")

        # Fetch data
        df = fetch_stock_data(symbol)
        if df is None:
            skipped += 1
            continue

        # Check filters
        passes, reason, price, volume = passes_filters(df)
        if not passes:
            skipped += 1
            continue

        # Analyze
        result = analyze_stock(symbol, df)
        results.append(result)

        if result['signals']:
            signal_types = [s[1] for s in result['signals']]
            signals_found.append(f"{symbol}.TO: {', '.join(signal_types)}")

            # In verbose mode, print each signal as found
            if verbose:
                print(f"  -> {symbol}.TO: {', '.join(signal_types)}")

    # Print results
    stocks_with_signals = [r for r in results if r['signals']]
    print(f"\nScan complete!")
    print(f"  Analyzed: {len(results)} stocks")
    print(f"  Skipped: {skipped} (no data or filtered out)")
    print(f"  Signals found: {len(stocks_with_signals)} stocks")

    if stocks_with_signals:
        print_results(stocks_with_signals)
    else:
        print("\nNo signals found matching your criteria.")


def scan_single(symbol: str):
    """Scan a single stock and show detailed analysis"""
    print(f"\nDetailed Analysis: {symbol}.TO")
    print("=" * 60)

    df = fetch_stock_data(symbol)
    if df is None:
        print("Error: Could not fetch data for this symbol")
        return

    passes, reason, price, volume = passes_filters(df)
    print(f"Price: ${price:.2f}")
    print(f"Avg Volume (20-day): {volume:,.0f}")
    print(f"Filter Status: {reason}")

    if not passes:
        print("\nNote: Stock does not pass filters but showing analysis anyway.")

    result = analyze_stock(symbol, df)

    print("\n--- RSI ---")
    rsi = result['rsi']
    print(f"Current RSI: {rsi['rsi_value']:.1f}")
    print(f"Signal: {rsi['signal'] or 'None'}")
    print(f"Description: {rsi['description']}")

    print("\n--- MACD ---")
    macd = result['macd']
    print(f"Histogram: {macd['histogram_value']:.4f}")
    print(f"Signal: {macd['signal'] or 'None'}")
    print(f"Description: {macd['description']}")

    print("\n--- Slow Stochastic ---")
    stoch = result['stochastic']
    print(f"%K: {stoch['k_value']:.1f}, %D: {stoch['d_value']:.1f}")
    print(f"Signal: {stoch['signal'] or 'None'}")
    print(f"Description: {stoch['description']}")

    print("\n--- Point & Figure ---")
    pf = result['pf']
    print(f"Current Direction: {pf['current_direction'] or 'N/A'}")
    print(f"Total Columns: {len(pf['columns'])}")
    print(f"Triple Top Breakout: {'YES - BUY SIGNAL' if pf['triple_top_breakout'] else 'No'}")

    print("\n--- Summary ---")
    if result['signals']:
        print("Active Signals:")
        for indicator, signal, desc in result['signals']:
            print(f"  [{signal}] {indicator}: {desc}")
    else:
        print("No active signals")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Scan single stock
        symbol = sys.argv[1].upper().replace(".TO", "")
        scan_single(symbol)
    else:
        # Full scan
        run_scanner()
