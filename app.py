#!/usr/bin/env python3
"""
TSX Stock Scanner - Web App
Mobile-friendly dashboard for viewing scan results
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import csv
import io

from scanner import get_tsx_symbols, fetch_stock_data, fetch_stock_info, passes_filters, analyze_stock, classify_signal_tier
from config import DEFAULT_INDICATORS, ALL_INDICATORS
from news import get_stock_news, get_analyst_ratings
from fear_greed import get_fear_greed_index
from storage import (
    get_watchlist, add_to_watchlist, remove_from_watchlist, is_in_watchlist,
    get_alerts, add_alert, remove_alert, get_alerts_for_symbol, check_alerts,
    get_scan_history, save_scan_history,
    save_backtest_results, get_backtest_results
)
from backtest import run_market_backtest

app = Flask(__name__)

# Global state for scan results
scan_state = {
    'last_scan': None,
    'results': [],
    'is_scanning': False,
    'stop_requested': False,
    'progress': 0,
    'total': 0,
    'error': None,
    'sector_breakdown': {},
    'auto_scan_enabled': False,
    'next_auto_scan': None
}


# Number of parallel workers for scanning
SCAN_WORKERS = 20  # Adjust based on performance (10-30 is typical)


def process_single_stock(symbol, indicators):
    """Process a single stock - used by parallel executor"""
    try:
        df = fetch_stock_data(symbol)
        if df is None:
            return None

        passes, reason, price, volume = passes_filters(df)
        if not passes:
            return None

        result = analyze_stock(symbol, df, indicators)

        if result['signals']:
            # Add tier classification
            tier = classify_signal_tier(result)
            result['tier'] = tier

            # Fetch additional info (sector, 52-week, dividend, etc.)
            stock_info = fetch_stock_info(symbol)
            result['info'] = stock_info

            # Fetch news and analyst ratings for Silver+ signals
            is_silver_plus = (
                tier.get('tier_buy') in ('GOLD', 'SILVER') or
                tier.get('tier_sell') in ('GOLD', 'SILVER')
            )
            if is_silver_plus:
                result['news'] = get_stock_news(symbol, max_articles=3)
                result['analyst'] = get_analyst_ratings(symbol)
            else:
                result['news'] = []
                result['analyst'] = None

            return result
        return None
    except Exception:
        return None


def run_scan_async(indicators=None):
    """Run scan in background thread with parallel processing"""
    global scan_state

    scan_state['is_scanning'] = True
    scan_state['stop_requested'] = False
    scan_state['progress'] = 0
    scan_state['error'] = None
    scan_state['results'] = []
    scan_state['sector_breakdown'] = {}
    scan_state['indicators_used'] = indicators or DEFAULT_INDICATORS

    try:
        symbols = get_tsx_symbols()
        scan_state['total'] = len(symbols)

        results = []
        sector_counts = {'buy': {}, 'sell': {}}
        completed = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(process_single_stock, symbol, indicators): symbol
                for symbol in symbols
            }

            # Process results as they complete
            for future in as_completed(future_to_symbol):
                # Check if stop was requested
                if scan_state['stop_requested']:
                    scan_state['error'] = 'Scan stopped by user'
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                completed += 1
                scan_state['progress'] = completed

                result = future.result()
                if result:
                    results.append(result)

                    # Track sector breakdown
                    tier = result.get('tier', {})
                    sector = result.get('info', {}).get('sector') or 'Unknown'
                    if tier.get('tier_buy') or any(s[1] in ('BUY', 'STRONG_BUY') for s in result['signals']):
                        sector_counts['buy'][sector] = sector_counts['buy'].get(sector, 0) + 1
                    if tier.get('tier_sell') or any(s[1] == 'SELL' for s in result['signals']):
                        sector_counts['sell'][sector] = sector_counts['sell'].get(sector, 0) + 1

        scan_state['results'] = results
        scan_state['last_scan'] = datetime.now().isoformat()
        scan_state['sector_breakdown'] = sector_counts

        # Save to history
        categorized = categorize_results(results)
        summary = {
            'gold_buys': len(categorized['gold_buys']),
            'gold_sells': len(categorized['gold_sells']),
            'silver_buys': len(categorized['silver_buys']),
            'silver_sells': len(categorized['silver_sells']),
            'regular_buys': len(categorized['regular_buys']),
            'regular_sells': len(categorized['regular_sells'])
        }
        save_scan_history(results, summary)

    except Exception as e:
        scan_state['error'] = str(e)
    finally:
        scan_state['is_scanning'] = False


def categorize_results(results):
    """Categorize results into tiers"""
    gold_buys = []
    gold_sells = []
    silver_buys = []
    silver_sells = []
    regular_buys = []
    regular_sells = []

    for r in results:
        tier = r.get('tier', {})

        if tier.get('tier_buy') == 'GOLD':
            gold_buys.append(r)
        elif tier.get('tier_sell') == 'GOLD':
            gold_sells.append(r)
        elif tier.get('tier_buy') == 'SILVER':
            silver_buys.append(r)
        elif tier.get('tier_sell') == 'SILVER':
            silver_sells.append(r)
        else:
            # Categorize regular signals
            for indicator, signal, desc in r['signals']:
                if signal in ('BUY', 'STRONG_BUY'):
                    regular_buys.append({**r, 'highlight_signal': (indicator, desc)})
                elif signal == 'SELL':
                    regular_sells.append({**r, 'highlight_signal': (indicator, desc)})

    return {
        'gold_buys': gold_buys,
        'gold_sells': gold_sells,
        'silver_buys': silver_buys,
        'silver_sells': silver_sells,
        'regular_buys': regular_buys,
        'regular_sells': regular_sells
    }


@app.route('/sw.js')
def service_worker():
    """Serve service worker from root"""
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')


@app.route('/')
def index():
    """Main dashboard"""
    categorized = categorize_results(scan_state['results'])
    fear_greed = get_fear_greed_index()
    watchlist = get_watchlist()

    return render_template('index.html',
        scan_state=scan_state,
        categorized=categorized,
        last_scan=scan_state['last_scan'],
        fear_greed=fear_greed,
        watchlist=watchlist
    )


@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start a new scan with optional indicator selection"""
    if scan_state['is_scanning']:
        return jsonify({'status': 'already_running'})

    # Get indicator selection from request
    data = request.get_json() or {}
    indicators = data.get('indicators')

    # Validate indicators if provided
    if indicators:
        valid_indicators = [i for i in indicators if i in ALL_INDICATORS]
        if not valid_indicators:
            return jsonify({'status': 'error', 'message': 'No valid indicators selected'})
        indicators = valid_indicators

    thread = threading.Thread(target=run_scan_async, args=(indicators,))
    thread.start()

    return jsonify({'status': 'started', 'indicators': indicators or DEFAULT_INDICATORS})


@app.route('/api/status')
def get_status():
    """Get current scan status"""
    return jsonify({
        'is_scanning': scan_state['is_scanning'],
        'progress': scan_state['progress'],
        'total': scan_state['total'],
        'last_scan': scan_state['last_scan'],
        'results_count': len(scan_state['results']),
        'error': scan_state['error']
    })


@app.route('/api/scan/stop', methods=['POST'])
def stop_scan():
    """Stop the current scan"""
    if scan_state['is_scanning']:
        scan_state['stop_requested'] = True
        return jsonify({'status': 'stopping'})
    return jsonify({'status': 'not_running'})


@app.route('/api/search')
def search_symbols():
    """Search for matching stock symbols"""
    query = request.args.get('q', '').upper().strip()
    if len(query) < 1:
        return jsonify([])

    symbols = get_tsx_symbols()
    matches = [s for s in symbols if s.startswith(query)][:10]
    return jsonify(matches)


@app.route('/api/indicators')
def get_available_indicators():
    """Get list of available indicators with metadata"""
    indicator_info = {
        'RSI': {'name': 'RSI', 'full_name': 'Relative Strength Index', 'default': True},
        'MACD': {'name': 'MACD', 'full_name': 'MACD Histogram', 'default': True},
        'SlowSto': {'name': 'SlowSto', 'full_name': 'Slow Stochastic', 'default': True},
        'P&F': {'name': 'P&F', 'full_name': 'Point & Figure', 'default': True},
        'BB': {'name': 'BB', 'full_name': 'Bollinger Bands', 'default': False},
        'OBV': {'name': 'OBV', 'full_name': 'On-Balance Volume', 'default': False},
        'ADX': {'name': 'ADX', 'full_name': 'Average Directional Index', 'default': False},
        'WilliamsR': {'name': 'WilliamsR', 'full_name': 'Williams %R', 'default': False}
    }

    return jsonify({
        'all_indicators': ALL_INDICATORS,
        'default_indicators': DEFAULT_INDICATORS,
        'indicator_info': indicator_info
    })


@app.route('/api/results')
def get_results():
    """Get scan results as JSON"""
    categorized = categorize_results(scan_state['results'])

    return jsonify({
        'last_scan': scan_state['last_scan'],
        'summary': {
            'gold_buys': len(categorized['gold_buys']),
            'gold_sells': len(categorized['gold_sells']),
            'silver_buys': len(categorized['silver_buys']),
            'silver_sells': len(categorized['silver_sells']),
            'regular_buys': len(categorized['regular_buys']),
            'regular_sells': len(categorized['regular_sells']),
        },
        'results': categorized
    })


@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """Detailed view for a single stock"""
    symbol = symbol.upper().strip()
    print(f"[DEBUG] Fetching stock: {symbol}")
    df = fetch_stock_data(symbol)
    if df is None:
        print(f"[DEBUG] Failed to fetch data for {symbol}")
        return render_template('error.html', message=f"Could not fetch data for {symbol}. The stock may be delisted or Yahoo Finance may be temporarily unavailable.")

    result = analyze_stock(symbol.upper(), df)
    tier = classify_signal_tier(result)
    result['tier'] = tier

    # Fetch additional info
    stock_info = fetch_stock_info(symbol.upper())
    result['info'] = stock_info

    passes, reason, price, volume = passes_filters(df)

    # Check if in watchlist
    in_watchlist = is_in_watchlist(symbol.upper())

    # Get alerts for this symbol
    alerts = get_alerts_for_symbol(symbol.upper())

    return render_template('stock.html',
        symbol=symbol.upper(),
        result=result,
        price=price,
        volume=volume,
        filter_status=reason,
        in_watchlist=in_watchlist,
        alerts=alerts
    )


# ============ WATCHLIST ROUTES ============

@app.route('/api/watchlist')
def api_get_watchlist():
    """Get watchlist"""
    return jsonify(get_watchlist())


@app.route('/api/watchlist/add', methods=['POST'])
def api_add_watchlist():
    """Add to watchlist"""
    data = request.get_json()
    symbol = data.get('symbol', '')
    success = add_to_watchlist(symbol)
    return jsonify({'success': success})


@app.route('/api/watchlist/remove', methods=['POST'])
def api_remove_watchlist():
    """Remove from watchlist"""
    data = request.get_json()
    symbol = data.get('symbol', '')
    success = remove_from_watchlist(symbol)
    return jsonify({'success': success})


@app.route('/watchlist')
def watchlist_page():
    """Watchlist page with stock details"""
    symbols = get_watchlist()
    stocks = []

    for symbol in symbols:
        df = fetch_stock_data(symbol)
        if df is None:
            continue

        result = analyze_stock(symbol, df)
        tier = classify_signal_tier(result)
        result['tier'] = tier

        stock_info = fetch_stock_info(symbol)
        result['info'] = stock_info

        passes, reason, price, volume = passes_filters(df)
        result['price'] = price
        result['volume'] = volume

        stocks.append(result)

    return render_template('watchlist.html', stocks=stocks)


# ============ ALERTS ROUTES ============

@app.route('/api/alerts')
def api_get_alerts():
    """Get all alerts"""
    return jsonify(get_alerts())


@app.route('/api/alerts/add', methods=['POST'])
def api_add_alert():
    """Add a price alert"""
    data = request.get_json()
    symbol = data.get('symbol', '')
    target_price = float(data.get('target_price', 0))
    direction = data.get('direction', 'above')

    alert = add_alert(symbol, target_price, direction)
    return jsonify(alert)


@app.route('/api/alerts/remove', methods=['POST'])
def api_remove_alert():
    """Remove an alert"""
    data = request.get_json()
    alert_id = float(data.get('alert_id', 0))
    success = remove_alert(alert_id)
    return jsonify({'success': success})


# ============ EXPORT ROUTES ============

@app.route('/api/export/csv')
def export_csv():
    """Export scan results to CSV"""
    categorized = categorize_results(scan_state['results'])

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['Symbol', 'Price', 'Tier', 'Signal Type', 'Signals', 'Sector', '52W High', '52W Low', 'Dividend Yield'])

    # All results
    for category, tier_name in [
        ('gold_buys', 'Gold Buy'),
        ('gold_sells', 'Gold Sell'),
        ('silver_buys', 'Silver Buy'),
        ('silver_sells', 'Silver Sell'),
        ('regular_buys', 'Buy'),
        ('regular_sells', 'Sell')
    ]:
        for stock in categorized.get(category, []):
            info = stock.get('info', {})
            signals = ', '.join([s[0] for s in stock.get('signals', [])])
            writer.writerow([
                f"{stock['symbol']}.TO",
                f"${stock['price']:.2f}",
                tier_name,
                'Buy' if 'buy' in tier_name.lower() else 'Sell',
                signals,
                info.get('sector', 'N/A'),
                f"${info.get('week_52_high', 0):.2f}" if info.get('week_52_high') else 'N/A',
                f"${info.get('week_52_low', 0):.2f}" if info.get('week_52_low') else 'N/A',
                f"{info.get('dividend_yield', 0):.2f}%" if info.get('dividend_yield') else 'N/A'
            ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=tsx_scan_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'}
    )


# ============ HISTORY ROUTES ============

@app.route('/api/history')
def api_get_history():
    """Get scan history"""
    return jsonify(get_scan_history())


@app.route('/history')
def history_page():
    """Scan history page"""
    history = get_scan_history()
    return render_template('history.html', history=history)


# ============ SECTOR BREAKDOWN ============

@app.route('/api/sectors')
def api_get_sectors():
    """Get sector breakdown from last scan"""
    return jsonify(scan_state.get('sector_breakdown', {}))


# ============ FILTERED RESULTS ============

@app.route('/api/results/filtered')
def get_filtered_results():
    """Get filtered scan results"""
    # Get filter parameters
    sector = request.args.get('sector')
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    min_cap = request.args.get('min_cap', type=float)  # in millions
    sort_by = request.args.get('sort', 'tier')  # tier, price, volume, dividend
    signal_type = request.args.get('type')  # buy, sell, all

    results = scan_state['results'].copy()

    # Apply filters
    filtered = []
    for r in results:
        info = r.get('info', {})

        # Sector filter
        if sector and info.get('sector') != sector:
            continue

        # Price filter
        if min_price and r['price'] < min_price:
            continue
        if max_price and r['price'] > max_price:
            continue

        # Market cap filter (in millions)
        if min_cap:
            cap = info.get('market_cap', 0) or 0
            if cap < min_cap * 1_000_000:
                continue

        # Signal type filter
        tier = r.get('tier', {})
        if signal_type == 'buy':
            if not (tier.get('tier_buy') or any(s[1] in ('BUY', 'STRONG_BUY') for s in r['signals'])):
                continue
        elif signal_type == 'sell':
            if not (tier.get('tier_sell') or any(s[1] == 'SELL' for s in r['signals'])):
                continue

        filtered.append(r)

    # Sort results
    if sort_by == 'price':
        filtered.sort(key=lambda x: x['price'], reverse=True)
    elif sort_by == 'volume':
        filtered.sort(key=lambda x: x.get('info', {}).get('volume_ratio', 0) or 0, reverse=True)
    elif sort_by == 'dividend':
        filtered.sort(key=lambda x: x.get('info', {}).get('dividend_yield', 0) or 0, reverse=True)
    elif sort_by == 'week52':
        filtered.sort(key=lambda x: x.get('info', {}).get('week_52_pct', 50) or 50)
    else:  # tier (default)
        def tier_score(r):
            tier = r.get('tier', {})
            if tier.get('tier_buy') == 'GOLD' or tier.get('tier_sell') == 'GOLD':
                return 0
            if tier.get('tier_buy') == 'SILVER' or tier.get('tier_sell') == 'SILVER':
                return 1
            return 2
        filtered.sort(key=tier_score)

    return jsonify({
        'count': len(filtered),
        'results': filtered
    })


# ============ AUTO-SCAN ============

auto_scan_timer = None

def schedule_auto_scan():
    """Schedule next auto scan at market open"""
    global auto_scan_timer

    now = datetime.now()
    # Market opens at 9:30 AM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    if now >= market_open:
        # If past market open, schedule for tomorrow
        market_open += timedelta(days=1)

    # Skip weekends
    while market_open.weekday() >= 5:
        market_open += timedelta(days=1)

    delay = (market_open - now).total_seconds()
    scan_state['next_auto_scan'] = market_open.isoformat()

    auto_scan_timer = threading.Timer(delay, auto_scan_trigger)
    auto_scan_timer.daemon = True
    auto_scan_timer.start()


def auto_scan_trigger():
    """Triggered by auto-scan timer"""
    if scan_state['auto_scan_enabled'] and not scan_state['is_scanning']:
        thread = threading.Thread(target=run_scan_async)
        thread.start()
    # Schedule next scan
    schedule_auto_scan()


@app.route('/api/autoscan/enable', methods=['POST'])
def enable_auto_scan():
    """Enable auto-scan at market open"""
    scan_state['auto_scan_enabled'] = True
    schedule_auto_scan()
    return jsonify({
        'enabled': True,
        'next_scan': scan_state['next_auto_scan']
    })


@app.route('/api/autoscan/disable', methods=['POST'])
def disable_auto_scan():
    """Disable auto-scan"""
    global auto_scan_timer
    scan_state['auto_scan_enabled'] = False
    if auto_scan_timer:
        auto_scan_timer.cancel()
    scan_state['next_auto_scan'] = None
    return jsonify({'enabled': False})


@app.route('/api/autoscan/status')
def auto_scan_status():
    """Get auto-scan status"""
    return jsonify({
        'enabled': scan_state['auto_scan_enabled'],
        'next_scan': scan_state['next_auto_scan']
    })


# ============ BACKTEST ROUTES ============

backtest_state = {
    'is_running': False,
    'stop_requested': False,
    'progress': 0,
    'total': 0,
    'error': None
}


def run_backtest_async(indicators=None):
    """Run backtest in background thread"""
    global backtest_state

    backtest_state['is_running'] = True
    backtest_state['stop_requested'] = False
    backtest_state['progress'] = 0
    backtest_state['error'] = None

    try:
        def progress_callback(current, total):
            backtest_state['progress'] = current
            backtest_state['total'] = total

        def stop_check():
            return backtest_state['stop_requested']

        results = run_market_backtest(progress_callback, indicators=indicators, stop_check=stop_check)

        if backtest_state['stop_requested']:
            backtest_state['error'] = 'Backtest stopped by user'

        # Save to storage
        save_backtest_results(results)

    except Exception as e:
        backtest_state['error'] = str(e)
    finally:
        backtest_state['is_running'] = False


@app.route('/backtest')
def backtest_page():
    """Backtest results page"""
    results = get_backtest_results()
    return render_template('backtest.html',
                           results=results,
                           is_running=backtest_state['is_running'])


@app.route('/api/backtest/run', methods=['POST'])
def start_backtest():
    """Start backtest analysis"""
    if backtest_state['is_running']:
        return jsonify({'status': 'already_running'})

    # Get selected indicators from request
    data = request.get_json() or {}
    indicators = data.get('indicators')

    # Validate indicators if provided
    if indicators:
        valid_indicators = ['RSI', 'MACD', 'SlowSto', 'P&F', 'BB', 'OBV', 'ADX', 'WilliamsR']
        indicators = [i for i in indicators if i in valid_indicators]
        if not indicators:
            indicators = None  # Fall back to all indicators

    thread = threading.Thread(target=run_backtest_async, args=(indicators,))
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/backtest/status')
def backtest_status():
    """Get backtest progress"""
    return jsonify({
        'is_running': backtest_state['is_running'],
        'progress': backtest_state['progress'],
        'total': backtest_state['total'],
        'error': backtest_state['error']
    })


@app.route('/api/backtest/stop', methods=['POST'])
def stop_backtest():
    """Stop the current backtest"""
    if backtest_state['is_running']:
        backtest_state['stop_requested'] = True
        return jsonify({'status': 'stopping'})
    return jsonify({'status': 'not_running'})


@app.route('/api/backtest/results')
def get_backtest_results_api():
    """Get backtest results as JSON"""
    results = get_backtest_results()
    if not results:
        return jsonify({'error': 'No backtest results available'})

    return jsonify(results)


@app.route('/api/test-fetch/<symbol>')
def test_fetch(symbol):
    """Debug endpoint to test yfinance"""
    import yfinance as yf

    result = {'step': 'start', 'symbol': symbol.upper()}

    try:
        result['step'] = 'creating ticker'
        tsx_symbol = f"{symbol.upper()}.TO"
        ticker = yf.Ticker(tsx_symbol)

        result['step'] = 'fetching history'
        df = ticker.history(period="5d")

        result['step'] = 'processing'
        if df is None or df.empty:
            result['status'] = 'empty'
            result['message'] = 'No data returned'
        else:
            result['status'] = 'success'
            result['rows'] = len(df)
            result['price'] = round(float(df['Close'].iloc[-1]), 2)

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['error_type'] = type(e).__name__

    return jsonify(result)


if __name__ == '__main__':
    # Run on all interfaces so it's accessible from phone
    # Using 8080 because 5000 is often used by AirPlay on macOS
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
