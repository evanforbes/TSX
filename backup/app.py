#!/usr/bin/env python3
"""
TSX Stock Scanner - Web App
Mobile-friendly dashboard for viewing scan results
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from datetime import datetime
import threading
import json
import os

from scanner import get_tsx_symbols, fetch_stock_data, passes_filters, analyze_stock, classify_signal_tier
from news import get_stock_news, get_analyst_ratings
from fear_greed import get_fear_greed_index

app = Flask(__name__)

# Global state for scan results
scan_state = {
    'last_scan': None,
    'results': [],
    'is_scanning': False,
    'progress': 0,
    'total': 0,
    'error': None
}


def run_scan_async():
    """Run scan in background thread"""
    global scan_state

    scan_state['is_scanning'] = True
    scan_state['progress'] = 0
    scan_state['error'] = None
    scan_state['results'] = []

    try:
        symbols = get_tsx_symbols()
        scan_state['total'] = len(symbols)

        results = []

        for i, symbol in enumerate(symbols, 1):
            scan_state['progress'] = i

            df = fetch_stock_data(symbol)
            if df is None:
                continue

            passes, reason, price, volume = passes_filters(df)
            if not passes:
                continue

            result = analyze_stock(symbol, df)

            if result['signals']:
                # Add tier classification
                tier = classify_signal_tier(result)
                result['tier'] = tier

                # Fetch news and analyst ratings for all Silver+ signals (buys and sells)
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

                results.append(result)

        scan_state['results'] = results
        scan_state['last_scan'] = datetime.now().isoformat()

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

    return render_template('index.html',
        scan_state=scan_state,
        categorized=categorized,
        last_scan=scan_state['last_scan'],
        fear_greed=fear_greed
    )


@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start a new scan"""
    if scan_state['is_scanning']:
        return jsonify({'status': 'already_running'})

    thread = threading.Thread(target=run_scan_async)
    thread.start()

    return jsonify({'status': 'started'})


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


@app.route('/api/search')
def search_symbols():
    """Search for matching stock symbols"""
    query = request.args.get('q', '').upper().strip()
    if len(query) < 1:
        return jsonify([])

    symbols = get_tsx_symbols()
    matches = [s for s in symbols if s.startswith(query)][:10]
    return jsonify(matches)


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
    df = fetch_stock_data(symbol.upper())
    if df is None:
        return render_template('error.html', message=f"Could not fetch data for {symbol}")

    result = analyze_stock(symbol.upper(), df)
    tier = classify_signal_tier(result)
    result['tier'] = tier

    passes, reason, price, volume = passes_filters(df)

    return render_template('stock.html',
        symbol=symbol.upper(),
        result=result,
        price=price,
        volume=volume,
        filter_status=reason
    )


if __name__ == '__main__':
    # Run on all interfaces so it's accessible from phone
    # Using 8080 because 5000 is often used by AirPlay on macOS
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
