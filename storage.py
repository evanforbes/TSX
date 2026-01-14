"""
Storage module for watchlist, alerts, and scan history
Uses JSON files for persistence
"""

import json
import os
from datetime import datetime
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WATCHLIST_FILE = os.path.join(DATA_DIR, 'watchlist.json')
ALERTS_FILE = os.path.join(DATA_DIR, 'alerts.json')
HISTORY_FILE = os.path.join(DATA_DIR, 'scan_history.json')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(filepath: str, default: any = None) -> any:
    """Load JSON file or return default"""
    if default is None:
        default = []
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(filepath: str, data: any):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ============ WATCHLIST ============

def get_watchlist() -> list:
    """Get all watchlist symbols"""
    return _load_json(WATCHLIST_FILE, [])


def add_to_watchlist(symbol: str) -> bool:
    """Add symbol to watchlist"""
    watchlist = get_watchlist()
    symbol = symbol.upper().replace('.TO', '')
    if symbol not in watchlist:
        watchlist.append(symbol)
        _save_json(WATCHLIST_FILE, watchlist)
        return True
    return False


def remove_from_watchlist(symbol: str) -> bool:
    """Remove symbol from watchlist"""
    watchlist = get_watchlist()
    symbol = symbol.upper().replace('.TO', '')
    if symbol in watchlist:
        watchlist.remove(symbol)
        _save_json(WATCHLIST_FILE, watchlist)
        return True
    return False


def is_in_watchlist(symbol: str) -> bool:
    """Check if symbol is in watchlist"""
    watchlist = get_watchlist()
    return symbol.upper().replace('.TO', '') in watchlist


# ============ PRICE ALERTS ============

def get_alerts() -> list:
    """Get all price alerts"""
    return _load_json(ALERTS_FILE, [])


def add_alert(symbol: str, target_price: float, direction: str = 'above') -> dict:
    """
    Add a price alert
    direction: 'above' or 'below'
    """
    alerts = get_alerts()
    symbol = symbol.upper().replace('.TO', '')

    alert = {
        'id': datetime.now().timestamp(),
        'symbol': symbol,
        'target_price': target_price,
        'direction': direction,  # 'above' or 'below'
        'created': datetime.now().isoformat(),
        'triggered': False,
        'triggered_at': None
    }

    alerts.append(alert)
    _save_json(ALERTS_FILE, alerts)
    return alert


def remove_alert(alert_id: float) -> bool:
    """Remove an alert by ID"""
    alerts = get_alerts()
    alerts = [a for a in alerts if a['id'] != alert_id]
    _save_json(ALERTS_FILE, alerts)
    return True


def check_alerts(symbol: str, current_price: float) -> list:
    """Check if any alerts are triggered for a symbol"""
    alerts = get_alerts()
    triggered = []

    symbol = symbol.upper().replace('.TO', '')

    for alert in alerts:
        if alert['symbol'] != symbol or alert['triggered']:
            continue

        if alert['direction'] == 'above' and current_price >= alert['target_price']:
            alert['triggered'] = True
            alert['triggered_at'] = datetime.now().isoformat()
            triggered.append(alert)
        elif alert['direction'] == 'below' and current_price <= alert['target_price']:
            alert['triggered'] = True
            alert['triggered_at'] = datetime.now().isoformat()
            triggered.append(alert)

    if triggered:
        _save_json(ALERTS_FILE, alerts)

    return triggered


def get_alerts_for_symbol(symbol: str) -> list:
    """Get all alerts for a specific symbol"""
    alerts = get_alerts()
    symbol = symbol.upper().replace('.TO', '')
    return [a for a in alerts if a['symbol'] == symbol]


# ============ SCAN HISTORY ============

MAX_HISTORY = 20  # Keep last 20 scans

def save_scan_history(results: list, summary: dict):
    """Save scan results to history"""
    history = _load_json(HISTORY_FILE, [])

    entry = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'gold_buys': [r['symbol'] for r in results if r.get('tier', {}).get('tier_buy') == 'GOLD'],
        'gold_sells': [r['symbol'] for r in results if r.get('tier', {}).get('tier_sell') == 'GOLD'],
        'silver_buys': [r['symbol'] for r in results if r.get('tier', {}).get('tier_buy') == 'SILVER'],
        'silver_sells': [r['symbol'] for r in results if r.get('tier', {}).get('tier_sell') == 'SILVER'],
        'total_signals': len(results)
    }

    history.insert(0, entry)
    history = history[:MAX_HISTORY]  # Keep only last N
    _save_json(HISTORY_FILE, history)


def get_scan_history() -> list:
    """Get scan history"""
    return _load_json(HISTORY_FILE, [])


def clear_scan_history():
    """Clear all scan history"""
    _save_json(HISTORY_FILE, [])


# ============ BACKTEST RESULTS ============

BACKTEST_FILE = os.path.join(DATA_DIR, 'backtest_results.json')


def save_backtest_results(results: dict):
    """Save backtest results to file"""
    _save_json(BACKTEST_FILE, results)


def get_backtest_results() -> Optional[dict]:
    """Load backtest results from file"""
    result = _load_json(BACKTEST_FILE, None)
    # _load_json returns [] when file doesn't exist, convert to None
    if result == [] or result is None:
        return None
    return result
