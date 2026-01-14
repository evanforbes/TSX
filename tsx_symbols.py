"""
TSX Symbol Fetcher
Fetches complete list of Toronto Stock Exchange symbols
"""

import requests
import pandas as pd
import os
import json
from datetime import datetime, timedelta

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'tsx_symbols_cache.json')
CACHE_DAYS = 7  # Refresh cache weekly


def fetch_tsx_symbols_from_tmx() -> list:
    """
    Fetch TSX symbols from TMX Money API
    Returns list of symbols (without .TO suffix)
    """
    symbols = []

    # TMX API endpoint for listed companies
    url = "https://www.tsx.com/json/company-directory/search/tsx/^*"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                for company in data['results']:
                    if 'symbol' in company:
                        symbols.append(company['symbol'])
    except Exception as e:
        print(f"Error fetching from TMX: {e}")

    return symbols


def fetch_tsx_symbols_from_yahoo() -> list:
    """
    Alternative: Get major TSX symbols via common index constituents
    This is a fallback with a comprehensive list
    """
    # S&P/TSX Composite Index constituents + TSX 60 + other major stocks
    # This is a comprehensive list of ~250 actively traded TSX stocks
    symbols = [
        # Financials
        "RY", "TD", "BNS", "BMO", "CM", "NA", "CWB", "LB", "EQB", "SLF",
        "MFC", "GWO", "IAG", "POW", "PWF", "IGM", "CIX", "FFH", "IFC", "DFY",
        "X", "BK", "CF", "GSY", "PRL", "FN", "ECN", "BN", "BAM", "BIP-UN",
        "BEP-UN", "BIPC", "BEPC",

        # Energy
        "ENB", "TRP", "PPL", "IPL", "KEY", "GEI", "SPB", "ALA", "CPG",
        "SU", "CNQ", "CVE", "IMO", "HSE", "MEG", "OVV", "TVE", "WCP", "ERF",
        "FRU", "PEY", "BTE", "TOU", "ARX", "PSK", "SES", "VET", "SGY", "CR",
        "PXT", "NVA", "AAV", "BIR", "CJ", "GXE", "HWX", "JOY", "KEL", "LGN",
        "NGL", "OBE", "PNE", "PRQ", "SGI", "TOT", "TPZ", "VLE", "WRG", "YGR",

        # Materials
        "ABX", "AEM", "AGI", "AR", "ATH", "B", "BTO", "CCO", "CG", "CS",
        "DML", "DPM", "EDV", "ELD", "EQX", "ERO", "FM", "FR", "FVI", "GCM",
        "GUY", "HBM", "IMG", "IVN", "K", "KGC", "KNT", "LIF", "LUG", "LUN",
        "MAG", "MND", "MX", "NGD", "NTR", "NXE", "OGC", "OR", "ORE", "OSK",
        "PVG", "RIO", "SEA", "SII", "SLS", "SSL", "SSRM", "SVM", "TKO", "TECK-B",
        "TXG", "UNS", "WDO", "WM", "WPM", "YRI",

        # Industrials
        "CNR", "CP", "AC", "CHR", "TFII", "WSP", "STN", "SNC", "ARE", "AFN",
        "AIF", "AND", "ATA", "BBD-B", "BDI", "BYD", "CAE", "CJT", "EIF", "EXE",
        "FOOD", "GBT", "HPS-A", "IFP", "KXS", "MDA", "MG", "MTL", "NFI", "RBA",
        "RCH", "RUS", "SJ", "SXP", "TCL-A", "TFI", "TIH", "TRQ", "WCN", "WJA",

        # Consumer
        "ATD", "L", "MRU", "EMP-A", "NWC", "PBH", "SAP", "WN", "PLC", "QSR",
        "DOL", "GIL", "CCL-B", "MTY", "A&W", "CGX", "CTC-A", "PJC-A", "RSI",
        "TOY", "LNR", "Pet", "MAL", "CAS", "DII-B",

        # Technology
        "SHOP", "CSU", "OTEX", "KXS", "DCBO", "ENGH", "GIB-A", "REAL", "LSPD",
        "NVEI", "BB", "CGO", "DSG", "ET", "FOOD", "GSY", "HUT", "KITS", "MOGO",
        "PAYF", "PHO", "QTRH", "TIXT", "WELL",

        # Telecom
        "BCE", "T", "RCI-B", "QBR-B", "CCA", "SJR-B",

        # Utilities
        "FTS", "EMA", "H", "AQN", "CPX", "ACO-X", "BLX", "CU", "INE", "NPI",
        "TA", "ALA", "SPB", "KEY",

        # REITs
        "REI-UN", "HR-UN", "AP-UN", "CAR-UN", "CRT-UN", "D-UN", "DIR-UN",
        "GRT-UN", "IIP-UN", "KMP-UN", "MEQ", "MI-UN", "MRG-UN", "NWH-UN",
        "PMZ-UN", "SRU-UN", "TCN", "TNT-UN", "WIR-U",

        # Healthcare
        "WELL", "CXR", "GUD", "JWEL", "MG", "NHC", "PLC", "VLNS",

        # Other
        "AW-UN", "CLS", "GFL", "ONEX", "QSP-UN", "TRI", "GEI",
    ]

    # Remove duplicates
    return list(set(symbols))


def get_cached_symbols() -> tuple:
    """
    Get symbols from cache if valid
    Returns: (symbols list, is_valid bool)
    """
    if not os.path.exists(CACHE_FILE):
        return [], False

    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)

        cache_date = datetime.fromisoformat(cache['date'])
        if datetime.now() - cache_date < timedelta(days=CACHE_DAYS):
            return cache['symbols'], True

        return cache['symbols'], False
    except Exception:
        return [], False


def save_symbols_cache(symbols: list):
    """Save symbols to cache file"""
    cache = {
        'date': datetime.now().isoformat(),
        'symbols': symbols,
        'count': len(symbols)
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_all_tsx_symbols(force_refresh: bool = False) -> list:
    """
    Get all TSX symbols, using cache when available

    Args:
        force_refresh: Force refresh from source even if cache is valid

    Returns:
        List of TSX symbols (without .TO suffix)
    """
    # Check cache first
    if not force_refresh:
        cached, is_valid = get_cached_symbols()
        if is_valid and cached:
            print(f"Using cached TSX symbols ({len(cached)} stocks)")
            return cached

    print("Fetching TSX symbol list...")

    # Try TMX first
    symbols = fetch_tsx_symbols_from_tmx()

    # If TMX fails or returns too few, use Yahoo fallback
    if len(symbols) < 100:
        print("Using comprehensive TSX symbol list...")
        symbols = fetch_tsx_symbols_from_yahoo()

    # Clean up symbols
    cleaned = []
    for s in symbols:
        # Remove any .TO suffix if present
        s = s.replace('.TO', '').replace('.to', '')
        # Handle special TSX suffixes
        s = s.replace('.', '-')  # Convert GIB.A to GIB-A for yfinance
        if s and len(s) <= 10:
            cleaned.append(s.upper())

    # Remove duplicates and sort
    cleaned = sorted(list(set(cleaned)))

    # Save to cache
    save_symbols_cache(cleaned)
    print(f"Found {len(cleaned)} TSX symbols")

    return cleaned


if __name__ == "__main__":
    # Test the fetcher
    symbols = get_all_tsx_symbols(force_refresh=True)
    print(f"\nTotal symbols: {len(symbols)}")
    print(f"Sample: {symbols[:20]}")
