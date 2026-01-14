# TSX Stock Scanner

A Python-based scanner for Toronto Stock Exchange (TSX) stocks that detects technical trading signals.

## Signals Detected

| Indicator | BUY Signal | SELL Signal |
|-----------|------------|-------------|
| **RSI** | Below 20 crosses up, or near 10 (strong buy) | Above 80 crosses down |
| **MACD** | Histogram turns positive | Histogram turns negative |
| **Slow Stochastic** | Below 20 (oversold) and rising | Above 80 (overbought) and falling |
| **Point & Figure** | Triple top breakout | - |

## Installation

```bash
cd tsx-stock-scanner
pip install -r requirements.txt
```

## Usage

### Scan All Stocks
```bash
python scanner.py
```

### Analyze Single Stock
```bash
python scanner.py RY      # Royal Bank
python scanner.py TD      # TD Bank
python scanner.py SHOP    # Shopify
```

## Configuration

Edit `config.py` to customize:

- **CUSTOM_SYMBOLS**: Add your own stock list
- **MIN_PRICE**: Minimum price filter (default: $5)
- **MIN_VOLUME**: Minimum volume filter (default: 500,000)
- **RSI settings**: Overbought/oversold levels
- **MACD settings**: EMA periods
- **Stochastic settings**: %K/%D periods
- **P&F settings**: Box size and reversal

## Adding Custom Stocks

Edit `config.py`:

```python
CUSTOM_SYMBOLS = [
    "RY", "TD", "ENB", "CNR",  # Add your symbols here
]
```

## Output Example

```
================================================================================
TSX STOCK SCANNER
Run Date: 2024-01-15 09:30:00
Filters: Price > $5.00, Volume > 500,000
================================================================================

Scanning 70 TSX stocks...

[1/70] Scanning RY.TO... No signals
[2/70] Scanning TD.TO... SIGNALS: BUY
...

================================================================================
BUY SIGNALS
================================================================================

TD.TO - $85.42
  [RSI] RSI crossed above 20 (was 18.5, now 22.3)

================================================================================
SUMMARY
================================================================================
Strong Buys: 0
Buy Signals: 3
Sell Signals: 2
Total Stocks Scanned: 65
```

## Notes

- Data is fetched from Yahoo Finance (free, no API key needed)
- TSX symbols use the `.TO` suffix automatically
- Default stock list includes ~70 major TSX stocks
- Lookback period is 1 year for Point & Figure analysis
