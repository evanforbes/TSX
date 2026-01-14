"""
News fetcher for TSX stocks
Pulls latest articles from Yahoo Finance
"""

import yfinance as yf
from datetime import datetime


def get_stock_news(symbol: str, max_articles: int = 3) -> list:
    """
    Fetch latest news articles for a TSX stock

    Args:
        symbol: Stock symbol without .TO suffix
        max_articles: Maximum number of articles to return

    Returns:
        List of dicts with title, link, publisher, published
    """
    tsx_symbol = f"{symbol}.TO"

    try:
        ticker = yf.Ticker(tsx_symbol)
        news = ticker.news

        if not news:
            return []

        articles = []
        for item in news[:max_articles]:
            # Handle new yfinance structure (nested under 'content')
            content = item.get('content', item)

            # Get the URL - try multiple locations
            url = '#'
            if 'clickThroughUrl' in content and content['clickThroughUrl']:
                url = content['clickThroughUrl'].get('url', '#')
            elif 'canonicalUrl' in content and content['canonicalUrl']:
                url = content['canonicalUrl'].get('url', '#')

            # Get publisher
            publisher = 'Yahoo Finance'
            if 'provider' in content and content['provider']:
                publisher = content['provider'].get('displayName', 'Yahoo Finance')

            # Get publish date
            pub_date = content.get('pubDate', '') or content.get('displayTime', '')

            article = {
                'title': content.get('title', 'No title'),
                'link': url,
                'publisher': publisher,
                'published': format_date(pub_date),
            }
            articles.append(article)

        return articles

    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []


def format_date(date_str: str) -> str:
    """Convert ISO date string to readable format"""
    if not date_str:
        return ''
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo)
        diff = now - dt

        if diff.days == 0:
            hours = diff.seconds // 3600
            if hours == 0:
                mins = diff.seconds // 60
                return f"{mins}m ago"
            return f"{hours}h ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        else:
            return dt.strftime("%b %d")
    except:
        return ''


def get_analyst_ratings(symbol: str) -> dict:
    """
    Fetch analyst ratings for a TSX stock

    Returns dict with:
        - strong_buy, buy, hold, sell, strong_sell counts
        - total: total number of analysts
        - rating: overall rating string
        - score: numeric score (1-5, 5 being strong buy)
    """
    tsx_symbol = f"{symbol}.TO"

    try:
        ticker = yf.Ticker(tsx_symbol)
        recs = ticker.recommendations_summary

        if recs is None or recs.empty:
            return None

        # Get current month data (first row)
        current = recs.iloc[0]

        strong_buy = int(current.get('strongBuy', 0))
        buy = int(current.get('buy', 0))
        hold = int(current.get('hold', 0))
        sell = int(current.get('sell', 0))
        strong_sell = int(current.get('strongSell', 0))

        total = strong_buy + buy + hold + sell + strong_sell

        if total == 0:
            return None

        # Calculate weighted score (5 = strong buy, 1 = strong sell)
        score = (strong_buy * 5 + buy * 4 + hold * 3 + sell * 2 + strong_sell * 1) / total

        # Determine overall rating
        if score >= 4.5:
            rating = "Strong Buy"
        elif score >= 3.5:
            rating = "Buy"
        elif score >= 2.5:
            rating = "Hold"
        elif score >= 1.5:
            rating = "Sell"
        else:
            rating = "Strong Sell"

        return {
            'strong_buy': strong_buy,
            'buy': buy,
            'hold': hold,
            'sell': sell,
            'strong_sell': strong_sell,
            'total': total,
            'rating': rating,
            'score': round(score, 2)
        }

    except Exception as e:
        print(f"Error fetching analyst ratings for {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test news
    print("=== News ===")
    articles = get_stock_news("RY")
    for a in articles:
        print(f"{a['title']}")
        print(f"  {a['publisher']} - {a['published']}")
        print(f"  {a['link']}\n")

    # Test analyst ratings
    print("=== Analyst Ratings ===")
    ratings = get_analyst_ratings("RY")
    if ratings:
        print(f"Rating: {ratings['rating']} ({ratings['score']}/5)")
        print(f"Strong Buy: {ratings['strong_buy']}, Buy: {ratings['buy']}, Hold: {ratings['hold']}, Sell: {ratings['sell']}, Strong Sell: {ratings['strong_sell']}")
