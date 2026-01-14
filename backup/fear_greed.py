"""
Fear and Greed Index fetcher
Pulls data from CNN's Fear and Greed Index API
"""

import requests
from datetime import datetime

CACHE = {
    'data': None,
    'timestamp': None
}

CACHE_DURATION = 300  # 5 minutes


def get_fear_greed_index() -> dict:
    """
    Fetch the current Fear and Greed Index from CNN

    Returns dict with:
        - score: 0-100 value
        - rating: 'extreme fear', 'fear', 'neutral', 'greed', 'extreme greed'
        - previous_close: yesterday's score
        - previous_week: last week's score
        - previous_month: last month's score
        - previous_year: last year's score
    """
    global CACHE

    # Check cache
    if CACHE['data'] and CACHE['timestamp']:
        age = (datetime.now() - CACHE['timestamp']).seconds
        if age < CACHE_DURATION:
            return CACHE['data']

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()
        fg = data.get('fear_and_greed', {})

        result = {
            'score': round(fg.get('score', 50)),
            'rating': fg.get('rating', 'neutral'),
            'previous_close': round(fg.get('previous_close', 50)),
            'previous_week': round(fg.get('previous_1_week', 50)),
            'previous_month': round(fg.get('previous_1_month', 50)),
            'previous_year': round(fg.get('previous_1_year', 50)),
            'timestamp': fg.get('timestamp', '')
        }

        # Cache the result
        CACHE['data'] = result
        CACHE['timestamp'] = datetime.now()

        return result

    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return None


def get_rating_label(score: int) -> str:
    """Convert score to human-readable label"""
    if score <= 25:
        return "Extreme Fear"
    elif score <= 45:
        return "Fear"
    elif score <= 55:
        return "Neutral"
    elif score <= 75:
        return "Greed"
    else:
        return "Extreme Greed"


if __name__ == "__main__":
    data = get_fear_greed_index()
    if data:
        print(f"Fear & Greed Index: {data['score']} ({data['rating'].title()})")
        print(f"Previous Close: {data['previous_close']}")
        print(f"Previous Week: {data['previous_week']}")
        print(f"Previous Month: {data['previous_month']}")
        print(f"Previous Year: {data['previous_year']}")
    else:
        print("Failed to fetch data")
