import sys
import os
sys.path.append('.')

# Test the cache function directly
try:
    from main import get_market_data_with_cache
    print("✅ Cache function imported successfully")

    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    print(f"🧪 Testing cache with tickers: {test_tickers}")

    cached_data = get_market_data_with_cache(test_tickers)
    print(f"📊 Cache returned data for {len(cached_data)} tickers")

    for ticker, data in cached_data.items():
        if data and data.get('price', 0) > 0:
            print(f"✅ {ticker}: ${data['price']:.2f} ({data.get('company_name', 'Unknown')})")
        else:
            print(f"❌ {ticker}: Invalid data - {data}")

except Exception as e:
    print(f"❌ Error testing cache: {e}")
    import traceback
    traceback.print_exc()