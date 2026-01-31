import asyncio
import httpx
import time

async def test_concurrent_analyzes():
    # 10 tickers NOT in cache (assuming these aren't cached)
    tickers = ["NVDA", "TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "NFLX", "AMD", "INTC"]

    async def analyze_ticker(ticker):
        start_time = time.time()
        print(f"🚀 Starting request for {ticker} at {start_time}")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"http://localhost:8000/analyze/{ticker}")
                end_time = time.time()
                print(f"✅ {ticker} completed in {end_time - start_time:.2f}s - Status: {response.status_code}")
                return response.status_code
        except Exception as e:
            end_time = time.time()
            print(f"❌ {ticker} failed in {end_time - start_time:.2f}s - Error: {e}")
            return None

    # Send all 10 requests simultaneously
    print("🎯 Sending 10 simultaneous analyze requests...")
    start_all = time.time()

    tasks = [analyze_ticker(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    end_all = time.time()
    print(f"🏁 All requests completed in {end_all - start_all:.2f}s")
    print(f"Results: {results}")

    return results

if __name__ == "__main__":
    asyncio.run(test_concurrent_analyzes())