"""
🔍 Production Health Monitor
Tests all critical endpoints to ensure everything is working
"""

import requests
import time
from datetime import datetime

BASE_URL = "https://tamtechaifinance-backend-production.up.railway.app"

print("=" * 70)
print("🔍 TamtechAI Production Health Check")
print("=" * 70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target: {BASE_URL}")
print("=" * 70)

tests = []

# Test 1: Health Check
print("\n📋 Test 1: Health Check Endpoint")
try:
    start = time.time()
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    duration = time.time() - start
    
    if r.status_code == 200:
        data = r.json()
        print(f"✅ PASSED ({duration:.2f}s)")
        print(f"   Status: {data.get('status')}")
        print(f"   Gemini API: {data.get('gemini_api')}")
        print(f"   Model: {data.get('model')}")
        print(f"   API Key Present: {data.get('api_key_present')}")
        
        if data.get('gemini_api') == 'configured':
            tests.append(("Health Check", True, duration))
        else:
            tests.append(("Health Check", False, duration))
            print("   ⚠️  WARNING: Gemini API not configured!")
    else:
        print(f"❌ FAILED - Status {r.status_code}")
        tests.append(("Health Check", False, duration))
except Exception as e:
    print(f"❌ FAILED - {e}")
    tests.append(("Health Check", False, 0))

# Test 2: Root Endpoint
print("\n📋 Test 2: Root Endpoint")
try:
    start = time.time()
    r = requests.get(f"{BASE_URL}/", timeout=10)
    duration = time.time() - start
    
    if r.status_code == 200:
        print(f"✅ PASSED ({duration:.2f}s)")
        print(f"   Response: {r.json()}")
        tests.append(("Root", True, duration))
    else:
        print(f"❌ FAILED - Status {r.status_code}")
        tests.append(("Root", False, duration))
except Exception as e:
    print(f"❌ FAILED - {e}")
    tests.append(("Root", False, 0))

# Test 3: Random Ticker Endpoint
print("\n📋 Test 3: Random Ticker Generator")
try:
    start = time.time()
    r = requests.get(f"{BASE_URL}/get-random-ticker-v2", timeout=10)
    duration = time.time() - start
    
    if r.status_code == 200:
        data = r.json()
        print(f"✅ PASSED ({duration:.2f}s)")
        print(f"   Random Ticker: {data.get('ticker')}")
        tests.append(("Random Ticker", True, duration))
    else:
        print(f"❌ FAILED - Status {r.status_code}")
        tests.append(("Random Ticker", False, duration))
except Exception as e:
    print(f"❌ FAILED - {e}")
    tests.append(("Random Ticker", False, 0))

# Test 4: Market Sentiment
print("\n📋 Test 4: Market Sentiment")
try:
    start = time.time()
    r = requests.get(f"{BASE_URL}/market-sentiment", timeout=10)
    duration = time.time() - start
    
    if r.status_code == 200:
        data = r.json()
        print(f"✅ PASSED ({duration:.2f}s)")
        print(f"   Sentiment: {data.get('sentiment')}")
        print(f"   Description: {data.get('description')}")
        tests.append(("Market Sentiment", True, duration))
    else:
        print(f"❌ FAILED - Status {r.status_code}")
        tests.append(("Market Sentiment", False, duration))
except Exception as e:
    print(f"❌ FAILED - {e}")
    tests.append(("Market Sentiment", False, 0))

# Test 5: Analysis Endpoint (Expected to fail for guests after limit)
print("\n📋 Test 5: Stock Analysis Endpoint")
try:
    start = time.time()
    r = requests.get(f"{BASE_URL}/analyze/AAPL", timeout=30)
    duration = time.time() - start
    
    if r.status_code == 200:
        print(f"✅ PASSED - Got analysis ({duration:.2f}s)")
        tests.append(("Stock Analysis", True, duration))
    elif r.status_code == 403:
        print(f"⚠️  Expected - Guest limit reached ({duration:.2f}s)")
        print(f"   Message: {r.json().get('detail')}")
        tests.append(("Stock Analysis", True, duration))  # Expected behavior
    elif r.status_code == 500:
        error = r.json().get('detail', 'Unknown error')
        print(f"❌ FAILED - Server Error ({duration:.2f}s)")
        print(f"   Error: {error}")
        if "refund" in error.lower():
            print(f"   ✅ Refund system working!")
        tests.append(("Stock Analysis", False, duration))
    else:
        print(f"❌ FAILED - Status {r.status_code} ({duration:.2f}s)")
        tests.append(("Stock Analysis", False, duration))
except Exception as e:
    print(f"❌ FAILED - {e}")
    tests.append(("Stock Analysis", False, 0))

# Summary
print("\n" + "=" * 70)
print("📊 TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, status, _ in tests if status)
failed = len(tests) - passed
avg_time = sum(t for _, _, t in tests) / len(tests) if tests else 0

for name, status, duration in tests:
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {name:25} - {duration:.2f}s")

print("=" * 70)
print(f"Total: {len(tests)} tests | Passed: {passed} | Failed: {failed}")
print(f"Average Response Time: {avg_time:.2f}s")

if failed == 0:
    print("\n🎉 ALL TESTS PASSED - System is healthy!")
    exit(0)
else:
    print(f"\n⚠️  {failed} test(s) failed - Check logs above")
    exit(1)
