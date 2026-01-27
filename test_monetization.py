"""
🧪 Testing Script for Strict Monetization System
Run this script to verify the backend implementation
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"  # Change to your backend URL
TEST_TICKER = "AAPL"

def test_credit_deduction():
    """Test that credits are deducted on every request"""
    print("\n" + "="*50)
    print("TEST 1: Credit Deduction on Every Request")
    print("="*50)
    
    # You'll need to replace this with a valid JWT token from a logged-in user
    token = "YOUR_JWT_TOKEN_HERE"
    headers = {"Authorization": f"Bearer {token}"}
    
    # First request
    print(f"\n📊 Requesting analysis for {TEST_TICKER} (Request #1)")
    response1 = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}", headers=headers)
    data1 = response1.json()
    
    credits_after_first = data1.get("credits_left")
    cache_hit_first = data1.get("cache_hit")
    cache_age_first = data1.get("cache_age_hours", 0)
    
    print(f"✅ Credits remaining: {credits_after_first}")
    print(f"✅ Cache hit: {cache_hit_first}")
    print(f"✅ Cache age: {cache_age_first} hours")
    
    # Second request (should still deduct credit even if cached)
    time.sleep(2)
    print(f"\n📊 Requesting analysis for {TEST_TICKER} (Request #2)")
    response2 = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}", headers=headers)
    data2 = response2.json()
    
    credits_after_second = data2.get("credits_left")
    cache_hit_second = data2.get("cache_hit")
    cache_age_second = data2.get("cache_age_hours", 0)
    
    print(f"✅ Credits remaining: {credits_after_second}")
    print(f"✅ Cache hit: {cache_hit_second}")
    print(f"✅ Cache age: {cache_age_second} hours")
    
    # Verify credit deduction
    if credits_after_first is not None and credits_after_second is not None:
        credit_diff = credits_after_first - credits_after_second
        if credit_diff == 1:
            print("\n✅ PASS: 1 credit was deducted for cached request")
        else:
            print(f"\n❌ FAIL: Expected 1 credit deducted, got {credit_diff}")
    else:
        print("\n⚠️ SKIP: Cannot verify (not logged in)")

def test_cache_functionality():
    """Test that cache is working (AI not called twice)"""
    print("\n" + "="*50)
    print("TEST 2: Cache Functionality")
    print("="*50)
    
    token = "YOUR_JWT_TOKEN_HERE"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📊 First request for {TEST_TICKER}")
    start_time1 = time.time()
    response1 = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}", headers=headers)
    duration1 = time.time() - start_time1
    data1 = response1.json()
    
    print(f"⏱️ Duration: {duration1:.2f}s")
    print(f"✅ Cache hit: {data1.get('cache_hit')}")
    
    time.sleep(1)
    
    print(f"\n📊 Second request for {TEST_TICKER}")
    start_time2 = time.time()
    response2 = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}", headers=headers)
    duration2 = time.time() - start_time2
    data2 = response2.json()
    
    print(f"⏱️ Duration: {duration2:.2f}s")
    print(f"✅ Cache hit: {data2.get('cache_hit')}")
    
    if data2.get('cache_hit'):
        print(f"\n✅ PASS: Second request was {duration1/duration2:.1f}x faster (cache working)")
    else:
        print("\n⚠️ WARNING: Second request was not cached")

def test_live_price():
    """Test that price is always live"""
    print("\n" + "="*50)
    print("TEST 3: Live Price Injection")
    print("="*50)
    
    token = "YOUR_JWT_TOKEN_HERE"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📊 Requesting {TEST_TICKER} twice with 5s gap")
    
    response1 = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}", headers=headers)
    data1 = response1.json()
    price1 = data1.get("data", {}).get("price")
    
    print(f"💰 Price #1: ${price1}")
    
    time.sleep(5)
    
    response2 = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}", headers=headers)
    data2 = response2.json()
    price2 = data2.get("data", {}).get("price")
    
    print(f"💰 Price #2: ${price2}")
    
    if price1 and price2:
        if price1 == price2:
            print("\n⚠️ INFO: Prices are identical (market may be unchanged)")
        else:
            print(f"\n✅ PASS: Price updated (${price1} → ${price2})")
    else:
        print("\n❌ FAIL: Could not retrieve price")

def test_guest_limiting():
    """Test guest IP-based limiting"""
    print("\n" + "="*50)
    print("TEST 4: Guest Trial Limiting")
    print("="*50)
    
    print(f"\n📊 Making 4 requests as guest (should fail on 4th)")
    
    for i in range(1, 5):
        response = requests.get(f"{BASE_URL}/analyze/{TEST_TICKER}")
        print(f"\nRequest #{i}: Status {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Success")
        elif response.status_code == 403:
            print(f"🚫 Blocked (expected after 3 attempts)")
        else:
            print(f"❌ Unexpected status: {response.status_code}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 STRICT MONETIZATION SYSTEM - TEST SUITE")
    print("="*60)
    print("\n⚠️ IMPORTANT: Replace 'YOUR_JWT_TOKEN_HERE' with a real token")
    print("⚠️ IMPORTANT: Ensure backend is running on localhost:8000")
    
    # Uncomment tests as needed
    # test_credit_deduction()
    # test_cache_functionality()
    # test_live_price()
    # test_guest_limiting()
    
    print("\n" + "="*60)
    print("✅ Tests complete! Check results above.")
    print("="*60)
