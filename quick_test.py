import requests
import time

API_BASE = "https://tamtechaifinance-backend-production.up.railway.app"

print("Waiting for Railway deployment...")
time.sleep(10)

# Test login
print("\nTesting backend after latest deployment...")
login_data = {
    "username": "tameraamr@gmail.com",
    "password": "12345678"
}

response = requests.post(f"{API_BASE}/token", data=login_data)
print(f"Login: {response.status_code}")

if response.status_code == 200:
    token = response.json().get('access_token')
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test trades
    trades_response = requests.get(f"{API_BASE}/journal/trades?limit=3", headers=headers)
    print(f"Trades: {trades_response.status_code}")
    if trades_response.status_code == 200:
        print(f"✓ {len(trades_response.json())} trades loaded")
    else:
        print(f"✗ Error: {trades_response.text[:200]}")
    
    # Test stats
    stats_response = requests.get(f"{API_BASE}/journal/stats", headers=headers)
    print(f"Stats: {stats_response.status_code}")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"✓ Total trades: {stats.get('total_trades')}")
    else:
        print(f"✗ Error: {stats_response.text[:200]}")
else:
    print(f"✗ Login failed: {response.text}")
