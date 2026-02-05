import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://tamtechaifinance-backend-production.up.railway.app"

# Test login
print("Testing login...")
login_data = {
    "email": "tameraamr@gmail.com",
    "password": input("Enter your password for tameraamr@gmail.com: ")
}

response = requests.post(f"{API_BASE}/login", json=login_data)
print(f"Login status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    token = data.get('token')
    print(f"✓ Login successful! Token: {token[:20]}...")
    
    # Test fetching trades
    print("\nFetching trades...")
    headers = {"Authorization": f"Bearer {token}"}
    trades_response = requests.get(f"{API_BASE}/journal/trades?limit=5", headers=headers)
    
    print(f"Trades fetch status: {trades_response.status_code}")
    
    if trades_response.status_code == 200:
        trades = trades_response.json()
        print(f"✓ Found {len(trades)} trades (showing first 5)")
        for trade in trades[:5]:
            print(f"  - {trade['pair_ticker']} {trade['order_type']} | P&L: ${trade.get('profit_loss_usd', 0)}")
    else:
        print(f"✗ Error: {trades_response.text}")
else:
    print(f"✗ Login failed: {response.text}")
