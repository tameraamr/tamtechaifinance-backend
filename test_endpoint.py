import requests

try:
    response = requests.get('http://localhost:8000/market-winners-losers', timeout=10)
    data = response.json()
    print('✅ Endpoint responded successfully')
    if 'winners' in data and data['winners']:
        winner = data['winners'][0]
        print(f'🏆 Top winner: {winner["ticker"]} at ${winner["price"]:.2f} ({winner["change_percent"]:+.2f}%)')
    if 'losers' in data and data['losers']:
        loser = data['losers'][0]
        print(f'📉 Top loser: {loser["ticker"]} at ${loser["price"]:.2f} ({loser["change_percent"]:+.2f}%)')
    print(f'📊 Total winners: {len(data.get("winners", []))}, Total losers: {len(data.get("losers", []))}')
except Exception as e:
    print(f'❌ Error: {e}')