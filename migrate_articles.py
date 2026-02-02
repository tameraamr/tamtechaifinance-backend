"""
Migration script to import existing static articles into the database.

Run this once after deploying the new Article model to populate the database
with your existing articles.

Usage:
    python backend/migrate_articles.py
"""

import requests
import os
from datetime import datetime

# Backend URL (use your Railway/local URL)
BACKEND_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")

# Your admin token (login first to get this from localStorage)
# For testing locally: get it from browser console after login
ADMIN_TOKEN = "YOUR_ADMIN_TOKEN_HERE"

# Existing articles to migrate
ARTICLES = [
    {
        "title": "MicroStrategy's Bitcoin Strategy: Why MSTR Could Hit $500 in 2026",
        "slug": "microstrategy-bitcoin-strategy-2026",
        "description": "Deep dive into MicroStrategy's aggressive Bitcoin accumulation strategy and why it could make MSTR one of the best-performing stocks of 2026.",
        "author": "TamtechAI Research",
        "hero_emoji": "₿",
        "hero_gradient": "orange,amber,yellow",
        "related_tickers": ["MSTR", "BTC-USD", "COIN", "HOOD"],
        "is_featured": 0,  # Don't auto-feature old articles
        "published": 1,
        "content": """
<h2>🏦 The Bold Bitcoin Play</h2>
<p>MicroStrategy (MSTR) has transformed from a traditional business intelligence company into the world's largest corporate Bitcoin holder, with over 190,000 BTC worth approximately $8 billion.</p>

<h3>The Strategy</h3>
<p>CEO Michael Saylor has adopted an aggressive "Bitcoin Standard" strategy:</p>
<ul>
  <li><strong>Leveraged BTC Purchases:</strong> Using convertible debt to buy more Bitcoin</li>
  <li><strong>Diamond Hands:</strong> Never selling, only accumulating</li>
  <li><strong>Corporate Treasury:</strong> Bitcoin as primary reserve asset</li>
</ul>

<h2>📊 Why This Could Work</h2>
<p>If Bitcoin reaches $150K by 2026 (conservative projection), MSTR's holdings would be worth $28.5 billion - more than 3x current value.</p>

<h3>Stock Price Leverage</h3>
<p>MSTR stock historically moves 2-3x Bitcoin's percentage gains due to:</p>
<ul>
  <li>Leveraged exposure to BTC</li>
  <li>No tax on unrealized gains</li>
  <li>Ability to issue more stock at premium to buy more BTC</li>
</ul>

<h2>⚠️ The Risks</h2>
<ul>
  <li><strong>Bitcoin volatility:</strong> If BTC crashes to $30K, MSTR could drop 70%+</li>
  <li><strong>Debt obligations:</strong> $2.4B in convertible notes maturing 2027-2028</li>
  <li><strong>Regulatory risk:</strong> Potential Bitcoin regulations could impact strategy</li>
</ul>

<h2>🎯 Our Verdict</h2>
<p><strong>High-Risk, High-Reward Buy</strong> for investors bullish on Bitcoin long-term. Not suitable for conservative portfolios.</p>

<p><strong>Price Target:</strong> $450-$500 by end of 2026 (assuming BTC reaches $150K)</p>
"""
    },
    {
        "title": "Is Gold in a Bubble? Why $3,000/oz is Just the Beginning",
        "slug": "is-gold-a-bubble-2026",
        "description": "Historical analysis shows gold isn't in a bubble - it's actually undervalued. Here's why $3,000/oz could be the floor, not the ceiling.",
        "author": "TamtechAI Research",
        "hero_emoji": "🏆",
        "hero_gradient": "yellow,amber,orange",
        "related_tickers": ["GLD", "GOLD", "NEM", "AEM"],
        "is_featured": 0,
        "published": 1,
        "content": """
<h2>💰 The $3,000 Question</h2>
<p>With gold recently hitting all-time highs above $2,800/oz, many investors are asking: is this a bubble?</p>

<p>Our AI analysis of 50+ years of gold price data suggests: <strong>Absolutely not.</strong></p>

<h2>📊 Historical Context</h2>

<h3>Inflation-Adjusted Peak</h3>
<p>Gold's 1980 peak of $850/oz equals approximately <strong>$3,400 in 2026 dollars</strong>.</p>
<p>Current price: $2,800 → Still 21% below inflation-adjusted ATH</p>

<h3>Gold-to-Money Supply Ratio</h3>
<p>With M2 money supply at $21 trillion (up 40% since 2020), gold should be priced at:</p>
<ul>
  <li><strong>Conservative:</strong> $3,200/oz</li>
  <li><strong>Historical average:</strong> $3,800/oz</li>
  <li><strong>Crisis scenario:</strong> $5,000+/oz</li>
</ul>

<h2>🚀 Why Gold Could Hit $5,000</h2>

<h3>1. Central Bank Buying</h3>
<p>Global central banks (China, Russia, India) bought a record 1,136 tonnes in 2024, continuing into 2026.</p>

<h3>2. De-Dollarization</h3>
<p>BRICS nations moving away from USD → increased gold reserves</p>

<h3>3. Debt Crisis</h3>
<p>US national debt at $36 trillion → gold as hedge against currency devaluation</p>

<h3>4. Geopolitical Uncertainty</h3>
<p>Multiple ongoing conflicts → safe-haven demand</p>

<h2>📈 Best Ways to Invest</h2>
<ul>
  <li><strong>GLD:</strong> Physical gold ETF (0.40% expense ratio)</li>
  <li><strong>Gold miners:</strong> NEM, GOLD (2-3x leverage to gold price)</li>
  <li><strong>Physical gold:</strong> Coins/bars (but watch premiums)</li>
</ul>

<h2>🎯 Our Verdict</h2>
<p><strong>Buy</strong> - Gold is not in a bubble. Target: $3,500 by end 2026, $5,000 by 2028.</p>
"""
    },
    {
        "title": "NVIDIA's AI Dominance: Why $200 is Possible by 2027",
        "slug": "nvidia-ai-dominance-2026",
        "description": "NVIDIA controls 92% of the AI chip market. With Blackwell chips launching and demand exploding, here's why the stock could double again.",
        "author": "TamtechAI Research",
        "hero_emoji": "⚡",
        "hero_gradient": "green,emerald,teal",
        "related_tickers": ["NVDA", "AMD", "INTC", "ARM"],
        "is_featured": 0,
        "published": 1,
        "content": """
<h2>🤖 The AI Chip King</h2>
<p>NVIDIA (NVDA) has become the single most important company in the AI revolution, controlling over 92% of the market for AI training chips.</p>

<h2>📊 The Numbers Are Insane</h2>

<h3>Revenue Growth</h3>
<ul>
  <li><strong>2023:</strong> $27 billion</li>
  <li><strong>2024:</strong> $79 billion</li>
  <li><strong>2025 (projected):</strong> $125 billion</li>
  <li><strong>2026 (our estimate):</strong> $180 billion</li>
</ul>

<h3>Gross Margins</h3>
<p>NVIDIA's H100 and upcoming Blackwell chips command <strong>70-80% gross margins</strong> - unheard of in the chip industry.</p>

<h2>🚀 Growth Catalysts for 2026-2027</h2>

<h3>1. Blackwell Architecture Launch</h3>
<p>New B100/B200 chips offer:</p>
<ul>
  <li>5x performance improvement over H100</li>
  <li>2x energy efficiency</li>
  <li>Already sold out through 2026</li>
</ul>

<h3>2. Software Moat (CUDA)</h3>
<p>NVIDIA's CUDA platform is the industry standard - AMD and Intel are 5+ years behind in software ecosystem.</p>

<h3>3. Sovereign AI Demand</h3>
<p>Countries building national AI infrastructure:</p>
<ul>
  <li>UAE: $50B AI investment</li>
  <li>Saudi Arabia: $40B</li>
  <li>Japan: $30B</li>
  <li>EU: $100B+ over 5 years</li>
</ul>

<h3>4. Automotive AI</h3>
<p>NVIDIA DRIVE platform powering autonomous vehicles - $14B pipeline through 2028.</p>

<h2>⚠️ Risks to Consider</h2>
<ul>
  <li><strong>Competition:</strong> AMD's MI300X gaining traction</li>
  <li><strong>China export restrictions:</strong> Lost ~$5B annual revenue</li>
  <li><strong>Valuation:</strong> Trading at 30x forward earnings (but justified by growth)</li>
  <li><strong>Customer concentration:</strong> Meta, Microsoft, Google = 40% of revenue</li>
</ul>

<h2>💹 Valuation Analysis</h2>
<p>At $180B revenue in 2026 with 50% net margins:</p>
<ul>
  <li><strong>Net income:</strong> $90 billion</li>
  <li><strong>25x PE multiple:</strong> $2.25 trillion market cap</li>
  <li><strong>Share price:</strong> ~$200 (split-adjusted)</li>
</ul>

<h2>🎯 Our Verdict</h2>
<p><strong>Strong Buy</strong> - NVIDIA remains the best way to invest in the AI revolution. Target: $180 by end 2026, $250+ by 2028.</p>

<p><em>Confidence Score: 89%</em></p>
"""
    }
]

def migrate_articles():
    """Migrate existing articles to database."""
    
    if ADMIN_TOKEN == "YOUR_ADMIN_TOKEN_HERE":
        print("❌ ERROR: Please set your admin token first!")
        print("\n📝 Steps to get your token:")
        print("1. Login to your account on the website")
        print("2. Open browser console (F12)")
        print("3. Run: localStorage.getItem('token')")
        print("4. Copy the token and paste it in this script")
        return
    
    headers = {
        "Authorization": f"Bearer {ADMIN_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print(f"🚀 Migrating {len(ARTICLES)} articles to database...\n")
    
    for i, article in enumerate(ARTICLES, 1):
        print(f"[{i}/{len(ARTICLES)}] Creating: {article['title']}")
        
        # Convert related_tickers list to JSON string
        payload = {
            **article,
            "related_tickers": str(article["related_tickers"])  # Convert list to string representation
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/admin/articles",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Created successfully (ID: {data['article']['id']})")
            else:
                print(f"  ❌ Error: {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    print("\n🎉 Migration complete!")
    print(f"\n📊 Check your articles at: {BACKEND_URL.replace(':8000', '')}/admin/articles")

if __name__ == "__main__":
    migrate_articles()
