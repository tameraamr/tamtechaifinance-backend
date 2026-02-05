import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

print("Checking trades in database...\n")

with engine.connect() as conn:
    # Check user
    user = conn.execute(text("SELECT id, email FROM users WHERE id = 8")).fetchone()
    if user:
        print(f"✓ User found: ID={user[0]}, Email={user[1]}")
    else:
        print("✗ User ID 8 not found!")
        
    # Check trades count
    result = conn.execute(text("SELECT COUNT(*) FROM trading_journal WHERE user_id = 8")).fetchone()
    print(f"✓ Total trades for user 8: {result[0]}")
    
    # Show sample trades
    trades = conn.execute(text("""
        SELECT id, pair_ticker, order_type, entry_price, profit_loss_usd, status, result 
        FROM trading_journal 
        WHERE user_id = 8 
        ORDER BY entry_time DESC 
        LIMIT 5
    """)).fetchall()
    
    print(f"\nSample trades:")
    for t in trades:
        print(f"  ID {t[0]}: {t[1]} {t[2]} @ {t[3]} | P&L: ${t[4]} | {t[5]} ({t[6]})")
