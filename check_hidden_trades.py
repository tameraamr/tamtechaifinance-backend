"""
Check which trades are being hidden due to the limit=50
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found!")
    exit(1)

# Fix URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
db = Session()

try:
    # Get trades 48-52 to see the cutoff
    result = db.execute(text("""
        SELECT 
            ROW_NUMBER() OVER (ORDER BY entry_time DESC) as row_num,
            id,
            pair_ticker,
            order_type,
            entry_time,
            profit_loss_usd,
            result
        FROM trading_journal
        ORDER BY entry_time DESC
        LIMIT 10 OFFSET 45
    """))
    
    print("🔍 TRADES AROUND THE 50-TRADE LIMIT:")
    print("="*100)
    print("Row | ID  | Pair     | Type | Entry Time          | P/L      | Result")
    print("="*100)
    for row in result:
        pl_str = f"${row[5]:7.2f}" if row[5] else "$   0.00"
        result_str = row[6] if row[6] else "open"
        print(f"{row[0]:3d} | {row[1]:3d} | {row[2]:8s} | {row[3]:4s} | {row[4]} | {pl_str} | {result_str}")
    
    print("\n" + "="*100)
    print("⚠️  Trades from row 51 onwards are NOT shown on the website (limit=50)")
    print("="*100)
    
    # Count hidden trades
    result = db.execute(text("""
        SELECT COUNT(*) 
        FROM (
            SELECT ROW_NUMBER() OVER (ORDER BY entry_time DESC) as row_num
            FROM trading_journal
        ) subq
        WHERE row_num > 50
    """))
    hidden_count = result.scalar()
    print(f"\n📊 HIDDEN TRADES: {hidden_count}")
    
    # Get the oldest visible trade and oldest hidden trade
    result = db.execute(text("""
        WITH ranked AS (
            SELECT 
                entry_time,
                pair_ticker,
                ROW_NUMBER() OVER (ORDER BY entry_time DESC) as row_num
            FROM trading_journal
        )
        SELECT 
            (SELECT entry_time FROM ranked WHERE row_num = 50) as last_visible,
            (SELECT entry_time FROM ranked WHERE row_num = 51) as first_hidden,
            (SELECT entry_time FROM ranked WHERE row_num = 1) as newest,
            (SELECT COUNT(*) FROM trading_journal) as total
    """))
    row = result.fetchone()
    print(f"\n📅 DATE BREAKDOWN:")
    print(f"  Newest trade (row 1):      {row[2]}")
    print(f"  Last VISIBLE trade (row 50): {row[0]}")
    print(f"  First HIDDEN trade (row 51): {row[1]}")
    print(f"  Total trades: {row[3]}")
    print(f"\n⚠️  You're missing {hidden_count} trades from {row[1]} backwards!")
        
finally:
    db.close()
