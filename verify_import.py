"""
Verify the import and check for duplicates
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
db = Session()

try:
    # Total count
    result = db.execute(text("SELECT COUNT(*) FROM trading_journal"))
    print(f"✅ TOTAL TRADES IN DATABASE: {result.scalar()}")
    
    # Date range
    result = db.execute(text("""
        SELECT 
            MIN(entry_time) as first,
            MAX(entry_time) as last
        FROM trading_journal
    """))
    row = result.fetchone()
    print(f"\n📅 DATE RANGE: {row[0]} to {row[1]}")
    
    # Check for potential duplicates (same time, pair, type)
    result = db.execute(text("""
        SELECT 
            entry_time,
            pair_ticker,
            order_type,
            COUNT(*) as count,
            STRING_AGG(CAST(entry_price AS TEXT), ', ') as entry_prices
        FROM trading_journal
        GROUP BY entry_time, pair_ticker, order_type
        HAVING COUNT(*) > 1
        ORDER BY entry_time
    """))
    
    duplicates = result.fetchall()
    if duplicates:
        print(f"\n⚠️  POTENTIAL DUPLICATES FOUND: {len(duplicates)}")
        print("="*100)
        for dup in duplicates[:10]:  # Show first 10
            print(f"{dup[0]} | {dup[1]:8s} | {dup[2]:4s} | Count: {dup[3]} | Entry prices: {dup[4]}")
    else:
        print(f"\n✅ NO DUPLICATES FOUND!")
    
    # Stats
    result = db.execute(text("""
        SELECT 
            result,
            COUNT(*) as count,
            SUM(profit_loss_usd) as total_pl
        FROM trading_journal
        WHERE result IS NOT NULL
        GROUP BY result
        ORDER BY result
    """))
    
    print(f"\n📊 RESULTS BREAKDOWN:")
    print("="*50)
    total_pl = 0
    for row in result:
        print(f"{row[0]:10s} | Count: {row[1]:3d} | Total P/L: ${row[2]:>10.2f}")
        total_pl += row[2] if row[2] else 0
    print("="*50)
    print(f"{'NET P/L':10s} | {'':8s} | Total P/L: ${total_pl:>10.2f}")
    
finally:
    db.close()
