"""
Check what trades are currently in the database
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
    # Check total trades
    result = db.execute(text("SELECT COUNT(*) FROM trading_journal"))
    total = result.scalar()
    print(f"📊 TOTAL TRADES IN DATABASE: {total}")
    
    # Check date range
    result = db.execute(text("""
        SELECT 
            MIN(entry_time) as first_trade,
            MAX(entry_time) as last_trade
        FROM trading_journal
    """))
    row = result.fetchone()
    print(f"\n📅 DATE RANGE:")
    print(f"  First trade: {row[0]}")
    print(f"  Last trade: {row[1]}")
    
    # Check first 10 trades
    result = db.execute(text("""
        SELECT 
            id,
            pair_ticker,
            order_type,
            entry_time,
            entry_price,
            exit_price,
            profit_loss_pips,
            profit_loss_usd,
            status,
            result
        FROM trading_journal
        ORDER BY entry_time ASC
        LIMIT 10
    """))
    
    print(f"\n📋 FIRST 10 TRADES IN DATABASE:")
    print("="*100)
    for row in result:
        exit_str = f"{row[5]:.2f}" if row[5] else "OPEN"
        result_str = row[9] if row[9] else row[8]
        print(f"{row[0]:3d} | {str(row[1]):8s} | {str(row[2]):4s} | {row[3]} | Entry: {row[4]:.5f} | Exit: {exit_str:>10s} | Pips: {row[6] if row[6] else 0:6.1f} | P/L: ${row[7] if row[7] else 0:8.2f} | {result_str}")
    
    # Check last 10 trades
    result = db.execute(text("""
        SELECT 
            id,
            pair_ticker,
            order_type,
            entry_time,
            entry_price,
            exit_price,
            profit_loss_pips,
            profit_loss_usd,
            status,
            result
        FROM trading_journal
        ORDER BY entry_time DESC
        LIMIT 10
    """))
    
    print(f"\n📋 LAST 10 TRADES IN DATABASE:")
    print("="*100)
    for row in result:
        exit_str = f"{row[5]:.2f}" if row[5] else "OPEN"
        result_str = row[9] if row[9] else row[8]
        print(f"{row[0]:3d} | {str(row[1]):8s} | {str(row[2]):4s} | {row[3]} | Entry: {row[4]:.5f} | Exit: {exit_str:>10s} | Pips: {row[6] if row[6] else 0:6.1f} | P/L: ${row[7] if row[7] else 0:8.2f} | {result_str}")
    
    # Check win/loss stats
    result = db.execute(text("""
        SELECT 
            result,
            COUNT(*) as count,
            SUM(profit_loss_usd) as total_pl
        FROM trading_journal
        WHERE result IS NOT NULL
        GROUP BY result
    """))
    
    print(f"\n📊 TRADE RESULTS:")
    print("="*50)
    for row in result:
        print(f"{row[0]:10s} | Count: {row[1]:3d} | Total P/L: ${row[2] if row[2] else 0:>8.2f}")
        
finally:
    db.close()
