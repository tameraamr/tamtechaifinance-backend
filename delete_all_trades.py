"""
Delete ALL trades and reimport clean
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
    # Check user IDs
    result = db.execute(text("""
        SELECT user_id, COUNT(*) 
        FROM trading_journal 
        GROUP BY user_id
    """))
    
    print("👥 TRADES BY USER:")
    for row in result:
        print(f"  User ID {row[0]}: {row[1]} trades")
    
    # Delete ALL trades
    print(f"\n🗑️  Deleting ALL trades from trading_journal...")
    result = db.execute(text("DELETE FROM trading_journal"))
    deleted = result.rowcount
    db.commit()
    print(f"✅ Deleted {deleted} trades")
    
    # Verify
    result = db.execute(text("SELECT COUNT(*) FROM trading_journal"))
    remaining = result.scalar()
    print(f"📊 Remaining trades: {remaining}")
    
finally:
    db.close()

print("\n✅ Ready for clean import!")
