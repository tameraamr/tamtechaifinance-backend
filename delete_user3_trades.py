"""
Delete trades from wrong user (ID 3) and prepare for correct import to user 8
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
    # Check current state
    result = db.execute(text("""
        SELECT user_id, COUNT(*) 
        FROM trading_journal 
        GROUP BY user_id
        ORDER BY user_id
    """))
    
    print("📊 CURRENT TRADES BY USER:")
    for row in result:
        print(f"  User ID {row[0]}: {row[1]} trades")
    
    # Delete from user 3
    print(f"\n🗑️  Deleting trades from User ID 3...")
    result = db.execute(text("DELETE FROM trading_journal WHERE user_id = 3"))
    deleted = result.rowcount
    db.commit()
    print(f"✅ Deleted {deleted} trades from User 3")
    
    # Verify
    result = db.execute(text("""
        SELECT user_id, COUNT(*) 
        FROM trading_journal 
        GROUP BY user_id
        ORDER BY user_id
    """))
    
    print(f"\n📊 AFTER DELETION:")
    for row in result:
        print(f"  User ID {row[0]}: {row[1]} trades")
    
finally:
    db.close()

print("\n✅ Ready for correct import to User ID 8!")
