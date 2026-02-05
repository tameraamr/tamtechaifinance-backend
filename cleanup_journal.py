import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

print("Cleaning up journal table...\n")

with engine.connect() as conn:
    # 1. Check current count
    result = conn.execute(text("SELECT COUNT(*) FROM trading_journal WHERE user_id = 8")).fetchone()
    print(f"Current trades: {result[0]}")
    
    # 2. Delete duplicates - keep only the latest 76 trades
    print("\nDeleting duplicate trades (keeping latest 76)...")
    conn.execute(text("""
        DELETE FROM trading_journal 
        WHERE user_id = 8 
        AND id NOT IN (
            SELECT id FROM trading_journal 
            WHERE user_id = 8 
            ORDER BY id DESC 
            LIMIT 76
        )
    """))
    conn.commit()
    
    # 3. Check new count
    result = conn.execute(text("SELECT COUNT(*) FROM trading_journal WHERE user_id = 8")).fetchone()
    print(f"✓ Cleaned! Remaining trades: {result[0]}")
    
    # 4. Drop unused columns
    print("\nDropping unused columns...")
    
    # Columns to drop (not used in your Excel or frontend)
    unused_columns = [
        'lot_type',
        'pips_gained', 
        'risk_amount_usd',
        'risk_percentage',
        'account_size_at_entry',
        'ai_trade_score',
        'ai_review',
        'created_at',
        'updated_at'
    ]
    
    for col in unused_columns:
        try:
            conn.execute(text(f"ALTER TABLE trading_journal DROP COLUMN IF EXISTS {col}"))
            print(f"  ✓ Dropped: {col}")
        except Exception as e:
            print(f"  ✗ Failed to drop {col}: {e}")
    
    conn.commit()

print("\n✅ Cleanup complete!")
print("\nRemaining columns: pair_ticker, asset_type, order_type, lot_size, entry_price, exit_price,")
print("stop_loss, take_profit, entry_time, exit_time, trading_session, market_trend, strategy,")
print("profit_loss_pips, profit_loss_usd, risk_reward_ratio, status, result, notes")
