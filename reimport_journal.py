"""
CORRECT Journal Import Script
Deletes old data and reimports from Excel with proper column mapping
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL not found!")
    exit(1)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
db = Session()

# Read Excel file
excel_path = r"c:\Users\tamer\Desktop\Tamer Journal.xlsx"
print(f"📖 Reading Excel file: {excel_path}")
df = pd.read_excel(excel_path)

print(f"✅ Found {len(df)} trades in Excel")
print(f"\nColumns: {df.columns.tolist()}\n")

# Clean and prepare data
df['Time Exec'] = pd.to_datetime(df['Time Exec'])

try:
    # Step 1: Get user ID (assuming first user or specific user)
    result = db.execute(text("SELECT id FROM users ORDER BY id LIMIT 1"))
    user_id = result.scalar()
    
    if not user_id:
        print("❌ No user found in database!")
        exit(1)
    
    print(f"👤 Using User ID: {user_id}")
    
    # Step 2: DELETE all existing trades for this user
    print(f"\n🗑️  Deleting existing trades...")
    result = db.execute(text(f"DELETE FROM trading_journal WHERE user_id = {user_id}"))
    deleted = result.rowcount
    db.commit()
    print(f"✅ Deleted {deleted} old trades")
    
    # Step 3: Import trades with CORRECT mapping
    print(f"\n📥 Importing {len(df)} trades...")
    
    imported = 0
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            # Determine asset type
            pair = str(row['Pair']).upper()
            if 'XAU' in pair or 'GOLD' in pair:
                asset_type = 'gold'
            elif 'US30' in pair or 'NAS' in pair or 'SPX' in pair or 'DOW' in pair:
                asset_type = 'indices'
            else:
                asset_type = 'forex'
            
            # Entry price is in "Stop" column, Exit price is in "Exit" column
            # Handle "NONE" text values
            def safe_float(val):
                if pd.isna(val) or val == '' or str(val).upper() == 'NONE':
                    return None
                try:
                    return float(val)
                except:
                    return None
            
            entry_price = safe_float(row['Stop'])
            exit_price = safe_float(row['Exit'])
            
            # Determine status and result
            status = 'closed' if pd.notna(exit_price) else 'open'
            
            result_str = None
            if pd.notna(row['Status']):
                status_val = str(row['Status']).strip().lower()
                if status_val == 'win':
                    result_str = 'win'
                elif status_val == 'loss':
                    result_str = 'loss'
                elif status_val == 'be' or status_val == 'breakeven':
                    result_str = 'breakeven'
            
            # Get other values using safe_float
            lot_size = safe_float(row['Lot Size']) or 0.01
            pips = safe_float(row['Pips/Points'])
            pl_usd = safe_float(row['P/L $'])
            rr = safe_float(row['R:R'])
            
            # Take profit and stop loss
            tp = safe_float(row['TP1'])
            sl = None  # Stop loss not clearly defined in Excel
            
            session = str(row['Session']) if pd.notna(row['Session']) else None
            bias = str(row['Bias']) if pd.notna(row['Bias']) else None
            
            # Insert into database
            insert_query = text("""
                INSERT INTO trading_journal (
                    user_id, pair_ticker, asset_type, order_type, lot_size,
                    entry_price, exit_price, entry_time, exit_time,
                    profit_loss_pips, profit_loss_usd, risk_reward_ratio,
                    status, result, trading_session, market_trend,
                    take_profit, stop_loss
                ) VALUES (
                    :user_id, :pair, :asset_type, :order_type, :lot_size,
                    :entry_price, :exit_price, :entry_time, :exit_time,
                    :pips, :pl_usd, :rr,
                    :status, :result, :session, :bias,
                    :tp, :sl
                )
            """)
            
            db.execute(insert_query, {
                'user_id': user_id,
                'pair': pair,
                'asset_type': asset_type,
                'order_type': str(row['Buy/Sell']),
                'lot_size': lot_size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': row['Time Exec'],
                'exit_time': None,  # Not explicitly in Excel
                'pips': pips,
                'pl_usd': pl_usd,
                'rr': rr,
                'status': status,
                'result': result_str,
                'session': session,
                'bias': bias,
                'tp': tp,
                'sl': sl
            })
            
            # Commit each trade individually
            db.commit()
            imported += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  ✅ Imported {idx + 1}/{len(df)} trades...")
                
        except Exception as e:
            errors += 1
            db.rollback()  # Rollback this trade only
            print(f"  ❌ Error on row {idx + 1}: {str(e)[:100]}")
            if errors <= 5:  # Only show details for first 5 errors
                print(f"     Pair: {row.get('Pair')}, Date: {row.get('Time Exec')}, Stop: {row.get('Stop')}, Exit: {row.get('Exit')}")
    
    print(f"\n{'='*80}")
    print(f"✅ IMPORT COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successfully imported: {imported} trades")
    if errors > 0:
        print(f"❌ Errors: {errors} trades")
    
    # Verify
    result = db.execute(text(f"SELECT COUNT(*) FROM trading_journal WHERE user_id = {user_id}"))
    final_count = result.scalar()
    print(f"\n📊 Total trades in database: {final_count}")
    
    # Show first 5 trades
    result = db.execute(text("""
        SELECT pair_ticker, order_type, entry_time, entry_price, exit_price, 
               profit_loss_pips, profit_loss_usd, result
        FROM trading_journal
        ORDER BY entry_time ASC
        LIMIT 5
    """))
    
    print(f"\n📋 First 5 trades:")
    print("="*100)
    for row in result:
        pair = str(row[0])
        order_type = str(row[1])
        exit_str = f"{row[4]:.2f}" if row[4] else "N/A"
        result_str = row[7] if row[7] else "N/A"
        print(f"{pair:8s} | {order_type:4s} | {row[2]} | Entry: {row[3]:.5f} | Exit: {exit_str:>10s} | Pips: {row[5] if row[5] else 0:6.1f} | P/L: ${row[6] if row[6] else 0:7.2f} | {result_str}")
    
except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    db.rollback()
    raise
finally:
    db.close()

print(f"\n{'='*80}")
print("🎉 ALL DONE! Your journal is now correctly imported!")
print(f"{'='*80}")
