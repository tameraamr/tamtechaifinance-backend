"""
CORRECT Journal Import for User ID 8
Entry = Column J, Exit = Column O, Stop Loss = Column K, Take Profit = Column L
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

print(f"✅ Found {len(df)} trades in Excel\n")

# Clean and prepare data
df['Time Exec'] = pd.to_datetime(df['Time Exec'])

USER_ID = 8  # CORRECT USER ID

try:
    # DELETE all existing trades for User 8
    print(f"🗑️  Deleting existing trades for User ID {USER_ID}...")
    result = db.execute(text(f"DELETE FROM trading_journal WHERE user_id = {USER_ID}"))
    deleted = result.rowcount
    db.commit()
    print(f"✅ Deleted {deleted} old trades\n")
    
    print(f"📥 Importing {len(df)} trades to User ID {USER_ID}...\n")
    
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
            
            # CORRECT MAPPING:
            # Entry price = Column J = "Entry"
            # Exit price = Column O = "Exit"
            # Stop Loss = Column K = "Stop"
            # Take Profit = Column L = "TP1"
            
            def safe_float(val):
                if pd.isna(val) or val == '' or str(val).upper() == 'NONE':
                    return None
                try:
                    return float(val)
                except:
                    return None
            
            entry_price = safe_float(row['Entry'])  # Column J
            exit_price = safe_float(row['Exit'])    # Column O
            stop_loss = safe_float(row['Stop'])     # Column K
            take_profit = safe_float(row['TP1'])    # Column L
            
            # Get other values
            lot_size = safe_float(row['Lot Size']) or 0.01
            pips = safe_float(row['Pips/Points'])
            pl_usd = safe_float(row['P/L $'])
            rr = safe_float(row['R:R'])
            
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
            
            session = str(row['Session']) if pd.notna(row['Session']) else None
            bias = str(row['Bias']) if pd.notna(row['Bias']) else None
            poi = str(row['POI']) if pd.notna(row['POI']) else None  # Column H - notes
            
            # Strategy is always CRT
            strategy = 'CRT'
            
            # Insert into database
            insert_query = text("""
                INSERT INTO trading_journal (
                    user_id, pair_ticker, asset_type, order_type, lot_size,
                    entry_price, exit_price, entry_time, exit_time,
                    profit_loss_pips, profit_loss_usd, risk_reward_ratio,
                    status, result, trading_session, market_trend,
                    take_profit, stop_loss, strategy, notes
                ) VALUES (
                    :user_id, :pair, :asset_type, :order_type, :lot_size,
                    :entry_price, :exit_price, :entry_time, :exit_time,
                    :pips, :pl_usd, :rr,
                    :status, :result, :session, :bias,
                    :tp, :sl, :strategy, :notes
                )
            """)
            
            db.execute(insert_query, {
                'user_id': USER_ID,
                'pair': pair,
                'asset_type': asset_type,
                'order_type': str(row['Buy/Sell']),
                'lot_size': lot_size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': row['Time Exec'],
                'exit_time': None,
                'pips': pips,
                'pl_usd': pl_usd,
                'rr': rr,
                'status': status,
                'result': result_str,
                'session': session,
                'bias': bias,
                'tp': take_profit,
                'sl': stop_loss,
                'strategy': strategy,
                'notes': poi
            })
            
            db.commit()
            imported += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  ✅ Imported {idx + 1}/{len(df)} trades...")
                
        except Exception as e:
            errors += 1
            db.rollback()
            print(f"  ❌ Error on row {idx + 1}: {str(e)[:200]}")
            if errors <= 3:
                print(f"     Pair: {row.get('Pair')}, Date: {row.get('Time Exec')}")
                print(f"     Entry (J): {row.get('Entry')}, Exit (O): {row.get('Exit')}")
                print(f"     Stop (K): {row.get('Stop')}, TP (L): {row.get('TP1')}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ IMPORT COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successfully imported: {imported} trades")
    if errors > 0:
        print(f"❌ Errors: {errors} trades")
    
    # Verify
    result = db.execute(text(f"SELECT COUNT(*) FROM trading_journal WHERE user_id = {USER_ID}"))
    final_count = result.scalar()
    print(f"\n📊 Total trades for User {USER_ID}: {final_count}")
    
    # Show first 5 trades
    result = db.execute(text("""
        SELECT pair_ticker, order_type, entry_time, entry_price, exit_price, 
               profit_loss_pips, profit_loss_usd, result, notes
        FROM trading_journal
        WHERE user_id = :user_id
        ORDER BY entry_time ASC
        LIMIT 5
    """), {'user_id': USER_ID})
    
    print(f"\n📋 First 5 trades:")
    print("="*120)
    for row in result:
        exit_str = f"{row[4]:.2f}" if row[4] else "OPEN"
        result_str = row[7] if row[7] else "open"
        notes = row[8] if row[8] else ""
        print(f"{row[0]:8s} | {row[1]:4s} | {row[2]} | Entry: {row[3]:.5f} | Exit: {exit_str:>10s} | Pips: {row[5] if row[5] else 0:6.1f} | P/L: ${row[6] if row[6] else 0:7.2f} | {result_str:8s} | {notes}")
    
    # Show stats
    result = db.execute(text("""
        SELECT 
            result,
            COUNT(*) as count,
            SUM(profit_loss_usd) as total_pl
        FROM trading_journal
        WHERE user_id = :user_id AND result IS NOT NULL
        GROUP BY result
    """), {'user_id': USER_ID})
    
    print(f"\n📊 RESULTS:")
    print("="*50)
    for row in result:
        print(f"{row[0]:10s} | Count: {row[1]:3d} | P/L: ${row[2]:>10.2f}")
    
except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    db.rollback()
    raise
finally:
    db.close()

print(f"\n{'='*80}")
print("🎉 ALL DONE! Your journal is correctly imported to User ID 8!")
print(f"{'='*80}")
