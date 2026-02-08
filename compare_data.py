"""
Compare Excel data vs Database data to find column mapping issues
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Read Excel
excel_path = r"c:\Users\tamer\Desktop\Tamer Journal.xlsx"
df_excel = pd.read_excel(excel_path)
df_excel['Time Exec'] = pd.to_datetime(df_excel['Time Exec'])

# Connect to DB
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
db = Session()

try:
    # Get first 10 trades from DB
    result = db.execute(text("""
        SELECT 
            id,
            pair_ticker,
            order_type,
            entry_time,
            entry_price,
            exit_price,
            lot_size,
            profit_loss_pips,
            profit_loss_usd,
            risk_reward_ratio,
            status,
            result
        FROM trading_journal
        ORDER BY entry_time ASC
        LIMIT 10
    """))
    
    print("="*120)
    print("COMPARISON: EXCEL vs DATABASE (First 10 Trades)")
    print("="*120)
    print("\nEXCEL COLUMNS:")
    print(df_excel.columns.tolist())
    
    print("\n" + "="*120)
    print("TRADE-BY-TRADE COMPARISON:")
    print("="*120)
    
    db_trades = result.fetchall()
    
    for i, (db_row, excel_idx) in enumerate(zip(db_trades, df_excel.head(10).iterrows())):
        excel_row = excel_idx[1]
        print(f"\n--- TRADE #{i+1} ---")
        print(f"Excel Date:   {excel_row['Time Exec']}")
        print(f"DB Date:      {db_row[3]}")
        print(f"MATCH: {'✅' if str(excel_row['Time Exec']) == str(db_row[3]) else '❌'}")
        
        print(f"\nExcel Pair:   {excel_row['Pair']}")
        print(f"DB Pair:      {db_row[1]}")
        print(f"MATCH: {'✅' if excel_row['Pair'] == db_row[1] else '❌'}")
        
        print(f"\nExcel Type:   {excel_row['Buy/Sell']}")
        print(f"DB Type:      {db_row[2]}")
        print(f"MATCH: {'✅' if excel_row['Buy/Sell'] == db_row[2] else '❌'}")
        
        print(f"\nExcel Lot:    {excel_row['Lot Size']}")
        print(f"DB Lot:       {db_row[6]}")
        print(f"MATCH: {'✅' if abs(excel_row['Lot Size'] - db_row[6]) < 0.001 else '❌'}")
        
        # Entry price - the Excel 'Entry' column seems wrong, let's check other columns
        print(f"\nExcel Exit Price: {excel_row['Exit']}")
        print(f"DB Entry Price:   {db_row[4]}")
        print(f"DB Exit Price:    {db_row[5]}")
        
        print(f"\nExcel Pips:   {excel_row['Pips/Points']}")
        print(f"DB Pips:      {db_row[7]}")
        print(f"MATCH: {'✅' if abs(excel_row['Pips/Points'] - db_row[7]) < 0.1 else '❌'}")
        
        print(f"\nExcel P/L $:  {excel_row['P/L $']}")
        print(f"DB P/L $:     {db_row[8]}")
        print(f"MATCH: {'✅' if abs(excel_row['P/L $'] - db_row[8]) < 0.01 else '❌'}")
        
        print(f"\nExcel R:R:    {excel_row['R:R']}")
        print(f"DB R:R:       {db_row[9]}")
        
        print(f"\nExcel Status: {excel_row['Status']}")
        print(f"DB Result:    {db_row[11]}")
        print(f"MATCH: {'✅' if excel_row['Status'].lower() == db_row[11] else '❌'}")
        
        print("="*120)
        
finally:
    db.close()
