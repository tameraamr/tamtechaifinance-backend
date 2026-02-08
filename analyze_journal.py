import pandas as pd
import os

# Read the Excel file
excel_path = r"c:\Users\tamer\Desktop\Tamer Journal.xlsx"

if os.path.exists(excel_path):
    print("✅ Found Excel file:", excel_path)
    print("\n" + "="*80)
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    print(f"\n📊 TOTAL ROWS: {len(df)}")
    print(f"📊 TOTAL COLUMNS: {len(df.columns)}")
    
    print("\n" + "="*80)
    print("📋 COLUMN NAMES:")
    print("="*80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    print("\n" + "="*80)
    print("📅 DATE RANGE:")
    print("="*80)
    
    # Try to find date columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'entry' in col.lower()]
    print(f"Potential date columns: {date_columns}")
    
    if date_columns:
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                valid_dates = df[col].dropna()
                if len(valid_dates) > 0:
                    print(f"\n{col}:")
                    print(f"  First date: {valid_dates.min()}")
                    print(f"  Last date: {valid_dates.max()}")
                    print(f"  Total valid dates: {len(valid_dates)}")
            except:
                pass
    
    print("\n" + "="*80)
    print("📊 FIRST 10 ROWS:")
    print("="*80)
    print(df.head(10).to_string())
    
    print("\n" + "="*80)
    print("📊 DATA TYPES:")
    print("="*80)
    print(df.dtypes)
    
    print("\n" + "="*80)
    print("📊 BASIC STATISTICS:")
    print("="*80)
    print(df.describe())
    
    # Save a CSV version for easy viewing
    csv_path = r"c:\Users\tamer\Desktop\TamtechAI Finance Tool\TamtechAI\backend\journal_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved CSV analysis to: {csv_path}")
    
else:
    print("❌ Excel file not found at:", excel_path)
    print("Please provide the correct path to your trading journal Excel file.")
