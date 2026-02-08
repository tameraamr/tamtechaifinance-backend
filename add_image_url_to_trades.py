#!/usr/bin/env python3
"""
Migration script to add image_url column to trading_journal table
This allows users to attach screenshots to their trades
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in environment")
    exit(1)

# Fix URL format for psycopg2
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

def add_image_url_column():
    """Add image_url column to trading_journal table if it doesn't exist"""

    try:
        engine = create_engine(DATABASE_URL)

        # Check if table exists
        inspector = inspect(engine)
        if 'trading_journal' not in inspector.get_table_names():
            print("❌ trading_journal table does not exist. Please run create_journal_tables.py first.")
            return False

        # Get existing columns
        columns = [col['name'] for col in inspector.get_columns('trading_journal')]

        with engine.connect() as conn:
            # Add image_url column if it doesn't exist
            if 'image_url' not in columns:
                print("⚙️ Running migration: Adding image_url column to trading_journal table...")
                conn.execute(text("ALTER TABLE trading_journal ADD COLUMN image_url TEXT"))
                conn.commit()
                print("✅ Migration complete: image_url column added")
                return True
            else:
                print("✅ Column 'image_url' already exists in trading_journal table")
                return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = add_image_url_column()
    exit(0 if success else 1)