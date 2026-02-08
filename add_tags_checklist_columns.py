#!/usr/bin/env python3
"""
Migration script to add tags and checklist columns to trading_journal table
Run this script to add the new columns for enhanced trade entry features
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

def migrate_database():
    """Add tags and checklist columns to trading_journal table"""

    try:
        engine = create_engine(DATABASE_URL)

        # Check if table exists
        inspector = inspect(engine)
        if 'trading_journal' not in inspector.get_table_names():
            print("❌ trading_journal table does not exist. Please run create_journal_tables.py first.")
            return

        # Get existing columns
        columns = [col['name'] for col in inspector.get_columns('trading_journal')]

        with engine.connect() as conn:
            # Add tags column if it doesn't exist
            if 'tags' not in columns:
                print("Adding tags column...")
                if DATABASE_URL.startswith("sqlite"):
                    conn.execute(text("ALTER TABLE trading_journal ADD COLUMN tags TEXT"))
                else:
                    conn.execute(text("ALTER TABLE trading_journal ADD COLUMN tags TEXT"))
                print("✓ Added tags column")
            else:
                print("✓ Tags column already exists")

            # Add checklist column if it doesn't exist
            if 'checklist' not in columns:
                print("Adding checklist column...")
                if DATABASE_URL.startswith("sqlite"):
                    conn.execute(text("ALTER TABLE trading_journal ADD COLUMN checklist TEXT"))
                else:
                    conn.execute(text("ALTER TABLE trading_journal ADD COLUMN checklist TEXT"))
                print("✓ Added checklist column")
            else:
                print("✓ Checklist column already exists")

            conn.commit()

        print("\n✅ Migration completed successfully!")
        print("New columns added: tags, checklist")

    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")

if __name__ == "__main__":
    print("🔄 Starting database migration for enhanced trade features...")
    migrate_database()