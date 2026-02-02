"""
Migration script to add image_url column to articles table
"""
import os
from sqlalchemy import create_engine, text

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in environment")
    exit(1)

# Fix postgres:// to postgresql:// for SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='articles' AND column_name='image_url'
        """))
        
        if result.fetchone():
            print("✅ Column 'image_url' already exists in articles table")
        else:
            # Add the column
            conn.execute(text("ALTER TABLE articles ADD COLUMN image_url TEXT"))
            conn.commit()
            print("✅ Successfully added 'image_url' column to articles table")
            
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
