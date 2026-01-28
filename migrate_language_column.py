"""
Database migration script to add language column to analysis_reports table
Run this ONCE before deploying the updated backend
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in environment variables")
    exit(1)

engine = create_engine(DATABASE_URL)

print("🔄 Starting database migration...")

try:
    with engine.connect() as conn:
        # Step 1: Add language column with default value 'en'
        print("📝 Step 1: Adding 'language' column...")
        conn.execute(text("""
            ALTER TABLE analysis_reports 
            ADD COLUMN IF NOT EXISTS language VARCHAR DEFAULT 'en'
        """))
        conn.commit()
        print("✅ Language column added")
        
        # Step 2: Drop old unique constraint on ticker
        print("📝 Step 2: Removing old unique constraint...")
        try:
            conn.execute(text("""
                ALTER TABLE analysis_reports 
                DROP CONSTRAINT IF EXISTS analysis_reports_ticker_key
            """))
            conn.commit()
            print("✅ Old constraint removed")
        except Exception as e:
            print(f"⚠️  Constraint might not exist (this is OK): {e}")
        
        # Step 3: Create composite unique index on (ticker, language)
        print("📝 Step 3: Creating composite index...")
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_ticker_language 
            ON analysis_reports (ticker, language)
        """))
        conn.commit()
        print("✅ Composite index created")
        
        # Step 4: Update existing records to have language = 'en'
        print("📝 Step 4: Updating existing records...")
        result = conn.execute(text("""
            UPDATE analysis_reports 
            SET language = 'en' 
            WHERE language IS NULL
        """))
        conn.commit()
        print(f"✅ Updated {result.rowcount} existing records")
        
        print("\n🎉 Migration completed successfully!")
        print("✅ You can now deploy the updated backend code")
        
except Exception as e:
    print(f"❌ Migration failed: {e}")
    print("\n⚠️  If you see 'column already exists', the migration was already run")
    print("⚠️  If you see other errors, check your database connection")
