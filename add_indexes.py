"""
Database Index Migration Script
Adds critical performance indexes identified in technical audit
"""
import os
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables")

# Fix URL format for psycopg2
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)

def check_index_exists(index_name: str) -> bool:
    """Check if an index already exists"""
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE indexname = '{index_name}'
            )
        """))
        return result.scalar()

def add_indexes():
    """Add all critical indexes"""
    indexes_to_add = [
        {
            "name": "idx_users_is_pro",
            "sql": "CREATE INDEX IF NOT EXISTS idx_users_is_pro ON users(is_pro) WHERE is_pro = 1",
            "description": "Index for Pro user lookups"
        },
        {
            "name": "idx_users_subscription_expiry",
            "sql": "CREATE INDEX IF NOT EXISTS idx_users_subscription_expiry ON users(subscription_expiry) WHERE subscription_expiry IS NOT NULL",
            "description": "Index for subscription expiry checks"
        },
        {
            "name": "idx_user_analysis_history_user_created",
            "sql": "CREATE INDEX IF NOT EXISTS idx_user_analysis_history_user_created ON user_analysis_history(user_id, created_at DESC)",
            "description": "Composite index for user history queries"
        },
        {
            "name": "idx_portfolio_user_id",
            "sql": "CREATE INDEX IF NOT EXISTS idx_portfolio_user_id ON portfolio_holdings(user_id)",
            "description": "Index for portfolio queries"
        },
        {
            "name": "idx_market_data_cache_last_updated",
            "sql": "CREATE INDEX IF NOT EXISTS idx_market_data_cache_last_updated ON market_data_cache(last_updated DESC)",
            "description": "Index for cache expiry queries"
        }
    ]
    
    print("🔧 Starting database index migration...\n")
    
    with engine.connect() as conn:
        for idx in indexes_to_add:
            try:
                if check_index_exists(idx["name"]):
                    print(f"✅ Index '{idx['name']}' already exists - {idx['description']}")
                else:
                    print(f"⚙️  Creating index '{idx['name']}' - {idx['description']}")
                    conn.execute(text(idx["sql"]))
                    conn.commit()
                    print(f"✅ Index '{idx['name']}' created successfully")
            except Exception as e:
                print(f"❌ Error creating index '{idx['name']}': {e}")
                conn.rollback()
    
    print("\n🎉 Index migration completed!")
    
    # Verify indexes
    print("\n📊 Verifying indexes...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT schemaname, tablename, indexname 
            FROM pg_indexes 
            WHERE tablename IN ('users', 'user_analysis_history', 'portfolio_holdings', 'market_data_cache')
            ORDER BY tablename, indexname
        """))
        
        print("\nCurrent indexes:")
        for row in result:
            print(f"  {row[1]}.{row[2]}")

if __name__ == "__main__":
    add_indexes()
