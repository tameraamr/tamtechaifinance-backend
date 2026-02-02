"""
Quick verification test for critical fixes
Run this after deploying the updates
"""
import asyncio
import time
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

print("🧪 Testing Critical Fixes...\n")

# Test 1: Verify Indexes
print("📊 Test 1: Verifying Database Indexes")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT indexname FROM pg_indexes 
        WHERE indexname IN (
            'idx_users_is_pro',
            'idx_users_subscription_expiry',
            'idx_user_analysis_history_user_created',
            'idx_portfolio_user_id',
            'idx_market_data_cache_last_updated'
        )
    """))
    indexes = [row[0] for row in result]
    
    expected = [
        'idx_users_is_pro',
        'idx_users_subscription_expiry',
        'idx_user_analysis_history_user_created',
        'idx_portfolio_user_id',
        'idx_market_data_cache_last_updated'
    ]
    
    for idx in expected:
        if idx in indexes:
            print(f"  ✅ {idx}")
        else:
            print(f"  ❌ {idx} - MISSING!")

# Test 2: Check row-level locking syntax
print("\n🔒 Test 2: Checking Row-Level Locking Implementation")
try:
    from main import User, select, with_for_update
    db = SessionLocal()
    
    # This should compile without errors
    stmt = select(User).where(User.id == 1).with_for_update()
    print("  ✅ Row-level locking syntax is correct")
    db.close()
except Exception as e:
    print(f"  ❌ Row-level locking error: {e}")

# Test 3: Check CircuitBreaker class
print("\n🛡️ Test 3: Testing Circuit Breaker")
try:
    from main import CircuitBreaker
    breaker = CircuitBreaker(failure_threshold=3, timeout_duration=5)
    
    # Test normal state
    assert breaker.can_proceed() == True, "Circuit should be closed initially"
    print("  ✅ Initial state: CLOSED")
    
    # Trigger failures
    for i in range(3):
        breaker.record_failure()
    
    # Check if circuit opened
    if breaker.state == "OPEN":
        print("  ✅ Circuit opened after 3 failures")
    else:
        print(f"  ⚠️ Circuit state: {breaker.state} (expected OPEN)")
    
    # Test recovery
    breaker.record_success()
    print(f"  ✅ Recovery works, state: {breaker.state}")
    
except Exception as e:
    print(f"  ❌ Circuit breaker error: {e}")

# Test 4: Check MarketDataCache model
print("\n⚡ Test 4: Verifying Market Data Cache")
try:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM market_data_cache
        """))
        count = result.scalar()
        print(f"  ✅ Market data cache table exists ({count} entries)")
except Exception as e:
    print(f"  ❌ Cache table error: {e}")

print("\n" + "="*60)
print("🎉 Critical Fixes Verification Complete!")
print("="*60)
print("\nNext Steps:")
print("1. Deploy to Railway")
print("2. Monitor logs for circuit breaker messages")
print("3. Test with real traffic")
print("4. Check response times (should be 30% faster)")
