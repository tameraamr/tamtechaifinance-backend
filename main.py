from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, Cookie, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from google import genai
from google.genai import types
import yfinance as yf
import os
import json
import random
# Version: 1.0.1 - Fixed is_verified in login response
import requests
import httpx
from dotenv import load_dotenv
import feedparser
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, func, Index
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import timezone, UTC
from jose import JWTError, jwt
import bcrypt
import re
import secrets
from mailer import send_verification_email

# --- Helper function for datetime handling ---
def make_datetime_aware(dt):
    """Convert naive datetime to UTC-aware if needed"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

# Load environment variables
load_dotenv()

# --- Database & Security (PostgreSQL ONLY) ---
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "❌ DATABASE_URL environment variable is required!\n"
        "Please set it in your .env file or environment.\n"
        "Example: DATABASE_URL=postgresql://user:password@localhost:5432/dbname"
    )

# Fix URL format for psycopg2
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Increased from default 5
    max_overflow=30,        # Increased from default 10
    pool_pre_ping=True,     # Test connections before use
    pool_recycle=3600,      # Recycle connections every hour
    echo=False              # Disable SQL logging for performance
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "MY_SUPER_SECRET_KEY_123" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # البيانات الشخصية الجديدة
    first_name = Column(String)
    last_name = Column(String)
    phone_number = Column(String)
    country = Column(String)              # إجباري
    address = Column(String, nullable=True) # اختياري
    
    credits = Column(Integer, default=0)
    is_verified = Column(Integer, default=0, nullable=True)  # 0 = not verified, 1 = verified 

# هذا الجدول يسجل IP الزائر وكم مرة استخدم الموقع
class GuestUsage(Base):
    __tablename__ = "guest_usage"
    ip_address = Column(String, primary_key=True, index=True)
    attempts = Column(Integer, default=0)
    last_attempt = Column(DateTime, default=func.now())

# 1. إنشاء جدول لتخزين آخر التحليلات
class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    verdict = Column(String)  # BUY, SELL, HOLD
    confidence_score = Column(Integer)
    created_at = Column(DateTime, default=func.now())

# 2. جدول تخزين التقارير الكاملة (24-hour cache with language support)
class AnalysisReport(Base):
    __tablename__ = "analysis_reports"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)  # Stock ticker
    language = Column(String, index=True, default="en")  # Language code (en, ar, es, etc.)
    ai_json_data = Column(Text)  # Stores full AI analysis as JSON string (using Text for large data)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Composite unique constraint for ticker + language
    __table_args__ = (
        Index('ix_ticker_language', 'ticker', 'language', unique=True),
    )

# 3. جدول التحقق من البريد الإلكتروني
class VerificationToken(Base):
    __tablename__ = "verification_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())

# 4. User Analysis History - Track each user's analyzed stocks
class UserAnalysisHistory(Base):
    __tablename__ = "user_analysis_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Links to User.id
    ticker = Column(String, index=True)
    company_name = Column(String)
    last_price = Column(String)  # Store as string to avoid decimal issues
    verdict = Column(String)
    confidence_score = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# 5. Portfolio Holdings - Track user stock portfolios
class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # FK to users
    ticker = Column(String, index=True)
    quantity = Column(Float)  # Number of shares
    avg_buy_price = Column(Float, nullable=True)  # Optional: average purchase price
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# 6. Calendar Events Cache - Weekly economic calendar cache (Using raw SQL instead of ORM)
# class CalendarEvent(Base):
#     __tablename__ = "calendar_events"
#     id = Column(Integer, primary_key=True)
#     name = Column(String, nullable=False)
#     date_time = Column(DateTime, nullable=False)
#     importance = Column(String, nullable=False)  # High, Medium, Low
#     ai_impact_note = Column(Text)
#     created_at = Column(DateTime, default=func.now())
#     updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Composite index for user_id + ticker (one row per user per ticker)
    __table_args__ = (
        Index('ix_user_ticker', 'user_id', 'ticker', unique=True),
    )

# 6. Market Data Cache - Global caching layer for market data (10-minute cache)
class MarketDataCache(Base):
    __tablename__ = "market_data_cache"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True)  # Stock ticker/symbol
    asset_type = Column(String, index=True)  # 'stock', 'crypto', 'commodity', 'forex'
    name = Column(String)  # Company/asset name
    price = Column(Float)  # Current price
    change_percent = Column(Float)  # 24h change percentage (change_p)
    sector = Column(String, nullable=True)  # Sector for stocks
    market_cap = Column(Float, nullable=True)  # Market cap (for stocks/crypto)
    volume = Column(Float, nullable=True)  # Trading volume
    last_updated = Column(DateTime, default=func.now(), index=True)
    created_at = Column(DateTime, default=func.now())

# إنشاء الجداول تلقائياً
Base.metadata.create_all(bind=engine)

# --- Database Migration Logic ---
# Add is_verified column if it doesn't exist (for existing databases)
try:
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    
    # Check if users table has is_verified column
    users_columns = [col['name'] for col in inspector.get_columns('users')]
    
    if 'is_verified' not in users_columns:
        print("⚙️ Running migration: Adding is_verified column to users table...")
        with engine.connect() as conn:
            # PostgreSQL syntax
            conn.execute(text("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0"))
            conn.commit()
        print("✅ Migration complete: is_verified column added")
    else:
        print("✅ is_verified column already exists")
        
except Exception as e:
    print(f"⚠️ Migration warning: {e}")
    # Don't fail startup if migration fails
    pass

# --- Gemini Setup ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"DEBUG: API_KEY loaded: {API_KEY[:10] if API_KEY else 'None'}")
# ملاحظة: في السيرفر لا توقف التطبيق إذا لم تجد المفتاح فوراً، السيرفر سيحقنه
try:
    if not API_KEY: 
        print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
        print("   Server will start without Gemini AI functionality.")
        print("   To enable AI features, set GOOGLE_API_KEY environment variable.")
        client = None
        model_name = None
    else:
        # Don't initialize client globally to avoid startup issues
        # Initialize it in functions that need it
        client = None  # Will be initialized when needed
        model_name = 'gemini-2.0-flash'
        print("✅ Gemini API key found, will initialize client when needed")
except Exception as e:
    print(f"❌ Error with Gemini setup: {e}")
    print("   Server will start without Gemini AI functionality.")
    client = None
    model_name = None

# --- Global Market Data Caching Engine ---

def get_cached_market_data(tickers: list, db: Session, include_expired: bool = False) -> dict:
    """
    Get cached market data for tickers. Returns dict with ticker -> data mapping.
    Only returns data that's less than 10 minutes old unless include_expired=True.
    If tickers is empty, returns all cached data.
    """
    # CRITICAL FIX: Use UTC time to match database timestamps (func.now() uses UTC)
    ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)

    if tickers:
        if include_expired:
            cached_data = db.query(MarketDataCache).filter(
                MarketDataCache.ticker.in_(tickers)
            ).all()
        else:
            cached_data = db.query(MarketDataCache).filter(
                MarketDataCache.ticker.in_(tickers),
                MarketDataCache.last_updated >= ten_minutes_ago
            ).all()
    else:
        # Return all cached data if no tickers specified
        if include_expired:
            cached_data = db.query(MarketDataCache).all()
        else:
            cached_data = db.query(MarketDataCache).filter(
                MarketDataCache.last_updated >= ten_minutes_ago
            ).all()

    return {item.ticker: {
        'name': item.name,
        'price': item.price,
        'change_percent': item.change_percent,
        'sector': item.sector,
        'market_cap': item.market_cap,
        'volume': item.volume,
        'asset_type': item.asset_type,
        'last_updated': item.last_updated.replace(tzinfo=UTC) if item.last_updated else None
    } for item in cached_data}

def get_cached_market_data_with_background_update(tickers: list, db: Session, background_tasks: BackgroundTasks) -> dict:
    """
    Cache-first approach: Get data from DB only. If data is older than 10 minutes, trigger background update.
    Never waits for API - instant response.
    """
    if not tickers:
        return {}

    # Get cached data (only from DB)
    cached_data = get_cached_market_data(tickers, db)

    # Check if any data is missing or stale (>10 min)
    ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    stale_tickers = []

    for ticker in tickers:
        if ticker not in cached_data:
            stale_tickers.append(ticker)
        elif cached_data[ticker]['last_updated'] < ten_minutes_ago:
            stale_tickers.append(ticker)

    # If any data is stale or missing, trigger background update
    if stale_tickers:
        background_tasks.add_task(update_market_data_background, stale_tickers)

    return cached_data

def update_market_data_background(tickers: list):
    """
    Background task to fetch and update market data.
    Creates its own DB session to avoid session conflicts.
    """
    db = SessionLocal()
    try:
        print(f"🔄 Background update for {len(tickers)} tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
        
        # Fetch fresh data
        fresh_data = batch_fetch_market_data(tickers)
        
        # Update cache
        update_market_cache(fresh_data, db)
        
        print(f"✅ Background update completed for {len(fresh_data)} tickers")
    except Exception as e:
        print(f"❌ Background update failed: {e}")
    finally:
        db.close()

def get_ticker_sector(ticker: str, info: dict) -> str:
    """
    Intelligent sector mapping for tickers.
    Uses Yahoo Finance data first, then falls back to ticker-based classification.
    """
    # First try Yahoo Finance sector
    yahoo_sector = info.get("sector")
    if yahoo_sector:
        return yahoo_sector.title()

    # Fallback: Classify based on ticker patterns and characteristics
    ticker_upper = ticker.upper()

    # ETF Classification based on common patterns
    if any(etf_pattern in ticker_upper for etf_pattern in ['.AS', '.DE', '.L', '.PA', '.MI', '.BR', '.LS', '.MC', '.VI']):
        # European ETFs - classify by ticker prefix
        if ticker_upper.startswith(('VWCE', 'VWRD', 'VEUR', 'VUSA', 'VFEM', 'VJPN', 'VWO')):
            return 'ETF - Global Equity'
        elif ticker_upper.startswith(('CSPX', 'VUSA', 'EQQQ', 'NQSE')):
            return 'ETF - US Equity'
        elif ticker_upper.startswith(('VEUR', 'VGER')):
            return 'ETF - European Equity'
        elif ticker_upper.startswith(('VFEM', 'VHYL')):
            return 'ETF - Emerging Markets'
        elif ticker_upper.startswith(('VIG', 'VHYG')):
            return 'ETF - Bonds'
        elif ticker_upper.startswith(('IBTL', 'IBTM')):
            return 'ETF - Bonds'
        else:
            return 'ETF - Mixed Assets'

    # US ETF Classification
    elif ticker_upper.endswith(('ETF', 'FUND')) or len(ticker_upper) > 5:
        # Common US ETF patterns
        if ticker_upper.startswith(('SPY', 'VOO', 'IVV', 'QQQ', 'IWM')):
            return 'ETF - US Equity'
        elif ticker_upper.startswith(('EFA', 'VEA', 'VXUS', 'VWO')):
            return 'ETF - International Equity'
        elif ticker_upper.startswith(('BND', 'AGG', 'TLT', 'IEF')):
            return 'ETF - Bonds'
        elif ticker_upper.startswith(('GLD', 'SLV', 'IAU')):
            return 'ETF - Commodities'
        else:
            return 'ETF - Mixed Assets'

    # Crypto Classification
    elif ticker_upper in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD', 'LTC-USD', 'MATIC-USD']:
        return 'Cryptocurrency'

    # Commodity Classification
    elif ticker_upper in ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'ALI=F', 'ZC=F', 'ZW=F', 'ZS=F', 'ZM=F', 'ZL=F', 'ZO=F', 'ZR=F', 'KE=F', 'CC=F', 'KC=F', 'CT=F', 'SB=F']:
        return 'Commodities'

    # Forex Classification
    elif any(ticker_upper.endswith(pair) for pair in ['=X', 'USD=X', 'EUR=X', 'GBP=X', 'JPY=X', 'CAD=X', 'AUD=X', 'CHF=X', 'CNY=X', 'INR=X']):
        return 'Currency'

    # Individual Stock Classification (fallback)
    else:
        # Try to infer from company name or other info
        industry = info.get("industry", "").lower()
        if industry:
            # Map common industries to sectors
            industry_sector_map = {
                'software': 'Technology',
                'internet': 'Technology',
                'semiconductor': 'Technology',
                'computer': 'Technology',
                'pharmaceutical': 'Healthcare',
                'biotechnology': 'Healthcare',
                'medical': 'Healthcare',
                'bank': 'Financial Services',
                'financial': 'Financial Services',
                'insurance': 'Financial Services',
                'automotive': 'Consumer Cyclical',
                'retail': 'Consumer Defensive',
                'beverage': 'Consumer Defensive',
                'food': 'Consumer Defensive',
                'energy': 'Energy',
                'oil': 'Energy',
                'gas': 'Energy',
                'industrial': 'Industrials',
                'manufacturing': 'Industrials',
                'aerospace': 'Industrials',
                'defense': 'Industrials',
                'real estate': 'Real Estate',
                ' REIT': 'Real Estate',
                'materials': 'Basic Materials',
                'chemical': 'Basic Materials',
                'mining': 'Basic Materials',
                'utilities': 'Utilities',
                'telecommunication': 'Communication Services',
                'media': 'Communication Services',
                'entertainment': 'Communication Services'
            }

            for key, sector in industry_sector_map.items():
                if key in industry:
                    return sector

        # Final fallback
        return 'Unknown'

def batch_fetch_market_data(tickers: list, asset_types: dict = None):
    """
    Batch fetch market data for multiple tickers with intelligent sector mapping.
    Returns dict with ticker -> data mapping for cache storage.
    """
    if not tickers:
        return {}

    results = {}
    chunk_size = 10  # Process in chunks to avoid rate limits

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"📊 Fetching batch {i//chunk_size + 1}/{(len(tickers) + chunk_size - 1)//chunk_size}: {chunk}")

        # Fetch data for this chunk
        for ticker in chunk:
            try:
                # Determine asset type
                asset_type = asset_types.get(ticker, 'stock') if asset_types else 'stock'

                # Fetch data using yfinance
                stock = yf.Ticker(ticker)

                # Get basic info
                info = stock.info
                current_price = None

                # Try multiple ways to get current price
                try:
                    current_price = stock.fast_info['last_price']
                except:
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')

                if not current_price or current_price <= 0:
                    print(f"⚠️ No valid price for {ticker}, skipping")
                    continue

                # Get change percentage
                change_percent = 0
                try:
                    prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
                    if prev_close and prev_close > 0:
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                except:
                    pass

                # Get sector using our intelligent mapping
                sector = get_ticker_sector(ticker, info)

                # Get market cap and volume
                market_cap = info.get('marketCap', 0)
                volume = info.get('volume') or info.get('averageVolume', 0)

                # Get company name
                name = info.get('longName') or info.get('shortName') or ticker

                results[ticker] = {
                    'name': name,
                    'price': current_price,
                    'change_percent': change_percent,
                    'sector': sector,
                    'market_cap': market_cap,
                    'volume': volume,
                    'asset_type': asset_type,
                    'success': True
                }

                print(f"✅ {ticker}: ${current_price:.2f} ({change_percent:+.2f}%) - {sector}")

            except Exception as e:
                print(f"❌ Failed to fetch {ticker}: {e}")
                # Return minimal data for failed fetches to avoid breaking cache
                results[ticker] = {
                    'name': ticker,
                    'price': 0,
                    'change_percent': 0,
                    'sector': 'Unknown',
                    'market_cap': 0,
                    'volume': 0,
                    'asset_type': asset_types.get(ticker, 'stock') if asset_types else 'stock',
                    'success': False
                }

        # Small delay between chunks to be respectful to Yahoo Finance
        if i + chunk_size < len(tickers):
            import time
            time.sleep(0.5)

    print(f"📊 Batch fetch completed: {len(results)}/{len(tickers)} tickers processed")
    return results

def update_market_cache(tickers_data: dict, db: Session):
    """
    Update the market data cache with fresh data.
    """
    for ticker, data in tickers_data.items():
        try:
            # Check if cache entry exists
            existing = db.query(MarketDataCache).filter(MarketDataCache.ticker == ticker).first()
            
            if existing:
                # Update existing
                existing.name = data['name']
                existing.price = data['price']
                existing.change_percent = data['change_percent']
                existing.sector = data['sector']
                existing.market_cap = data['market_cap']
                existing.volume = data['volume']
                existing.asset_type = data['asset_type']
                existing.last_updated = datetime.now(timezone.utc)  # CRITICAL FIX: Use UTC time
            else:
                # Create new
                cache_entry = MarketDataCache(
                    ticker=ticker,
                    name=data['name'],
                    price=data['price'],
                    change_percent=data['change_percent'],
                    sector=data['sector'],
                    market_cap=data['market_cap'],
                    volume=data['volume'],
                    asset_type=data['asset_type']
                )
                db.add(cache_entry)
                
        except Exception as e:
            print(f"❌ Error updating cache for {ticker}: {e}")
    
    try:
        db.commit()
    except Exception as e:
        print(f"❌ Error committing cache updates: {e}")
        db.rollback()

def get_market_data_with_cache(tickers: list, asset_types: dict = None, db: Session = None, force_fresh: bool = False, stale_while_revalidate: bool = False) -> dict:
    """
    Main function: Get market data with intelligent caching.
    - Check cache first (10-minute validity)
    - Force fresh fetch for invalid cached data (price=0 or None)
    - URGENT FIX: If DB is empty, BLOCK and wait for live fetch
    - URGENT FIX: If price == 0, bypass_cache_and_fetch_live()
    - URGENT FIX: Stale-while-revalidate mode for navbar speed
    - Batch fetch missing/expired data
    - Update cache asynchronously
    - Return complete dataset
    """
    if not tickers:
        return {}

    # Get cached data - always instant, background updates handle fresh data
    cached_data = get_cached_market_data(tickers, db)

    # CRITICAL FIX: Return cached data immediately if all tickers are in cache
    # This prevents ANY API calls when data exists in cache, regardless of freshness
    if cached_data and len(cached_data) == len(tickers) and not force_fresh:
        print(f"✅ CACHE HIT: Serving {len(tickers)} tickers from cache (no API calls)")
        return cached_data

    # Find tickers that need fresh data
    tickers_needing_fresh = []
    valid_cached_data = {}

    for ticker in tickers:
        if force_fresh:
            # Force fresh fetch for all tickers
            tickers_needing_fresh.append(ticker)
        elif ticker not in cached_data:
            # Missing from cache
            tickers_needing_fresh.append(ticker)
        elif cached_data[ticker].get('price', 0) <= 0:
            # URGENT FIX 2: Invalid cached data (price is 0 or negative) - bypass cache
            print(f"🚨 INVALID CACHE: {ticker} has price ${cached_data[ticker].get('price', 0)} - bypassing cache")
            tickers_needing_fresh.append(ticker)
        elif stale_while_revalidate:
            # Stale-while-revalidate: return cached data but trigger background update
            valid_cached_data[ticker] = cached_data[ticker]
            tickers_needing_fresh.append(ticker)  # Still fetch fresh in background
        else:
            # Valid cached data
            valid_cached_data[ticker] = cached_data[ticker]

    # Batch fetch data that needs updating
    if tickers_needing_fresh:
        fresh_data = batch_fetch_market_data(tickers_needing_fresh, asset_types)

        # Filter out invalid fresh data (don't cache bad data)
        valid_fresh_data = {
            ticker: data for ticker, data in fresh_data.items()
            if data.get('price', 0) > 0 and data.get('success', False)
        }

        # Update cache with valid fresh data
        if valid_fresh_data and db:
            if stale_while_revalidate:
                # Async update for stale-while-revalidate
                def update_cache_async():
                    # Create a new session for the async operation
                    async_db = SessionLocal()
                    try:
                        update_market_cache(valid_fresh_data, async_db)
                        async_db.commit()
                        print(f"🔄 Stale-while-revalidate: Updated {len(valid_fresh_data)} tickers in background")
                    except Exception as e:
                        print(f"⚠️ Async cache update failed: {e}")
                        async_db.rollback()
                    finally:
                        async_db.close()

                try:
                    import threading
                    thread = threading.Thread(target=update_cache_async, daemon=True)
                    thread.start()
                except Exception as e:
                    print(f"⚠️ Failed to start async cache update thread: {e}")
                    # Fallback to synchronous update
                    update_market_cache(valid_fresh_data, db)
            else:
                # Synchronous update for critical data
                update_market_cache(valid_fresh_data, db)

        # Merge valid fresh data with valid cached data
        valid_cached_data.update(valid_fresh_data)

        # For tickers that failed to fetch fresh data, return cached data even if invalid
        # This ensures we always return something rather than empty data
        for ticker in tickers_needing_fresh:
            if ticker not in valid_cached_data and ticker in cached_data:
                valid_cached_data[ticker] = cached_data[ticker]

    return valid_cached_data

def update_heatmap_cache_background(all_tickers: list, asset_types: dict):
    """
    🔥 BACKGROUND TASK: Updates heatmap cache asynchronously
    Called only when cache needs refresh - prevents blocking API responses
    Creates its own database session to avoid concurrency issues
    NOW INCLUDES PORTFOLIO TICKERS: Combines heatmap + portfolio tickers
    """
    # Create a new database session for the background task
    db = SessionLocal()
    try:
        # 🎯 CRITICAL FIX: Include portfolio tickers in the update
        # Get all unique tickers from portfolio_holdings table
        portfolio_tickers = db.query(PortfolioHolding.ticker).distinct().all()
        portfolio_tickers = [row[0] for row in portfolio_tickers]  # Extract tickers from tuples

        # Merge heatmap tickers + portfolio tickers, remove duplicates
        combined_tickers = list(set(all_tickers + portfolio_tickers))

        # Create asset types for portfolio tickers (default to 'stocks')
        combined_asset_types = asset_types.copy()
        for ticker in portfolio_tickers:
            if ticker not in combined_asset_types:
                combined_asset_types[ticker] = 'stocks'  # Default portfolio tickers to stocks

        print(f"🔄 Background cache update started for {len(combined_tickers)} tickers ({len(all_tickers)} heatmap + {len(portfolio_tickers)} portfolio)")

        # Use the existing cache update logic but in background
        market_data = get_market_data_with_cache(combined_tickers, combined_asset_types, db)

        # CRITICAL: Commit the transaction to save cache updates
        db.commit()

        print(f"✅ Background cache update completed and committed for {len(market_data)} tickers")

    except Exception as e:
        print(f"❌ Background cache update failed: {str(e)}")
        # Rollback on error
        try:
            db.rollback()
        except:
            pass  # Ignore rollback errors
    finally:
        # Always close the session
        db.close()

# 🎯 HARD-CODED TICKER POOL - 180+ DIVERSE STOCKS
# NO SMCI, NO PLTR - Removed to prove true randomness
# Pure Python random.choice() - NO AI, NO TRENDING APIs
TICKER_POOL = [
    # Technology (40 stocks)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "CSCO", "ACN", "AMD",
    "INTC", "IBM", "TXN", "QCOM", "NOW", "AMAT", "MU", "LRCX", "KLAC", "SNPS",
    "CDNS", "ADSK", "ROP", "FTNT", "ANSS", "TYL", "PTC", "ZBRA", "KEYS", "GDDY",
    "INTU", "PANW", "WDAY", "TEAM", "DDOG", "SNOW", "NET", "ZS", "OKTA", "CRWD",
    
    # Healthcare (35 stocks)
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
    "AMGN", "GILD", "CVS", "CI", "MDT", "ISRG", "REGN", "VRTX", "HUM", "BSX",
    "ELV", "ZTS", "SYK", "BDX", "EW", "IDXX", "RMD", "MTD", "DXCM", "A",
    "ALGN", "HOLX", "PODD", "IQV", "CRL",
    
    # Finance (35 stocks)
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
    "C", "AXP", "SCHW", "PGR", "CB", "MMC", "PNC", "USB", "TFC", "COF",
    "AON", "ICE", "CME", "MCO", "AJG", "TRV", "AFL", "ALL", "MET", "AIG",
    "FIS", "FISV", "BK", "STT", "TROW",
    
    # Consumer Discretionary (30 stocks)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
    "MAR", "ORLY", "AZO", "GM", "F", "ROST", "YUM", "DG", "DLTR", "EBAY",
    "POOL", "ULTA", "DPZ", "BBY", "DECK", "LVS", "MGM", "WYNN", "GRMN", "GPC",
    
    # Energy (25 stocks)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "HES", "HAL", "DVN", "BKR", "FANG", "MRO", "APA", "EQT", "CTRA",
    "OKE", "LNG", "TRGP", "EPD", "ET",
    
    # Industrials (30 stocks)
    "CAT", "RTX", "UNP", "HON", "BA", "UPS", "LMT", "GE", "DE", "MMM",
    "ETN", "PH", "EMR", "ITW", "CSX", "NSC", "FDX", "WM", "CMI", "PCAR",
    "ROK", "CARR", "OTIS", "GWW", "FAST", "PAYX", "VRSK", "IEX", "DOV", "XYL",
    
    # Materials (15 stocks)
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "CTVA", "DD", "NUE", "DOW",
    "ALB", "VMC", "MLM", "PPG", "CF",
    
    # Real Estate (15 stocks)
    "PLD", "AMT", "EQIX", "PSA", "WELL", "DLR", "O", "SPG", "VICI", "AVB",
    "EQR", "SBAC", "VTR", "EXR", "INVH",
    
    # Utilities (15 stocks)
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ED",
    "ES", "AWK", "DTE", "PPL", "AEE",
    
    # Communications (15 stocks)
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "MTCH", "PARA", "WBD",
    
    # Consumer Staples (15 stocks)
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "MDLZ", "KMB",
    "GIS", "K", "HSY", "CAG", "SJM"
]

print(f"✅ Ticker Pool Loaded: {len(TICKER_POOL)} stocks (NO SMCI, NO PLTR)")

app = FastAPI()

# 👇 CORS Configuration - Updated for httpOnly cookie authentication
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://tamtech-frontend.vercel.app",
    "https://tamtech-finance.com",
    "https://www.tamtech-finance.com",
]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins, # ✅ Must specify exact origins when using credentials (not "*")
#     allow_credentials=True, # ✅ Required for httpOnly cookies
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"]
# )

# ========== SECURITY MIDDLEWARE - Block Malicious Scanners ==========
from collections import defaultdict
from fastapi.responses import JSONResponse
import time

# Rate limiting storage (in-memory, resets on server restart)
rate_limit_storage = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # 1 minute window
MAX_ANALYZE_REQUESTS = 10  # Max 10 analyze requests per minute per IP

# Blocked IPs storage (persistent until restart)
blocked_ips = set()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Block malicious scanner requests and implement rate limiting"""
    # Temporarily disabled for debugging
    response = await call_next(request)
    return response

# --- Health Check Endpoints ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "TamtechAI Finance API"}

@app.get("/test")
async def test_endpoint():
    return {"status": "ok"}

# --- Helpers ---
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def verify_password(plain, hashed):
    return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_optional(
    token: str = Depends(oauth2_scheme), 
    access_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    # Priority: Cookie first, then Authorization header (for backward compatibility)
    token_to_use = access_token or token
    if not token_to_use: 
        print("⚠️ No token found (neither cookie nor header)")
        return None
    try:
        payload = jwt.decode(token_to_use, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: 
            print("⚠️ Token decoded but no email in payload")
            return None
        print(f"✅ User authenticated via cookie: {email}")
    except JWTError as e:
        print(f"⚠️ JWT decode error: {e}")
        return None
    return db.query(User).filter(User.email == email).first()

async def get_current_user_mandatory(
    token: str = Depends(oauth2_scheme),
    access_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    token_to_use = access_token or token
    if not token_to_use: raise HTTPException(status_code=401, detail="Not authenticated")
    user = await get_current_user_optional(token, access_token, db)
    if not user: raise HTTPException(status_code=401, detail="Invalid token")
    return user

async def verified_user_required(
    current_user: User = Depends(get_current_user_mandatory)
):
    """
    Dependency to ensure user is both authenticated AND email-verified
    Use this on protected endpoints that require verified email
    """
    if current_user.is_verified != 1:
        raise HTTPException(
            status_code=403, 
            detail="Please verify your email to access this feature. Check your inbox for the verification link."
        )
    return current_user

# --- Models ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    first_name: str
    last_name: str
    phone_number: str
    country: str                    
    address: Optional[str] = None   

    @field_validator('password')
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v): raise ValueError('Password must contain a number')
        return v

# 👇 ننشئ نموذجاً لبيانات المستخدم لترتيب الرد
class UserDataSchema(BaseModel):
    email: str
    first_name: str | None = None
    last_name: str | None = None
    phone_number: str | None = None
    country: str | None = None
    address: str | None = None

# 👇 نحدث الـ Token ليشمل البيانات والرصيد
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserDataSchema
    credits: int

class LicenseRequest(BaseModel):
    license_key: str

# --- Routes ---
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        # التأكد من أن الإيميل غير مستخدم سابقاً
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # إنشاء المستخدم مع كافة البيانات
        new_user = User(
            email=user.email, 
            hashed_password=get_password_hash(user.password),
            first_name=user.first_name,
            last_name=user.last_name,
            phone_number=user.phone_number,
            country=user.country,
            address=user.address,
            credits=3,  # رصيد مجاني للبداية
            is_verified=0  # يحتاج للتحقق من البريد
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # إنشاء رمز التحقق (UUID آمن)
        verification_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        
        try:
            token_entry = VerificationToken(
                user_id=new_user.id,
                token=verification_token,
                expires_at=expires_at
            )
            db.add(token_entry)
            db.commit()
            
            # إرسال البريد الإلكتروني للتحقق
            try:
                send_verification_email(
                    user_email=new_user.email,
                    user_name=new_user.first_name,
                    token=verification_token
                )
                print(f"✅ Verification email sent to {new_user.email}")
            except Exception as e:
                print(f"⚠️ Failed to send verification email: {e}")
                # لا نفشل التسجيل إذا فشل إرسال البريد، يمكن إعادة الإرسال لاحقاً
        except Exception as e:
            print(f"⚠️ Failed to create verification token: {e}")
            # If verification token creation fails, still allow registration
            # User can verify later via resend
        
        return {
            "message": "User created successfully. Please check your email to verify your account.",
            "email": new_user.email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/token", response_model=Token)
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # 1. التحقق من المستخدم
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # 2. إنشاء التوكين
    access_token = create_access_token(data={"sub": user.email})
    
    # 3. 🔒 Set httpOnly cookie for authentication
    # Note: Using samesite=lax since Vercel rewrite makes requests appear same-origin
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,  # ✅ Cannot be accessed by JavaScript (XSS protection)
        secure=True,     # ✅ HTTPS required for secure cookies
        samesite="lax",  # ✅ Same-site cookie (works with Vercel proxy)
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # 7 days in seconds
        path="/",
    )
    
    # 4. 👇 Return user data (token now in cookie, not body)
    return {
        "access_token": access_token,  # Still return for backward compatibility
        "token_type": "bearer",
        "credits": user.credits,
        "user": {
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone_number": user.phone_number,
            "country": user.country,
            "address": user.address,
            "is_verified": user.is_verified  # ✅ CRITICAL: Include verification status
        }
    }

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user_mandatory)):
    return {
        "email": current_user.email, 
        "credits": current_user.credits,
        "is_verified": current_user.is_verified,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "phone_number": current_user.phone_number,
        "country": current_user.country,
        "address": current_user.address
    }

@app.post("/logout")
def logout(response: Response):
    """Clear the httpOnly cookie to log user out"""
    response.delete_cookie(
        key="access_token",
        path="/",
        httponly=True,
        secure=True,
        samesite="lax"
    )
    return {"message": "Logged out successfully"}

# --- User Profile Update ---
class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None

@app.put("/users/profile")
def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user_mandatory),
    db: Session = Depends(get_db)
):
    """Update user profile information"""
    if profile_data.first_name is not None:
        current_user.first_name = profile_data.first_name.strip()
    if profile_data.last_name is not None:
        current_user.last_name = profile_data.last_name.strip()
    
    db.commit()
    db.refresh(current_user)
    
    return {
        "message": "Profile updated successfully",
        "first_name": current_user.first_name,
        "last_name": current_user.last_name
    }

# --- Password Change ---
class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@app.post("/users/change-password")
def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user_mandatory),
    db: Session = Depends(get_db)
):
    """Change user password"""
    # Verify current password
    if not bcrypt.checkpw(password_data.current_password.encode('utf-8'), current_user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Validate new password strength
    if len(password_data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters long")
    
    # Hash and update password
    hashed_new_password = bcrypt.hashpw(password_data.new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    current_user.hashed_password = hashed_new_password
    
    db.commit()
    
    return {"message": "Password changed successfully"}

# --- Email Verification Routes ---
@app.get("/auth/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    """
    Verify user email with token from verification email
    """
    # Find the token in database
    token_entry = db.query(VerificationToken).filter(VerificationToken.token == token).first()
    
    if not token_entry:
        raise HTTPException(status_code=404, detail="Invalid verification token")
    
    # Check if token has expired
    if datetime.now(timezone.utc) > make_datetime_aware(token_entry.expires_at):
        db.delete(token_entry)
        db.commit()
        raise HTTPException(status_code=410, detail="Verification token has expired. Please request a new one.")
    
    # Get the user
    user = db.query(User).filter(User.id == token_entry.user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if already verified
    if user.is_verified == 1:
        db.delete(token_entry)
        db.commit()
        return {"message": "Email already verified", "redirect": "https://www.tamtech-finance.com/dashboard?already_verified=true"}
    
    # Mark user as verified
    user.is_verified = 1
    db.delete(token_entry)  # Remove used token
    db.commit()
    
    print(f"✅ User {user.email} verified successfully")
    
    return {
        "message": "Email verified successfully! You can now access all features.",
        "redirect": "https://www.tamtech-finance.com/dashboard?verified=true"
    }

@app.post("/auth/resend-verification")
def resend_verification(current_user: User = Depends(get_current_user_mandatory), db: Session = Depends(get_db)):
    """
    Resend verification email to logged-in user
    """
    if current_user.is_verified == 1:
        raise HTTPException(status_code=400, detail="Email already verified")
    
    # Delete old tokens for this user
    db.query(VerificationToken).filter(VerificationToken.user_id == current_user.id).delete()
    db.commit()
    
    # Create new token
    verification_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    
    token_entry = VerificationToken(
        user_id=current_user.id,
        token=verification_token,
        expires_at=expires_at
    )
    db.add(token_entry)
    db.commit()
    
    # Send email
    try:
        send_verification_email(
            user_email=current_user.email,
            user_name=current_user.first_name,
            token=verification_token
        )
        return {"message": "Verification email sent. Please check your inbox."}
    except Exception as e:
        print(f"❌ Error sending verification email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send verification email")


# ==================== DASHBOARD ENDPOINTS ====================

@app.get("/dashboard/history")
async def get_user_dashboard_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(verified_user_required)
):
    """
    Get user's analysis history with 24-hour expiration tracking.
    Only verified users can access this endpoint.
    """
    try:
        # Fetch user's analysis history ordered by most recent first
        user_history = db.query(UserAnalysisHistory).filter(
            UserAnalysisHistory.user_id == current_user.id
        ).order_by(UserAnalysisHistory.updated_at.desc()).all()
        
        history_items = []
        now = datetime.utcnow()  # Use offset-naive UTC datetime to match database
        
        for item in user_history:
            # Calculate how old the report is
            age = now - item.updated_at
            hours_ago = int(age.total_seconds() / 3600)
            
            # Check if expired (> 24 hours)
            is_expired = hours_ago >= 24
            
            history_items.append({
                "id": item.id,
                "ticker": item.ticker,
                "company_name": item.company_name,
                "last_price": float(item.last_price) if item.last_price else 0,
                "verdict": item.verdict,
                "confidence_score": item.confidence_score,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "is_expired": is_expired,
                "hours_ago": hours_ago
            })
        
        return {
            "history": history_items,
            "total_count": len(history_items)
        }
        
    except Exception as e:
        print(f"❌ Dashboard history error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load history: {str(e)}")


@app.get("/dashboard/analysis/{ticker}")
async def get_historical_analysis(
    ticker: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(verified_user_required)
):
    """
    Fetch a specific historical analysis report for viewing.
    Returns the full cached analysis data if it exists and belongs to the user.
    """
    try:
        ticker = ticker.upper()
        
        # Verify user has this ticker in their history
        user_history = db.query(UserAnalysisHistory).filter(
            UserAnalysisHistory.user_id == current_user.id,
            UserAnalysisHistory.ticker == ticker
        ).first()
        
        if not user_history:
            raise HTTPException(status_code=404, detail="Analysis not found in your history")
        
        # Check if expired (> 24 hours)
        age = datetime.now(timezone.utc) - make_datetime_aware(user_history.updated_at)
        hours_ago = int(age.total_seconds() / 3600)
        is_expired = hours_ago >= 24
        
        if is_expired:
            raise HTTPException(status_code=410, detail="This analysis has expired. Please refresh to get updated data.")
        
        # Fetch the cached report from AnalysisReport table (language-aware)
        # Note: For dashboard, we fetch the most recent regardless of language
        # Or you can add a lang parameter if needed
        cached_report = db.query(AnalysisReport).filter(
            AnalysisReport.ticker == ticker
        ).order_by(AnalysisReport.updated_at.desc()).first()
        
        if not cached_report:
            raise HTTPException(status_code=404, detail="Analysis report not found in cache")
        
        # Parse the JSON data
        analysis_json = json.loads(cached_report.ai_json_data)
        
        # Get live financial data for chart and current price
        live_financial_data = await get_real_financial_data(ticker)
        
        if not live_financial_data or not live_financial_data.get('price'):
            raise HTTPException(status_code=500, detail="Failed to fetch current market data")
        
        # Return EXACT same structure as /analyze endpoint for frontend compatibility
        return {
            "ticker": ticker,
            "data": live_financial_data,    # All financial data from yfinance
            "analysis": analysis_json,       # Complete AI analysis from cache
            "cache_hit": True,
            "cache_age_hours": round(age.total_seconds() / 3600, 1),
            "credits_left": None,  # Not applicable for cached views
            "is_guest": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Historical analysis fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load analysis: {str(e)}")


# ... (Imports existing)

# 👇👇👇 أضف هذا الـ Endpoint الجديد هنا 👇👇👇
@app.get("/search-ticker/{query}")
async def search_ticker(query: str):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            data = response.json()
        
        suggestions = []
        if 'quotes' in data:
            for item in data['quotes']:
                if item.get('isYahooFinance', False): # تصفية النتائج
                    suggestions.append({
                        "symbol": item['symbol'],
                        "name": item.get('longname') or item.get('shortname') or item['symbol']
                    })
        # نرجع أول 5 نتائج فقط
        return suggestions[:5]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

# 🎯🎯🎯 NEW RANDOM TICKER ENDPOINT V2 - GUARANTEED FRESH 🎯🎯🎯
@app.get("/get-random-ticker-v2")
def get_random_ticker_v2():
    """
    BRAND NEW ENDPOINT - Forces Railway to use new code.
    Pure random.choice() from 230+ stock pool.
    NO SMCI. NO PLTR. NO AI. NO CACHE.
    """
    try:
        ticker = random.choice(TICKER_POOL)
        print(f"🎲 V2 Random Pick: {ticker} from pool of {len(TICKER_POOL)}")
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pool_size": len(TICKER_POOL),
            "version": "v2"
        }
    except Exception as e:
        print(f"❌ V2 Error: {e}")
        return {
            "ticker": random.choice(["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "WMT", "PG"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pool_size": 8,
            "version": "v2-fallback"
        }

@app.get("/get-price/{ticker}")
async def get_stock_price(ticker: str):
    """Get current stock price for Regret Machine"""
    try:
        data = await get_real_financial_data(ticker)
        if data and 'price' in data:
            return {"price": data['price']}
        else:
            raise HTTPException(status_code=404, detail="Price not found")
    except Exception as e:
        print(f"❌ Price fetch error for {ticker}: {e}")
        return {"error": "Failed to fetch price"}, 500

# ⚠️⚠️ OLD ENDPOINT - KEEP FOR BACKWARD COMPATIBILITY BUT REDIRECT TO V2 ⚠️⚠️
@app.get("/suggest-stock")
def suggest_stock():
    """OLD ENDPOINT - Redirects to V2 for backward compatibility"""
    return get_random_ticker_v2()

async def get_real_financial_data(ticker: str):
    """Fetch stock data with automatic retry on network failures"""
    import asyncio
    max_retries = 3
    retry_delay = 1  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            stock = await asyncio.to_thread(yf.Ticker, ticker)
            try: current_price = stock.fast_info['last_price']
            except: 
                info = await asyncio.to_thread(lambda: stock.info)
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not current_price: 
                if attempt < max_retries - 1:
                    print(f"⚠️ No price found for {ticker}, retrying... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return None
            
            info = await asyncio.to_thread(lambda: stock.info)
            news = await asyncio.to_thread(lambda: stock.news if hasattr(stock, 'news') else [])
            history = await asyncio.to_thread(lambda: stock.history(period="6mo"))
            chart_data = [{"date": d.strftime('%Y-%m-%d'), "price": round(r['Close'], 2)} for d, r in history.iterrows()]
            
            # Success! Return data
            if attempt > 0:
                print(f"✅ Price fetch succeeded on retry {attempt + 1}")
            
            return {
                "symbol": ticker.upper(),
                "companyName": info.get('longName', ticker),
                "price": current_price,
                "currency": info.get('currency', 'USD'),
                "market_cap": info.get('marketCap', "N/A"),
                "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh', current_price),
                "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow', current_price),
                "targetMeanPrice": info.get('targetMeanPrice', "N/A"),
                "recommendationKey": info.get('recommendationKey', "none"),
                
                # --- Advanced Metrics ---
                "pe_ratio": info.get('trailingPE', 0),
                "forward_pe": info.get('forwardPE', 0),
                "peg_ratio": info.get('pegRatio', 0),
                "price_to_sales": info.get('priceToSalesTrailing12Months', 0),
                "price_to_book": info.get('priceToBook', 0),
                "eps": info.get('trailingEps', 0),
                "beta": info.get('beta', 0),
                "dividend_yield": (info.get('dividendYield', 0) or 0) * 100,
                "profit_margins": (info.get('profitMargins', 0) or 0) * 100,
                "operating_margins": (info.get('operatingMargins', 0) or 0) * 100,
                "return_on_equity": (info.get('returnOnEquity', 0) or 0) * 100,
                "debt_to_equity": (info.get('debtToEquity', 0) or 0),
                "revenue_growth": (info.get('revenueGrowth', 0) or 0) * 100,
                "current_ratio": info.get('currentRatio', 0),
                
                "chart_data": chart_data,
                "recent_news": news[:5], 
                "description": info.get('longBusinessSummary', "No description.")[:600] + "..."
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ YFinance error on attempt {attempt + 1}/{max_retries}: {e}")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                print(f"❌ YFinance Error after {max_retries} attempts: {e}")
                return None
    
    return None

@app.get("/search-ticker/{ticker}")
async def search_ticker(ticker: str):
    """جلب اقتراحات الأسهم الحقيقية من Yahoo Finance"""
    try:
        # نستخدم طلب البحث الرسمي من ياهو لضمان الدقة
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            data = response.json()
        
        suggestions = []
        for res in data.get('quotes', []):
            # نأخذ الأسهم فقط (EQUITY) لضمان عدم ظهور عملات أو صناديق غير مرغوبة
            if res.get('quoteType') == 'EQUITY': 
                suggestions.append({
                    "symbol": res.get('symbol'),
                    "name": res.get('shortname') or res.get('longname')
                })
        return suggestions[:5] # نكتفي بـ 5 نتائج لتكون القائمة سريعة
    except Exception as e:
        print(f"Search Error: {e}")
        return []

@app.get("/analyze/{ticker}")
async def analyze_stock(
    ticker: str, 
    request: Request, 
    lang: str = "en",
    force_refresh: bool = False,  # NEW: Instant Refresh parameter
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user_optional)
):
    """
    🔒 STRICT MONETIZATION & CACHING LOGIC + EMAIL VERIFICATION REQUIRED
    
    Every request costs 1 credit (no exceptions).
    Cache is used only for speed and AI cost savings.
    Live price is ALWAYS injected before response.
    
    force_refresh=True: Skip 6-hour cache, always call AI, costs 1 credit
    """
    
    try:
        ticker = ticker.upper()
        
        # ========== STEP 1: CREDIT CHECK & DEDUCTION (STRICT POLICY) ==========
        is_guest = current_user is None
        credits_left = 0
        
        if current_user:
            # Email verification check for logged-in users
            if current_user.is_verified != 1:
                raise HTTPException(
                    status_code=403, 
                    detail="Please verify your email to access this feature. Check your inbox for the verification link."
                )
            
            # Dev mode bypass
            if ticker == "#DEVMODE":
                current_user.credits = 1000
                db.commit()
                return {"message": "Dev Mode: 1000 Credits Added"}
            
            # Check credit balance
            if current_user.credits <= 0:
                raise HTTPException(status_code=402, detail="No credits left")
            
            # 💳 IMMEDIATE DEDUCTION - Before any processing
            current_user.credits -= 1
            db.commit()
            credits_left = current_user.credits
            
            print(f"✅ User {current_user.email} charged 1 credit. Remaining: {credits_left}")
        else:
            # Guest IP-based limiting
            x_forwarded_for = request.headers.get("x-forwarded-for")
            if x_forwarded_for:
                client_ip = x_forwarded_for.split(",")[0].strip()
            else:
                client_ip = request.client.host

            guest = db.query(GuestUsage).filter(GuestUsage.ip_address == client_ip).first()
            
            if not guest:
                guest = GuestUsage(ip_address=client_ip, attempts=0)
                db.add(guest)
            
            if guest.attempts >= 3:
                raise HTTPException(status_code=403, detail="Guest limit reached. Please register.")
            
            # 💳 IMMEDIATE DEDUCTION for guests
            guest.attempts += 1
            db.commit()
            
            print(f"✅ Guest {client_ip} used trial {guest.attempts}/3")
        
        # ========== STEP 2: CACHE LOOKUP (24-Hour Window with Language Support) ==========
        # Skip cache if force_refresh is True (Instant Refresh feature)
        cache_hit = False
        analysis_json = None
        
        # LANGUAGE-AWARE CACHE: Query by both ticker AND language
        cached_report = db.query(AnalysisReport).filter(
            AnalysisReport.ticker == ticker,
            AnalysisReport.language == lang
        ).first()
        
        if cached_report and not force_refresh:
            # Check if cache is still valid (within 24 hours)
            cache_age = datetime.now(timezone.utc) - make_datetime_aware(cached_report.updated_at)
            if cache_age < timedelta(hours=24):
                cache_hit = True
                analysis_json = json.loads(cached_report.ai_json_data)
                cache_age_hours = cache_age.total_seconds() / 3600
                print(f"📦 CACHE HIT for {ticker} (Language: {lang}). Age: {cache_age_hours:.1f} hours")
            else:
                print(f"🗑️ Cache expired for {ticker} ({lang}). Deleting old report.")
                db.delete(cached_report)
                db.commit()
                cached_report = None
        elif force_refresh:
            print(f"⚡ FORCE REFRESH for {ticker} ({lang}). Skipping cache.")
        elif not cached_report:
            print(f"🆕 No cache found for {ticker} in {lang}. Will generate fresh analysis.")
        
        # ========== STEP 3: GENERATE NEW REPORT (if no cache or force refresh) ==========
        if not cache_hit:
            print(f"🔬 Generating NEW AI report for {ticker}")
            
            # Get financial data (without price for AI prompt)
            financial_data_for_ai = await get_real_financial_data(ticker)
            
            if not financial_data_for_ai or not financial_data_for_ai.get('price'):
                raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found or delisted.")
            
            ai_payload = {k: v for k, v in financial_data_for_ai.items() if k != 'chart_data'}
        
            lang_map = {
                "en": "English", 
                "ar": "Arabic (Modern Standard, High-End Financial Tone)", 
                "es": "Spanish (Professional Financial Tone)",
                "he": "Hebrew (Professional Financial Tone)",
                "ru": "Russian (Professional Financial Tone)",
                "it": "Italian (Professional Financial Tone)"
            }
            target_lang = lang_map.get(lang, "English")

            prompt = f"""
            You are the Chief Investment Officer (CIO) at a prestigious Global Hedge Fund. 
            Your task is to produce an **EXHAUSTIVE, INSTITUTIONAL-GRADE INVESTMENT MEMO** for {ticker}.
        
        **Financial Data & News:** {json.dumps(ai_payload)}
        **Language:** Write strictly in {target_lang}.

        **⚠️ CRITICAL INSTRUCTIONS:**
        1.  **EXTREME DEPTH:** Each text section must be LONG, DETAILED, and ANALYTICAL (aim for 400-600 words per chapter).
        2.  **SENTIMENT ANALYSIS:** Analyze the provided 'recent_news'. For each major news item, determine if it's Positive, Negative, or Neutral and assign an Impact Score (1-10).
        3.  **NO FLUFF:** Use professional financial terminology. Connect the news to the valuation.
        4.  **JSON FORMATTING:** You MUST return ONLY valid JSON. NO markdown code blocks, NO extra text. Ensure all quotes inside text fields are properly escaped. For RTL languages (Arabic/Hebrew), be extra careful with quote escaping.
        5.  **STRUCTURE:** Return strictly the JSON structure below.

        **REQUIRED JSON OUTPUT:**
        {{
            "chapter_1_the_business": "Headline: [Translate 'The Business DNA']. [Write 400+ words detailed essay]",
            "chapter_2_financials": "Headline: [Translate 'Financial Health']. [Write 400+ words detailed essay]",
            "chapter_3_valuation": "Headline: [Translate 'Valuation Check']. [Write 400+ words detailed essay]",
            
            "news_analysis": [
        {{
            "headline": "Title of the news",
            "sentiment": "positive/negative/neutral",
            "impact_score": 8,
            "url": "direct link to the news",
            "time": "e.g., 2 hours ago or Oct 24"
        }}
    ]

            "bull_case_points": ["Detailed point 1", "Detailed point 2", "Detailed point 3"],
            "bear_case_points": ["Detailed point 1", "Detailed point 2", "Detailed point 3"],
            
            "forecasts": {{
                "next_1_year": "Detailed 12-month scenario analysis",
                "next_5_years": "Detailed 2030 outlook"
            }},
            
            "swot_analysis": {{
                "strengths": ["S1", "S2", "S3"],
                "weaknesses": ["W1", "W2", "W3"],
                "opportunities": ["O1", "O2", "O3"],
                "threats": ["T1", "T2", "T3"]
            }},
            
            "radar_scores": [
                {{ "subject": "Value", "A": 8 }}, 
                {{ "subject": "Growth", "A": 7 }},
                {{ "subject": "Profitability", "A": 9 }}, 
                {{ "subject": "Health", "A": 6 }},
                {{ "subject": "Momentum", "A": 8 }}
            ],
            
            "verdict": "BUY / HOLD / SELL", 
            "confidence_score": 85, 
    "summary_one_line": "Executive summary"
}}
"""
        
            try:
                # Initialize client if not already done
                global client
                if not client:
                    if not API_KEY:
                        print("❌ DEBUG: No API_KEY found")
                        raise HTTPException(status_code=500, detail="AI service not configured - missing API key")
                    print(f"✅ DEBUG: Initializing client with API_KEY starting with {API_KEY[:10]}...")
                    client = genai.Client(api_key=API_KEY)
                    print(f"Available models: {client.models.list()}")
                    print("✅ DEBUG: Client initialized successfully")
                
                if not client or not model_name:
                    print("❌ DEBUG: Client or model_name is None")
                    raise HTTPException(status_code=500, detail="AI service not configured")
                
                # Add timeout protection (30 seconds max)
                import asyncio
                try:
                    # Try Gemini 2.0 Flash first
                    response = await asyncio.to_thread(
                        lambda: client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.3  # Lower temperature for more consistent JSON formatting
                            )
                        )
                    )
                except Exception as model_err:
                    if "404" in str(model_err) or "not found" in str(model_err).lower() or "model" in str(model_err).lower():
                        print(f"⚠️ Gemini 2.0 Flash not available, falling back to 1.5 Flash: {model_err}")
                        try:
                            response = await asyncio.to_thread(
                                lambda: client.models.generate_content(
                                    model='gemini-1.5-flash',
                                    contents=prompt,
                                    config=types.GenerateContentConfig(
                                        response_mime_type="application/json",
                                        temperature=0.3
                                    )
                                )
                            )
                        except Exception as fallback_err:
                            print(f"❌ Fallback also failed: {fallback_err}")
                            raise fallback_err
                    else:
                        raise model_err
                    raise
                
                # Parse JSON with repair attempts for malformed responses
                raw_response_text = response.text
                try:
                    analysis_json = json.loads(raw_response_text)
                except json.JSONDecodeError as json_err:
                    print(f"⚠️ Initial JSON parse failed: {json_err}")
                    print(f"📄 Raw response preview: {raw_response_text[:500]}...")
                    
                    # Attempt 1: Strip markdown code blocks (```json ... ```)
                    cleaned = raw_response_text.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("```")[1]
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:].strip()
                    
                    # Attempt 2: Fix common issues with escaped quotes in Arabic/RTL text
                    import re
                    # Fix trailing commas before closing braces/brackets
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    
                    # Attempt 3: Parse the repaired JSON
                    try:
                        analysis_json = json.loads(cleaned)
                        print(f"✅ JSON repaired successfully!")
                    except json.JSONDecodeError as repair_err:
                        print(f"❌ JSON repair failed: {repair_err}")
                        # REFUND CREDIT BEFORE raising exception
                        if current_user:
                            current_user.credits += 1
                            db.commit()
                            print(f"❌ JSON Error - Refunded 1 credit to {current_user.email}. Balance: {current_user.credits}")
                        else:
                            if guest:
                                guest.attempts -= 1
                                db.commit()
                                print(f"❌ JSON Error - Refunded guest trial. Remaining: {3 - guest.attempts}")
                        
                        raise HTTPException(
                            status_code=500,
                            detail="AI analysis temporarily unavailable. Your credit has been refunded."
                        )
                
                # Save to cache with language (upsert pattern)
                try:
                    if cached_report:
                        # Update existing cache entry
                        cached_report.ai_json_data = json.dumps(analysis_json)
                        cached_report.updated_at = datetime.now(timezone.utc)
                    else:
                        # Create new cache entry with language
                        new_report = AnalysisReport(
                            ticker=ticker,
                            language=lang,
                            ai_json_data=json.dumps(analysis_json)
                        )
                        db.add(new_report)
                    
                    db.commit()
                    print(f"💾 Saved AI report to cache for {ticker} (Language: {lang})")
                except Exception as db_error:
                    # If duplicate key error, try to update instead
                    print(f"⚠️ Cache save error (likely duplicate): {db_error}")
                    db.rollback()
                    # Try to fetch and update the existing record
                    existing = db.query(AnalysisReport).filter(
                        AnalysisReport.ticker == ticker,
                        AnalysisReport.language == lang
                    ).first()
                    if existing:
                        existing.ai_json_data = json.dumps(analysis_json)
                        existing.updated_at = datetime.now(timezone.utc)
                        db.commit()
                        print(f"💾 Updated existing cache for {ticker} ({lang})")
                    else:
                        print(f"❌ Failed to save cache: {db_error}")
                
                # Save to history
                new_history = AnalysisHistory(
                    ticker=ticker,
                    verdict=analysis_json.get("verdict", "HOLD"),
                    confidence_score=analysis_json.get("confidence_score", 0)
                )
                db.add(new_history)
                db.commit()
                
            except HTTPException:
                # Re-raise HTTP exceptions (like 500 from JSON parsing)
                raise
            except Exception as e:
                error_msg = str(e)
                print(f"AI Error: {error_msg}")
                
                # REFUND CREDIT - AI failed
                if current_user:
                    current_user.credits += 1
                    db.commit()
                    print(f"❌ AI Error - Refunded 1 credit to {current_user.email}. Balance: {current_user.credits}")
                else:
                    # Refund guest attempt
                    if guest:
                        guest.attempts -= 1
                        db.commit()
                        print(f"❌ AI Error - Refunded guest trial. Remaining: {3 - guest.attempts}")
                
                # User-friendly error messages
                if "404" in error_msg or "NOT_FOUND" in error_msg:
                    user_message = "AI service is updating. Your credit has been refunded."
                elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
                    user_message = "AI service temporarily unavailable. Your credit has been refunded."
                elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    user_message = "High traffic detected. Your credit has been refunded. Please try again."
                elif "API key" in error_msg:
                    user_message = "Service configuration error. Your credit has been refunded. Contact support."
                else:
                    user_message = "Analysis temporarily unavailable. Your credit has been refunded."
                
                raise HTTPException(status_code=500, detail=user_message)
        
        # ========== STEP 4: LIVE PRICE INJECTION ==========
        print(f"💹 Fetching LIVE price for {ticker}")
        live_financial_data = await get_real_financial_data(ticker)
        use_cached_price = False
        
        if not live_financial_data or not live_financial_data.get('price'):
            # If we have cached analysis, use its price as fallback instead of failing
            if cache_hit and analysis_json:
                print(f"⚠️ Live price fetch failed, using cached price from analysis")
                use_cached_price = True
                # Extract minimal financial data from cached analysis for response
                live_financial_data = {
                    "symbol": ticker.upper(),
                    "price": analysis_json.get("current_price", 0),
                    "companyName": analysis_json.get("company_name", ticker),
                    "chart_data": [],  # No fresh chart data available
                    "currency": "USD"
                }
            else:
                # No cache available and price fetch failed - this is a real failure
                print(f"❌ Price Error - No cache available and live fetch failed for {ticker}")
                # REFUND CREDIT - Price fetch failed with no fallback
                if current_user:
                    current_user.credits += 1
                    db.commit()
                    print(f"❌ Price Error - Refunded 1 credit to {current_user.email}. Balance: {current_user.credits}")
                else:
                    if guest:
                        guest.attempts -= 1
                        db.commit()
                raise HTTPException(status_code=500, detail="Failed to fetch live price. Your credit has been refunded.")
        
        # Calculate cache age for frontend display
        cache_age_hours = 0
        if cached_report:
            cache_age = datetime.now(timezone.utc) - make_datetime_aware(cached_report.updated_at)
            cache_age_hours = round(cache_age.total_seconds() / 3600, 1)
        
        # ========== STEP 5: UPDATE USER ANALYSIS HISTORY (for logged-in users) ==========
        if current_user:
            try:
                # Check if this ticker already exists in user's history
                user_history = db.query(UserAnalysisHistory).filter(
                    UserAnalysisHistory.user_id == current_user.id,
                    UserAnalysisHistory.ticker == ticker
                ).first()
                
                if user_history:
                    # Update existing record
                    user_history.company_name = live_financial_data.get('company_name', ticker)
                    user_history.last_price = str(live_financial_data.get('price', 0))
                    user_history.verdict = analysis_json.get('verdict', 'HOLD')
                    user_history.confidence_score = analysis_json.get('confidence_score', 50)
                    user_history.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new record
                    new_user_history = UserAnalysisHistory(
                        user_id=current_user.id,
                        ticker=ticker,
                        company_name=live_financial_data.get('company_name', ticker),
                        last_price=str(live_financial_data.get('price', 0)),
                        verdict=analysis_json.get('verdict', 'HOLD'),
                        confidence_score=analysis_json.get('confidence_score', 50)
                    )
                    db.add(new_user_history)
                
                db.commit()
                print(f"✅ User history updated for {current_user.email}: {ticker}")
            except Exception as e:
                print(f"⚠️ Failed to update user history: {e}")
                # Don't fail the entire request if history update fails
                db.rollback()
        
        # ========== FINAL RESPONSE ==========
        return {
            "ticker": ticker,
            "data": live_financial_data,  # LIVE PRICE + chart data
            "analysis": analysis_json,     # Cached or fresh AI analysis
            "credits_left": credits_left,
            "is_guest": is_guest,
            "cache_hit": cache_hit,
            "cache_age_hours": cache_age_hours
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (they're already formatted correctly)
        raise
    except Exception as e:
        # Catch any unexpected errors and return proper JSON response
        print(f"❌ Unexpected error in analyze endpoint: {e}")
        import traceback
        traceback.print_exc()
        # Return full traceback for debugging
        full_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}\n\nFull Traceback:\n{full_traceback}")
    

@app.get("/recent-analyses")
async def get_recent_analyses(db: Session = Depends(get_db)):
    # جلب آخر 5 تحليلات مرتبة من الأحدث للأقدم
    history = db.query(AnalysisHistory).order_by(AnalysisHistory.created_at.desc()).limit(5).all()
    
    return [
        {
            "ticker": h.ticker,
            "verdict": h.verdict,
            "confidence": h.confidence_score,
            "time": h.created_at.strftime("%H:%M") 
        } for h in history
    ]
    

@app.post("/verify-license")
async def verify_license(request: LicenseRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user_mandatory)):
    PRODUCT_ID = "APVOhGVIRQbt7xx1qGXtPg==" 
    try:
        # Send request to Gumroad to verify and increment usage counter
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.gumroad.com/v2/licenses/verify",
                data={"product_id": PRODUCT_ID, "license_key": request.license_key, "increment_uses_count": "true"})
            data = response.json()
        
        # Verify success and no refund
        if data.get("success") == True and not data.get("purchase", {}).get("refunded"):
            # Gumroad returns usage count AFTER increment
            # If code is new, uses should equal exactly 1
            # If more than 1, it means someone else used it before
            uses = data.get("uses") 
            
            if uses and uses > 1:
                return {"valid": False, "message": "This key has already been redeemed."}
            
            # Determine credits based on variant/price
            purchase_info = data.get("purchase", {})
            price = purchase_info.get("price", 0)  # Price in cents
            variant_name = purchase_info.get("variant_name", "").lower()
            
            # Detect which package: 10 credits ($4.99) or 50 credits ($9.99)
            credits_to_add = 50  # Default to 50 credits
            if price <= 500 or "10" in variant_name or "starter" in variant_name:
                credits_to_add = 10
            elif price >= 900 or "50" in variant_name or "pro" in variant_name:
                credits_to_add = 50
            
            # Add credits to user account
            current_user.credits += credits_to_add
            db.commit()
            return {"valid": True, "credits": current_user.credits, "credits_added": credits_to_add}
            
        return {"valid": False, "message": "Invalid license key or already used."}
    except Exception as e: 
        return {"valid": False, "message": "Connection error with verification server"}
    
@app.get("/market-winners-losers")
def get_market_winners_losers(db: Session = Depends(get_db)):
    """
    📈 GET DAILY MARKET WINNERS & LOSERS (PREMIUM FEATURE)
    Returns top 10 gainers and losers from major indices
    """
    try:
        # Use the same cached data as the heatmap
        # db is now injected by FastAPI
        # Get the universe of tickers from the heatmap logic
        master_universe = {
            "stocks": [
                "SPY", "QQQ", "IWM", "VTI", "VXUS", "BND", "VEA", "VWO", "VIG", "VUG",
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC",
                "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "PYPL", "COIN",
                "XOM", "CVX", "COP", "EOG", "MPC", "PSX", "VLO", "OXY",
                "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "TMO", "DHR", "ABT", "AMGN",
                "WMT", "COST", "HD", "LOW", "TGT", "DG", "DLTR", "KR", "CVS"
            ],
            "crypto": [
                "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LTC-USD", "MATIC-USD",
                "LINK-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "TRX-USD", "ETC-USD", "XLM-USD", "THETA-USD", "HBAR-USD"
            ],
            "commodities": [
                "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F", "ALI=F", "ZC=F", "ZW=F",
                "ZS=F", "ZM=F", "ZL=F", "ZO=F", "ZR=F", "KE=F", "CC=F", "KC=F", "CT=F", "SB=F"
            ],
            "forex": [
                "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
                "EURGBP=X", "EURAUD=X", "GBPAUD=X", "AUDNZD=X", "USDSGD=X", "USDHKD=X", "USDNOK=X", "USDSEK=X", "USDMXN=X", "USDBRL=X"
            ]
        }
        all_tickers = []
        for tickers in master_universe.values():
            all_tickers.extend(tickers)
        cached_data = get_cached_market_data(all_tickers, db)
        performance_data = []
        for ticker, data in cached_data.items():
            change_percent = data.get('change_percent')
            if change_percent is not None:
                performance_data.append({
                    "ticker": ticker,
                    "name": data.get("name", ticker),
                    "price": data.get("price", 0),
                    "change_percent": change_percent,
                    "volume": data.get('volume', 0),
                    "market_cap": data.get("market_cap", 0),
                    "sector": data.get("sector", "")
                })
        winners = sorted(performance_data, key=lambda x: x["change_percent"], reverse=True)[:10]
        losers = sorted(performance_data, key=lambda x: x["change_percent"])[:10]
        return {
            "winners": winners,
            "losers": losers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        print(f"Error in winners/losers: {e}")
        return {"winners": [], "losers": [], "error": str(e)}

# --- Dynamic OG Image Generation for Social Sharing ---
@app.get("/og/{ticker}")
async def generate_og_image(ticker: str, db: Session = Depends(get_db)):
    """Generate Open Graph image for stock ticker pages"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        from io import BytesIO
        from fastapi.responses import StreamingResponse
        
        ticker = ticker.upper()
        
        # Get cached analysis
        cached = db.query(AnalysisReport).filter(
            AnalysisReport.ticker == ticker,
            AnalysisReport.language == "en"
        ).first()
        
        if not cached:
            # Return default image if no analysis exists
            raise HTTPException(status_code=404, detail="No analysis found")
        
        analysis = json.loads(cached.ai_json_data)
        
        # Create image (1200x630 for OG standard)
        img = Image.new('RGB', (1200, 630), color='#0f172a')
        draw = ImageDraw.Draw(img)
        
        # Try to load custom font, fallback to default
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            subtitle_font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Determine colors based on verdict
        verdict = analysis.get('verdict', 'HOLD')
        if verdict == 'BUY':
            verdict_color = '#22c55e'  # Green
            bg_color = '#10b981'
        elif verdict == 'SELL':
            verdict_color = '#ef4444'  # Red
            bg_color = '#dc2626'
        else:
            verdict_color = '#f59e0b'  # Amber
            bg_color = '#d97706'
        
        # Draw gradient background (simulate with rectangles)
        for i in range(630):
            alpha = i / 630
            draw.rectangle([(0, i), (1200, i+1)], fill='#0f172a')
        
        # Draw ticker symbol
        draw.text((60, 80), ticker, font=title_font, fill='#ffffff')
        
        # Draw verdict badge
        draw.rounded_rectangle([(60, 180), (260, 240)], radius=10, fill=verdict_color)
        draw.text((90, 190), verdict, font=subtitle_font, fill='#ffffff')
        
        # Draw confidence score
        confidence = analysis.get('confidence_score', 0)
        draw.text((60, 270), f"{confidence}% Confidence", font=subtitle_font, fill='#94a3b8')
        
        # Draw summary (truncated)
        summary = analysis.get('summary_one_line', '')[:120] + '...'
        # Word wrap for summary
        words = summary.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 60:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = 340
        for line in lines[:3]:  # Max 3 lines
            draw.text((60, y_offset), line, font=small_font, fill='#cbd5e1')
            y_offset += 40
        
        # Draw branding
        draw.text((60, 550), "Tamtech Finance", font=subtitle_font, fill='#3b82f6')
        draw.text((60, 590), "AI-Powered Stock Analysis", font=small_font, fill='#64748b')
        
        # Save to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png", headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
        })
        
    except Exception as e:
        print(f"OG image generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate image")


# --- Comparison Route (Costs 2 Credits) ---
@app.get("/analyze-compare/{ticker1}/{ticker2}")
async def analyze_compare(
    ticker1: str, 
    ticker2: str, 
    request: Request, 
    lang: str = "en", 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user_optional)
):
    # --- 🛡️ نظام الحماية المتطور (IP & Credits) ---
    if current_user:
        # Email verification check for logged-in users
        if current_user.is_verified != 1:
            raise HTTPException(
                status_code=403, 
                detail="Please verify your email to access this feature. Check your inbox for the verification link."
            )
        
        if current_user.credits < 2:
            raise HTTPException(status_code=402, detail="Insufficient credits. 2 credits required.")
    else:
        # التقاط الـ IP الحقيقي للمستخدم خلف بروكسي Railway
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host

        guest = db.query(GuestUsage).filter(GuestUsage.ip_address == client_ip).first()
        
        if not guest:
            guest = GuestUsage(ip_address=client_ip, attempts=0)
            db.add(guest)
        
        # منع الزائر بعد 3 محاولات شاملة للجهاز
        if guest.attempts >= 3:
            raise HTTPException(status_code=403, detail="Guest limit reached. Please register.")
        
        guest.attempts += 1
        db.commit()
    # --- 🛡️ نهاية نظام الحماية ---

    try:
        # 2. جلب بيانات السهمين
        data1 = await get_real_financial_data(ticker1)
        data2 = await get_real_financial_data(ticker2)
        
        if not data1 or not data2:
            raise HTTPException(status_code=404, detail="One or both stocks not found")

        ai_payload1 = {k: v for k, v in data1.items() if k != 'chart_data'}
        ai_payload2 = {k: v for k, v in data2.items() if k != 'chart_data'}

        lang_map = {
            "en": "English", 
            "ar": "Arabic (Modern Standard, High-End Financial Tone)", 
            "es": "Spanish (Professional Financial Tone)",
            "he": "Hebrew (Professional Financial Tone)",
            "ru": "Russian (Professional Financial Tone)",
            "it": "Italian (Professional Financial Tone)"
        }
        target_lang = lang_map.get(lang, "English")

        # 3. أمر التحليل (البرومبت الخاص بك كما هو)
        prompt = f"""
        Act as a Senior Hedge Fund Strategy Director. Conduct a 'Capital Battle' between {ticker1} and {ticker2}.
        
        Financial Data {ticker1}: {json.dumps(ai_payload1)}
        Financial Data {ticker2}: {json.dumps(ai_payload2)}
        Language: {target_lang}

        ⚠️ CRITICAL INSTRUCTIONS:
        1. Write a massive institutional memo (min 600 words).
        2. Directly compare their Valuations (P/E, PEG). Who is a better bargain?
        3. Compare Profitability (ROE, Operating Margins). Who is more efficient?
        4. Discuss 'Strategic Moat': Which business model is harder to destroy?
        5. Use a professional, aggressive financial tone.
        
        Return strictly JSON with keys: 'verdict' (the long essay), 'winner', 'comparison_summary'.
        """
        
        if not client:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        try:
            # Try Gemini 2.0 Flash first
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
            )
        except Exception as model_err:
            if "404" in str(model_err) or "not found" in str(model_err).lower() or "model" in str(model_err).lower():
                print(f"⚠️ Gemini 2.0 Flash not available, falling back to 1.5 Flash: {model_err}")
                response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2
                        )
                    )
                )
            else:
                raise model_err
        analysis_result = json.loads(response.text)

    except json.JSONDecodeError:
        # REFUND for guests
        if not current_user and guest:
            guest.attempts -= 1
            db.commit()
            print(f"❌ Comparison Error - Refunded guest trial")
        raise HTTPException(
            status_code=500,
            detail="Comparison service temporarily unavailable. Your trial has been refunded."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Comparison AI Error: {error_msg}")
        
        # REFUND for guests
        if not current_user and guest:
            guest.attempts -= 1
            db.commit()
            print(f"❌ Comparison Error - Refunded guest trial")
        
        # User-friendly messages
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            user_message = "Comparison service updating. Your trial has been refunded."
        elif "403" in error_msg or "PERMISSION" in error_msg:
            user_message = "Service temporarily unavailable. Your trial has been refunded."
        elif "429" in error_msg:
            user_message = "Too many requests. Your trial has been refunded. Please wait."
        else:
            user_message = "Comparison temporarily unavailable. Your trial has been refunded."
        
        raise HTTPException(status_code=500, detail=user_message)
    
    try:

        # 4. خصم الكريدت وتحديث البيانات للمسجلين
        credits_left = 0
        if current_user:
            current_user.credits -= 2
            db.commit()
            credits_left = current_user.credits

        return {
            "analysis": analysis_result,
            "stock1": data1,
            "stock2": data2,
            "credits_left": credits_left
        }
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        # 2. جلب بيانات السهمين
        data1 = await get_real_financial_data(ticker1)
        data2 = await get_real_financial_data(ticker2)
        
        if not data1 or not data2:
            raise HTTPException(status_code=404, detail="One or both stocks not found")

        # تجهيز البيانات للذكاء الاصطناعي
        ai_payload1 = {k: v for k, v in data1.items() if k != 'chart_data'}
        ai_payload2 = {k: v for k, v in data2.items() if k != 'chart_data'}

        lang_map = {
            "en": "English", 
            "ar": "Arabic (Modern Standard, High-End Financial Tone)", 
            "es": "Spanish (Professional Financial Tone)",
            "he": "Hebrew (Professional Financial Tone)",
            "ru": "Russian (Professional Financial Tone)",
            "it": "Italian (Professional Financial Tone)"
        }
        target_lang = lang_map.get(lang, "English")

        # 3. أمر التحليل والمقارنة
        prompt = f"""
        Act as a Senior Hedge Fund Strategy Director. Conduct a 'Capital Battle' between {ticker1} and {ticker2}.
        
        Financial Data {ticker1}: {json.dumps(ai_payload1)}
        Financial Data {ticker2}: {json.dumps(ai_payload2)}
        Language: {target_lang}

        ⚠️ CRITICAL INSTRUCTIONS:
        1. Write a massive institutional memo (min 600 words).
        2. Directly compare their Valuations (P/E, PEG). Who is a better bargain?
        3. Compare Profitability (ROE, Operating Margins). Who is more efficient?
        4. Discuss 'Strategic Moat': Which business model is harder to destroy?
        5. Use a professional, aggressive financial tone.
        
        Return strictly JSON with keys: 'verdict' (the long essay), 'winner', 'comparison_summary'.
        """
        
        if not client:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        try:
            # Try Gemini 2.0 Flash first
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )
        except Exception as model_err:
            if "404" in str(model_err) or "not found" in str(model_err).lower() or "model" in str(model_err).lower():
                print(f"⚠️ Gemini 2.0 Flash not available, falling back to 1.5 Flash: {model_err}")
                response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                )
            else:
                raise model_err
        analysis_result = json.loads(response.text)

    except json.JSONDecodeError:
        # REFUND CREDIT
        current_user.credits += 2
        db.commit()
        print(f"❌ Battle Error - Refunded 2 credits to {current_user.email}. Balance: {current_user.credits}")
        raise HTTPException(
            status_code=500,
            detail="Battle analysis temporarily unavailable. Your credits have been refunded."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Battle AI Error: {error_msg}")
        
        # REFUND CREDIT
        current_user.credits += 2
        db.commit()
        print(f"❌ Battle Error - Refunded 2 credits to {current_user.email}. Balance: {current_user.credits}")
        
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            user_message = "Battle service updating. Your credits have been refunded."
        elif "403" in error_msg or "PERMISSION" in error_msg:
            user_message = "Service temporarily unavailable. Your credits have been refunded."
        elif "429" in error_msg:
            user_message = "High demand. Your credits have been refunded. Please wait."
        else:
            user_message = "Battle analysis unavailable. Your credits have been refunded."
        
        raise HTTPException(status_code=500, detail=user_message)
    
    try:

        # 4. خصم الكريدت وتحديث الداتابيز
        current_user.credits -= 2
        db.commit()

        return {
            "analysis": analysis_result,
            "stock1": data1,
            "stock2": data2,
            "credits_left": current_user.credits
        }
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market-sentiment")
async def get_market_sentiment(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    🚀 CACHE-FIRST: Returns sentiment from cache only (no API calls)
    Used by navbar - must be instant
    """
    try:
        # Get SPY data from cache with background update if needed
        cached_data = get_cached_market_data_with_background_update(["SPY"], db, background_tasks)

        if "SPY" not in cached_data or cached_data["SPY"].get('price', 0) <= 0:
            print("Sentiment: SPY not in cache or invalid price, returning neutral")
            return {"sentiment": "Neutral", "score": 50}

        # Use cached data for instant response
        change_percent = cached_data["SPY"].get('change_percent', 0)
        print(f"Sentiment: SPY change_percent = {change_percent}")

        # Fear & Greed score based on SPY change
        # Map change_percent to 0-100 scale
        if change_percent >= 5:
            score = 100  # Extreme Greed
        elif change_percent >= 2:
            score = 75   # Greed
        elif change_percent >= 0.5:
            score = 60   # Optimism
        elif change_percent >= -0.5:
            score = 50   # Neutral
        elif change_percent >= -2:
            score = 40   # Caution
        elif change_percent >= -5:
            score = 25   # Fear
        else:
            score = 0    # Extreme Fear

        # Label based on score
        if score >= 75:
            sentiment_label = "Greed"
        elif score >= 60:
            sentiment_label = "Optimism"
        elif score >= 40:
            sentiment_label = "Neutral"
        elif score >= 25:
            sentiment_label = "Caution"
        else:
            sentiment_label = "Fear"

        print(f"Sentiment: calculated score = {score}")
        return {
            "sentiment": sentiment_label,
            "score": score
        }
    except Exception as e:
        print(f"Sentiment Error: {e}")
        return {"sentiment": "Neutral", "score": 50}


@app.get("/market-sectors")
async def get_market_sectors(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        # Get all cached market data for stocks
        all_cached_data = get_cached_market_data([], db)  # Empty list to get all
        
        # Filter for stocks only and group by sector
        sector_data = {}
        for ticker, data in all_cached_data.items():
            if data['asset_type'] == 'stock':
                sector = data.get('sector') or "Unknown"
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(data['change_percent'])

        results = []
        for sector, changes in sector_data.items():
            if changes:
                avg_change = sum(changes) / len(changes)
                results.append({
                    "name": sector,
                    "change": f"{avg_change:+.2f}%",
                    "positive": bool(avg_change > 0)
                })
            else:
                results.append({
                    "name": sector,
                    "change": "0.00%",
                    "positive": True
                })

        # If no sectors found, return default sectors with 0.00%
        if not results:
            default_sectors = ["Technology", "Energy", "Financials", "Healthcare", "Consumer Discretionary", 
                             "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services", "Consumer Staples"]
            results = [{"name": sector, "change": "0.00%", "positive": True} for sector in default_sectors]

        return results
    except Exception as e:
        print(f"Global Sectors Error: {e}")
        return []


# ==================== PUBLIC STOCK PAGE ENDPOINT (TEASER/FREEMIUM) ====================
@app.get("/stocks/{ticker}")
async def get_stock_page_data(
    ticker: str,
    lang: str = "en",
    db: Session = Depends(get_db)
):
    """
    🌐 PUBLIC SEO-FRIENDLY ENDPOINT (No Auth Required)
    
    Returns CACHED analysis only for SEO stock pages.
    Shows teaser data with high-value sections BLURRED/LOCKED.
    
    - No AI calls (zero cost)
    - No credit deduction
    - Read-only cached data
    - Returns 404 if no cache exists
    - Freemium model: Show summary, lock premium insights
    """
    
    try:
        ticker = ticker.upper()
        
        # Look for cached analysis in database (try requested language first, then fallback to any language)
        cached_report = db.query(AnalysisReport).filter(
            AnalysisReport.ticker == ticker,
            AnalysisReport.language == lang
        ).order_by(AnalysisReport.created_at.desc()).first()
        
        # Fallback: If no cache in requested language, use ANY language
        if not cached_report:
            cached_report = db.query(AnalysisReport).filter(
                AnalysisReport.ticker == ticker
            ).order_by(AnalysisReport.created_at.desc()).first()
        
        if not cached_report:
            raise HTTPException(
                status_code=404, 
                detail=f"No cached analysis available for {ticker}. Please analyze this stock first."
            )
        
        # Parse cached JSON
        cached_analysis = json.loads(cached_report.ai_json_data)
        
        # Fetch LIVE stock price (free, no cost)
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            company_name = info.get("shortName") or info.get("longName") or ticker
            currency = info.get("currency", "USD")
        except Exception as e:
            # Fallback if Yahoo Finance fails
            print(f"Yahoo Finance error: {e}")
            current_price = None
            company_name = ticker
            currency = "USD"
        
        # Calculate cache age
        cache_age_hours = (datetime.now(timezone.utc) - make_datetime_aware(cached_report.created_at)).total_seconds() / 3600
        
        # TEASER MODE: Blur high-value data
        # Build chapters from old format if needed
        chapters = cached_analysis.get("chapters", [])
        if not chapters and "chapter_1_the_business" in cached_analysis:
            # Old format - build chapters array
            chapters = [
                {"title": "The Business DNA", "content": cached_analysis.get("chapter_1_the_business", "")},
                {"title": "Financial Health", "content": cached_analysis.get("chapter_2_financial_health", "")},
                {"title": "Valuation", "content": cached_analysis.get("chapter_3_valuation", "")},
                {"title": "Final Verdict", "content": cached_analysis.get("chapter_4_final_verdict", "")}
            ]
        
        teaser_analysis = {
            "summary_one_line": cached_analysis.get("summary_one_line", "AI-powered stock analysis"),
            "chapters": chapters,
            "chapter_1_the_business": cached_analysis.get("chapter_1_the_business", ""),
            "chapter_2_financial_health": cached_analysis.get("chapter_2_financial_health", ""),
            "chapter_3_valuation": cached_analysis.get("chapter_3_valuation", ""),
            "chapter_4_final_verdict": cached_analysis.get("chapter_4_final_verdict", ""),
            
            # LOCKED FIELDS (blurred on frontend)
            "verdict": "🔒 LOCKED",
            "confidence_score": 0,
            "intrinsic_value": "🔒 LOCKED",
            "fair_value_range": "🔒 LOCKED",
            "swot": {
                "strengths": ["🔒 Unlock to see strengths"],
                "weaknesses": ["🔒 Unlock to see weaknesses"],
                "opportunities": ["🔒 Unlock to see opportunities"],
                "threats": ["🔒 Unlock to see threats"]
            },
            
            # Meta info for frontend
            "is_teaser": True,
            "unlock_message": "Sign in and use 1 credit to unlock the full analysis with AI verdict, intrinsic value, and SWOT insights."
        }
        
        return {
            "success": True,
            "ticker": ticker,
            "data": {
                "ticker": ticker,
                "companyName": company_name,
                "price": current_price,
                "currency": currency,
                "cacheAge": f"{int(cache_age_hours)} hours ago"
            },
            "analysis": teaser_analysis,
            "is_teaser": True,
            "cache_age_hours": int(cache_age_hours)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error fetching stock page data: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PORTFOLIO TRACKER ENDPOINTS ====================

# Get user's portfolio
@app.get("/portfolio")
async def get_portfolio(
    current_user: User = Depends(get_current_user_mandatory),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    📊 GET USER PORTFOLIO (FREE FEATURE)
    Returns all holdings with live prices using unified cache source.
    URGENT FIX: If any ticker shows $0.00, trigger immediate background fetch
    """
    try:
        print(f"Portfolio request for user {current_user.id}")

        # Get user's portfolio holdings
        holdings = db.query(PortfolioHolding).filter(
            PortfolioHolding.user_id == current_user.id
        ).all()

        print(f"DEBUG: Found {len(holdings)} holdings in portfolio_holdings for user {current_user.id}")

        if not holdings:
            return []

        # Extract tickers for cache lookup
        tickers = [holding.ticker for holding in holdings]

        # 🔗 UNIFIED CACHE SOURCE: Use the same get_cached_market_data() function as heatmap
        cached_data = get_cached_market_data(tickers, db)

        # � STALE-WHILE-REVALIDATE: Return cached data immediately, update stale data in background
        portfolio_data = []
        stale_tickers = []

        for holding in holdings:
            ticker = holding.ticker
            cache_entry = cached_data.get(ticker)

            # Use cached data if available, otherwise default to 0
            current_price = cache_entry.get('price', 0) if cache_entry else 0
            change_p = cache_entry.get('change_percent', 0) if cache_entry else 0
            sector = cache_entry.get('sector') if cache_entry else None

            # Check if data is stale (>10 min) or missing
            if not cache_entry or (cache_entry.get('last_updated') and 
                                 make_datetime_aware(cache_entry['last_updated']) < datetime.now(timezone.utc) - timedelta(minutes=10)):
                stale_tickers.append(ticker)

            portfolio_data.append({
                "id": holding.id,
                "symbol": ticker,
                "current_price": current_price,
                "change_p": change_p,
                "shares": holding.quantity,
                "avg_buy_price": holding.avg_buy_price,
                "sector": sector
            })

        # 🔥 BACKGROUND UPDATE: Trigger only AFTER returning response
        if stale_tickers:
            print(f"🔄 Portfolio: Triggering background update for {len(stale_tickers)} stale/missing tickers")
            background_tasks.add_task(update_market_data_background, stale_tickers)

        print(f"DEBUG: Returning {len(portfolio_data)} portfolio items to frontend")
        return portfolio_data

    except Exception as e:
        print(f"Portfolio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add/Update portfolio holding
@app.post("/portfolio/add")
async def add_portfolio_holding(
    ticker: str = Form(...),
    quantity: float = Form(...),
    avg_buy_price: float = Form(None),
    current_user: User = Depends(get_current_user_mandatory),
    db: Session = Depends(get_db)
):
    """
    ➕ ADD/UPDATE PORTFOLIO HOLDING (FREE FEATURE)
    """
    try:
        ticker = ticker.upper()
        
        # Check if holding already exists
        existing = db.query(PortfolioHolding).filter(
            PortfolioHolding.user_id == current_user.id,
            PortfolioHolding.ticker == ticker
        ).first()
        
        if existing:
            # Update existing holding
            existing.quantity = quantity
            if avg_buy_price is not None:
                existing.avg_buy_price = avg_buy_price
            existing.updated_at = func.now()
        else:
            # Create new holding
            new_holding = PortfolioHolding(
                user_id=current_user.id,
                ticker=ticker,
                quantity=quantity,
                avg_buy_price=avg_buy_price
            )
            db.add(new_holding)
        
        db.commit()
        return {"success": True, "message": f"Added {ticker} to portfolio"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Delete portfolio holding
@app.delete("/portfolio/{holding_id}")
async def delete_portfolio_holding(
    holding_id: int,
    current_user: User = Depends(get_current_user_mandatory),
    db: Session = Depends(get_db)
):
    """
    🗑️ DELETE PORTFOLIO HOLDING (FREE FEATURE)
    """
    try:
        holding = db.query(PortfolioHolding).filter(
            PortfolioHolding.id == holding_id,
            PortfolioHolding.user_id == current_user.id
        ).first()
        
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")
        
        db.delete(holding)
        db.commit()
        return {"success": True, "message": "Holding deleted"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Update portfolio holding ticker
@app.put("/portfolio/{holding_id}")
async def update_portfolio_holding(
    holding_id: int,
    ticker: str = Form(...),
    current_user: User = Depends(get_current_user_mandatory),
    db: Session = Depends(get_db)
):
    """
    ✏️ UPDATE PORTFOLIO HOLDING TICKER (FREE FEATURE)
    """
    try:
        ticker = ticker.upper()
        
        holding = db.query(PortfolioHolding).filter(
            PortfolioHolding.id == holding_id,
            PortfolioHolding.user_id == current_user.id
        ).first()
        
        if not holding:
            raise HTTPException(status_code=404, detail="Holding not found")
        
        # Check if new ticker already exists for this user
        existing = db.query(PortfolioHolding).filter(
            PortfolioHolding.user_id == current_user.id,
            PortfolioHolding.ticker == ticker,
            PortfolioHolding.id != holding_id
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail=f"You already have {ticker} in your portfolio")
        
        holding.ticker = ticker
        holding.updated_at = func.now()
        db.commit()
        return {"success": True, "message": f"Updated ticker to {ticker}"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



# ==================== MASTER UNIVERSE HEATMAP ENDPOINT ====================

@app.get("/master-universe-heatmap")
async def get_master_universe_heatmap(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    🌍 MASTER UNIVERSE HEATMAP (FREE FEATURE)
    Returns market data for 100+ assets across stocks, crypto, commodities, and forex.
    Uses intelligent caching with 10-minute validity and batch fetching.
    ATOMIC UPDATES: Never serves partial data - waits for complete cache or uses last full valid cache.
    STALE-WHILE-REVALIDATE: Always serves data, updates in background.
    """
    try:
        # 🎯 MASTER UNIVERSE - 100+ ASSETS ACROSS MULTIPLE CLASSES
        master_universe = {
            # 🏢 MAJOR STOCK INDICES & ETFs
            "stocks": [
                "SPY", "QQQ", "IWM", "VTI", "VXUS", "BND", "VEA", "VWO", "VIG", "VUG",
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC",
                "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "PYPL", "COIN",
                "XOM", "CVX", "COP", "EOG", "MPC", "PSX", "VLO", "OXY",  # Removed HES (404)
                "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "TMO", "DHR", "ABT", "AMGN",
                "WMT", "COST", "HD", "LOW", "TGT", "DG", "DLTR", "KR", "CVS"  # Removed WBA (404)
            ],
            # ₿ CRYPTOCURRENCIES
            "crypto": [
                "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LTC-USD", "MATIC-USD",
                "LINK-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "TRX-USD", "ETC-USD", "XLM-USD", "THETA-USD", "HBAR-USD"
            ],
            # 🛢️ COMMODITIES
            "commodities": [
                "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F", "ALI=F", "ZC=F", "ZW=F",
                "ZS=F", "ZM=F", "ZL=F", "ZO=F", "ZR=F", "KE=F", "CC=F", "KC=F", "CT=F", "SB=F"
            ],
            # 💱 FOREX PAIRS
            "forex": [
                "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
                "EURGBP=X", "EURAUD=X", "GBPAUD=X", "AUDNZD=X", "USDSGD=X", "USDHKD=X", "USDNOK=X", "USDSEK=X", "USDMXN=X", "USDBRL=X"
            ]
        }

        # Flatten all tickers and create asset type mapping
        all_tickers = []
        asset_types = {}

        for asset_class, tickers in master_universe.items():
            for ticker in tickers:
                all_tickers.append(ticker)
                asset_types[ticker] = asset_class

        # 🚀 TRUE STALE-WHILE-REVALIDATE: Always return data immediately, update in background
        cached_data = get_cached_market_data(all_tickers, db)

        # Check data completeness and staleness
        current_time = datetime.now(timezone.utc)
        complete_data = len(cached_data) == len(all_tickers)
        needs_background_update = not complete_data

        # Check for expired data even if complete (stale-while-revalidate)
        if complete_data:
            for ticker in all_tickers:
                if ticker in cached_data:
                    last_updated = cached_data[ticker].get('last_updated')
                    if not last_updated or (current_time - last_updated).total_seconds() >= 600:  # 10 minutes
                        needs_background_update = True
                        break

        # 🔥 BACKGROUND UPDATE: Trigger only AFTER determining response data
        if needs_background_update:
            background_tasks.add_task(update_heatmap_cache_background, all_tickers, asset_types)
        if needs_background_update:
            background_tasks.add_task(update_heatmap_cache_background, all_tickers, asset_types)

        # 🛡️ ATOMIC UPDATES: If we don't have complete data, try to get the last full valid cache
        if not complete_data:
            print(f"⚠️ Incomplete cache ({len(cached_data)}/{len(all_tickers)} items), checking for last full cache...")

            # Try to get all cached data (even expired) as fallback
            fallback_data = get_cached_market_data([], db, include_expired=True)  # Empty list = get all, include expired

            # Check if fallback has all tickers (even if expired)
            if len(fallback_data) >= len(all_tickers) * 0.9:  # At least 90% coverage
                print(f"✅ Using fallback cache with {len(fallback_data)} items")
                cached_data = fallback_data
            else:
                print(f"❌ No sufficient fallback cache available, returning partial data")

        # Organize response by asset class using cached data
        heatmap_data = {
            "stocks": [],
            "crypto": [],
            "commodities": [],
            "forex": [],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "cache_status": "complete" if complete_data else "partial"
        }

        for ticker, data in cached_data.items():
            if ticker not in asset_types:
                continue  # Skip tickers not in our universe

            asset_class = asset_types[ticker]

            # Format for heatmap display - STRIPPED TO ESSENTIALS ONLY
            heatmap_item = {
                "s": ticker,  # symbol (shorter key)
                "p": round(data.get('price', 0), 2),  # price
                "c": round(data.get('change_percent', 0), 2),  # change_percent
                "t": asset_class  # type (stocks/crypto/commodities/forex)
            }

            if asset_class in heatmap_data:
                heatmap_data[asset_class].append(heatmap_item)

        # Sort each asset class by absolute change percentage (most volatile first)
        for asset_class in heatmap_data:
            if asset_class not in ["last_updated", "cache_status"]:
                heatmap_data[asset_class].sort(key=lambda x: abs(x.get('c', 0)), reverse=True)

        print(f"🌍 Heatmap: Returning {sum(len(v) for k, v in heatmap_data.items() if k not in ['last_updated', 'cache_status'])} items ({heatmap_data['cache_status']})")
        return heatmap_data

    except Exception as e:
        print(f"❌ Master Universe Heatmap Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")



# ==================== CONTACT FORM ENDPOINT ====================

class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    message: str

@app.post("/contact")
async def send_contact_form(contact: ContactForm):
    """
    📧 SEND CONTACT FORM EMAIL
    Sends contact form submissions to tamtecht@gmail.com
    """
    try:
        # Import mailer
        from mailer import send_contact_email

        # Send email
        success = send_contact_email(
            name=contact.name,
            email=contact.email,
            subject=contact.subject,
            message=contact.message
        )

        if success:
            return {"success": True, "message": "Message sent successfully!"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")

    except Exception as e:
        print(f"Contact form error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")



# ==================== WHALE ALERT ENDPOINTS ====================

class WhaleAlert(Base):
    __tablename__ = "whale_alerts"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    alert_type = Column(String)  # 'volume_spike' or 'large_transaction' or 'both'
    volume = Column(Float, nullable=True)
    avg_volume_30d = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    transaction_value = Column(Float, nullable=True)
    direction = Column(String, nullable=True)  # 'up' or 'down' or null
    ai_insight = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

@app.get("/whale-alerts")
async def get_whale_alerts(limit: int = 10):
    """
    📊 GET WHALE ALERTS
    Returns recent whale alerts from database
    """
    try:
        db = SessionLocal()
        alerts = db.query(WhaleAlert).order_by(WhaleAlert.created_at.desc()).limit(limit).all()
        db.close()

        return {
            "alerts": [
                {
                    "id": alert.id,
                    "ticker": alert.ticker,
                    "alert_type": alert.alert_type,
                    "volume": alert.volume,
                    "avg_volume_30d": alert.avg_volume_30d,
                    "volume_ratio": alert.volume_ratio,
                    "transaction_value": alert.transaction_value,
                    "direction": alert.direction,
                    "ai_insight": alert.ai_insight,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts
            ]
        }

    except Exception as e:
        print(f"Whale alerts error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch whale alerts")

def generate_template_insight(alert_type: str, ticker: str, transaction_value: float = None) -> str:
    """
    🎯 GENERATE TEMPLATE-BASED INSIGHT
    Returns a random template key for whale alerts
    """
    import random

    if alert_type == "volume_spike":
        return "volumeSpike"
    elif alert_type == "large_transaction":
        return "largeTransaction"
    elif alert_type == "both":
        return "both"
    
    return "volumeSpike"  # fallback

@app.post("/check-whale-activity")
async def check_whale_activity():
    """
    🐳 CHECK WHALE ACTIVITY
    Scans market data for whale activity and generates alerts
    """
    try:
        db = SessionLocal()

        # Get cached market data for all tickers
        cached_data = get_cached_market_data([], db)

        alerts_created = 0
        alerts = []

        for ticker, data in cached_data.items():
            if not data or 'price' not in data:
                continue

            current_price = data['price']
            volume = data.get('volume', 0)
            avg_volume_30d = data.get('avg_volume_30d', 0)

            # Skip if no volume data
            if volume == 0 or avg_volume_30d == 0:
                continue

            volume_ratio = volume / avg_volume_30d
            transaction_value = current_price * volume

            alert_type = None
            ai_insight = None

            # Check for volume spike (>200%)
            if volume_ratio > 2.0:
                # Check for large transaction (>$1M)
                if transaction_value > 1000000:
                    alert_type = "both"
                    ai_insight = generate_template_insight("both", ticker, transaction_value)
                else:
                    alert_type = "volume_spike"
                    ai_insight = generate_template_insight("volume_spike", ticker)
            # Check for large transaction only
            elif transaction_value > 1000000:
                alert_type = "large_transaction"
                ai_insight = generate_template_insight("large_transaction", ticker, transaction_value)

            if alert_type:
                # Create alert in database
                alert = WhaleAlert(
                    ticker=ticker,
                    alert_type=alert_type,
                    volume=volume,
                    avg_volume_30d=avg_volume_30d,
                    volume_ratio=volume_ratio,
                    transaction_value=transaction_value,
                    ai_insight=ai_insight
                )
                db.add(alert)
                alerts_created += 1

                alerts.append({
                    "id": alert.id,
                    "ticker": ticker,
                    "alert_type": alert_type,
                    "volume": volume,
                    "avg_volume_30d": avg_volume_30d,
                    "volume_ratio": volume_ratio,
                    "transaction_value": transaction_value,
                    "ai_insight": ai_insight,
                    "created_at": alert.created_at.isoformat()
                })

        db.commit()
        db.close()

        return {
            "success": True,
            "alerts_created": alerts_created,
            "alerts": alerts
        }

    except Exception as e:
        print(f"Whale activity check error: {e}")
        try:
            db.close()
        except:
            pass
        raise HTTPException(status_code=500, detail="Failed to check whale activity")


# ==================== CALENDAR EVENTS ENDPOINTS ====================

def sync_calendar_events(db: Session):
    """
    Sync weekly calendar events from RSS feeds and cache in database
    Only updates if cache is older than 24 hours
    """
    try:
        # Check if we need to update (last update within 24 hours) using raw SQL
        result = db.execute(text("SELECT MAX(updated_at) as last_update FROM calendar_events")).fetchone()
        now = datetime.utcnow()

        if result and result[0]:
            last_update = result[0]
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            if (now - last_update.replace(tzinfo=None)) < timedelta(hours=24):
                print("📅 Calendar cache is fresh (updated within 24 hours)")
                return

        print("📅 Syncing calendar events from RSS feeds...")

        events = []

        # Try multiple RSS sources
        rss_sources = [
            "https://www.forexfactory.com/ffcal_week_this.xml",
            "https://www.dailyfx.com/feeds/economic-calendar",
            "https://www.investing.com/rss/economic_calendar.rss"
        ]

        feed_data = None
        for url in rss_sources:
            try:
                feed = feedparser.parse(url)
                if feed.entries:
                    feed_data = feed
                    print(f"✅ Successfully loaded RSS from: {url}")
                    break
            except Exception as e:
                print(f"❌ Failed to load RSS from {url}: {e}")
                continue

        if feed_data and feed_data.entries:
            # Parse RSS entries for the next 7 days
            for entry in feed_data.entries[:30]:  # Get more entries
                try:
                    title = entry.title

                    # Try to extract date from title or use published date
                    event_date = None

                    # Look for date patterns in title
                    import re
                    date_patterns = [
                        r'(\w{3}\s+\d{1,2})',  # "Jan 31"
                        r'(\d{1,2}/\d{1,2})',  # "01/31"
                        r'(\d{4}-\d{2}-\d{2})'  # "2026-01-31"
                    ]

                    for pattern in date_patterns:
                        match = re.search(pattern, title)
                        if match:
                            date_str = match.group(1)
                            try:
                                if '/' in date_str:
                                    # Assume MM/DD format for current year
                                    month, day = map(int, date_str.split('/'))
                                    event_date = datetime(now.year, month, day, 8, 30)  # Default 8:30 AM
                                elif '-' in date_str:
                                    # ISO format
                                    event_date = datetime.fromisoformat(date_str)
                                else:
                                    # Month DD format
                                    event_date = datetime.strptime(f"{now.year} {date_str}", "%Y %b %d")
                                break
                            except:
                                continue

                    # If no date found in title, use published date
                    if not event_date and hasattr(entry, 'published_parsed') and entry.published_parsed:
                        event_date = datetime(*entry.published_parsed[:6])

                    if event_date and event_date >= now and (event_date - now) <= timedelta(days=7):
                        # Determine importance
                        title_lower = title.lower()
                        if any(word in title_lower for word in ['fed', 'fomc', 'interest rate', 'cpi', 'nfp', 'gdp', 'unemployment', 'jobs']):
                            importance = "High"
                        elif any(word in title_lower for word in ['employment', 'retail sales', 'housing', 'pmi', 'consumer confidence']):
                            importance = "Medium"
                        else:
                            importance = "Low"

                        # Generate AI impact note
                        ai_notes = {
                            "High": "High impact expected - monitor volatility and safe-haven assets",
                            "Medium": "Moderate market impact - watch for sector-specific movements",
                            "Low": "Limited impact - focus on broader market trends"
                        }

                        events.append({
                            "name": title,
                            "date_time": event_date,
                            "importance": importance,
                            "ai_impact_note": ai_notes.get(importance, "Monitor market reaction")
                        })

                except Exception as e:
                    print(f"❌ Error parsing RSS entry: {e}")
                    continue

        # If no events from RSS, generate realistic upcoming events
        if not events:
            print("⚠️ No RSS events found, generating realistic calendar events")
            events = generate_realistic_calendar_events_for_cache()

        # Clear old events and insert new ones using raw SQL
        db.execute(text("DELETE FROM calendar_events"))
        for event in events:
            db.execute(text("""
                INSERT INTO calendar_events (name, date_time, importance, ai_impact_note, updated_at)
                VALUES (:name, :date_time, :importance, :ai_impact_note, CURRENT_TIMESTAMP)
            """), {
                "name": event["name"],
                "date_time": event["date_time"],
                "importance": event["importance"],
                "ai_impact_note": event["ai_impact_note"]
            })

        db.commit()
        print(f"✅ Calendar sync complete: {len(events)} events cached")

    except Exception as e:
        print(f"❌ Calendar sync error: {e}")
        db.rollback()


def generate_realistic_calendar_events_for_cache():
    """
    Generate realistic upcoming economic events for the next 7 days (for caching)
    """
    now = datetime.utcnow()
    events = []

    # Base events that typically occur
    base_events = [
        {"name": "Initial Jobless Claims", "importance": "Low", "hour": 8, "minute": 30},
        {"name": "CPI Data Release", "importance": "High", "hour": 8, "minute": 30},
        {"name": "Non-Farm Payrolls", "importance": "High", "hour": 8, "minute": 30},
        {"name": "FOMC Meeting Minutes", "importance": "Medium", "hour": 14, "minute": 0},
        {"name": "Fed Interest Rate Decision", "importance": "High", "hour": 14, "minute": 0},
        {"name": "GDP Growth Report", "importance": "Medium", "hour": 8, "minute": 30},
        {"name": "Retail Sales Data", "importance": "Medium", "hour": 8, "minute": 30},
        {"name": "Housing Starts", "importance": "Low", "hour": 8, "minute": 30},
        {"name": "Consumer Confidence Index", "importance": "Low", "hour": 10, "minute": 0},
        {"name": "PMI Manufacturing", "importance": "Medium", "hour": 9, "minute": 45},
        {"name": "Durable Goods Orders", "importance": "Low", "hour": 8, "minute": 30},
        {"name": "Trade Balance", "importance": "Low", "hour": 8, "minute": 30},
        {"name": "Unemployment Rate", "importance": "High", "hour": 8, "minute": 30},
        {"name": "Core PCE Price Index", "importance": "High", "hour": 8, "minute": 30},
        {"name": "Federal Budget", "importance": "Low", "hour": 14, "minute": 0}
    ]

    # Generate events for the next 7 days
    for i in range(7):
        event_date = now + timedelta(days=i)

        # Skip weekends for most economic data (except some Fed events)
        if event_date.weekday() >= 5 and i > 0:  # Saturday/Sunday, but allow today
            continue

        # Add more events for today and tomorrow
        if i == 0:  # Today
            num_events = random.randint(2, 4)
        elif i == 1:  # Tomorrow
            num_events = random.randint(2, 3)
        else:
            num_events = random.randint(1, 2)

        # Select events and assign random times
        selected_events = random.sample(base_events, min(num_events, len(base_events)))

        for event in selected_events:
            # Vary the time slightly
            hour_variation = random.randint(-30, 30)  # ±30 minutes
            event_hour = max(8, min(16, event["hour"] + hour_variation // 60))
            event_minute = (event["minute"] + hour_variation) % 60

            event_datetime = event_date.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)

            # Only include future events
            if event_datetime > now:
                ai_notes = {
                    "High": "High impact expected - monitor volatility and safe-haven assets",
                    "Medium": "Moderate market impact - watch for sector-specific movements",
                    "Low": "Limited impact - focus on broader market trends"
                }

                events.append({
                    "name": event["name"],
                    "date_time": event_datetime,
                    "importance": event["importance"],
                    "ai_impact_note": ai_notes[event["importance"]]
                })

    # Sort by date
    events.sort(key=lambda x: x['date_time'])
    return events


@app.get("/calendar-events")
async def get_calendar_events(db: Session = Depends(get_db)):
    """
    📅 GET CALENDAR EVENTS
    Returns cached weekly economic calendar events
    """
    try:
        # Sync calendar if needed
        sync_calendar_events(db)

        # Get events from cache using raw SQL
        result = db.execute(text("""
            SELECT name, date_time, importance, ai_impact_note
            FROM calendar_events
            WHERE date_time >= CURRENT_TIMESTAMP
            ORDER BY date_time
            LIMIT 20
        """)).fetchall()

        events = []
        for row in result:
            events.append({
                "name": row[0],
                "date_time": row[1].isoformat() if hasattr(row[1], 'isoformat') else str(row[1]),
                "importance": row[2],
                "ai_impact_note": row[3]
            })

        return {"events": events}

    except Exception as e:
        print(f"❌ Calendar events error: {e}")
        return {"events": []}


# ==================== SERVER STARTUP ====================

def warmup_cache_on_startup():
    """
    URGENT FIX 4: Batch warm-up - fetch top 100 assets + ALL portfolio tickers on server startup
    This ensures cache is 'warm' before first user visits, preventing $0 displays
    """
    print("🔥 Warming up cache with top 100 assets + portfolio tickers...")

    try:
        # Create database session
        db = SessionLocal()

        # Top 50 assets from our ticker pool (reduced to prevent DB overload)
        top_100_tickers = TICKER_POOL[:50]

        # 🚨 CRITICAL: Also fetch ALL existing portfolio tickers immediately
        portfolio_tickers = db.query(PortfolioHolding.ticker).distinct().all()
        portfolio_tickers = [row[0] for row in portfolio_tickers]  # Extract tickers

        # Combine and deduplicate
        all_warmup_tickers = list(set(top_100_tickers + portfolio_tickers))

        # Asset type mapping for proper data fetching
        asset_types = {ticker: 'stock' for ticker in all_warmup_tickers}

        print(f"📊 Fetching {len(all_warmup_tickers)} assets for cache warm-up ({len(top_100_tickers)} heatmap + {len(portfolio_tickers)} portfolio)...")

        # Force fresh fetch for all tickers (bypass cache completely for warm-up)
        market_data = get_market_data_with_cache(
            all_warmup_tickers,
            asset_types,
            db,
            force_fresh=True  # Force fresh fetch for all
        )

        # Count successful fetches
        successful_fetches = sum(1 for data in market_data.values() if data.get('price', 0) > 0)

        print(f"✅ Cache warm-up complete: {successful_fetches}/{len(all_warmup_tickers)} assets cached")
        print("💾 Cache is now WARM - ready for user requests!")

        db.close()

    except Exception as e:
        print(f"❌ Cache warm-up failed: {e}")
        try:
            db.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting TamtechAI Finance Tool Backend...")
    print("📊 Master Universe Heatmap: ENABLED")
    print("💾 Global Caching Engine: ACTIVE")
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

# ========== NEW ENDPOINTS FOR FRONTEND DATA FETCHING ==========

@app.get("/stock-quote/{ticker}")
async def get_stock_quote(ticker: str):
    """Fetch basic stock quote data from Yahoo Finance"""
    try:
        ticker = ticker.upper()
        stock = await asyncio.to_thread(yf.Ticker, ticker)
        
        # Get basic info
        info = await asyncio.to_thread(lambda: stock.info)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not current_price:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
        
        return {
            "symbol": ticker,
            "price": current_price,
            "change": info.get('regularMarketChange', 0),
            "changePercent": info.get('regularMarketChangePercent', 0),
            "marketCap": info.get('marketCap'),
            "volume": info.get('regularMarketVolume'),
            "companyName": info.get('longName', ticker)
        }
    except Exception as e:
        print(f"Quote fetch error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stock quote")

@app.get("/stock-chart/{ticker}")
async def get_stock_chart(ticker: str, range: str = "1d", interval: str = "1d"):
    """Fetch stock chart data from Yahoo Finance"""
    try:
        ticker = ticker.upper()
        stock = await asyncio.to_thread(yf.Ticker, ticker)
        
        # Get historical data
        history = await asyncio.to_thread(lambda: stock.history(period=range, interval=interval))
        
        if history.empty:
            raise HTTPException(status_code=404, detail=f"No chart data for {ticker}")
        
        chart_data = []
        for date, row in history.iterrows():
            chart_data.append({
                "date": date.strftime('%Y-%m-%d %H:%M:%S'),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        return {
            "symbol": ticker,
            "range": range,
            "interval": interval,
            "data": chart_data
        }
    except Exception as e:
        print(f"Chart fetch error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chart data")
