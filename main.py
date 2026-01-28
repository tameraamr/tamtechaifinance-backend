from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, Cookie, Form
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
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
import bcrypt
import re
import secrets
from mailer import send_verification_email

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

engine = create_engine(DATABASE_URL)

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
    last_attempt = Column(DateTime, default=datetime.utcnow)

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
    
    # Composite index for user_id + ticker (one row per user per ticker)
    __table_args__ = (
        Index('ix_user_ticker', 'user_id', 'ticker', unique=True),
    )

# 6. Portfolio Audits - Track AI audit history
class PortfolioAudit(Base):
    __tablename__ = "portfolio_audits"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    audit_json = Column(Text)  # Stores AI audit result as JSON
    portfolio_health_score = Column(Integer)  # 0-100 score
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
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
# ملاحظة: في السيرفر لا توقف التطبيق إذا لم تجد المفتاح فوراً، السيرفر سيحقنه
if not API_KEY: 
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
    client = None
    model_name = None
else:
    try:
        # Initialize new google.genai client
        client = genai.Client(api_key=API_KEY)
        # Use the latest, fastest model for real-time financial analysis
        # gemini-2.5-flash is the newest and most optimized model (January 2026)
        model_name = 'gemini-2.5-flash'
        print("✅ Gemini API initialized with gemini-2.5-flash")
    except Exception as e:
        print(f"❌ Error initializing Gemini client: {e}")
        client = None
        model_name = None

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
    "https://tamtech-frontend.vercel.app",
    "https://tamtech-finance.com",
    "https://www.tamtech-finance.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # ✅ Must specify exact origins when using credentials (not "*")
    allow_credentials=True, # ✅ Required for httpOnly cookies
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

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
    path = request.url.path.lower()
    
    # Get client IP
    x_forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.client.host
    
    # 1. Check if IP is permanently blocked
    if client_ip in blocked_ips:
        print(f"🚫 Blocked IP attempted access: {client_ip}")
        return JSONResponse(status_code=403, content={"detail": "Access denied"})
    
    # 2. Block malicious scanner paths
    malicious_patterns = [
        "wp-admin", "wp-login", "wordpress", ".env", "config.php", 
        "xmlrpc.php", ".git", "phpMyAdmin", "admin.php", "setup.php",
        "wp-config", "db_config", ".htaccess", "backup.sql"
    ]
    
    if any(pattern in path for pattern in malicious_patterns):
        print(f"🚨 SECURITY ALERT: Malicious scan detected from {client_ip} - Path: {path}")
        # Add to blocked IPs after 3 malicious attempts
        blocked_ips.add(client_ip)
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    
    # 3. Rate limiting for /analyze endpoint
    if "/analyze/" in path and request.method == "GET":
        current_time = time.time()
        # Clean old requests outside the time window
        rate_limit_storage[client_ip] = [
            timestamp for timestamp in rate_limit_storage[client_ip]
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
        
        # Check if limit exceeded
        if len(rate_limit_storage[client_ip]) >= MAX_ANALYZE_REQUESTS:
            print(f"⚠️ Rate limit exceeded for {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a minute and try again."}
            )
        
        # Add current request timestamp
        rate_limit_storage[client_ip].append(current_time)
    
    # Continue with request
    response = await call_next(request)
    return response

# --- Health Check Endpoints ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "TamtechAI Finance API"}

@app.get("/health")
async def health_check():
    """Check if API and Gemini are working"""
    gemini_status = "configured" if (client and model_name) else "not_configured"
    return {
        "status": "healthy",
        "gemini_api": gemini_status,
        "model": model_name if model_name else "none",
        "api_key_present": bool(API_KEY)
    }

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
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
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
    if datetime.utcnow() > token_entry.expires_at:
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
    expires_at = datetime.utcnow() + timedelta(hours=24)
    
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
        now = datetime.utcnow()
        
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
        age = datetime.utcnow() - user_history.updated_at
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
        live_financial_data = get_real_financial_data(ticker)
        
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
def search_ticker(query: str):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
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
            "timestamp": datetime.utcnow().isoformat(),
            "pool_size": len(TICKER_POOL),
            "version": "v2"
        }
    except Exception as e:
        print(f"❌ V2 Error: {e}")
        return {
            "ticker": random.choice(["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "WMT", "PG"]),
            "timestamp": datetime.utcnow().isoformat(),
            "pool_size": 8,
            "version": "v2-fallback"
        }

# ⚠️⚠️ OLD ENDPOINT - KEEP FOR BACKWARD COMPATIBILITY BUT REDIRECT TO V2 ⚠️⚠️
@app.get("/suggest-stock")
def suggest_stock():
    """OLD ENDPOINT - Redirects to V2 for backward compatibility"""
    return get_random_ticker_v2()

def get_real_financial_data(ticker: str):
    """Fetch stock data with automatic retry on network failures"""
    max_retries = 3
    retry_delay = 1  # Start with 1 second
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            try: current_price = stock.fast_info['last_price']
            except: 
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not current_price: 
                if attempt < max_retries - 1:
                    print(f"⚠️ No price found for {ticker}, retrying... (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return None
            
            info = stock.info
            news = stock.news if hasattr(stock, 'news') else []
            history = stock.history(period="6mo")
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
def search_ticker(ticker: str):
    """جلب اقتراحات الأسهم الحقيقية من Yahoo Finance"""
    try:
        # نستخدم طلب البحث الرسمي من ياهو لضمان الدقة
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
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
            cache_age = datetime.utcnow() - cached_report.updated_at
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
            financial_data_for_ai = get_real_financial_data(ticker)
            
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
                if not client or not model_name:
                    raise HTTPException(status_code=500, detail="AI service not configured")
                
                # Add timeout protection (30 seconds max)
                import asyncio
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.3  # Lower temperature for more consistent JSON formatting
                        )
                    )
                except Exception as timeout_err:
                    if "timeout" in str(timeout_err).lower():
                        raise Exception("Request timeout - AI took too long")
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
                        cached_report.updated_at = datetime.utcnow()
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
                        existing.updated_at = datetime.utcnow()
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
        live_financial_data = get_real_financial_data(ticker)
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
            cache_age = datetime.utcnow() - cached_report.updated_at
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
                    user_history.updated_at = datetime.utcnow()
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
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

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
        response = requests.post("https://api.gumroad.com/v2/licenses/verify",
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
    
@app.get("/market-pulse")
def get_market_pulse():
    try:
        # رموز ياهو فاينانس الرسمية
        tickers = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "NVDA": "NVIDIA",
            "BTC-USD": "Bitcoin",
            "GC=F": "GOLD"
        }
        
        pulse_data = []
        for sym, name in tickers.items():
            stock = yf.Ticker(sym)
            info = stock.fast_info
            current_price = info['last_price']
            
            # حساب نسبة التغيير
            prev_close = stock.info.get('previousClose', current_price)
            change = ((current_price - prev_close) / prev_close) * 100
            
            pulse_data.append({
                "name": name,
                "price": f"{current_price:,.2f}" if "Bitcoin" not in name else f"{current_price:,.0f}",
                "change": f"{'+' if change > 0 else ''}{change:.2f}%",
                "up": change > 0
            })
        return pulse_data
    except Exception as e:
        print(f"Error fetching pulse: {e}")
        return []

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
        data1 = get_real_financial_data(ticker1)
        data2 = get_real_financial_data(ticker2)
        
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
        
        if not client or not model_name:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
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
        data1 = get_real_financial_data(ticker1)
        data2 = get_real_financial_data(ticker2)
        
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
        
        if not client or not model_name:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
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
async def get_market_sentiment():
    try:
        # جلب بيانات 30 يوم للحسابات السريعة
        spy_ticker = yf.Ticker("SPY").history(period="30d")
        if spy_ticker.empty:
            return {"sentiment": "Neutral", "score": 50}
            
        # 1. حساب الزخم القصير (Current Price vs 20-day Average)
        ma20 = spy_ticker['Close'].rolling(window=20).mean().iloc[-1]
        current_spy = spy_ticker['Close'].iloc[-1]
        
        # 2. حساب الـ RSI المبسط (قوة الشراء والبيع)
        delta = spy_ticker['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
        
        if loss == 0:
            rsi = 100
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        # دمج الـ RSI مع الزخم ليعطي نتيجة واقعية
        # إذا الـ RSI عالي جداً (فوق 70) السوق في طمع (Greed)
        momentum_impact = ((current_spy - ma20) / ma20 * 100) * 10 # تكبير التأثير ليتحرك السكور
        final_score = (rsi * 0.7) + (momentum_impact * 0.3) + 20 
        
        # ضمان أن السكور بين 5 و 95
        final_score = max(5, min(95, int(final_score)))
        
        sentiment_label = "Neutral"
        if final_score > 70: sentiment_label = "Extreme Greed"
        elif final_score > 55: sentiment_label = "Greed"
        elif final_score < 30: sentiment_label = "Extreme Fear"
        elif final_score < 45: sentiment_label = "Fear"
        
        return {
            "sentiment": sentiment_label,
            "score": final_score
        }
    except Exception as e:
        print(f"Sentiment Error: {e}")
        return {"sentiment": "Neutral", "score": 50}


@app.get("/market-sectors")
async def get_market_sectors():
    try:
        # رموز الصناديق التي تمثل كامل قطاعات السوق الأمريكي (11 قطاعاً)
        sector_tickers = {
            "Technology": "XLK",
            "Energy": "XLE",
            "Financials": "XLF",
            "Healthcare": "XLV",
            "Consumer Disc": "XLY",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication": "XLC",
            "Consumer Staples": "XLP",
            "S&P 500 Index": "SPY",
        }
        results = []
        
        for name, sym in sector_tickers.items():
            try:
                stock = yf.Ticker(sym)
                hist = stock.history(period="2d")
                
                if hist is not None and not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    
                    if prev_price != 0:
                        change = ((current_price - prev_price) / prev_price) * 100
                        results.append({
                            "name": name,
                            "change": f"{change:+.2f}%",
                            "positive": bool(change > 0)
                        })
                else:
                    results.append({"name": name, "change": "0.00%", "positive": True})
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
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
        cache_age_hours = (datetime.utcnow() - cached_report.created_at).total_seconds() / 3600
        
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
    db: Session = Depends(get_db)
):
    """
    📊 GET USER PORTFOLIO (FREE FEATURE)
    Returns all holdings with live prices and P&L calculations.
    """
    try:
        holdings = db.query(PortfolioHolding).filter(
            PortfolioHolding.user_id == current_user.id
        ).all()
        
        portfolio_data = []
        total_value = 0
        total_cost = 0
        
        for holding in holdings:
            # Fetch live price
            price_error = False
            try:
                stock = yf.Ticker(holding.ticker)
                info = stock.info
                current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                
                # If price is 0, mark as error
                if current_price == 0:
                    price_error = True
                    
                company_name = info.get("shortName") or info.get("longName") or holding.ticker
            except:
                current_price = 0
                company_name = holding.ticker
                price_error = True
            
            # Calculate P&L
            market_value = current_price * holding.quantity
            cost_basis = (holding.avg_buy_price or 0) * holding.quantity
            pnl = market_value - cost_basis if holding.avg_buy_price else 0
            pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            total_value += market_value
            total_cost += cost_basis
            
            portfolio_data.append({
                "id": holding.id,
                "ticker": holding.ticker,
                "company_name": company_name,
                "quantity": holding.quantity,
                "avg_buy_price": holding.avg_buy_price,
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "price_error": price_error
            })
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "success": True,
            "holdings": portfolio_data,
            "summary": {
                "total_value": total_value,
                "total_cost": total_cost,
                "total_pnl": total_pnl,
                "total_pnl_percent": total_pnl_percent,
                "holdings_count": len(holdings)
            }
        }
    except Exception as e:
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


# AI Portfolio Audit (PREMIUM - 5 CREDITS)
@app.post("/portfolio/audit")
async def audit_portfolio(
    language: str = Form("en"),
    current_user: User = Depends(get_current_user_mandatory),
    db: Session = Depends(get_db)
):
    """
    🤖 AI PORTFOLIO AUDIT (PREMIUM FEATURE - 5 CREDITS)
    
    Analyzes entire portfolio for:
    - Diversification score
    - Correlation risks
    - Sector exposure
    - Overall portfolio health score
    - Personalized recommendations
    """
    try:
        # Email verification check
        if current_user.is_verified != 1:
            raise HTTPException(
                status_code=403,
                detail="Please verify your email to access this feature."
            )
        
        # Credit check - costs 5 credits
        if current_user.credits < 5:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient credits. Portfolio audit costs 5 credits. You have {current_user.credits} credits."
            )
        
        # Get all holdings
        holdings = db.query(PortfolioHolding).filter(
            PortfolioHolding.user_id == current_user.id
        ).all()
        
        if not holdings:
            raise HTTPException(
                status_code=400,
                detail="Your portfolio is empty. Add stocks first before running an audit."
            )
        
        # DEDUCT 5 CREDITS IMMEDIATELY
        current_user.credits -= 5
        db.commit()
        
        # Fetch live data for all holdings
        portfolio_summary = []
        total_value = 0
        
        for holding in holdings:
            try:
                stock = yf.Ticker(holding.ticker)
                info = stock.info
                current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")
                market_cap = info.get("marketCap", 0)
                
                market_value = current_price * holding.quantity
                total_value += market_value
                
                portfolio_summary.append({
                    "ticker": holding.ticker,
                    "quantity": holding.quantity,
                    "current_price": current_price,
                    "market_value": market_value,
                    "sector": sector,
                    "industry": industry,
                    "market_cap": market_cap
                })
            except:
                continue
        
        # Calculate weights
        for item in portfolio_summary:
            item["weight_percent"] = (item["market_value"] / total_value * 100) if total_value > 0 else 0
        
        # Build AI prompt
        prompt = f"""You are a professional portfolio analyst. Analyze this investment portfolio and provide a comprehensive audit.

PORTFOLIO HOLDINGS:
{json.dumps(portfolio_summary, indent=2)}

TOTAL PORTFOLIO VALUE: ${total_value:,.2f}

Provide your analysis in the following JSON format:
{{
  "portfolio_health_score": <0-100 integer>,
  "diversification_score": <0-100 integer>,
  "risk_level": "<LOW/MEDIUM/HIGH/VERY HIGH>",
  "summary": "<One paragraph overall assessment>",
  "sector_exposure": {{
    "<sector_name>": <percentage as float>
  }},
  "correlations": [
    {{"pair": ["TICKER1", "TICKER2"], "correlation": "<HIGH/MEDIUM/LOW>", "risk": "<explanation>"}}
  ],
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
  "rebalancing_suggestions": [
    {{"action": "<BUY/SELL/HOLD>", "ticker": "<TICKER>", "reason": "<explanation>"}}
  ]
}}

Focus on:
1. Diversification across sectors and market caps
2. Correlation risks (e.g., tech-heavy portfolios)
3. Concentration risk (any single stock >25% is concerning)
4. Sector balance
5. Actionable rebalancing advice

IMPORTANT: Respond in {language} language. Use proper translations for all financial terms."""

        # Call Gemini API with client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )
        
        # Clean and parse response - robust handling of Gemini's response format
        response_text = response.text
        
        # Debug: Log raw response
        print(f"🔍 Raw Gemini response (first 500 chars): {repr(response_text[:500])}")
        print(f"🔍 Raw Gemini response length: {len(response_text)}")
        
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        if '```' in response_text:
            # Extract content between triple backticks
            parts = response_text.split('```')
            if len(parts) >= 3:
                # Get the middle part (between opening and closing ```)
                response_text = parts[1]
                # Remove language identifier if present (e.g., "json\n{...}")
                if response_text.strip().startswith('json'):
                    response_text = response_text.strip()[4:]
        
        # Remove all leading/trailing whitespace, newlines, and any non-JSON characters
        response_text = response_text.strip()
        
        # Find the first '{' and last '}' to extract pure JSON
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            print(f"❌ No JSON braces found in: {repr(response_text[:500])}")
            raise ValueError("No valid JSON object found in response")
        
        response_text = response_text[start_idx:end_idx+1]
        
        # Debug: Log cleaned JSON
        print(f"🔍 Cleaned JSON (first 500 chars): {repr(response_text[:500])}")
        print(f"🔍 Cleaned JSON length: {len(response_text)}")
        
        try:
            audit_result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"❌ Failed JSON string: {repr(response_text[:500])}")
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
        
        # Validate and sanitize audit result
        print(f"🔍 Parsed audit_result keys: {list(audit_result.keys())}")
        print(f"🔍 Strengths: {audit_result.get('strengths', 'NOT_FOUND')}")
        print(f"🔍 Weaknesses: {audit_result.get('weaknesses', 'NOT_FOUND')}")
        print(f"🔍 Recommendations: {audit_result.get('recommendations', 'NOT_FOUND')}")
        
        audit_result = {
            "portfolio_health_score": audit_result.get("portfolio_health_score", 0),
            "diversification_score": audit_result.get("diversification_score", 0),
            "risk_level": audit_result.get("risk_level", "UNKNOWN"),
            "summary": audit_result.get("summary", "Analysis completed but summary not available."),
            "sector_exposure": audit_result.get("sector_exposure", {}),
            "correlations": audit_result.get("correlations", []),
            "strengths": audit_result.get("strengths", []) if isinstance(audit_result.get("strengths"), list) else [],
            "weaknesses": audit_result.get("weaknesses", []) if isinstance(audit_result.get("weaknesses"), list) else [],
            "recommendations": audit_result.get("recommendations", []) if isinstance(audit_result.get("recommendations"), list) else [],
            "rebalancing_suggestions": audit_result.get("rebalancing_suggestions", []) if isinstance(audit_result.get("rebalancing_suggestions"), list) else []
        }
        
        # Save audit to database
        audit_record = PortfolioAudit(
            user_id=current_user.id,
            audit_json=json.dumps(audit_result),
            portfolio_health_score=audit_result.get("portfolio_health_score", 0)
        )
        db.add(audit_record)
        db.commit()
        
        return {
            "success": True,
            "credits_remaining": current_user.credits,
            "audit": audit_result,
            "portfolio_value": total_value,
            "holdings_count": len(holdings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Portfolio audit error: {e}")
        print(traceback.format_exc())
        
        # Refund credits on error
        current_user.credits += 5
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")