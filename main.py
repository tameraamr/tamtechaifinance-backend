from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, Cookie
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
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
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

# 2. جدول تخزين التقارير الكاملة (6-hour cache)
class AnalysisReport(Base):
    __tablename__ = "analysis_reports"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, unique=True)  # One report per ticker
    ai_json_data = Column(Text)  # Stores full AI analysis as JSON string (using Text for large data)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

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
    # Initialize new google.genai client
    client = genai.Client(api_key=API_KEY)
    # Use the latest, fastest model for real-time financial analysis
    # gemini-2.5-flash is the newest and most optimized model (January 2026)
    model_name = 'gemini-2.5-flash'

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
        
        # Fetch the cached report from AnalysisReport table
        cached_report = db.query(AnalysisReport).filter(AnalysisReport.ticker == ticker).first()
        
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
    try:
        stock = yf.Ticker(ticker)
        try: current_price = stock.fast_info['last_price']
        except: 
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not current_price: return None
        
        info = stock.info
        news = stock.news if hasattr(stock, 'news') else []
        history = stock.history(period="6mo")
        chart_data = [{"date": d.strftime('%Y-%m-%d'), "price": round(r['Close'], 2)} for d, r in history.iterrows()]
        
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
        print(f"YFinance Error: {e}")
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
        
        # ========== STEP 2: CACHE LOOKUP (6-Hour Window) ==========
        # Skip cache if force_refresh is True (Instant Refresh feature)
        cache_hit = False
        analysis_json = None
        cached_report = db.query(AnalysisReport).filter(AnalysisReport.ticker == ticker).first()
        
        if cached_report and not force_refresh:
            # Check if cache is still valid (within 24 hours)
            cache_age = datetime.utcnow() - cached_report.updated_at
            if cache_age < timedelta(hours=24):
                cache_hit = True
                analysis_json = json.loads(cached_report.ai_json_data)
                cache_age_hours = cache_age.total_seconds() / 3600
                print(f"📦 CACHE HIT for {ticker}. Age: {cache_age_hours:.1f} hours")
            else:
                print(f"🗑️ Cache expired for {ticker}. Deleting old report.")
                db.delete(cached_report)
                db.commit()
                cached_report = None
        elif force_refresh:
            print(f"⚡ FORCE REFRESH for {ticker}. Skipping cache.")
        
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
        4.  **FORMAT:** Return strictly the JSON structure below.

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
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                analysis_json = json.loads(response.text)
                
                # Save to cache (upsert pattern)
                if cached_report:
                    cached_report.ai_json_data = json.dumps(analysis_json)
                    cached_report.updated_at = datetime.utcnow()
                else:
                    new_report = AnalysisReport(
                        ticker=ticker,
                        ai_json_data=json.dumps(analysis_json)
                    )
                    db.add(new_report)
                
                db.commit()
                print(f"💾 Saved AI report to cache for {ticker}")
                
                # Save to history
                new_history = AnalysisHistory(
                    ticker=ticker,
                    verdict=analysis_json.get("verdict", "HOLD"),
                    confidence_score=analysis_json.get("confidence_score", 0)
                )
                db.add(new_history)
                db.commit()
                
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail="AI analysis temporarily unavailable. Please try again in a moment."
                )
            except Exception as e:
                error_msg = str(e)
                print(f"AI Error: {error_msg}")
                
                # User-friendly error messages
                if "404" in error_msg or "NOT_FOUND" in error_msg:
                    user_message = "AI service is updating. Please try again in a few seconds."
                elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
                    user_message = "AI service temporarily unavailable. Our team has been notified."
                elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    user_message = "High traffic detected. Please wait a moment and try again."
                elif "API key" in error_msg:
                    user_message = "Service configuration error. Please contact support."
                else:
                    user_message = "Analysis temporarily unavailable. Please try again."
                
                raise HTTPException(status_code=500, detail=user_message)
        
        # ========== STEP 4: LIVE PRICE INJECTION ==========
        print(f"💹 Fetching LIVE price for {ticker}")
        live_financial_data = get_real_financial_data(ticker)
        
        if not live_financial_data or not live_financial_data.get('price'):
            raise HTTPException(status_code=500, detail="Failed to fetch live price")
        
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
        # نحن نرسل الطلب لـ Gumroad لزيادة العداد
        response = requests.post("https://api.gumroad.com/v2/licenses/verify",
            data={"product_id": PRODUCT_ID, "license_key": request.license_key, "increment_uses_count": "true"})
        data = response.json()
        
        # التأكد من نجاح التحقق وعدم وجود استرجاع أموال
        if data.get("success") == True and not data.get("purchase", {}).get("refunded"):
            # 👇 التعديل الجوهري هنا:
            # Gumroad يرجع عدد الاستخدامات *بعد* الزيادة. 
            # إذا كان الكود جديداً، يجب أن يكون عدد الاستخدامات (uses) يساوي 1 بالضبط.
            # إذا كان أكثر من 1، فهذا يعني أن شخصاً آخر استخدمه قبله.
            uses = data.get("uses") 
            
            if uses and uses > 1:
                return {"valid": False, "message": "This key has already been redeemed."}
            
            # إذا وصل الكود هنا، معناه أول مرة يستخدم
            current_user.credits += 50
            db.commit()
            return {"valid": True, "credits": current_user.credits}
            
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
        raise HTTPException(
            status_code=500,
            detail="Analysis service temporarily unavailable. Please try again."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Comparison AI Error: {error_msg}")
        
        # User-friendly messages
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            user_message = "Comparison service updating. Try again in a moment."
        elif "403" in error_msg or "PERMISSION" in error_msg:
            user_message = "Service temporarily unavailable. Please try again later."
        elif "429" in error_msg:
            user_message = "Too many requests. Please wait 30 seconds."
        else:
            user_message = "Comparison temporarily unavailable. Please try again."
        
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
        raise HTTPException(
            status_code=500,
            detail="Battle analysis service temporarily unavailable. Try again shortly."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Battle AI Error: {error_msg}")
        
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            user_message = "Battle service updating. Please retry in a moment."
        elif "403" in error_msg or "PERMISSION" in error_msg:
            user_message = "Service temporarily unavailable. Try again later."
        elif "429" in error_msg:
            user_message = "High demand. Please wait 30 seconds and retry."
        else:
            user_message = "Battle analysis unavailable. Please try again."
        
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