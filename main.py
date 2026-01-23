from fastapi import FastAPI, HTTPException, Depends, status, Request 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
import google.generativeai as genai
import yfinance as yf
import os
import json
import random
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
import bcrypt
import re

# --- Database & Security (Deployment Ready) ---
# 👇 هذا الكود الذكي يختار القاعدة المناسبة تلقائياً
# --- Database & Security (Deployment Ready) ---
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # تعديل الرابط ليعمل مع مكتبة psycopg2 سواء كان من Render أو Supabase
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
    
    engine = create_engine(DATABASE_URL)
else:
    # Localhost (SQLite)
    SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

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

# إنشاء الجداول تلقائياً
Base.metadata.create_all(bind=engine)

# --- Gemini Setup ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
# ملاحظة: في السيرفر لا توقف التطبيق إذا لم تجد المفتاح فوراً، السيرفر سيحقنه
if not API_KEY: 
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY)

try:
    # نستخدم الموديل الأقوى والأذكى للتحليل العميق
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
except:
    model = genai.GenerativeModel('gemini-1.5-pro')

app = FastAPI()

# 👇 أضف هذا القسم المحدث هنا بالضبط للسماح بالدومين الجديد
origins = [
    "http://localhost:3000",
    "https://tamtech-frontend.vercel.app",
    "https://tamtech-finance.com",
    "https://www.tamtech-finance.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # يسمح لـ www.tamtech-finance.com وغيرها بالوصول
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"] # أضف هذا السطر لضمان رؤية المتصفح للرد
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

async def get_current_user_optional(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token: return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: return None
    except JWTError: return None
    return db.query(User).filter(User.email == email).first()

async def get_current_user_mandatory(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token: raise HTTPException(status_code=401, detail="Not authenticated")
    user = await get_current_user_optional(token, db)
    if not user: raise HTTPException(status_code=401, detail="Invalid token")
    return user

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
        credits=3 # رصيد مجاني للبداية
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # 1. التحقق من المستخدم
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # 2. إنشاء التوكين
    access_token = create_access_token(data={"sub": user.email})
    
    # 3. 👇 التغيير الكبير: إرجاع البيانات كاملة
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "credits": user.credits, # ✅ الرصيد وصل!
        "user": {
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone_number": user.phone_number,
            "country": user.country,
            "address": user.address
        }
    }

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user_mandatory)):
    return {"email": current_user.email, "credits": current_user.credits}


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

# 👇👇👇 نقطة الاتصال لاقتراح سهم ذكي 👇👇👇
@app.get("/suggest-stock")
def suggest_stock():
    """Asks the AI to pick a high-potential stock dynamically with a temperature boost for variety."""
    try:
        # أضفنا طلب التنوع (randomly pick) وتعديل الـ Temperature لزيادة العشوائية
        prompt = "Act as a senior financial analyst. Pick ONE high-potential stock ticker from the US market (NYSE/NASDAQ) that is trending or has a strong growth catalyst. Return ONLY the ticker symbol (e.g. TSLA). Do not repeat NVDA every time, try to be diverse. Return ONLY the symbol."
        
        # استخدام الـ Generation Config لزيادة العشوائية (Temperature)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9, # كلما زاد الرقم زاد التنوع (بين 0 و 1)
            )
        )
        ticker = response.text.strip().replace("\n", "").replace(" ", "").upper()
        clean_ticker = re.sub(r'[^A-Z]', '', ticker)
        return {"ticker": clean_ticker}
    except:
        # قائمة احتياطية في حال فشل الـ AI عشان ما يعطي دايماً NVDA
        import random
        backups = ["MSFT", "AAPL", "AMD", "META", "GOOGL", "AMZN", "PYPL", "V"]
        return {"ticker": random.choice(backups)}

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
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user_optional)
):
    # --- 🛡️ نظام التقاط الـ IP الحقيقي والحماية ---
    if current_user:
        if current_user.credits <= 0: raise HTTPException(status_code=402, detail="No credits left")
        if ticker == "#DEVMODE":
            current_user.credits = 1000
            db.commit()
            return {"message": "Dev Mode: 1000 Credits Added"}
    else:
        # 👇 التعديل الجوهري: الحصول على الـ IP الحقيقي من هيدرز Railway
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
        
        guest.attempts += 1
        db.commit()
    # --- 🛡️ نهاية نظام الحماية ---
    
    # 1. جلب البيانات
    financial_data = get_real_financial_data(ticker)
    
    # 👇 التعديل هنا: إذا لم نجد بيانات، أو وجدنا بيانات لكن بدون سعر (سهم وهمي) -> وقف فوراً
    if not financial_data or not financial_data.get('price'):
        raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found or delisted.")
    
    ai_payload = {k: v for k, v in financial_data.items() if k != 'chart_data'}
    
    lang_map = {
        "en": "English", 
        "ar": "Arabic (Modern Standard, High-End Financial Tone)", 
        "it": "Italian (Professional Financial Tone)"
    }
    target_lang = lang_map.get(lang, "English")

    # 👇 الـ Prompt المحدث - دسم جداً ويطلب تحليل الأخبار
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
        # استخدام الموديل لإنتاج الرد
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        credits_left = 0
        if current_user:
            current_user.credits -= 1
            db.commit()
            credits_left = current_user.credits

        # تحليل الرد والتأكد من تحويله لـ JSON
        analysis_json = json.loads(response.text)

        # 👇 هذا الكود يحفظ السهم والنتيجة في جدول التاريخ
        new_history = AnalysisHistory(
            ticker=ticker.upper(),
            verdict=analysis_json.get("verdict", "HOLD"),
            confidence_score=analysis_json.get("confidence_score", 0)
        )
        db.add(new_history)
        db.commit() # حفظ في قاعدة البيانات
        
        return {
            "ticker": ticker.upper(), 
            "data": financial_data, 
            "analysis": analysis_json, 
            "credits_left": credits_left,
            "is_guest": current_user is None
        }
    except Exception as e:
        print(f"AI Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

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

        lang_map = {"en": "English", "ar": "Arabic", "it": "Italian"}
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
        
        response = model.generate_content(
    prompt, 
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2  # 👈 هذا السطر السحري
    }
)
        analysis_result = json.loads(response.text)

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

        lang_map = {"en": "English", "ar": "Arabic", "it": "Italian"}
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
        
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        analysis_result = json.loads(response.text)

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
            "Consumer Staples": "XLP"
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