from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator
import google.generativeai as genai
import yfinance as yf
import os
import json
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String
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
    credits = Column(Integer, default=0) 

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    @validator('password')
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v): raise ValueError('Password must contain a number')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

class LicenseRequest(BaseModel):
    license_key: str

# --- Routes ---
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return {"message": "User created"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return {"access_token": create_access_token(data={"sub": user.email}), "token_type": "bearer"}

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user_mandatory)):
    return {"email": current_user.email, "credits": current_user.credits}

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

@app.get("/analyze/{ticker}")
async def analyze_stock(
    ticker: str, 
    lang: str = "en", 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user_optional)
):
    if current_user:
        if current_user.credits <= 0: raise HTTPException(status_code=402, detail="No credits left")
        if ticker == "#DEVMODE":
            current_user.credits = 1000
            db.commit()
            return {"message": "Dev Mode: 1000 Credits Added"}
    
    financial_data = get_real_financial_data(ticker)
    if not financial_data: raise HTTPException(status_code=404, detail="Stock not found.")
    
    ai_payload = {k: v for k, v in financial_data.items() if k != 'chart_data'}
    
    lang_map = {
        "en": "English", 
        "ar": "Arabic (Modern Standard, High-End Financial Tone)", 
        "it": "Italian (Professional Financial Tone)"
    }
    target_lang = lang_map.get(lang, "English")

    prompt = f"""
    You are the Chief Investment Officer (CIO) at a prestigious Global Hedge Fund. 
    Your task is to produce an **EXHAUSTIVE, INSTITUTIONAL-GRADE INVESTMENT MEMO** for {ticker}.
    
    **Financial Data:** {json.dumps(ai_payload)}
    **Language:** Write strictly in {target_lang}.

    **⚠️ CRITICAL INSTRUCTIONS - READ CAREFULLY:**
    1.  **EXTREME DEPTH:** Do NOT act like a chatbot. Act like a Equity Research Analyst writing a paid report. Each text section must be LONG, DETAILED, and ANALYTICAL (aim for 300-500 words per chapter).
    2.  **NO FLUFF:** Don't explain what "P/E" means. Use the metrics to draw conclusions. Connect the dots between Macroeconomics, Industry Trends, and Company Specifics.
    3.  **CONSISTENCY:** Base your verdict STRICTLY on the provided data (Valuation, Growth, Health). Do not hallucinate. If the stock is overvalued, say SELL/HOLD. If undervalued, say BUY.
    4.  **FORMAT:** Return strictly the JSON structure below.

    **REQUIRED JSON OUTPUT:**
    {{
        "chapter_1_the_business": "Headline: [Translate 'The Business DNA']. Write a comprehensive 'Deep Dive' essay (approx 400 words). Analyze the business model's durability, the 'Economic Moat' (Network Effects, Switching Costs, Cost Advantage), and the competitive landscape. Discuss Unit Economics and supply chain resilience.",
        
        "chapter_2_financials": "Headline: [Translate 'Financial Health']. Write a forensic financial analysis essay (approx 400 words). Discuss Capital Allocation (ROIC vs WACC), Operating Leverage, Free Cash Flow (FCF) conversion, and Balance Sheet strength. Analyze the quality of earnings (organic vs inorganic growth).",
        
        "chapter_3_valuation": "Headline: [Translate 'Valuation Check']. Write a detailed valuation essay (approx 400 words). Perform a mental DCF (Discounted Cash Flow) analysis. Compare current multiples (P/E, PEG, EV/EBITDA) to historical averages and peer groups. Is the stock priced for perfection? What is the Margin of Safety?",
        
        "bull_case_points": [
            "Extremely detailed, specific bull argument #1 (e.g., specific product launch impact).",
            "Extremely detailed, specific bull argument #2 (e.g., margin expansion catalyst).",
            "Extremely detailed, specific bull argument #3."
        ],
        
        "bear_case_points": [
            "Extremely detailed, specific bear argument #1 (e.g., specific regulatory threat).",
            "Extremely detailed, specific bear argument #2 (e.g., valuation compression risk).",
            "Extremely detailed, specific bear argument #3."
        ],
        
        "forecasts": {{
            "next_1_year": "A detailed 12-month scenario analysis. Discuss short-term catalysts, earnings revisions, and sentiment shifts.",
            "next_5_years": "A long-term structural thesis (2030 outlook). Discuss TAM (Total Addressable Market) expansion, secular trends, and terminal value."
        }},
        
        "swot_analysis": {{
            "strengths": ["Detailed Strength 1", "Detailed Strength 2", "Detailed Strength 3"],
            "weaknesses": ["Detailed Weakness 1", "Detailed Weakness 2", "Detailed Weakness 3"],
            "opportunities": ["Detailed Opportunity 1", "Detailed Opportunity 2", "Detailed Opportunity 3"],
            "threats": ["Detailed Threat 1", "Detailed Threat 2", "Detailed Threat 3"]
        }},
        
        "radar_scores": [
            {{ "subject": "Value", "A": 8 }}, 
            {{ "subject": "Growth", "A": 7 }},
            {{ "subject": "Profitability", "A": 9 }}, 
            {{ "subject": "Health", "A": 6 }},
            {{ "subject": "Momentum", "A": 8 }}
        ],
        
        "verdict": "BUY / HOLD / SELL / STRONG BUY", 
        "confidence_score": 85, 
        "summary_one_line": "A decisive, executive summary of the final investment recommendation."
    }}
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        credits_left = 0
        if current_user:
            current_user.credits -= 1
            db.commit()
            credits_left = current_user.credits

        return {
            "ticker": ticker.upper(), 
            "data": financial_data, 
            "analysis": json.loads(response.text), 
            "credits_left": credits_left,
            "is_guest": current_user is None
        }
    except Exception as e:
        print(f"AI Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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