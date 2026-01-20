import google.generativeai as genai
import os

# ⚠️ ضع مفتاحك الحقيقي هنا
os.environ["GOOGLE_API_KEY"] = "AIzaSyDZH6I2ZDMfi7q9fXjIUQSzXpchRH-gZ28"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("جاري البحث عن الموديلات المتاحة لحسابك...")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")