from google import genai
import os

# Load API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found")
    print("\n🔧 Setup Instructions:")
    print("   1. Get your API key: https://makersuite.google.com/app/apikey")
    print("   2. Set in PowerShell: $env:GOOGLE_API_KEY=\"your_key_here\"")
    print("   3. Or create .env file with: GOOGLE_API_KEY=your_key_here")
    exit(1)

client = genai.Client(api_key=API_KEY)

print("=" * 60)
print("🔍 Available Gemini Models for Financial Analysis")
print("=" * 60)

try:
    available_models = []
    models_response = client.models.list()
    for m in models_response:
        # New API: check capabilities
        available_models.append(m)
        print(f"\n✅ Model: {m.name}")
        print(f"   Display Name: {m.display_name if hasattr(m, 'display_name') else 'N/A'}")
        if hasattr(m, 'description'):
            desc = m.description[:100] if len(m.description) > 100 else m.description
            print(f"   Description: {desc}...")
    
    print("\n" + "=" * 60)
    print(f"📊 Total Models Available: {len(available_models)}")
    print("=" * 60)
    
    # Recommended models for financial analysis
    print("\n🎯 RECOMMENDED FOR REAL-TIME FINANCIAL ANALYSIS:")
    print("   1. gemini-1.5-flash (FASTEST - Best for real-time)")
    print("   2. gemini-1.5-flash-8b (Ultra-fast, lighter)")
    print("   3. gemini-1.5-pro (Most capable, slower)")
    print("\n⚠️  AVOID: gemini-2.0-* models (experimental, unstable)")
    
except Exception as e:
    print(f"❌ Error: {e}")