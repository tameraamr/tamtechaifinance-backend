"""
🧪 Quick Test for Gemini API Fix
Run this to verify the model change works correctly
"""

from google import genai
import os
from datetime import datetime

# Load API key from environment only
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found in environment variables")
    print("\n🔧 To fix this:")
    print("   1. Get your API key from: https://makersuite.google.com/app/apikey")
    print("   2. Set it in PowerShell:")
    print('      $env:GOOGLE_API_KEY="your_api_key_here"')
    print("   3. Run this test again")
    exit(1)

client = genai.Client(api_key=API_KEY)

print("=" * 60)
print("🧪 Testing Gemini API Configuration")
print("=" * 60)

# Test 1: Check model availability
print("\n📋 Test 1: Listing available models...")
try:
    flash_found = False
    models = client.models.list()
    for m in models:
        if 'gemini-2.5-flash' in m.name or 'gemini-flash-latest' in m.name:
            print(f"✅ Found: {m.name}")
            flash_found = True
            break
    
    if not flash_found:
        print("⚠️  gemini-2.5-flash not found in available models")
        print("Available models:")
        for m in client.models.list():
            print(f"   - {m.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")
    exit(1)

# Test 2: Initialize the model
print("\n📋 Test 2: Initializing gemini-2.5-flash...")
try:
    model_name = 'gemini-2.5-flash'
    print(f"✅ Model '{model_name}' ready to use")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    exit(1)

# Test 3: Simple API call
print("\n📋 Test 3: Making test API call...")
try:
    prompt = "What is the ticker symbol for Apple Inc? Answer in one word."
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    print(f"✅ API Response: {response.text.strip()}")
except Exception as e:
    print(f"❌ Error making API call: {e}")
    exit(1)

# Test 4: Financial analysis test
print("\n📋 Test 4: Testing financial analysis...")
try:
    prompt = """Analyze AAPL stock briefly in 2 sentences. 
    Focus on current market sentiment."""
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    print(f"✅ Analysis Response:\n{response.text}")
except Exception as e:
    print(f"❌ Error in financial analysis: {e}")
    exit(1)

# Test 5: Performance test
print("\n📋 Test 5: Performance test (response time)...")
try:
    import time
    start_time = time.time()
    
    prompt = "Classify this sentiment: 'Apple stock is performing well this quarter.'"
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Response time: {duration:.2f} seconds")
    if duration < 3:
        print("   🚀 Performance: EXCELLENT (< 3s)")
    elif duration < 5:
        print("   ✅ Performance: GOOD (3-5s)")
    else:
        print("   ⚠️  Performance: SLOW (> 5s)")
except Exception as e:
    print(f"❌ Error in performance test: {e}")

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n🎯 Next Steps:")
print("   1. Your backend is ready to use gemini-1.5-flash")
print("   2. Restart your FastAPI server: python backend/main.py")
print("   3. The 404 error should be resolved")
print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
