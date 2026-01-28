"""
🔐 Security Checker - Verify API Keys Are Not Exposed
Run this before committing code to Git
"""

import os
import re
from pathlib import Path

print("=" * 60)
print("🔐 API Key Security Checker")
print("=" * 60)

# Files to check
backend_dir = Path(__file__).parent
files_to_check = [
    "main.py",
    "check_models.py",
    "test_gemini_fix.py",
    "mailer.py",
]

# Pattern to detect API keys
api_key_pattern = re.compile(r'AIza[0-9A-Za-z_-]{35}')

print("\n📋 Checking Python files for hardcoded API keys...")

issues_found = False

for filename in files_to_check:
    filepath = backend_dir / filename
    if not filepath.exists():
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = api_key_pattern.findall(content)
        
        if matches:
            print(f"\n❌ SECURITY ISSUE: {filename}")
            print(f"   Found {len(matches)} hardcoded API key(s)")
            print(f"   Please use environment variables instead!")
            issues_found = True
        else:
            print(f"✅ {filename} - Clean")

# Check .env is in .gitignore
print("\n📋 Checking .gitignore...")
gitignore_path = backend_dir / ".gitignore"

if gitignore_path.exists():
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
        if '.env' in gitignore_content:
            print("✅ .env is in .gitignore")
        else:
            print("❌ WARNING: .env is NOT in .gitignore")
            print("   Add '.env' to .gitignore immediately!")
            issues_found = True
else:
    print("⚠️  No .gitignore found - create one!")
    issues_found = True

# Check if .env file exists
print("\n📋 Checking environment files...")
env_path = backend_dir / ".env"
env_example_path = backend_dir / ".env.example"

if env_path.exists():
    print("✅ .env file exists (contains your real API keys)")
    
    # Make sure it's not in Git
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'ls-files', str(env_path)],
            capture_output=True,
            text=True,
            cwd=backend_dir
        )
        if result.stdout.strip():
            print("❌ CRITICAL: .env is tracked by Git!")
            print("   Run: git rm --cached .env")
            issues_found = True
        else:
            print("✅ .env is not tracked by Git")
    except:
        print("⚠️  Could not check Git status (Git may not be installed)")
else:
    print("⚠️  .env file not found - create one from .env.example")

if env_example_path.exists():
    print("✅ .env.example exists (safe template)")
    
    # Check if .env.example contains real keys
    with open(env_example_path, 'r', encoding='utf-8', errors='ignore') as f:
        example_content = f.read()
        if api_key_pattern.search(example_content):
            print("❌ WARNING: .env.example contains a real API key!")
            print("   Replace with placeholder: GOOGLE_API_KEY=your_api_key_here")
            issues_found = True
else:
    print("⚠️  .env.example not found - create one for documentation")

print("\n" + "=" * 60)

if issues_found:
    print("❌ SECURITY ISSUES FOUND - FIX BEFORE COMMITTING!")
    print("=" * 60)
    print("\n🔧 Quick Fixes:")
    print("   1. Remove hardcoded keys from Python files")
    print("   2. Add '.env' to .gitignore")
    print("   3. Run: git rm --cached .env (if tracked)")
    print("   4. Only commit .env.example (with placeholders)")
    exit(1)
else:
    print("✅ ALL SECURITY CHECKS PASSED!")
    print("=" * 60)
    print("\n🎯 Safe to commit code to Git")
    print("   Your API keys are protected!")
    exit(0)
