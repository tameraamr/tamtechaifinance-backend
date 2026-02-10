import sys
import importlib

packages = [
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "dotenv",
    "google.genai",
    "yfinance",
    "requests",
    "httpx",
    "jose",
    "passlib",
    "bcrypt",
    "multipart",
    "psycopg2",
    "pydantic",
    "email_validator",
    "resend",
    "PIL",
    "feedparser"
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

failed = []
for package in packages:
    try:
        importlib.import_module(package)
        print(f"[OK] {package}")
    except ImportError as e:
        print(f"[FAIL] {package}: {e}")
        failed.append(package)

if failed:
    print(f"\nFailed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nAll packages imported successfully.")
    sys.exit(0)
