import asyncio
import sys
import os
from pathlib import Path

# Add backend-chatkit to path
sys.path.insert(0, str(Path(__file__).parent))

from main import app, store, server
from fastapi.testclient import TestClient

print("=" * 80)
print("TESTING CHATKIT BACKEND")
print("=" * 80)

# Create test client
client = TestClient(app)

# Test 1: Health endpoint
print("\n[TEST 1] Testing /health endpoint...")
response = client.get("/health")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

# Test 2: Debug threads endpoint
print("\n[TEST 2] Testing /debug/threads endpoint...")
response = client.get("/debug/threads")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

# Test 3: ChatKit endpoint (basic connection test)
print("\n[TEST 3] Testing /chatkit endpoint connection...")
test_payload = b'{"type": "threads.list", "params": {"limit": 10}}'
try:
    response = client.post("/chatkit", content=test_payload)
    print(f"  Status: {response.status_code}")
    print(f"  Response type: {response.headers.get('content-type', 'N/A')}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)}")

print("\n" + "=" * 80)
print("BACKEND TESTS COMPLETE")
print("=" * 80)
print("\nTo start the backend server:")
print("  cd backend-chatkit")
print("  python main.py")
print("\nTo start the frontend:")
print("  cd frontend")
print("  npm start")
