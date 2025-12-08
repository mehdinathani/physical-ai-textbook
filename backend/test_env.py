"""
Simple test script to verify environment setup without triggering protobuf issues
"""
import os
from dotenv import load_dotenv

def test_environment():
    """Test that environment variables are properly loaded."""
    print("Testing environment setup...")

    # Load environment variables
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    print(f"QDRANT_URL: {'Found' if qdrant_url else 'Missing'}")
    print(f"QDRANT_API_KEY: {'Found' if qdrant_api_key else 'Missing'}")
    print(f"GOOGLE_API_KEY: {'Found' if google_api_key else 'Missing'}")

    if not all([qdrant_url, qdrant_api_key, google_api_key]):
        print("ERROR: Missing required environment variables in .env file")
        return False

    print("Environment variables are properly set!")
    return True

if __name__ == "__main__":
    test_environment()