#!/usr/bin/env python3
"""
Simple test script for the Azure RAG Web App
"""

import requests
import json
import sys

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_chat_endpoint(base_url, question):
    """Test the chat endpoint"""
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"question": question},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat test passed")
            print(f"Question: {data.get('question', 'N/A')}")
            print(f"Answer: {data.get('answer', 'N/A')[:200]}...")
            return True
        else:
            print(f"❌ Chat test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat test error: {e}")
        return False

def main():
    """Main test function"""
    # Default to localhost if no URL provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    print(f"🧪 Testing Azure RAG Web App at: {base_url}")
    print("=" * 50)
    
    # Test health endpoint
    health_ok = test_health_endpoint(base_url)
    
    # Test chat endpoint
    chat_ok = test_chat_endpoint(base_url, "What is the main purpose of this chatbot?")
    
    print("=" * 50)
    if health_ok and chat_ok:
        print("🎉 All tests passed! Your app is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 