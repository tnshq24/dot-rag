#!/usr/bin/env python3
"""
Test script to verify all backend routes
Run this after deploying the backend to Azure Web App
"""

import requests
import json
import sys

def test_backend_routes(base_url):
    """Test all backend routes"""
    
    print(f"Testing backend at: {base_url}")
    print("=" * 50)
    
    # Test 1: Health check (public route)
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Login (get JWT token)
    print("\n2. Testing login...")
    try:
        login_data = {
            "email": "admin@xyz.com",
            "password": "admin"
        }
        response = requests.post(f"{base_url}/login", json=login_data)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("token"):
                print("✅ Login successful")
                token = data["token"]
                print(f"User ID: {data.get('user_id')}")
                print(f"Email: {data.get('email')}")
                print(f"Token: {token[:20]}...")
            else:
                print("❌ Login failed - no token received")
                return False
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Login error: {e}")
        return False
    
    # Test 3: Check auth with token
    print("\n3. Testing authentication check...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{base_url}/check_auth", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("authenticated"):
                print("✅ Authentication check passed")
                print(f"User: {data.get('email')}")
                print(f"Admin: {data.get('isadmin')}")
            else:
                print("❌ Authentication check failed")
                return False
        else:
            print(f"❌ Authentication check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Authentication check error: {e}")
        return False
    
    # Test 4: Get available files
    print("\n4. Testing available files...")
    try:
        response = requests.get(f"{base_url}/available_files", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("✅ Available files endpoint working")
            print(f"Files count: {len(data.get('files', []))}")
        else:
            print(f"❌ Available files failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Available files error: {e}")
    
    # Test 5: Chat endpoint
    print("\n5. Testing chat endpoint...")
    try:
        chat_data = {
            "question": "Hello, how are you?",
            "conversation_id": "test-conversation",
            "session_id": "test-session",
            "file_names": []
        }
        response = requests.post(f"{base_url}/chat", json=chat_data, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat endpoint working")
            print(f"Answer: {data.get('answer', '')[:100]}...")
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    # Test 6: Logout
    print("\n6. Testing logout...")
    try:
        response = requests.post(f"{base_url}/logout", headers=headers)
        if response.status_code == 200:
            print("✅ Logout successful")
        else:
            print(f"❌ Logout failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Logout error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_routes.py <backend_url>")
        print("Example: python test_routes.py https://your-app.azurewebsites.net")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    success = test_backend_routes(base_url)
    
    if success:
        print("\n🎉 Backend is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Backend has issues!")
        sys.exit(1) 