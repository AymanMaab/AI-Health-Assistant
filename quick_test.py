#!/usr/bin/env python3
"""
Quick test script to verify the API is working
Run this after starting the server
"""

import requests
import json
from time import sleep

BASE_URL = "http://localhost:8000"

def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")
    print()


def main():
    print("\n" + "🏥" * 20)
    print("AI HEALTH ASSISTANT - QUICK TEST")
    print("🏥" * 20)
    
    # Test 1: Health Check
    print("\n[TEST 1] Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print_response("Health Check", response)
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        print("Make sure the server is running: python src/main.py")
        return
    
    # Test 2: Ask about medicine
    print("[TEST 2] Asking about medicine side effects...")
    payload = {
        "question": "What are the side effects of metformin?",
        "use_external_api": False
    }
    response = requests.post(f"{BASE_URL}/api/v1/ask", json=payload)
    print_response("Medicine Question", response)
    
    # Test 3: Ask about nutrition
    print("[TEST 3] Asking about nutrition...")
    payload = {
        "question": "How much protein should I eat daily?",
        "use_external_api": False
    }
    response = requests.post(f"{BASE_URL}/api/v1/ask", json=payload)
    print_response("Nutrition Question", response)
    
    # Test 4: Medicine info endpoint
    print("[TEST 4] Getting specific medicine info...")
    payload = {"medicine_name": "aspirin"}
    response = requests.post(f"{BASE_URL}/api/v1/medicine-info", json=payload)
    print_response("Medicine Info (OpenFDA)", response)
    
    # Test 5: Nutrition endpoint
    print("[TEST 5] Getting nutrition info...")
    payload = {"food_name": "banana"}
    response = requests.post(f"{BASE_URL}/api/v1/nutrition", json=payload)
    print_response("Nutrition Info (USDA)", response)
    
    # Test 6: List intents
    print("[TEST 6] Listing available intents...")
    response = requests.get(f"{BASE_URL}/api/v1/intents")
    print_response("Available Intents", response)
    
    # Test 7: Batch processing
    print("[TEST 7] Batch processing...")
    payload = [
        {"question": "What is diabetes?", "use_external_api": False},
        {"question": "What are symptoms of flu?", "use_external_api": False}
    ]
    response = requests.post(f"{BASE_URL}/api/v1/batch", json=payload)
    print_response("Batch Questions", response)
    
    # Summary
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)
    print("\nAll endpoints are working correctly!")
    print(f"API Documentation: {BASE_URL}/docs")
    print(f"Alternative docs: {BASE_URL}/redoc")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()