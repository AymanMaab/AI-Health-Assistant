import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_ask_question():
    """Test main ask endpoint"""
    payload = {
        "question": "What are the side effects of aspirin?",
        "use_external_api": False
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "intent" in data
    assert "confidence" in data
    assert "question" in data
    assert data["question"] == payload["question"]


def test_ask_question_nutrition():
    """Test nutrition question"""
    payload = {
        "question": "How much protein should I eat daily?",
        "use_external_api": False
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["intent"] == "nutrition_advice" or "nutrition" in data["intent"]


def test_medicine_info():
    """Test medicine info endpoint"""
    payload = {
        "medicine_name": "aspirin"
    }
    response = client.post("/api/v1/medicine-info", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "medicine" in data
    assert "data" in data
    assert data["medicine"] == "aspirin"


def test_nutrition():
    """Test nutrition endpoint"""
    payload = {
        "food_name": "banana"
    }
    response = client.post("/api/v1/nutrition", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "food" in data
    assert "data" in data
    assert data["food"] == "banana"


def test_list_intents():
    """Test list intents endpoint"""
    response = client.get("/api/v1/intents")
    assert response.status_code == 200
    data = response.json()
    assert "intents" in data
    assert "descriptions" in data
    assert len(data["intents"]) > 0
    assert "medicine_info" in data["intents"]
    assert "nutrition_advice" in data["intents"]


def test_batch_questions():
    """Test batch processing"""
    payload = [
        {"question": "What are side effects of metformin?", "use_external_api": False},
        {"question": "How much protein do I need?", "use_external_api": False}
    ]
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "results" in data
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_invalid_question_too_short():
    """Test with invalid input - question too short"""
    payload = {
        "question": "Hi",  # Too short (min 5 chars)
        "use_external_api": False
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 422  # Validation error


def test_invalid_question_too_long():
    """Test with invalid input - question too long"""
    payload = {
        "question": "A" * 501,  # Too long (max 500 chars)
        "use_external_api": False
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 422  # Validation error


def test_invalid_medicine_name():
    """Test medicine info with empty name"""
    payload = {
        "medicine_name": "X"  # Too short
    }
    response = client.post("/api/v1/medicine-info", json=payload)
    assert response.status_code == 422  # Validation error


def test_batch_too_many_queries():
    """Test batch with too many queries"""
    payload = [
        {"question": f"Question {i}?", "use_external_api": False}
        for i in range(11)  # More than 10
    ]
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 400  # Bad request


def test_missing_required_fields():
    """Test with missing required fields"""
    # Missing 'question' field
    payload = {
        "use_external_api": False
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 422  # Validation error


def test_response_structure():
    """Test that response has all expected fields"""
    payload = {
        "question": "What causes diabetes?",
        "use_external_api": False
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check all required fields
    required_fields = ["question", "answer", "intent", "confidence", "source", "timestamp"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Check data types
    assert isinstance(data["answer"], str)
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 1


if __name__ == "__main__":
    print("Running API tests...")
    print("=" * 60)
    pytest.main([__file__, "-v", "--tb=short"])