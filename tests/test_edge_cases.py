from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_ocr_empty_body():
    """Test that an empty body to /v1/ocr returns 400."""
    response = client.post("/v1/ocr")
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty request body"

def test_ocr_jobs_empty_body():
    """Test that an empty body to /v1/ocr/jobs returns 400."""
    response = client.post("/v1/ocr/jobs")
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty request body"
