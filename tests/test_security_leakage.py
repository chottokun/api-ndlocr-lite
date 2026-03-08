from fastapi.testclient import TestClient
from src.api.main import app
import base64

client = TestClient(app, raise_server_exceptions=False)

def test_exception_leakage_base64_invalid():
    # Invalid base64
    payload = {"image": "data:image/jpeg;base64,invalid_base64!!!"}
    response = client.post("/v1/ocr", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request: Invalid image data or format"

def test_exception_leakage_non_image():
    # Not an image
    payload = {"image": "data:image/jpeg;base64," + base64.b64encode(b"not an image").decode()}
    response = client.post("/v1/ocr", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request: Invalid image data or format"

def test_exception_leakage_multipart_non_image():
    # Multipart upload with non-image data
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/v1/ocr", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request: Invalid image data or format"
