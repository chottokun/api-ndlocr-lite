import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import base64

client = TestClient(app)

def test_ocr_base64_invalid_json():
    response = client.post(
        "/v1/ocr",
        content="invalid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "Invalid request" in response.json()["detail"]

def test_ocr_base64_missing_image_field():
    response = client.post(
        "/v1/ocr",
        json={"not_image": "some value"}
    )
    assert response.status_code == 400
    assert "Invalid request" in response.json()["detail"]

def test_ocr_base64_malformed_padding():
    # Base64 with incorrect length (must be multiple of 4 if padded, but standard decoder is lenient)
    # However, 'abc' (3 chars) is definitely invalid for standard base64 decoding.
    response = client.post(
        "/v1/ocr",
        json={"image": "data:image/jpeg;base64,abc"}
    )
    assert response.status_code == 400
    assert "Invalid request" in response.json()["detail"]

def test_ocr_base64_valid_base64_but_not_image():
    encoded = base64.b64encode(b"not an image").decode()
    response = client.post(
        "/v1/ocr",
        json={"image": f"data:image/jpeg;base64,{encoded}"}
    )
    assert response.status_code == 400
    assert "Invalid request" in response.json()["detail"]

def test_ocr_body_too_large():
    # Default MAX_BODY_SIZE is 15MB.
    large_body = "a" * (15 * 1024 * 1024 + 100)
    response = client.post(
        "/v1/ocr",
        json={"image": large_body}
    )
    assert response.status_code == 413
    assert "Request body too large" in response.json()["detail"]
