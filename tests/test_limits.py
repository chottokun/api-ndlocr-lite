import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import src.api.main
from PIL import Image
import io
import base64

client = TestClient(app)

def test_file_too_large(monkeypatch):
    # Mock MAX_IMAGE_SIZE to 100 bytes
    monkeypatch.setattr(src.api.main, "MAX_IMAGE_SIZE", 100)
    data = b"a" * 101
    response = client.post(
        "/v1/ocr",
        files={"file": ("test.jpg", data, "image/jpeg")}
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "File too large"

def test_body_too_large(monkeypatch):
    # Mock MAX_BODY_SIZE to 100 bytes
    monkeypatch.setattr(src.api.main, "MAX_BODY_SIZE", 100)
    # Create a payload larger than 100 bytes
    payload = {"image": "data:image/jpeg;base64," + base64.b64encode(b"a" * 100).decode()}
    response = client.post("/v1/ocr", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large"

def test_image_pixels_too_large(monkeypatch):
    # Mock MAX_PIXELS to 100 pixels
    monkeypatch.setattr(src.api.main, "MAX_PIXELS", 100)
    # Create an image with 11x10 = 110 pixels (> 100)
    img = Image.new('RGB', (11, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/v1/ocr",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions too large"
