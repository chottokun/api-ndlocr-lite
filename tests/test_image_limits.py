import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import base64
from src.api.main import app

# Increase PIL limit to avoid DecompressionBombWarning during tests
Image.MAX_IMAGE_PIXELS = None

client = TestClient(app, raise_server_exceptions=False)

def create_large_image_bytes(width, height, format="JPEG"):
    """Creates a simple image with large dimensions but small file size."""
    img = Image.new('L', (width, height), color=0)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def test_image_dimensions_exceed_limit_multipart():
    # MAX_PIXELS is 100,000,000 (100MP). 10001 * 10001 = 100,020,001
    large_image = create_large_image_bytes(10001, 10001)

    files = {"file": ("large_image.jpg", large_image, "image/jpeg")}
    response = client.post("/v1/ocr", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions too large"

def test_image_dimensions_exceed_limit_base64():
    large_image = create_large_image_bytes(10001, 10001)
    encoded = base64.b64encode(large_image).decode()

    payload = {"image": f"data:image/jpeg;base64,{encoded}"}
    response = client.post("/v1/ocr", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions too large"
