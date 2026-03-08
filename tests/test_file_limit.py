import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import src.api.main

def test_file_too_large(monkeypatch):
    # Set MAX_IMAGE_SIZE to a very small value for testing
    monkeypatch.setattr(src.api.main, "MAX_IMAGE_SIZE", 100)

    with TestClient(app) as client:
        # Create a "file" that is 101 bytes long
        large_content = b"a" * 101
        files = {"file": ("large_image.jpg", large_content, "image/jpeg")}

        response = client.post("/v1/ocr", files=files)

        assert response.status_code == 413
        assert response.json()["detail"] == "File too large"

def test_file_within_limit(monkeypatch):
    # Set MAX_IMAGE_SIZE to a value large enough
    monkeypatch.setattr(src.api.main, "MAX_IMAGE_SIZE", 100)

    with TestClient(app) as client:
        # Create a "file" that is exactly 100 bytes long
        content = b"a" * 100
        files = {"file": ("small.jpg", content, "image/jpeg")}

        response = client.post("/v1/ocr", files=files)

        # It should NOT be 413. Since it's just "a"*100, it's an invalid image.
        # So it should return 400.
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid request: Invalid image data or format"
