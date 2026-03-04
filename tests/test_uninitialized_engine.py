import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import io
from PIL import Image
from src.api.main import app
import time

client = TestClient(app)

def get_small_image_bytes():
    img = Image.new('RGB', (1, 1), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_health_endpoint_engine_none():
    with patch("src.api.main.engine", None):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "engine_ready": False}

def test_ocr_endpoint_engine_none():
    image_bytes = get_small_image_bytes()
    with patch("src.api.main.engine", None):
        response = client.post(
            "/v1/ocr",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "Engine not initialized"}

def test_ocr_job_engine_none():
    image_bytes = get_small_image_bytes()
    with patch("src.api.main.engine", None):
        response = client.post(
            "/v1/ocr/jobs",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll for failure
        for _ in range(10):
            res = client.get(f"/v1/ocr/jobs/{job_id}")
            assert res.status_code == 200
            data = res.json()
            if data["status"] == "failed":
                assert data["error"] == "Engine not initialized"
                return
            time.sleep(0.1)
        pytest.fail("Job did not fail as expected")
