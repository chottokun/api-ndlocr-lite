import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from src.api.main import app
import time

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def sample_image_path():
    path = Path("extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg")
    if not path.exists():
        pytest.skip(f"Sample image {path} not found")
    return path

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ocr_upload(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/v1/ocr",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "pages" in data
    assert len(data["pages"]) > 0
    assert "第8章" in data["pages"][0]["markdown"]

def test_ocr_base64(client, sample_image_path):
    import base64
    with open(sample_image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    response = client.post(
        "/v1/ocr",
        json={"image": f"data:image/jpeg;base64,{encoded}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "pages" in data
    assert "第8章" in data["pages"][0]["markdown"]

def test_ocr_job_workflow(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/v1/ocr/jobs",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    job_data = response.json()
    job_id = job_data["job_id"]
    assert job_data["status"] == "pending"

    # Poll for completion
    max_retries = 60
    completed = False
    for _ in range(max_retries):
        response = client.get(f"/v1/ocr/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] == "completed":
            completed = True
            assert "result" in data
            assert "第8章" in data["result"]["pages"][0]["markdown"]
            break
        elif data["status"] == "failed":
            pytest.fail(f"Job failed: {data.get('error')}")
        time.sleep(1)

    assert completed, "Job did not complete in time"

def test_get_non_existent_job(client):
    response = client.get("/v1/ocr/jobs/non-existent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"

def test_ocr_upload_file_too_large(client):
    # MAX_IMAGE_SIZE is 10MB by default
    large_content = b"a" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/v1/ocr",
        files={"file": ("large.jpg", large_content, "image/jpeg")}
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]

def test_ocr_image_dimensions_too_large(client):
    import io
    from PIL import Image
    # MAX_PIXELS is 100MP by default. 10001 * 10001 > 100,000,000
    img = Image.new('RGB', (10001, 10001))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/v1/ocr",
        files={"file": ("large_dim.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Image dimensions too large" in response.json()["detail"]

def test_ocr_no_image(client):
    # Empty request body case
    response = client.post("/v1/ocr")
    assert response.status_code == 400
    # The current implementation returns "Empty request body" or "No image provided"

def test_ocr_engine_not_initialized(client, sample_image_path, monkeypatch):
    from src.api import main
    monkeypatch.setattr(main.app.state, "engine", None)

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/v1/ocr",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 503
    assert "Engine not initialized" in response.json()["detail"]

def test_ocr_job_engine_not_initialized(client, sample_image_path, monkeypatch):
    from src.api import main
    monkeypatch.setattr(main.app.state, "engine", None)

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/v1/ocr/jobs",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # Background task should set status to failed
    time.sleep(1)
    response = client.get(f"/v1/ocr/jobs/{job_id}")
    assert response.json()["status"] == "failed"
    assert "Engine not initialized" in response.json()["error"]

def test_ocr_internal_error(client, sample_image_path, monkeypatch):
    from src.api import main

    def mock_ocr(*args, **kwargs):
        raise Exception("Simulated OCR error")

    monkeypatch.setattr(main.app.state.engine, "ocr", mock_ocr)

    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/v1/ocr",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 500
    assert "An internal error occurred during OCR processing" in response.json()["detail"]
