import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from src.api.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def sample_image_path():
    path = Path("extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg")
    if not path.exists():
        pytest.skip(f"Sample image {path} not found")
    return path

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_ocr_upload(sample_image_path):
    with TestClient(app) as client:
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

def test_ocr_base64(sample_image_path):
    import base64
    with TestClient(app) as client:
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

def test_ocr_job_workflow(sample_image_path):
    import time
    with TestClient(app) as client:
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
