import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from pathlib import Path
import time

from src.api.main import app, process_ocr_job, InMemoryJobStore
from src.schemas.ocr import OCRJobResult

@pytest.fixture
def sample_image():
    return Image.new('RGB', (100, 100), color='red')

@pytest.fixture
def sample_image_bytes(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format='JPEG')
    return buf.getvalue()

def test_process_ocr_job_uninitialized_unit(sample_image):
    """Unit test for process_ocr_job with engine=None."""
    job_id = "test-job-id"
    job_store = InMemoryJobStore()
    job_store.set(job_id, OCRJobResult(job_id=job_id, status="pending"))

    # Call process_ocr_job with engine=None
    process_ocr_job(job_id, sample_image, "test.jpg", None, job_store)

    job = job_store.get(job_id)
    assert job.status == "failed"
    assert job.error == "Engine not initialized"

def test_process_ocr_job_uninitialized_integration(monkeypatch, sample_image_bytes):
    """Integration test for OCR job workflow when engine is uninitialized."""
    # We need to ensure app.state.engine is None when the job is created

    with TestClient(app) as client:
        # Manually set engine to None in app state
        app.state.engine = None

        response = client.post(
            "/v1/ocr/jobs",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll for completion (it should fail immediately since engine is None)
        max_retries = 5
        failed = False
        for _ in range(max_retries):
            response = client.get(f"/v1/ocr/jobs/{job_id}")
            assert response.status_code == 200
            data = response.json()
            if data["status"] == "failed":
                assert data["error"] == "Engine not initialized"
                failed = True
                break
            time.sleep(0.1)

        assert failed, "Job did not fail as expected"
