import pytest
import io
import time
from unittest.mock import MagicMock, patch
from PIL import Image
from fastapi.testclient import TestClient
from src.api.main import app, process_ocr_job, InMemoryJobStore
from src.schemas.ocr import OCRJobResult

def test_process_ocr_job_engine_exception():
    # Setup
    job_id = "test-job-id"
    job_store = InMemoryJobStore()
    job = OCRJobResult(job_id=job_id, status="pending")
    job_store.set(job_id, job)

    mock_engine = MagicMock()
    mock_engine.ocr.side_effect = Exception("Test engine error")

    mock_img = MagicMock(spec=Image.Image)

    # Execute
    process_ocr_job(job_id, mock_img, "test.jpg", mock_engine, job_store)

    # Verify
    updated_job = job_store.get(job_id)
    assert updated_job.status == "failed"
    assert updated_job.error == "An internal error occurred during OCR processing"

def test_process_ocr_job_engine_none():
    # Setup
    job_id = "test-job-id-none"
    job_store = InMemoryJobStore()
    job = OCRJobResult(job_id=job_id, status="pending")
    job_store.set(job_id, job)

    mock_img = MagicMock(spec=Image.Image)

    # Execute
    process_ocr_job(job_id, mock_img, "test.jpg", None, job_store)

    # Verify
    updated_job = job_store.get(job_id)
    assert updated_job.status == "failed"
    assert updated_job.error == "Engine not initialized"

def test_ocr_job_engine_failure_integration():
    # Create a small valid image for the request
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    with TestClient(app) as client:
        # Mock engine.ocr to raise exception
        with patch.object(app.state.engine, 'ocr', side_effect=Exception("Simulated engine failure")):
            response = client.post(
                "/v1/ocr/jobs",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )

            assert response.status_code == 200
            job_data = response.json()
            job_id = job_data["job_id"]

            # Poll for failure
            max_retries = 10
            failed = False
            for _ in range(max_retries):
                response = client.get(f"/v1/ocr/jobs/{job_id}")
                assert response.status_code == 200
                data = response.json()
                if data["status"] == "failed":
                    failed = True
                    assert data["error"] == "An internal error occurred during OCR processing"
                    break
                time.sleep(0.1)

            assert failed, "Job should have failed due to engine exception"
