import pytest
from pydantic import ValidationError
from src.schemas.ocr import (
    OCRBoundingBox,
    OCRLine,
    OCRPage,
    OCRResponse,
    OCRJobResponse,
    OCRJobResult,
    OCRRequest
)

def test_ocr_bounding_box():
    # Valid data
    valid_points = [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]]
    bbox = OCRBoundingBox(points=valid_points)
    assert bbox.points == valid_points

    # Invalid data - not a list
    with pytest.raises(ValidationError):
        OCRBoundingBox(points="invalid")

    # Invalid data - nested list with wrong types
    with pytest.raises(ValidationError):
        OCRBoundingBox(points=[["a", "b"]])

def test_ocr_line():
    # Valid data
    valid_data = {
        "id": 1,
        "text": "Hello World",
        "confidence": 0.95,
        "boundingBox": [[0, 0], [0, 10], [10, 10], [10, 0]]
    }
    line = OCRLine(**valid_data)
    assert line.id == 1
    assert line.text == "Hello World"
    assert line.confidence == 0.95

    # Missing required field
    invalid_data = valid_data.copy()
    del invalid_data["text"]
    with pytest.raises(ValidationError):
        OCRLine(**invalid_data)

def test_ocr_page():
    line_data = {
        "id": 1,
        "text": "Hello",
        "confidence": 0.9,
        "boundingBox": [[0, 0], [1, 1]]
    }
    page_data = {
        "index": 0,
        "markdown": "# Page 1",
        "width": 1000,
        "height": 2000,
        "lines": [line_data]
    }
    page = OCRPage(**page_data)
    assert page.index == 0
    assert len(page.lines) == 1
    assert isinstance(page.lines[0], OCRLine)

def test_ocr_response():
    page_data = {
        "index": 0,
        "markdown": "test",
        "width": 100,
        "height": 100,
        "lines": []
    }
    response_data = {
        "model": "test-model",
        "pages": [page_data]
    }
    response = OCRResponse(**response_data)
    assert response.model == "test-model"
    assert response.usage == {}  # Default factory

def test_ocr_job_response():
    data = {"job_id": "123", "status": "pending"}
    job_resp = OCRJobResponse(**data)
    assert job_resp.job_id == "123"
    assert job_resp.status == "pending"

def test_ocr_job_result():
    # Result case
    res_data = {
        "job_id": "123",
        "status": "completed",
        "result": {
            "model": "m",
            "pages": [],
            "usage": {}
        }
    }
    result = OCRJobResult(**res_data)
    assert result.status == "completed"
    assert result.result is not None
    assert result.error is None

    # Error case
    err_data = {
        "job_id": "123",
        "status": "failed",
        "error": "Something went wrong"
    }
    result = OCRJobResult(**err_data)
    assert result.status == "failed"
    assert result.error == "Something went wrong"
    assert result.result is None

def test_ocr_request():
    # Valid
    req = OCRRequest(image="base64string")
    assert req.image == "base64string"
    assert req.model == "ndlocr-lite"

    # Max length exceeded (15MB = 15 * 1024 * 1024 = 15728640)
    too_large = "a" * (15 * 1024 * 1024 + 1)
    with pytest.raises(ValidationError):
        OCRRequest(image=too_large)
