import pytest
import base64
import json
from unittest.mock import patch
import pydantic
from fastapi.testclient import TestClient
from pathlib import Path
from src.api.main import app

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

def test_openai_vision_api_success(client, sample_image_path):
    # Encode sample image to base64
    with open(sample_image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                    }
                ]
            }
        ]
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    
    message = data["choices"][0]["message"]
    assert "role" in message
    assert message["role"] == "assistant"
    assert message["content"] is not None
    assert "第8章" in message["content"]
    
    # Check tool_calls containing details
    assert "tool_calls" in message
    assert len(message["tool_calls"]) > 0
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_ocr_details"
    
    arguments = json.loads(tool_call["function"]["arguments"])
    assert "pages" in arguments
    assert len(arguments["pages"]) > 0
    page = arguments["pages"][0]
    
    # Ensure new fields isVertical and isTextline are mapped and exist in details
    assert len(page["lines"]) > 0
    for line in page["lines"]:
        assert "isVertical" in line
        assert line["isVertical"] in ["true", "false"]
        assert "isTextline" in line
        assert line["isTextline"] == "true"

def test_openai_vision_api_no_image(client):
    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image without actually providing one"}
                ]
            }
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "No image provided" in response.json()["detail"]

def test_openai_vision_api_invalid_image(client):
    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this invalid image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,invalidbase64data!!!"}
                    }
                ]
            }
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "Invalid request" in response.json()["detail"]

def test_ocr_lines_new_fields_in_standard_api(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        response = client.post(
            "/v1/ocr",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "pages" in data
    assert len(data["pages"]) > 0
    page = data["pages"][0]
    assert len(page["lines"]) > 0
    for line in page["lines"]:
        assert "isVertical" in line
        assert line["isVertical"] in ["true", "false"]
        assert "isTextline" in line
        assert line["isTextline"] == "true"


def test_openai_vision_api_validation_error(client, sample_image_path):
    with open(sample_image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                    }
                ]
            }
        ]
    }

    class DummyModel(pydantic.BaseModel):
        val: int

    with patch("src.api.main._engine_result_to_ocr_page") as mock_map:
        try:
            DummyModel(val="invalid")
        except pydantic.ValidationError as e:
            mock_map.side_effect = e

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 500
        assert "internal error" in response.json()["detail"].lower()

def test_openai_vision_api_too_large_image(client):
    from PIL import Image
    import io
    # Create a 10001x10001 image in memory (over MAX_PIXELS limit)
    img = Image.new("RGB", (10001, 10001), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this too large image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                    }
                ]
            }
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "dimensions too large" in response.json()["detail"].lower()

def test_openai_vision_api_empty_messages(client):
    payload = {
        "model": "ndlocr-lite",
        "messages": []
    }
    response = client.post("/v1/chat/completions", json=payload)
    # The endpoint catches empty messages list and throws 400 Bad Request
    assert response.status_code == 400
    assert "No image provided" in response.json()["detail"]


