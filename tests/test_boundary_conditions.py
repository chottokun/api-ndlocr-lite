import io
import pytest
from unittest.mock import MagicMock, AsyncMock
from starlette.requests import ClientDisconnect
from starlette.requests import Request
from starlette.datastructures import Headers
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app, _get_image_from_request
from src.core.engine import NDLOCREngine

@pytest.mark.asyncio
async def test_get_image_from_request_disconnect_file():
    # Mocking file upload exception during read
    mock_request = MagicMock(spec=Request)
    mock_file = AsyncMock()
    mock_file.read.side_effect = ClientDisconnect()

    with pytest.raises(HTTPException) as exc_info:
        await _get_image_from_request(mock_request, mock_file)

    assert exc_info.value.status_code == 499
    assert exc_info.value.detail == "Client closed connection"


@pytest.mark.asyncio
async def test_get_image_from_request_disconnect_body():
    # Mocking stream request chunk reading exception
    mock_request = MagicMock(spec=Request)
    mock_request.headers = Headers()

    async def mock_stream():
        raise ClientDisconnect()
        yield b""  # unreachable, but defines it as generator/async iterator

    mock_request.stream = mock_stream

    with pytest.raises(HTTPException) as exc_info:
        await _get_image_from_request(mock_request, None)

    assert exc_info.value.status_code == 499
    assert exc_info.value.detail == "Client closed connection"


def test_ocr_endpoint_early_disconnect(monkeypatch):
    # If client is already disconnected when hitting the endpoint
    async def mock_is_disconnected(self):
        return True

    monkeypatch.setattr(Request, "is_disconnected", mock_is_disconnected)

    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 499
        assert response.json()["detail"] == "Client closed connection"


def test_ocr_endpoint_post_run_disconnect(monkeypatch):
    # Client disconnects right after engine.ocr runs
    disconnect_call_count = 0

    async def mock_is_disconnected(self):
        nonlocal disconnect_call_count
        disconnect_call_count += 1
        # First check (before engine ocr) -> False
        # Second check (after engine ocr) -> True
        return disconnect_call_count > 1

    monkeypatch.setattr(Request, "is_disconnected", mock_is_disconnected)

    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 499
        assert response.json()["detail"] == "Client closed connection"


def test_engine_shutdown_with_pending_tasks():
    # Verify that engine shutdown terminates and cancels pending tasks in executor
    engine = NDLOCREngine(device="cpu")

    # Mocking actual model execution to avoid overhead or dependencies
    engine._load_models = MagicMock()

    # Setup some heavy task in executor
    def dummy_task():
        import time
        time.sleep(10)

    future = engine.executor.submit(dummy_task)
    assert future.running() or future.pending()

    # Shutdown engine
    engine.shutdown()

    # Submitting a task post-shutdown should raise a RuntimeError
    with pytest.raises(RuntimeError):
        engine.executor.submit(dummy_task)


def test_openai_vision_endpoint_early_disconnect(monkeypatch):
    # Test OpenAI chat completion endpoint when client is disconnected early
    async def mock_is_disconnected(self):
        return True

    monkeypatch.setattr(Request, "is_disconnected", mock_is_disconnected)

    img = Image.new('RGB', (50, 50), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    import base64
    b64_str = "data:image/jpeg;base64," + base64.b64encode(img_byte_arr.getvalue()).decode()

    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_str}}
                ]
            }
        ]
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 499
        assert response.json()["detail"] == "Client closed connection"


def test_openai_vision_endpoint_post_run_disconnect(monkeypatch):
    # Test OpenAI chat completion endpoint when client disconnects right after engine.ocr runs
    disconnect_call_count = 0

    async def mock_is_disconnected(self):
        nonlocal disconnect_call_count
        disconnect_call_count += 1
        return disconnect_call_count > 1

    monkeypatch.setattr(Request, "is_disconnected", mock_is_disconnected)

    img = Image.new('RGB', (50, 50), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    import base64
    b64_str = "data:image/jpeg;base64," + base64.b64encode(img_byte_arr.getvalue()).decode()

    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_str}}
                ]
            }
        ]
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 499
        assert response.json()["detail"] == "Client closed connection"


def test_create_ocr_job_early_disconnect(monkeypatch):
    # Test /v1/ocr/jobs endpoint when client is disconnected early
    async def mock_is_disconnected(self):
        return True

    monkeypatch.setattr(Request, "is_disconnected", mock_is_disconnected)

    img = Image.new('RGB', (50, 50), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/jobs",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 499
        assert response.json()["detail"] == "Client closed connection"


