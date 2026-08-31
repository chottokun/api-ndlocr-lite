from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks, Depends
from contextlib import asynccontextmanager
import asyncio
import io
import uuid
import json
import binascii
import PIL
from PIL import Image
import base64
import pydantic
from typing import Optional, Dict, Any
import os
import logging

import time
from src.core.engine import NDLOCREngine
from src.schemas.ocr import OCRResponse, OCRPage, OCRLine, OCRJobResponse, OCRJobResult
from src.schemas.openai import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionChoiceMessage
from src.api.job_store import InMemoryJobStore
from src.api.image_utils import (
    _get_image_from_request,
    MAX_IMAGE_SIZE as MAX_IMAGE_SIZE,
    MAX_BODY_SIZE as MAX_BODY_SIZE,
    MAX_PIXELS as MAX_PIXELS,
)

# Configure logging to provide visibility into API operations and background tasks
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _engine_result_to_ocr_page(result: Dict[str, Any], index: int = 0) -> OCRPage:
    """
    Maps the raw output dictionary from NDLOCREngine to the OCRPage Pydantic model.

    Args:
        result: Dictionary containing 'text', 'img_info', and 'lines' from the engine.
        index: Page index (defaults to 0 as currently single-page is supported).
    """
    return OCRPage(
        index=index,
        markdown=result["text"],
        width=result["img_info"]["width"],
        height=result["img_info"]["height"],
        lines=[
            OCRLine(
                id=line["id"],
                text=line["text"],
                confidence=line["confidence"],
                boundingBox=line["boundingBox"],
                isVertical=line.get("isVertical"),
                isTextline=line.get("isTextline"),
                class_index=line.get("class_index")
            ) for line in result["lines"]
        ]
    )

JOB_STORE_MAX_SIZE = int(os.getenv("JOB_STORE_MAX_SIZE", "1000"))
JOB_STORE_TTL_SECONDS = float(os.getenv("JOB_STORE_TTL_SECONDS", "3600.0"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    Initializes the OCR engine (loading ONNX models) and job store on startup.
    Ensures graceful shutdown of engine resources (thread pools).
    """
    logger.info("Initializing NDLOCR Engine and Job Store...")
    # Initialize engine on CPU by default.
    # Models are loaded once and stored in app.state for sharing across requests.
    enable_tcy = os.getenv("ENABLE_TCY", "false").lower() == "true"
    app.state.engine = NDLOCREngine(device="cpu", enable_tcy=enable_tcy)
    app.state.job_store = InMemoryJobStore(max_size=JOB_STORE_MAX_SIZE, ttl_seconds=JOB_STORE_TTL_SECONDS)
    yield
    logger.info("Shutting down...")
    # Clean up engine resources (e.g., ThreadPoolExecutor)
    if hasattr(app.state, "engine") and app.state.engine is not None:
        app.state.engine.shutdown()

app = FastAPI(title="NDLOCR-Lite API", lifespan=lifespan)

# Dependency injection helpers to access shared state
def get_engine(request: Request) -> Optional[NDLOCREngine]:
    return getattr(request.app.state, "engine", None)

def get_job_store(request: Request) -> Optional[InMemoryJobStore]:
    return getattr(request.app.state, "job_store", None)

def process_ocr_job(job_id: str, img: Image.Image, filename: str, engine: NDLOCREngine, job_store: InMemoryJobStore):
    """
    Background worker function for asynchronous OCR processing.
    Updates the job status in the JobStore throughout the process.
    """
    job = job_store.get(job_id)
    if job is None:
        return

    if engine is None:
        job.status = "failed"
        job.error = "Engine not initialized"
        return

    try:
        job.status = "processing"
        # Synchronous call to engine.ocr (run in background task thread)
        result = engine.ocr(img, img_name=filename)
        
        # Convert engine output to API schema
        page = _engine_result_to_ocr_page(result)
        
        job.result = OCRResponse(
            model="ndlocr-lite",
            pages=[page],
            usage={"pages": 1}
        )
        job.status = "completed"
    except Exception:
        # Log unexpected errors to aid debugging while keeping client error messages generic
        logger.exception("An error occurred during background OCR processing")
        job.status = "failed"
        job.error = "An internal error occurred during OCR processing"

@app.post("/v1/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    engine: NDLOCREngine = Depends(get_engine),
):
    """
    Synchronous OCR endpoint.
    Processes the provided image (file or base64) and returns results immediately.
    """
    # Extract image from multipart/form-data or JSON body
    img, filename = await _get_image_from_request(request, file)
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client closed connection")

    try:
        # Run CPU-bound OCR processing in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, engine.ocr, img, filename)

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client closed connection")

        # Convert and return response
        page = _engine_result_to_ocr_page(result)
        return OCRResponse(model="ndlocr-lite", pages=[page], usage={"pages": 1})
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("An error occurred during synchronous OCR processing")
        raise HTTPException(status_code=500, detail="An internal error occurred during OCR processing")

@app.post("/v1/ocr/jobs", response_model=OCRJobResponse)
async def create_ocr_job(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
    engine: NDLOCREngine = Depends(get_engine),
    job_store: InMemoryJobStore = Depends(get_job_store),
):
    """
    Asynchronous OCR endpoint.
    Accepts an image, initializes a background job, and returns a job_id for status polling.
    """
    # Extract image first to ensure it's valid before accepting the job
    img, filename = await _get_image_from_request(request, file)
    
    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client closed connection")

    job_id = str(uuid.uuid4())
    job_store.set(job_id, OCRJobResult(job_id=job_id, status="pending"))
    
    # Delegate processing to background task
    background_tasks.add_task(process_ocr_job, job_id, img, filename, engine, job_store)
    
    return OCRJobResponse(job_id=job_id, status="pending")

@app.get("/v1/ocr/jobs/{job_id}", response_model=OCRJobResult)
async def get_ocr_job(job_id: str, job_store: InMemoryJobStore = Depends(get_job_store)):
    """
    Poll the status and results of an asynchronous OCR job.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def openai_vision_endpoint(
    raw_request: Request,
    request: ChatCompletionRequest,
    engine: Optional[NDLOCREngine] = Depends(get_engine),
):
    """
    OpenAI-compatible Vision API endpoint.
    Extracts image from the messages and returns OCR results as a chat completion.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if await raw_request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client closed connection")

    img_data = None
    for message in request.messages:
        if message.role == "user" and isinstance(message.content, list):
            for part in message.content:
                if hasattr(part, "type") and part.type == "image_url":
                    img_data = part.image_url.url
                    break
        if img_data:
            break

    if not img_data:
        raise HTTPException(status_code=400, detail="No image provided in messages")

    try:
        # Remove data URI prefix if present
        header, encoded = img_data.split(",", 1) if "," in img_data else (None, img_data)
        contents = await asyncio.to_thread(base64.b64decode, encoded)
        img = await asyncio.to_thread(Image.open, io.BytesIO(contents))

        # Dimension check
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(status_code=400, detail="Image dimensions too large")

        if await raw_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client closed connection")

        # Run OCR
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, engine.ocr, img, "openai_image.jpg")

        if await raw_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client closed connection")

        # Format response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(
                        role="assistant",
                        content=result["text"],
                        tool_calls=[{
                            "id": f"call_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "name": "get_ocr_details",
                                "arguments": json.dumps({"pages": [_engine_result_to_ocr_page(result).model_dump()]})
                            }
                        }]
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": 0, # Tokens are not really applicable here but kept for compatibility
                "completion_tokens": len(result["text"]),
                "total_tokens": len(result["text"])
            }
        )
    except HTTPException:
        raise
    except pydantic.ValidationError:
        logger.exception("ValidationError occurred during OpenAI-compatible OCR processing")
        raise HTTPException(status_code=500, detail="An internal error occurred during OCR processing")
    except (binascii.Error, PIL.UnidentifiedImageError, ValueError) as e:
        logger.warning(f"Invalid image in OpenAI request: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid request: Invalid image data or format")
    except Exception:
        logger.exception("An error occurred during OpenAI-compatible OCR processing")
        raise HTTPException(status_code=500, detail="An internal error occurred during OCR processing")

@app.get("/health")
async def health(request: Request):
    """
    Health check endpoint.
    Indicates if the API is running and if the OCR engine has been initialized.
    """
    engine_ready = hasattr(request.app.state, "engine") and request.app.state.engine is not None
    return {"status": "ok", "engine_ready": engine_ready}
