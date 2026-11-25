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
from dotenv import find_dotenv, load_dotenv
from src.core.engine import NDLOCREngine
from src.schemas.ocr import OCRResponse, OCRPage, OCRLine, OCRTable, OCRJobResponse, OCRJobResult
from src.schemas.openai import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionChoiceMessage
from src.api.job_store import InMemoryJobStore
from src.api.image_utils import (
    _get_image_from_request,
    _get_image_and_options_from_request,
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
    raw_tables = result.get("tables")
    tables = [OCRTable(**table) for table in raw_tables] if raw_tables is not None else None
    return OCRPage(
        index=index,
        markdown=result.get("markdown", result["text"]),
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
        ],
        tables=tables,
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
    # .env ファイルが存在する場合は読み込む（既に設定済みの環境変数は上書きしない）
    load_dotenv(find_dotenv(usecwd=True), override=False)
    logger.info("Initializing NDLOCR Engine and Job Store...")
    # Initialize engine on the device selected by the DEVICE env var
    # ("cpu" by default, "cuda" for GPU builds). Models are loaded once
    # and stored in app.state for sharing across requests.
    device = os.getenv("DEVICE", "cpu").lower()
    if device not in ("cpu", "cuda"):
        logger.warning("Unknown DEVICE=%r; falling back to cpu", device)
        device = "cpu"
    enable_tcy = os.getenv("ENABLE_TCY", "false").lower() == "true"
    enable_character_inclusion = os.getenv("ENABLE_CHARACTER_INCLUSION", "false").lower() == "true"
    enable_rich_markdown = os.getenv("ENABLE_RICH_MARKDOWN", "false").lower() == "true"
    enable_table_structure = os.getenv("ENABLE_TABLE_STRUCTURE", "false").lower() == "true"
    app.state.engine = NDLOCREngine(
        device=device,
        enable_tcy=enable_tcy,
        enable_character_inclusion=enable_character_inclusion,
        enable_rich_markdown=enable_rich_markdown,
        enable_table_structure=enable_table_structure,
    )
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

def _resolve_ocr_options(
    query_character_inclusion: Optional[bool],
    query_rich_markdown: Optional[bool],
    query_table_structure: Optional[bool],
    body_options: Optional[Dict[str, Optional[bool]]] = None,
) -> Dict[str, Optional[bool]]:
    """クエリパラメータ > JSON ボディ > サーバーデフォルト(.env) の優先順位でオプションを解決する。"""
    body_options = body_options or {}
    return {
        "character_inclusion": (
            query_character_inclusion
            if query_character_inclusion is not None
            else body_options.get("character_inclusion")
        ),
        "rich_markdown": (
            query_rich_markdown
            if query_rich_markdown is not None
            else body_options.get("rich_markdown")
        ),
        "table_structure": (
            query_table_structure
            if query_table_structure is not None
            else body_options.get("table_structure")
        ),
    }


def _call_engine_ocr(
    engine: NDLOCREngine,
    img: Image.Image,
    filename: str,
    options: Optional[Dict[str, Optional[bool]]] = None,
) -> Dict[str, Any]:
    """オプション未指定時は従来通り位置引数のみで engine.ocr を呼び出し互換性を保つ。"""
    if not options or all(value is None for value in options.values()):
        return engine.ocr(img, filename)
    return engine.ocr(
        img,
        filename,
        enable_character_inclusion=options.get("character_inclusion"),
        enable_rich_markdown=options.get("rich_markdown"),
        enable_table_structure=options.get("table_structure"),
    )


def process_ocr_job(
    job_id: str,
    img: Image.Image,
    filename: str,
    engine: NDLOCREngine,
    job_store: InMemoryJobStore,
    options: Optional[Dict[str, Optional[bool]]] = None,
):
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
        result = _call_engine_ocr(engine, img, filename, options)
        
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
    character_inclusion: Optional[bool] = None,
    rich_markdown: Optional[bool] = None,
    table_structure: Optional[bool] = None,
    engine: NDLOCREngine = Depends(get_engine),
):
    """
    Synchronous OCR endpoint.
    Processes the provided image (file or base64) and returns results immediately.
    """
    # Extract image and optional parameters from multipart/form-data or JSON body
    img, filename, body_options = await _get_image_and_options_from_request(request, file)
    options = _resolve_ocr_options(
        character_inclusion, rich_markdown, table_structure, body_options
    )
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client closed connection")

    try:
        # Run CPU-bound OCR processing in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call_engine_ocr, engine, img, filename, options)

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
    character_inclusion: Optional[bool] = None,
    rich_markdown: Optional[bool] = None,
    table_structure: Optional[bool] = None,
    engine: NDLOCREngine = Depends(get_engine),
    job_store: InMemoryJobStore = Depends(get_job_store),
):
    """
    Asynchronous OCR endpoint.
    Accepts an image, initializes a background job, and returns a job_id for status polling.
    """
    # Extract image first to ensure it's valid before accepting the job
    img, filename, body_options = await _get_image_and_options_from_request(request, file)
    options = _resolve_ocr_options(
        character_inclusion, rich_markdown, table_structure, body_options
    )
    
    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client closed connection")

    job_id = str(uuid.uuid4())
    job_store.set(job_id, OCRJobResult(job_id=job_id, status="pending"))
    
    # Delegate processing to background task
    background_tasks.add_task(process_ocr_job, job_id, img, filename, engine, job_store, options)
    
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
    character_inclusion: Optional[bool] = None,
    rich_markdown: Optional[bool] = None,
    table_structure: Optional[bool] = None,
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
        options = _resolve_ocr_options(
            character_inclusion,
            rich_markdown,
            table_structure,
            {
                "character_inclusion": request.character_inclusion,
                "rich_markdown": request.rich_markdown,
                "table_structure": request.table_structure,
            },
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call_engine_ocr, engine, img, "openai_image.jpg", options)

        if await raw_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client closed connection")

        # Format response
        response_text = result.get("markdown", result["text"])
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(
                        role="assistant",
                        content=response_text,
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
                "completion_tokens": len(response_text),
                "total_tokens": len(response_text)
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
