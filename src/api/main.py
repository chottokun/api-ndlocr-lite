from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body, BackgroundTasks, Depends
from contextlib import asynccontextmanager
import asyncio
import io
import uuid
import json
import binascii
import PIL
from PIL import Image
import base64
from typing import List, Optional, Dict, Any
import os
import logging

from src.core.engine import NDLOCREngine
from src.schemas.ocr import OCRResponse, OCRPage, OCRLine, OCRRequest, OCRJobResponse, OCRJobResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InMemoryJobStore:
    """In-memory job store for OCR results."""
    def __init__(self):
        self._jobs: Dict[str, OCRJobResult] = {}

    def get(self, job_id: str) -> Optional[OCRJobResult]:
        return self._jobs.get(job_id)

    def set(self, job_id: str, result: OCRJobResult):
        self._jobs[job_id] = result

    def exists(self, job_id: str) -> bool:
        return job_id in self._jobs

def _engine_result_to_ocr_page(result: Dict[str, Any], index: int = 0) -> OCRPage:
    """Helper to convert engine result to OCRPage schema."""
    return OCRPage(
        index=index,
        markdown=result["text"],
        width=result["img_info"]["width"],
        height=result["img_info"]["height"],
        lines=[
            OCRLine(
                id=l["id"],
                text=l["text"],
                confidence=l["confidence"],
                boundingBox=l["boundingBox"]
            ) for l in result["lines"]
        ]
    )

# Security limits
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024)) # Default 10MB
MAX_BODY_SIZE = int(os.getenv("MAX_BODY_SIZE", 15 * 1024 * 1024))   # Default 15MB
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 100_000_000))            # Default 100MP

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing NDLOCR Engine and Job Store...")
    app.state.engine = NDLOCREngine(device="cpu")
    app.state.job_store = InMemoryJobStore()
    yield
    logger.info("Shutting down...")
    if hasattr(app.state, "engine") and app.state.engine is not None:
        app.state.engine.shutdown()

app = FastAPI(title="NDLOCR-Lite API", lifespan=lifespan)

def process_ocr_job(job_id: str, img: Image.Image, filename: str, engine: NDLOCREngine, job_store: InMemoryJobStore):
    job = job_store.get(job_id)
    if job is None:
        return

    if engine is None:
        job.status = "failed"
        job.error = "Engine not initialized"
        return

    try:
        job.status = "processing"
        result = engine.ocr(img, img_name=filename)
        
        page = _engine_result_to_ocr_page(result)
        
        job.result = OCRResponse(
            model="ndlocr-lite",
            pages=[page],
            usage={"pages": 1}
        )
        job.status = "completed"
    except Exception:
        logger.exception("An error occurred during background OCR processing")
        job.status = "failed"
        job.error = "An internal error occurred during OCR processing"

@app.post("/v1/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    # Same as before... but uses existing implementation
    img, filename = await _get_image_from_request(request, file)
    
    engine: NDLOCREngine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, engine.ocr, img, filename)

        page = _engine_result_to_ocr_page(result)
        return OCRResponse(model="ndlocr-lite", pages=[page], usage={"pages": 1})
    except Exception:
        logger.exception("An error occurred during synchronous OCR processing")
        raise HTTPException(status_code=500, detail="An internal error occurred during OCR processing")

@app.post("/v1/ocr/jobs", response_model=OCRJobResponse)
async def create_ocr_job(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    img, filename = await _get_image_from_request(request, file)
    
    job_id = str(uuid.uuid4())
    job_store: InMemoryJobStore = request.app.state.job_store
    engine: NDLOCREngine = request.app.state.engine

    job_store.set(job_id, OCRJobResult(job_id=job_id, status="pending"))
    
    background_tasks.add_task(process_ocr_job, job_id, img, filename, engine, job_store)
    
    return OCRJobResponse(job_id=job_id, status="pending")

@app.get("/v1/ocr/jobs/{job_id}", response_model=OCRJobResult)
async def get_ocr_job(request: Request, job_id: str):
    job_store: InMemoryJobStore = request.app.state.job_store
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

async def _get_image_from_request(request: Request, file: Optional[UploadFile]):
    img = None
    filename = "image.jpg"
    if file:
        contents = await file.read(MAX_IMAGE_SIZE + 1)
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        img = Image.open(io.BytesIO(contents))
        filename = file.filename or "uploaded_image.jpg"
    else:
        try:
            cl = request.headers.get("Content-Length")
            if cl and int(cl) > MAX_BODY_SIZE:
                raise HTTPException(status_code=413, detail="Request body too large")

            body_bytes = b""
            async for chunk in request.stream():
                body_bytes += chunk
                if len(body_bytes) > MAX_BODY_SIZE:
                    raise HTTPException(status_code=413, detail="Request body too large")

            if not body_bytes:
                raise HTTPException(status_code=400, detail="Empty request body")

            body = json.loads(body_bytes)
            ocr_req = OCRRequest(**body)
            header, encoded = ocr_req.image.split(",", 1) if "," in ocr_req.image else (None, ocr_req.image)
            contents = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(contents))
            filename = "base64_image.jpg"
        except HTTPException:
            raise
        except (binascii.Error, PIL.UnidentifiedImageError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        except Exception:
            logger.exception("Unexpected error while parsing image from request")
            raise HTTPException(status_code=500, detail="An internal error occurred while processing the request")
    
    if img is None:
        raise HTTPException(status_code=400, detail="No image provided")

    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(status_code=400, detail="Image dimensions too large")

    return img, filename

@app.get("/health")
async def health(request: Request):
    engine_ready = hasattr(request.app.state, "engine") and request.app.state.engine is not None
    return {"status": "ok", "engine_ready": engine_ready}
