from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body, BackgroundTasks
from contextlib import asynccontextmanager
import io
import uuid
from PIL import Image
import base64
from typing import List, Optional, Dict, Any

from src.core.engine import NDLOCREngine
from src.schemas.ocr import OCRResponse, OCRPage, OCRLine, OCRRequest, OCRJobResponse, OCRJobResult

# Global engine instance
engine: Optional[NDLOCREngine] = None
# In-memory job store
jobs: Dict[str, OCRJobResult] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("[INFO] Initializing NDLOCR Engine...")
    engine = NDLOCREngine(device="cpu")
    yield
    print("[INFO] Shutting down...")
    engine = None

app = FastAPI(title="NDLOCR-Lite API", lifespan=lifespan)

def process_ocr_job(job_id: str, img: Image.Image, filename: str):
    global engine
    if engine is None:
        jobs[job_id].status = "failed"
        jobs[job_id].error = "Engine not initialized"
        return

    try:
        jobs[job_id].status = "processing"
        result = engine.ocr(img, img_name=filename)
        
        page = OCRPage(
            index=0,
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
        
        jobs[job_id].result = OCRResponse(
            model="ndlocr-lite",
            pages=[page],
            usage={"pages": 1}
        )
        jobs[job_id].status = "completed"
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)

@app.post("/v1/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    # Same as before... but uses existing implementation
    img, filename = await _get_image_from_request(request, file)
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        result = engine.ocr(img, img_name=filename)
        # Result conversion...
        page = OCRPage(
            index=0,
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
        return OCRResponse(model="ndlocr-lite", pages=[page], usage={"pages": 1})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ocr/jobs", response_model=OCRJobResponse)
async def create_ocr_job(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    img, filename = await _get_image_from_request(request, file)
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = OCRJobResult(job_id=job_id, status="pending")
    
    background_tasks.add_task(process_ocr_job, job_id, img, filename)
    
    return OCRJobResponse(job_id=job_id, status="pending")

@app.get("/v1/ocr/jobs/{job_id}", response_model=OCRJobResult)
async def get_ocr_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def _get_image_from_request(request: Request, file: Optional[UploadFile]):
    img = None
    filename = "image.jpg"
    if file:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        filename = file.filename or "uploaded_image.jpg"
    else:
        try:
            body = await request.json()
            ocr_req = OCRRequest(**body)
            header, encoded = ocr_req.image.split(",", 1) if "," in ocr_req.image else (None, ocr_req.image)
            contents = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(contents))
            filename = "base64_image.jpg"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    if img is None:
        raise HTTPException(status_code=400, detail="No image provided")
    return img, filename

@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None}
