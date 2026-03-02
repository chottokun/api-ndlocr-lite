from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class OCRLine(BaseModel):
    id: int
    text: str
    confidence: float
    boundingBox: List[List[int]]

class OCRPage(BaseModel):
    index: int
    markdown: str
    width: int
    height: int
    lines: List[OCRLine]

class OCRResponse(BaseModel):
    model: str
    pages: List[OCRPage]
    usage: Dict[str, Any] = Field(default_factory=dict)

class OCRJobResponse(BaseModel):
    job_id: str
    status: str # "pending", "processing", "completed", "failed"

class OCRJobResult(BaseModel):
    job_id: str
    status: str
    result: Optional[OCRResponse] = None
    error: Optional[str] = None

class OCRRequest(BaseModel):
    image: str = Field(..., max_length=15 * 1024 * 1024) # Base64 encoded image (limit 15MB)
    model: Optional[str] = "ndlocr-lite"
