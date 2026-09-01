import os
import io
import json
import base64
import binascii
import asyncio
import logging
from typing import Optional
from PIL import Image
import PIL
import pydantic
from fastapi import Request, UploadFile, HTTPException
from starlette.requests import ClientDisconnect
from src.schemas.ocr import OCRRequest

logger = logging.getLogger(__name__)

# Security and resource limits
# These limits prevent DoS attacks via large files or excessive pixel counts.
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024)) # Default 10MB
MAX_BODY_SIZE = int(os.getenv("MAX_BODY_SIZE", 15 * 1024 * 1024))   # Default 15MB
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 100_000_000))            # Default 100MP

async def _get_image_from_request(request: Request, file: Optional[UploadFile]):
    """
    Internal helper to extract a PIL Image from the HTTP request.
    Supports:
    - Multipart file upload (via 'file' field)
    - JSON body with base64 encoded image (via 'image' field)

    Includes security checks for body size, file size, and image dimensions.
    """
    import src.api.main as main_mod

    max_img_size = getattr(main_mod, "MAX_IMAGE_SIZE", MAX_IMAGE_SIZE)
    max_body_size = getattr(main_mod, "MAX_BODY_SIZE", MAX_BODY_SIZE)
    max_pixels = getattr(main_mod, "MAX_PIXELS", MAX_PIXELS)

    img = None
    filename = "image.jpg"
    try:
        if file:
            # Handle multipart/form-data
            try:
                contents = await file.read(max_img_size + 1)
            except ClientDisconnect:
                logger.warning("Client disconnected during file upload")
                raise HTTPException(status_code=499, detail="Client closed connection")
            if len(contents) > max_img_size:
                raise HTTPException(status_code=413, detail="File too large")
            img = await asyncio.to_thread(Image.open, io.BytesIO(contents))
            filename = file.filename or "uploaded_image.jpg"
        else:
            # Handle JSON body (Base64)
            cl = request.headers.get("Content-Length")
            if cl and int(cl) > max_body_size:
                raise HTTPException(status_code=413, detail="Request body too large")

            body_bytes = b""
            try:
                async for chunk in request.stream():
                    body_bytes += chunk
                    if len(body_bytes) > max_body_size:
                        raise HTTPException(status_code=413, detail="Request body too large")
            except ClientDisconnect:
                logger.warning("Client disconnected during body streaming")
                raise HTTPException(status_code=499, detail="Client closed connection")

            if not body_bytes:
                raise HTTPException(status_code=400, detail="Empty request body")

            body = json.loads(body_bytes)
            ocr_req = OCRRequest(**body)
            # Remove data URI prefix if present
            header, encoded = ocr_req.image.split(",", 1) if "," in ocr_req.image else (None, ocr_req.image)
            contents = await asyncio.to_thread(base64.b64decode, encoded)
            img = await asyncio.to_thread(Image.open, io.BytesIO(contents))
            filename = "base64_image.jpg"
    except HTTPException:
        raise
    except (binascii.Error, PIL.UnidentifiedImageError, ValueError, pydantic.ValidationError) as e:
        # Catch image decoding errors and return 400 Bad Request
        logger.warning(f"Invalid image request: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid request: Invalid image data or format")
    except Exception as e:
        # Check if it's already an HTTPException (raised from file size/body size checks)
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Unexpected error while parsing image from request")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request")

    if img is None:
        raise HTTPException(status_code=400, detail="No image provided")

    # Final dimension check to prevent memory exhaustion
    if img.width * img.height > max_pixels:
        raise HTTPException(status_code=400, detail="Image dimensions too large")

    return img, filename
