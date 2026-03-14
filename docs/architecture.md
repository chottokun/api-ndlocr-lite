# Architecture Documentation

This document describes the architecture and data flow of the NDLOCR-Lite API.

## System Overview

The system is a FastAPI-based wrapper around the [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite) engine. It provides a RESTful API for both synchronous and asynchronous OCR processing, following an OpenAI-compatible response schema.

### Components

1.  **FastAPI Application (`src/api/main.py`)**: Handles HTTP requests, authentication (if any), input validation, and job orchestration.
2.  **NDLOCR Engine (`src/core/engine.py`)**: Wraps the underlying OCR models and logic. It manages model loading (ONNX), detection, reading order analysis, and character recognition.
3.  **Job Store (`InMemoryJobStore`)**: A simple in-memory storage for tracking the status and results of background OCR jobs.
4.  **NDLOCR-Lite (Submodule)**: The core OCR engine provided by NDL, including models for layout detection and character recognition.

---

## High-Level Architecture

```mermaid
graph TD
    Client[Client]
    API[FastAPI /v1/ocr]
    Engine[NDLOCR Engine]
    Sub[NDLOCR-Lite Submodule]
    JS[InMemoryJobStore]

    Client -->|POST /v1/ocr| API
    Client -->|POST /v1/ocr/jobs| API
    Client -->|GET /v1/ocr/jobs/{id}| API

    API -->|sync/async call| Engine
    API <-->|read/write| JS

    Engine -->|uses| Sub
```

---

## Detailed Data Flows

### 1. Synchronous OCR Flow (`POST /v1/ocr`)

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Engine

    Client->>API: POST /v1/ocr (Image/JSON)
    API->>API: Validate & Extract Image
    API->>Engine: engine.ocr(image)
    Note over Engine: Detection -> XML -> Reading Order -> Recognition
    Engine-->>API: Result (Dict)
    API->>API: Map to OCRResponse Schema
    API-->>Client: 200 OK (OCRResponse)
```

### 2. Asynchronous OCR Job Flow (`POST /v1/ocr/jobs`)

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant JobStore
    participant BackgroundTask
    participant Engine

    Client->>API: POST /v1/ocr/jobs
    API->>API: Validate & Extract Image
    API->>JobStore: Create Job (pending)
    API->>BackgroundTask: Start process_ocr_job
    API-->>Client: 202 Accepted (job_id)

    loop Background
        BackgroundTask->>JobStore: Update Status (processing)
        BackgroundTask->>Engine: engine.ocr(image)
        Engine-->>BackgroundTask: Result
        BackgroundTask->>JobStore: Store Result & Status (completed/failed)
    end

    Client->>API: GET /v1/ocr/jobs/{id}
    API->>JobStore: Fetch Job
    JobStore-->>API: Job Data
    API-->>Client: 200 OK (OCRJobResult)
```

### 3. Engine OCR Pipeline (`NDLOCREngine.ocr`)

The engine processes images in the following stages:

```mermaid
graph TD
    Start[Input PIL Image] --> Convert[Convert to RGB Numpy]
    Convert --> Detect[Layout Detection - DEIM]
    Detect --> XML[Convert Detections to XML]
    XML --> RO[Reading Order Analysis - XY-Cut]
    RO --> Extract[Extract Line Images]
    Extract --> Cascade{Recognition Cascade}

    Cascade -->|Small/Medium Char| R30[PARSEQ-30/50]
    Cascade -->|Long/Complex| R100[PARSEQ-100]

    R30 -->|Length Check| R50[Fallback to 50/100]

    R100 --> Final[Consolidate Results]
    Final --> Output[Return JSON-compatible Dict]
```

---

## Technical Details

### Security Measures
- **Input Validation**: Strict limits on image dimensions and file sizes.
- **XXE Protection**: Uses `defusedxml` for all XML parsing to prevent XML-based attacks.
- **Sanitization**: Image filenames are sanitized before being processed in internal XML structures.

### Performance Optimization
- **Model Caching**: Models are loaded once during the FastAPI lifespan and shared across requests.
- **Parallel Recognition**: Line-level recognition is parallelized using a `ThreadPoolExecutor` within the engine.
- **Recognition Cascade**: Uses smaller, faster models for simple cases and cascades to larger models only when necessary.
