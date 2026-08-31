import time
import threading
from typing import Optional, Dict
from src.schemas.ocr import OCRJobResult

class InMemoryJobStore:
    """
    In-memory job store for OCR results.
    Used to track the status and final output of asynchronous background jobs.
    In a production environment, this should be replaced with a persistent store like Redis.
    Includes time-to-live (TTL) eviction and LRU eviction to prevent unbounded memory growth.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self._jobs: Dict[str, OCRJobResult] = {}
        self._timestamps: Dict[str, float] = {}  # job_id -> last access timestamp
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def _evict_expired(self, now: float):
        """Removes expired jobs from the store. Must be called under lock."""
        expired_ids = [
            job_id for job_id, ts in list(self._timestamps.items())
            if now - ts > self._ttl_seconds
        ]
        for job_id in expired_ids:
            self._jobs.pop(job_id, None)
            self._timestamps.pop(job_id, None)

    def _evict_lru(self):
        """Removes the oldest (least recently accessed) job. Must be called under lock."""
        if not self._timestamps:
            return
        # Find key with the minimum timestamp
        oldest_job_id = min(self._timestamps, key=self._timestamps.get)
        self._jobs.pop(oldest_job_id, None)
        self._timestamps.pop(oldest_job_id, None)

    def get(self, job_id: str) -> Optional[OCRJobResult]:
        """Retrieves a job result by its ID and updates its access time."""
        with self._lock:
            now = time.time()
            self._evict_expired(now)
            if job_id in self._jobs:
                self._timestamps[job_id] = now
                return self._jobs[job_id]
            return None

    def set(self, job_id: str, result: OCRJobResult):
        """Stores or updates a job result, evicting expired or LRU items if full."""
        with self._lock:
            now = time.time()
            self._evict_expired(now)

            # If still over limit and adding a new key would exceed max_size
            if len(self._jobs) >= self._max_size and job_id not in self._jobs:
                self._evict_lru()

            self._jobs[job_id] = result
            self._timestamps[job_id] = now

    def exists(self, job_id: str) -> bool:
        """Checks if a job with the given ID exists."""
        with self._lock:
            now = time.time()
            self._evict_expired(now)
            return job_id in self._jobs
