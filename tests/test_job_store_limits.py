import time
import threading
from src.api.main import InMemoryJobStore
from src.schemas.ocr import OCRJobResult

def test_job_store_max_size_eviction():
    # Store with max_size=3
    store = InMemoryJobStore(max_size=3, ttl_seconds=3600.0)

    job1 = OCRJobResult(job_id="job1", status="pending")
    job2 = OCRJobResult(job_id="job2", status="pending")
    job3 = OCRJobResult(job_id="job3", status="pending")
    job4 = OCRJobResult(job_id="job4", status="pending")

    store.set("job1", job1)
    store.set("job2", job2)
    store.set("job3", job3)

    # All should exist
    assert store.exists("job1")
    assert store.exists("job2")
    assert store.exists("job3")

    # Access job1 and job2 to update their timestamps (making job3 the least recently used)
    # Note: Because they are created very close in time, we want to make sure order is clear.
    # To be extremely safe, we will simulate or just set them with slight artificial delays or let the system access them.
    store.get("job1")
    time.sleep(0.01)
    store.get("job2")
    time.sleep(0.01)

    # Now we insert job4, which should trigger LRU eviction on job3
    store.set("job4", job4)

    assert store.exists("job1")
    assert store.exists("job2")
    assert store.exists("job4")
    assert not store.exists("job3")

def test_job_store_ttl_eviction():
    # Store with max_size=10, ttl_seconds=0.05
    store = InMemoryJobStore(max_size=10, ttl_seconds=0.05)

    job1 = OCRJobResult(job_id="job1", status="pending")
    store.set("job1", job1)

    assert store.exists("job1")

    # Sleep past TTL
    time.sleep(0.06)

    # exists/get/set should trigger eviction and return None / False
    assert not store.exists("job1")
    assert store.get("job1") is None

def test_job_store_thread_safety():
    # Ensure no race conditions under high concurrent set/get operations
    store = InMemoryJobStore(max_size=10, ttl_seconds=10.0)

    num_threads = 20
    ops_per_thread = 100
    errors = []

    def worker(thread_idx):
        try:
            for i in range(ops_per_thread):
                job_id = f"thread-{thread_idx}-job-{i}"
                job = OCRJobResult(job_id=job_id, status="pending")
                store.set(job_id, job)
                store.get(job_id)
                store.exists(job_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Encountered thread safety errors: {errors}"
