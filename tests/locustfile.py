import time
import base64
from locust import HttpUser, task, between
from pathlib import Path

SAMPLE_IMAGE_PATH = Path("extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg")

class OCRUser(HttpUser):
    wait_time = between(1, 5)
    image_b64 = None

    def on_start(self):
        # Cache base64 image on startup to avoid redundant file I/O
        if SAMPLE_IMAGE_PATH.exists() and OCRUser.image_b64 is None:
            with open(SAMPLE_IMAGE_PATH, "rb") as f:
                OCRUser.image_b64 = base64.b64encode(f.read()).decode()

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(5)
    def ocr_sync(self):
        if not SAMPLE_IMAGE_PATH.exists():
            return

        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            self.client.post(
                "/v1/ocr",
                files={"file": ("sample.jpg", f, "image/jpeg")}
            )

    @task(3)
    def ocr_async_workflow(self):
        if not SAMPLE_IMAGE_PATH.exists():
            return

        # Create job
        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            response = self.client.post(
                "/v1/ocr/jobs",
                files={"file": ("sample.jpg", f, "image/jpeg")}
            )

        if response.status_code != 200:
            return

        job_id = response.json().get("job_id")
        if not job_id:
            return

        # Poll for results
        max_retries = 10
        for _ in range(max_retries):
            time.sleep(2)
            res = self.client.get(f"/v1/ocr/jobs/{job_id}", name="/v1/ocr/jobs/[job_id]")
            if res.status_code == 200:
                data = res.json()
                if data["status"] in ["completed", "failed"]:
                    break
            else:
                break

    @task(4)
    def openai_vision_api(self):
        if not self.image_b64:
            return

        payload = {
            "model": "ndlocr-lite",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{self.image_b64}"}
                        }
                    ]
                }
            ]
        }
        self.client.post("/v1/chat/completions", json=payload)

