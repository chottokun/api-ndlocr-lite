import asyncio
import base64
import httpx
import sys
import os

async def test_openai_api():
    # Create a small white 1x1 pixel image in base64
    # (Simplified for testing, actual valid JPEG or PNG is better)
    # This is a 1x1 white PNG
    img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    payload = {
        "model": "ndlocr-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ]
    }

    print("Testing /v1/chat/completions...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # We need the server running. Since I'm in a sandbox,
            # I'll start uvicorn in background if not already running.
            response = await client.post("http://localhost:8000/v1/chat/completions", json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Start server in background
    import subprocess
    import time

    # Check if port 8000 is already in use
    try:
        subprocess.run(["lsof", "-i", ":8000"], check=True, capture_output=True)
        print("Server already running on 8000")
    except subprocess.CalledProcessError:
        print("Starting uvicorn...")
        # Start uvicorn. Use a small image limit for testing if needed.
        proc = subprocess.Popen(["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
                                env={**os.environ, "PYTHONPATH": "."})
        time.sleep(15) # Wait for engine to initialize

    asyncio.run(test_openai_api())

    # Note: I'm not killing the server here to allow further tests if needed.
