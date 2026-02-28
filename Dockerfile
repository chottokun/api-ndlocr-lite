FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project files
COPY . .

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Set PYTHONPATH to include src and submodule src
ENV PYTHONPATH="/app:/app/extern/ndlocr-lite/src"

# Run the application
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
