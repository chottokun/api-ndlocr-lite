FROM python:3.13-slim

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade "setuptools>=78.1.1" pip

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user and set up work directory
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app
RUN chown appuser:appuser /home/appuser/app

USER appuser

# Copy only the dependency files first for better caching
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
# Optional: also copy submodule's dependency file if needed, 
# although normally top-level handles it.

# Install dependencies strictly into /home/appuser/app/.venv (production only, no cache)
RUN uv sync --frozen --no-dev --no-cache --no-install-project

# Copy the rest of the project files
COPY --chown=appuser:appuser . .

# Final sync to install the project itself
RUN uv sync --frozen --no-dev --no-cache

# Expose port
EXPOSE 8000

# Set PYTHONPATH to include src and submodule src
ENV PYTHONPATH="/home/appuser/app:/home/appuser/app/extern/ndlocr-lite/src"

# Run the application
CMD ["/home/appuser/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
