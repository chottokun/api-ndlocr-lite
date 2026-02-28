#!/bin/bash
set -e

# Docker Container Load Test Script
# Usage: ./run_docker_load_tests.sh [cpu|gpu|both]

TARGET="${1:-cpu}"
LOCUST_USERS=5
LOCUST_SPAWN_RATE=1
RUN_TIME=30s

# Read ports from .env or use defaults
CPU_PORT="${API_CPU_PORT:-8000}"
GPU_PORT="${API_GPU_PORT:-8001}"

# Source .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
    CPU_PORT="${API_CPU_PORT:-8000}"
    GPU_PORT="${API_GPU_PORT:-8001}"
fi

wait_for_health() {
    local port=$1
    local name=$2
    echo "[INFO] Waiting for $name container (port $port) to be healthy..."
    MAX_RETRIES=120
    RETRY_COUNT=0
    while ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "[ERROR] $name container timed out after ${MAX_RETRIES}s"
            return 1
        fi
        sleep 1
    done
    echo "[INFO] $name container is ready."
}

run_load_test() {
    local port=$1
    local name=$2
    echo ""
    echo "============================================"
    echo "  Load Test: $name (port $port)"
    echo "  Users: $LOCUST_USERS, Spawn Rate: $LOCUST_SPAWN_RATE"
    echo "  Duration: $RUN_TIME"
    echo "============================================"
    echo ""

    uv run locust -f tests/locustfile.py \
        --host "http://localhost:$port" \
        --headless \
        -u "$LOCUST_USERS" \
        -r "$LOCUST_SPAWN_RATE" \
        --run-time "$RUN_TIME" \
        --only-summary
}

test_ocr_quick() {
    local port=$1
    local name=$2
    echo "[INFO] Quick OCR test on $name (port $port)..."
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        "http://localhost:$port/v1/ocr" \
        -F "file=@extern/ndlocr-lite/resource/digidepo_3048008_0025.jpg")
    if [ "$response" = "200" ]; then
        echo "[OK] OCR test passed ($name)"
    else
        echo "[FAIL] OCR test failed with status $response ($name)"
        return 1
    fi
}

case "$TARGET" in
    cpu)
        echo "[INFO] === CPU Container Test ==="
        docker compose up -d api-cpu
        wait_for_health "$CPU_PORT" "CPU"
        test_ocr_quick "$CPU_PORT" "CPU"
        run_load_test "$CPU_PORT" "CPU"
        docker compose down
        ;;
    gpu)
        echo "[INFO] === GPU Container Test ==="
        docker compose --profile gpu up -d api-gpu
        wait_for_health "$GPU_PORT" "GPU"
        test_ocr_quick "$GPU_PORT" "GPU"
        run_load_test "$GPU_PORT" "GPU"
        docker compose --profile gpu down
        ;;
    both)
        echo "[INFO] === CPU + GPU Comparison Test ==="
        echo ""

        # CPU Test
        echo "[INFO] --- Phase 1: CPU Container ---"
        docker compose up -d api-cpu
        wait_for_health "$CPU_PORT" "CPU"
        test_ocr_quick "$CPU_PORT" "CPU"
        run_load_test "$CPU_PORT" "CPU"
        docker compose down
        echo ""

        # GPU Test
        echo "[INFO] --- Phase 2: GPU Container ---"
        docker compose --profile gpu up -d api-gpu
        wait_for_health "$GPU_PORT" "GPU"
        test_ocr_quick "$GPU_PORT" "GPU"
        run_load_test "$GPU_PORT" "GPU"
        docker compose --profile gpu down
        echo ""

        echo "============================================"
        echo "  Comparison complete!"
        echo "  Check the Locust summaries above."
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 [cpu|gpu|both]"
        echo "  cpu  - Test CPU container only (default)"
        echo "  gpu  - Test GPU container only"
        echo "  both - Test both and compare results"
        exit 1
        ;;
esac

echo "[INFO] Load tests completed."
