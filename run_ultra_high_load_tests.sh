#!/bin/bash
set -e

# Ultra-High Load Configuration
API_PORT=8002
LOCUST_USERS=20
LOCUST_SPAWN_RATE=5
RUN_TIME=1m

echo "[INFO] Starting API server on port $API_PORT..."
# Ensure we use a fresh log file
rm -f ultra_server.log
PYTHONPATH=. uv run uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT > ultra_server.log 2>&1 &
SERVER_PID=$!

# Function to kill the server on exit
cleanup() {
    echo "[INFO] Cleaning up..."
    kill $SERVER_PID || true
}
trap cleanup EXIT

# Wait for server to be ready
echo "[INFO] Waiting for server to be ready..."
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:$API_PORT/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[ERROR] Server timed out"
        exit 1
    fi
    sleep 1
done
echo "[INFO] Server is ready."

# Run Locust in headless mode
echo "[INFO] Running Ultra-High Load tests ($RUN_TIME, $LOCUST_USERS users)..."
uv run locust -f tests/locustfile.py --host http://localhost:$API_PORT --headless -u $LOCUST_USERS -r $LOCUST_SPAWN_RATE --run-time $RUN_TIME --only-summary

echo "[INFO] Ultra-High Load tests completed."
