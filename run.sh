#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Starting FinSight AI..."
echo "Open on this Mac: http://127.0.0.1:8000"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
