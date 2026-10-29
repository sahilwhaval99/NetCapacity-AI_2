#!/usr/bin/env bash
# 1-Click Starter Script for NetCapacity AI (macOS / Linux)
set -e

echo "============================================================"
echo " Starting NetCapacity AI (Backend + Frontend)"
echo "============================================================"

# Start Backend on port 8000
cd "$(dirname "$0")/backend"
python3 main.py &
BACKEND_PID=$!
echo "[OK] FastAPI Backend started on http://127.0.0.1:8000 (PID: $BACKEND_PID)"

# Start Frontend on port 5173
cd "../frontend"
if [ ! -d "node_modules" ]; then
  echo "[INFO] Installing frontend npm packages..."
  npm install
fi
echo "[OK] Starting React + Vite Frontend on http://localhost:5173"
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT
npm run dev
