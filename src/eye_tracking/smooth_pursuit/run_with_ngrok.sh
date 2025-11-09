#!/bin/bash

echo "Starting Smooth Pursuit Backend with ngrok..."
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null
then
    echo "ngrok not found. Install it first:"
    echo "  brew install ngrok"
    echo "  Or download from: https://ngrok.com/download"
    exit 1
fi

# Check if backend dependencies are installed
if ! python3 -c "import fastapi" &> /dev/null
then
    echo "Backend dependencies not installed. Run:"
    echo "  ./install.sh"
    exit 1
fi

echo "1. Starting backend on port 8000..."
source venv/bin/activate
python backend.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

echo "2. Starting ngrok tunnel..."
ngrok http 8000 --log=stdout &
NGROK_PID=$!

# Wait for ngrok to start
sleep 3

echo ""
echo "Getting ngrok URL..."
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)

if [ -z "$NGROK_URL" ]; then
    echo "Could not get ngrok URL. Check ngrok web interface at:"
    echo "  http://localhost:4040"
else
    echo ""
    echo "=========================================="
    echo "Backend is now accessible at:"
    echo "  $NGROK_URL"
    echo "=========================================="
    echo ""
    echo "Update your mobile app config:"
    echo "  File: mobile_app/app_code/config.js"
    echo "  Change BACKEND_URL to: \"$NGROK_URL\""
    echo ""
fi

echo "View ngrok dashboard: http://localhost:4040"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
trap "kill $BACKEND_PID $NGROK_PID 2>/dev/null" EXIT
wait

