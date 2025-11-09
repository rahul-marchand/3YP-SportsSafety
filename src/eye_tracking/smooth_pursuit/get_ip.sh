#!/bin/bash

echo "Finding your local IP address..."
echo ""

IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

if [ -z "$IP" ]; then
    echo "Could not find local IP. Make sure you're connected to WiFi."
    exit 1
fi

echo "Your local IP address: $IP"
echo ""
echo "Update your mobile app config:"
echo "  File: mobile_app/app_code/config.js"
echo "  Change BACKEND_URL to: \"http://$IP:8000\""
echo ""
echo "Then start the backend:"
echo "  python3 backend.py"
echo ""
echo "Make sure your phone is on the same WiFi network!"

