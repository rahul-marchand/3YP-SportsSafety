#!/bin/bash

echo "Installing smooth pursuit backend dependencies..."

pip3 install -r requirements.txt

echo ""
echo "Installation complete!"
echo ""
echo "To test the backend:"
echo "  python3 test_backend.py"
echo ""
echo "To run the backend server:"
echo "  python3 backend.py"
echo ""
echo "Server will run on http://0.0.0.0:8000"

