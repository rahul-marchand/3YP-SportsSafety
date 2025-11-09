"""
Quick test script to verify backend setup.
Run: python test_backend.py
"""

import sys
from pathlib import Path

print("Testing smooth pursuit backend setup...")

# Check dependencies
try:
    import cv2
    print("OpenCV: OK")
except ImportError:
    print("OpenCV: MISSING - run: pip install opencv-python")

try:
    import torch
    print(f"PyTorch: OK (version {torch.__version__})")
except ImportError:
    print("PyTorch: MISSING - run: pip install torch torchvision")

try:
    import fastapi
    print("FastAPI: OK")
except ImportError:
    print("FastAPI: MISSING - run: pip install fastapi uvicorn")

try:
    import numpy as np
    print("NumPy: OK")
except ImportError:
    print("NumPy: MISSING - run: pip install numpy")

try:
    import pandas as pd
    print("Pandas: OK")
except ImportError:
    print("Pandas: MISSING - run: pip install pandas")

# Check model exists
model_path = Path(__file__).parent.parent / "mobilenet" / "model.py"
if model_path.exists():
    print(f"MobileNet model: OK")
else:
    print(f"MobileNet model: NOT FOUND at {model_path}")

# Check OpenCV cascades
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if Path(cascade_path).exists():
    print("Haar cascades: OK")
else:
    print("Haar cascades: MISSING")

# Test angle calculation
from backend import AngleCalculator

calc = AngleCalculator(
    screen_width_cm=13.2,
    screen_height_cm=6.1,
    screen_width_px=2532,
    screen_height_px=1170,
)

theta_h, theta_v = calc.calculate_angles(
    dot_x_px=1500,
    dot_y_px=585,
    camera_x_px=287,
    camera_y_px=585,
    distance_cm=30,
)

print(f"\nAngle calculation test:")
print(f"  Dot at (1500, 585), Camera at (287, 585), Distance 30cm")
print(f"  Calculated angles: theta_h={theta_h:.2f}°, theta_v={theta_v:.2f}°")
print(f"  Expected: theta_h ~ 14-15°, theta_v ~ 0°")

if abs(theta_h - 14.5) < 2.0 and abs(theta_v) < 1.0:
    print("  PASS")
else:
    print("  FAIL - check angle calculation")

print("\nSetup complete. To run backend:")
print("  python backend.py")
print("\nThen test with:")
print("  curl http://localhost:8000/")

