SMOOTH PURSUIT PI - RASPBERRY PI 5 EYE TRACKING SYSTEM
======================================================

Three-component system for smooth pursuit assessment using a
Raspberry Pi 5 headset with beam splitter optics.

COMPONENTS
----------
1. Data Collection (runs on Pi)    - Captures eye video + dot positions
2. Training (runs on cloud/GPU)    - Trains MobileNetV2 gaze model
3. Inference Engine (runs on Pi)   - Real-time smooth pursuit assessment

QUICK START
-----------

1. Data Collection (on Pi):
   pip install -r requirements_pi.txt
   python data_collection.py --distance 40 --duration 30

2. Prepare Dataset (on training machine):
   pip install -r requirements_train.txt
   python prepare_dataset.py --sessions_dir ~/smooth_pursuit_data/sessions

3. Train Model:
   python train_model.py --dataset_dir ./dataset --distance 40

4. Export to ONNX:
   python export_onnx.py --checkpoint checkpoints/model_40cm.pth

5. Run Inference (on Pi):
   python inference_engine.py --model model_40cm.onnx --distance 40

FILE STRUCTURE
--------------
config.py              - Shared configuration (display, camera, geometry)
angle_calculator.py    - Gaze angle calculation from dot position
camera_manager.py      - Dual CSI camera wrapper (picamera2)
display_manager.py     - Pygame stimulus display (moving dot)
mcu_interface.py       - Optional MCU UART communication
data_collection.py     - Main data collection application
inference_engine.py    - Real-time inference and diagnosis
prepare_dataset.py     - Decode video + pair with ground-truth angles
train_model.py         - MobileNetV2 training script
export_onnx.py         - PyTorch to ONNX model conversion

SESSION DATA FORMAT
-------------------
Each collection session creates:
  sessions/session_YYYYMMDD_HHMMSS_XXcm/
    metadata.json        - Session config and summary
    dot_positions.csv    - Frame-by-frame dot position + angles
    left_eye.mp4         - Left camera encoded video
    right_eye.mp4        - Right camera encoded video

HARDWARE REQUIREMENTS
---------------------
- Raspberry Pi 5 (4GB+ RAM)
- Two MIPI CSI-2 cameras (640x480 @ 120fps)
- HDMI or DSI stimulus display
- IR LED illumination (850nm)
- Optional: RP2040 MCU for timing/illumination control
