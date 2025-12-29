SMOOTH PURSUIT EYE TRACKING SYSTEM
===================================

Two-phone beam splitter system for collecting eye tracking data and training 
gaze estimation models for smooth pursuit diagnosis.

QUICK START:
------------
1. Install: source venv/bin/activate && pip install -r requirements.txt
2. Start backend: python backend.py
3. Start ngrok: ngrok http 8000
4. Open phones: https://YOUR-NGROK-URL.ngrok-free.app/?mode=display (or camera)
5. Control: python control.py

TERMINAL COMMANDS:
------------------
Interactive Menu:
  python control.py                    # Shows menu with all options

Status & Info:
  python control.py status             # Check phone connections
  python control.py models             # List trained models
  python control.py data               # List training data

Data Collection:
  python control.py start              # Interactive: prompts for distance
  python control.py stop               # Stop recording/inference

Training:
  python control.py train              # Interactive: prompts for all options
  python train_smooth_pursuit.py --distance 30 --data_dir Dataset/smooth_pursuit_data

Inference:
  python control.py test               # Interactive: select model, enter distance
  python control.py test-stop          # Stop inference and show results

Data Management:
  python control.py delete             # Interactive: delete models or data

DESIGN CHOICES:
---------------
1. Single Distance Field: Store only total_distance_cm (display_to_splitter + 
   splitter_to_eye). For 45° beam splitter, geometry is isometric - only total 
   distance needed for angle calculations.

2. Full-Face Processing: Process full-face images (both eyes) instead of 
   cropping. More robust, provides redundant info, includes head pose context.

3. Zero Angle Reference: Screen center = 0° (straight-ahead gaze). Natural 
   reference point for angle calculations.

4. Timestamp Pairing: Pair dot positions with eye images using synchronized 
   timestamps (±100ms window). Ensures accurate training data.

5. Distance-Based Filtering: Database stores distance per session, training 
   filters by distance. Models are distance-specific.

6. Terminal Control: All control via terminal, phones stay on display/camera 
   URLs. Prevents mode switching, better for beam splitter setup.

7. In-Memory Augmentation: Augmentation during training, originals never 
   modified. Preserves data integrity.

8. Robust Training: Early stopping, LR warmup, gradient clipping. Prevents 
   overfitting, stabilizes training.

9. Comprehensive Metrics: Angle errors, velocity errors, smooth pursuit gain, 
   lag. Full diagnosis requires all metrics.

10. SQLite Database: Metadata in SQLite, images on disk. Fast queries, tracks 
    history, maintains integrity.

ARCHITECTURE:
-------------
Display Phone → WebSocket (dot position) → Backend
Camera Phone → WebSocket (eye image) → Backend
Backend → Pairs by timestamp → Calculates angles → Stores to DB

DATA FLOW:
----------
Collection: Display sends dot → Camera sends image → Backend pairs → Calculates 
            angle → Saves to DB

Training: Query DB by distance → Load images → Split train/val → Train 
          MobileNetV2 → Save model

Inference: Load model → Predict angles → Compare to expected → Calculate 
           metrics → Store results

FILES:
------
backend.py              - FastAPI server
control.py              - Terminal control (interactive menu)
database.py             - SQLite database management
train_smooth_pursuit.py - Training script
static/index.html       - Web interface for phones
Dataset/                - Data storage (sessions + DB)
