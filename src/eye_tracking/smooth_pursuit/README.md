# Smooth Pursuit Eye Tracking - Beam Splitter System

Two-phone coordinated system for collecting accurate eye tracking training data using a beam splitter setup.

## Features

- White background with black dot stimulus
- Two-phone coordination (display + camera)
- Beam splitter geometry correction
- Server time synchronization for accurate data pairing
- Terminal control (no screen touching needed)
- SQLite database storage
- MobileNetV2 training pipeline

## Quick Start

### Prerequisites
```bash
# Install dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### Basic Usage

1. **Start Backend**
```bash
python backend.py
```

2. **Open Phones** (before mounting)
```
Display: http://localhost:8000/?mode=display&distance=30
Camera:  http://localhost:8000/?mode=camera&distance=30
```

3. **Control Recording**
```bash
python control.py status   # Check connections
python control.py start    # Start recording
python control.py stop     # Stop recording
```

4. **Train Model**
```bash
python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30 --epochs 50
```

## Complete Documentation

**For full setup instructions, see:**
- **[BEAM_SPLITTER_SETUP.md](BEAM_SPLITTER_SETUP.md)** - Complete hardware and software setup guide
- **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** - Training and inference workflow

## System Architecture

```
Display Phone (30 Hz)         Camera Phone (10 Hz)
      |                              |
      | [Time sync on connect]       | [Time sync on connect]
      |                              |
      | dot positions                | eye frames
      | + synced timestamps          | + synced timestamps
      ↓                              ↓
           Backend Server
      (pairs by timestamp ±100ms)
              ↓
      Database + JPG Files
```

## Key Features

### Time Synchronization
Both phones sync clocks with server (<10ms accuracy) to ensure eye images are paired with correct dot positions for accurate training data.

### Beam Splitter Correction
Automatically calculates correct gaze angles accounting for 45° beam splitter geometry:
```python
theta_h = arctan(dot_x_cm / eye_to_beamsplitter_distance)
theta_v = arctan(dot_y_cm / eye_to_beamsplitter_distance)
```

### Terminal Control
Control both phones from laptop - no screen touching needed once mounted in beam splitter.

## Files

- `backend.py` - FastAPI server with WebSocket coordination
- `control.py` - Terminal control script
- `database.py` - SQLite database management
- `train_smooth_pursuit.py` - Training script
- `static/index.html` - Web interface (display/camera modes)
- `BEAM_SPLITTER_SETUP.md` - Complete setup guide
- `TRAINING_WORKFLOW.md` - Training guide

## URL Parameters

- `mode=display` - Display phone (shows moving dot)
- `mode=camera` - Camera phone (captures eye)
- `distance=X` - Distance in cm from eye to beam splitter

## Data Location

```
Dataset/smooth_pursuit_data/
├── session_20251121_HHMMSS_30cm/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── smooth_pursuit_data.db
```

## Troubleshooting

See [BEAM_SPLITTER_SETUP.md](BEAM_SPLITTER_SETUP.md#troubleshooting) for detailed troubleshooting.

Quick checks:
```bash
python control.py status  # Check connections
# Look for: Display connected: True, Camera connected: True
```

## Training Requirements

- Collect 8-10 sessions (60 seconds each)
- Results in ~5000-6000 frames
- Train for 50 epochs
- Model saved to: `mobilenet/checkpoints/model_30cm.pth`

## Contact

For issues, check the troubleshooting section in BEAM_SPLITTER_SETUP.md

