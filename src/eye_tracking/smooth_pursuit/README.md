# Smooth Pursuit Eye Tracking System

**Two-phone beam splitter system for collecting eye tracking data and training gaze estimation models for smooth pursuit diagnosis.**

---

## 🚀 Quick Start

### 1. Setup
```bash
cd src/eye_tracking/smooth_pursuit
source venv/bin/activate  # or: python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Backend
```bash
python backend.py
```
Server runs on `http://localhost:8000` (keep terminal open)

### 3. Start Ngrok (for phone access)
```bash
# In a NEW terminal
ngrok http 8000 --log=stdout
```
Copy the HTTPS URL (e.g., `https://xxxxx.ngrok-free.app`)

### 4. Open Phones
**Display Phone** (shows moving dot):
```
https://YOUR-NGROK-URL.ngrok-free.app/?mode=display
```

**Camera Phone** (captures eye):
```
https://YOUR-NGROK-URL.ngrok-free.app/?mode=camera
```

### 5. Control from Terminal
```bash
# In a NEW terminal
cd src/eye_tracking/smooth_pursuit
source venv/bin/activate
python control.py
```

---

## 📋 Terminal Commands

### Interactive Menu (Recommended)
```bash
python control.py
```
Shows menu with options:
1. Check System Status
2. Collect Training Data
3. Train Model
4. Test Model (Inference)
5. Stop All
6. Exit

### Command Line Commands

#### Status & Information
```bash
python control.py status    # Check phone connections and system status
python control.py models    # List all trained models
python control.py data      # List all training data (sessions)
```

#### Data Collection
```bash
python control.py start     # Interactive: prompts for total distance (cm)
python control.py stop      # Stop recording/inference
```

#### Training
```bash
python control.py train     # Interactive: prompts for distance, epochs, mode
```

#### Inference/Testing
```bash
python control.py test      # Interactive: select model, enter distance
python control.py test-stop # Stop inference and show results
```

#### Data Management
```bash
python control.py delete    # Interactive: delete models or data
```

### Training Script (Direct)
```bash
# Basic training
python train_smooth_pursuit.py --distance 30 --data_dir Dataset/smooth_pursuit_data

# With all options
python train_smooth_pursuit.py \
    --distance 30 \
    --data_dir Dataset/smooth_pursuit_data \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --warmup_epochs 5 \
    --early_stop_patience 10

# Resume interrupted training
python train_smooth_pursuit.py --distance 30 --data_dir Dataset/smooth_pursuit_data --resume auto

# Fine-tune from another model
python train_smooth_pursuit.py --distance 35 --data_dir Dataset/smooth_pursuit_data --fine_tune model_30cm.pth
```

---

## 🔧 Design Choices

### 1. **Single Distance Field (Total Distance)**
- **Decision**: Store only `distance_cm` representing total distance (display_to_splitter + splitter_to_eye)
- **Rationale**: For a 45° beam splitter, the geometry is isometric - the virtual image appears at the sum of the two distances. We only need the total distance to calculate angles correctly.
- **Implementation**: All angle calculations use `theta = arctan(displacement_cm / total_distance_cm)`

### 2. **Full-Face Image Processing**
- **Decision**: Process full-face images (both eyes visible) instead of cropping to single eye
- **Rationale**: 
  - More robust (no detection failures)
  - Provides redundant information (both eyes)
  - Includes head pose context
  - Industry standard approach (iTracker, MPIIGaze)
- **Implementation**: Simply resize full camera frame to 224x224, let CNN learn what matters

### 3. **Zero Angle Reference**
- **Decision**: Screen center is 0° (straight-ahead gaze)
- **Rationale**: Natural reference point, matches expected behavior
- **Implementation**: 
  - `theta_h = arctan((dot_x_px - screen_center_x) * px_to_cm / distance_cm)`
  - `theta_v = arctan((dot_y_px - screen_center_y) * px_to_cm / distance_cm)`
  - Positive horizontal = right, negative = left
  - Positive vertical = up, negative = down

### 4. **Timestamp-Based Pairing**
- **Decision**: Pair dot positions with eye images using synchronized timestamps (±100ms window)
- **Rationale**: Ensures accurate training data - each eye image paired with correct dot position
- **Implementation**: 
  - Both phones sync clocks with server (5 round-trip measurements, median offset)
  - Backend maintains buffer of recent dot positions
  - Matches frames to closest dot position by timestamp

### 5. **Distance-Based Data Filtering**
- **Decision**: Database stores distance with each session, training filters by distance
- **Rationale**: Models are distance-specific - training on mixed distances would hurt performance
- **Implementation**: `get_sessions_by_distance(distance_cm)` queries database, training script only uses matching sessions

### 6. **Terminal-Controlled Workflow**
- **Decision**: All control via terminal, phones stay on display/camera URLs
- **Rationale**: 
  - Prevents accidental mode switching
  - Ensures phones stay in correct mode
  - Better for beam splitter setup (phones mounted, can't touch screens)
- **Implementation**: `control.py` provides interactive menu and command-line interface

### 7. **In-Memory Data Augmentation**
- **Decision**: Augmentation happens during training, original images never modified
- **Rationale**: Preserves original data integrity, allows different augmentation strategies per training run
- **Implementation**: `transforms.ColorJitter`, `RandomRotation`, `RandomErasing` applied in-memory

### 8. **Robust Training Features**
- **Decision**: Include early stopping, learning rate warmup, gradient clipping
- **Rationale**: Prevents overfitting, stabilizes training, improves generalization
- **Implementation**: 
  - Early stopping: stops if validation loss doesn't improve for N epochs
  - LR warmup: gradually increases learning rate from small value
  - Gradient clipping: prevents exploding gradients

### 9. **Comprehensive Inference Metrics**
- **Decision**: Calculate angle errors, velocity errors, smooth pursuit gain, and lag
- **Rationale**: Full smooth pursuit diagnosis requires all these metrics
- **Implementation**: 
  - Real-time: angle errors, velocity calculations
  - Post-processing: lag via cross-correlation
  - Database storage: all metrics stored for analysis

### 10. **SQLite Database**
- **Decision**: Use SQLite for metadata, images stored on disk
- **Rationale**: 
  - Fast queries by distance
  - Tracks training history
  - Maintains data integrity (foreign keys)
  - Portable (single file)
- **Implementation**: Tables for sessions, frames, models, training_runs, inference_sessions, inference_results

---

## 📊 System Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Display Phone  │         │  Camera Phone   │
│  (Moving Dot)   │         │  (Eye Camera)   │
└────────┬────────┘         └────────┬────────┘
         │                            │
         │ WebSocket                  │ WebSocket
         │ (x, y, timestamp)          │ (image, timestamp)
         │                            │
         └────────────┬───────────────┘
                      │
              ┌───────▼────────┐
              │  Backend.py   │
              │  (FastAPI)    │
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌───▼────┐  ┌───▼────┐
    │ Buffer  │  │ Angle  │  │Storage │
    │(Pair by │  │Calc    │  │(DB +   │
    │timestamp)│  │(Beam   │  │ Images)│
    └─────────┘  │Splitter)│  └────────┘
                 └─────────┘
```

**Key Features:**
- **Time Synchronization**: Both phones sync with server (<10ms accuracy)
- **Timestamp Pairing**: Matches dot positions with eye images (±100ms window)
- **Beam Splitter Correction**: Calculates angles accounting for 45° geometry
- **Distance-Based Filtering**: Database automatically filters by distance

---

## 📁 File Structure

```
smooth_pursuit/
├── backend.py              # FastAPI server
├── control.py              # Terminal control (interactive menu)
├── database.py             # SQLite database management
├── train_smooth_pursuit.py # Training script
├── static/
│   └── index.html          # Web interface for phones
├── Dataset/
│   ├── smooth_pursuit_data/
│   │   └── session_YYYYMMDD_HHMMSS_XXcm/
│   │       └── frame_*.jpg
│   └── smooth_pursuit_data.db
└── mobilenet/
    └── checkpoints/
        └── model_XXcm.pth
```

---

## 🗄️ Database Structure

**`sessions`** - Recording sessions
- `distance_cm` - Total distance (eye to virtual image)
- `session_id` - Unique identifier
- `frame_count` - Number of frames
- `created_at` - Timestamp

**`frames`** - Individual frames
- `session_id` - Links to session
- `theta_h`, `theta_v` - Gaze angles (degrees)
- `distance_cm` - Distance (for filtering)
- `timestamp` - Frame timestamp

**`models`** - Trained models
- `model_name` - e.g., "model_30cm.pth"
- `distance_cm` - Distance model was trained for
- `best_val_loss` - Validation loss
- `model_path` - File path

**`inference_sessions`** - Inference/test runs
- `model_id` - Which model was tested
- `distance_cm` - Distance used during test
- Metrics: errors, velocities, lag, diagnosis

---

## 🔑 Key Concepts

### Total Distance
- **What**: Sum of display_to_splitter + splitter_to_eye distances
- **Why**: For 45° beam splitter, virtual image appears at this total distance
- **Usage**: Single parameter for all angle calculations

### Timestamp Pairing
- Both phones sync clocks with server
- Backend pairs dot positions with eye images by timestamp
- Window: ±100ms (configurable)
- Ensures accurate training data

### Beam Splitter Geometry
- Accounts for 45° beam splitter angle
- Isometric mapping: virtual image distance = total distance
- Formula: `theta = arctan(displacement_cm / total_distance_cm)`

---

## 🐛 Troubleshooting

### Phones Not Connecting
```bash
python control.py status
```
**Check:**
- Backend is running (`python backend.py`)
- Ngrok is running (`ngrok http 8000`)
- Correct URLs on phones (use ngrok HTTPS URL)
- Both phones show "Connected" status

### No Data Collected
- Check both phones are connected before starting
- Verify distance is set correctly
- Check terminal for error messages
- Ensure phones stay on the web page (don't lock screen)

### Training Fails
- Verify data exists: `python control.py data`
- Check database: Sessions should have `distance_cm` matching training distance
- Ensure enough data: Need at least 2 sessions for train/val split

### Model Not Found During Inference
- Check model exists: `python control.py models`
- Verify model name matches exactly
- Check database has model record

---

## 💡 Tips

1. **Always check status first**: `python control.py status`
2. **Use same distance**: Keep distance consistent during data collection
3. **Collect enough data**: 8-10 sessions minimum for good training
4. **Monitor training**: Watch validation loss, use early stopping
5. **Test with same distance**: Use same distance for inference as training

---

## 📚 Workflow Summary

### Data Collection
1. Start backend: `python backend.py`
2. Start ngrok: `ngrok http 8000`
3. Open phones on display/camera URLs
4. Run: `python control.py start` (enter distance)
5. Collect data for 60 seconds
6. Run: `python control.py stop`

### Training
1. Run: `python control.py train`
2. Select distance (e.g., 30cm)
3. Choose training mode (from scratch, resume, fine-tune)
4. Enter epochs and other parameters
5. Wait for training to complete

### Inference
1. Run: `python control.py test`
2. Select model from list
3. Enter distance (should match training distance)
4. Test for 60 seconds
5. Run: `python control.py test-stop` to see results

---

**For help, run**: `python control.py help`
