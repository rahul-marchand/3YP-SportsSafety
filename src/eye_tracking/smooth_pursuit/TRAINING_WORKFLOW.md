# Smooth Pursuit Training Workflow

## System Overview

You now have a complete white background with black dot smooth pursuit system that:
1. Collects eye tracking data with gaze angles
2. Stores data in SQLite database
3. Trains MobileNetV2 models for gaze estimation
4. Tests trained models with real-time inference

## Step 1: Collect Data

### Start the Backend
```bash
cd /Users/suleimanmahmood/Documents/3YP/src/eye_tracking/smooth_pursuit
source venv/bin/activate
python backend.py
```

The backend runs at `http://localhost:8000`

### Recording Sessions
1. Open the web interface on your phone/device (landscape mode required)
2. Set viewing distance (e.g., 30cm)
3. Click "START RECORDING"
4. Follow the black dot as it moves left-right on white background
5. Record for 30-60 seconds per session
6. Click "STOP RECORDING"

**Tip**: Collect multiple sessions (5-10) at the same distance for best results.

### Data Storage
- **Images**: `Dataset/smooth_pursuit_data/session_YYYYMMDD_HHMMSS_XXcm/*.jpg`
- **Metadata**: SQLite database at `Dataset/smooth_pursuit_data.db`
- Each frame stores: image file, gaze angles (theta_h, theta_v), timestamp

## Step 2: Train a Model

Once you have collected enough data (aim for 500+ frames):

```bash
cd /Users/suleimanmahmood/Documents/3YP/src/eye_tracking/smooth_pursuit
source venv/bin/activate

# Train on 30cm distance data
python train_smooth_pursuit.py \
  --data_dir Dataset/smooth_pursuit_data \
  --distance 30 \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001
```

### What Happens During Training:
1. Loads all sessions from database with `distance_cm = 30`
2. Splits sessions into train (85%) and validation (15%)
3. Trains MobileNetV2 to predict gaze angles from eye images
4. Saves best model: `mobilenet/checkpoints/model_30cm.pth`
5. Records training metadata in database

### Training Parameters:
- `--distance`: Distance in cm (must match your recording sessions)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--val_split`: Validation split ratio (default: 0.15)

## Step 3: Test Your Model

### Load Model for Inference
1. Start the backend (if not already running)
2. Open web interface
3. Click "Refresh Models" - you should see `model_30cm.pth`
4. Select the model from dropdown
5. Set the same viewing distance (30cm)
6. Click "START TEST"

### During Testing:
- Follow the black dot as it moves
- Real-time prediction errors shown: "Error: X.XX° H, X.XX° V"
- Click "STOP TEST" to see summary metrics:
  - Mean Error Horizontal
  - Mean Error Vertical
  - RMS Error
  - Number of samples

### Good Performance:
- Mean error < 2-3 degrees is excellent
- Mean error < 5 degrees is acceptable
- Higher error? Collect more data or train longer

## Model Architecture

**MobileNetV2** with custom regression head:
- Input: 224x224 RGB image (full face)
- Backbone: Pretrained MobileNetV2 (ImageNet)
- Head: 1280 → 512 → 2 (theta_h, theta_v)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam with ReduceLROnPlateau scheduler

## Data Collection Tips

### For Best Results:
1. **Lighting**: Good, consistent lighting (avoid shadows)
2. **Distance**: Keep consistent distance during recording
3. **Head Position**: Keep head relatively stable
4. **Multiple Sessions**: Collect 5-10 sessions at same distance
5. **Different Times**: Record at different times of day
6. **Variety**: Try different head angles slightly

### Minimum Data Requirements:
- Bare minimum: 3-4 sessions (300-400 frames)
- Recommended: 8-10 sessions (800-1000 frames)
- Ideal: 15+ sessions (1500+ frames)

## Database Queries

Check your collected data:

```python
from database import SmoothPursuitDB

db = SmoothPursuitDB()

# Check sessions by distance
sessions = db.get_sessions_by_distance(30)
print(f"30cm sessions: {len(sessions)}")
for s in sessions:
    print(f"  Session {s['session_id']}: {s['frame_count']} frames")

# Check trained models
models = db.get_all_models()
for m in models:
    print(f"\nModel: {m['model_name']}")
    print(f"  Distance: {m['distance_cm']}cm")
    print(f"  Val Loss: {m['best_val_loss']:.4f}")
    print(f"  Epochs: {m['total_epochs']}")
    
    train_ids, val_ids = db.get_model_training_sessions(m['id'])
    print(f"  Training sessions: {len(train_ids)}")
    print(f"  Validation sessions: {len(val_ids)}")

db.close()
```

## Troubleshooting

### "No sessions found for distance Xcm"
- You haven't collected any data at that distance yet
- Collect data first using the web interface

### "No models found - train one first"
- No trained models exist yet
- Collect data, then run `train_smooth_pursuit.py`

### High prediction errors during testing
- Collect more training data (aim for 1000+ frames)
- Train for more epochs (try 100 epochs)
- Ensure test distance matches training distance
- Check lighting conditions are similar to training data

### Model not appearing in web interface
- Check `mobilenet/checkpoints/` directory for `.pth` files
- Click "Refresh Models" button
- Restart backend if needed

## File Structure

```
smooth_pursuit/
├── backend.py                  # FastAPI backend
├── database.py                 # SQLite database interface
├── train_smooth_pursuit.py     # Training script
├── static/
│   └── index.html             # Web interface (white bg, black dot)
├── Dataset/
│   ├── smooth_pursuit_data.db  # SQLite database
│   └── smooth_pursuit_data/    # Image storage
│       └── session_*/          # Session directories
└── mobilenet/
    └── checkpoints/           # Trained models
        └── model_30cm.pth     # Your trained model
```

## Next Steps

1. Start backend: `python backend.py`
2. Collect data: Open web interface, record 8-10 sessions at 30cm
3. Train model: `python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30`
4. Test model: Use web interface inference mode

Your system is ready to go!

