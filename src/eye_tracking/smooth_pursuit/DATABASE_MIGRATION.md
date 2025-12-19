# Database Migration Guide

## Overview

The smooth pursuit system now uses **SQLite** for persistent data storage instead of CSV files. This provides:

1. **Distance-based filtering**: Automatically filters sessions by distance when training
2. **Training history**: Tracks which sessions were used to train each model
3. **Model metadata**: Stores training parameters, validation loss, and notes
4. **Data integrity**: Foreign key relationships ensure data consistency

## Database Schema

### Tables

1. **sessions**: Recording sessions with device configuration
   - `id`, `session_id`, `distance_cm`, `created_at`
   - Device specs: `screen_width_cm`, `screen_height_cm`, `screen_width_px`, etc.
   - `frame_count`: Number of frames in session

2. **frames**: Individual frames with gaze angles
   - `id`, `session_id` (FK), `frame_number`, `frame_filename`
   - `theta_h`, `theta_v`, `distance_cm`, `timestamp`

3. **models**: Trained model records
   - `id`, `model_name`, `distance_cm`, `model_path`
   - Training metadata: `best_val_loss`, `total_epochs`, `batch_size`, `learning_rate`

4. **training_runs**: Links models to sessions used for training
   - `model_id` (FK), `session_id` (FK), `split_type` ('train' or 'val')

## Key Benefits

### 1. Distance-Based Training
**Before**: Training script loaded ALL sessions regardless of distance
```bash
# Would mix 30cm and 40cm data!
python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30
```

**After**: Automatically filters by distance
```bash
# Only uses sessions with distance_cm = 30
python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30
```

### 2. Training History Tracking
You can now query:
- Which sessions were used to train `model_30cm.pth`?
- What was the validation loss for each model?
- When was each model trained?

### 3. Data Integrity
- Foreign keys ensure frames belong to valid sessions
- Training runs link models to their source sessions
- No orphaned data

## Usage

### Recording Data (Automatic)
The backend automatically creates database records when recording:
- Session metadata stored in `sessions` table
- Frame data stored in `frames` table
- Images still saved to disk (for training compatibility)

### Training Models
Training script now:
1. Queries database for sessions matching the specified distance
2. Splits sessions into train/val sets
3. Records model in `models` table
4. Links training sessions in `training_runs` table

```bash
python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30 --epochs 50
```

### Querying the Database

You can use SQLite tools or Python to query:

```python
from database import SmoothPursuitDB

db = SmoothPursuitDB()

# Get all models for 30cm
models = db.get_all_models()
for model in models:
    if model["distance_cm"] == 30:
        print(f"Model: {model['model_name']}, Val Loss: {model['best_val_loss']}")
        
        # Get training sessions
        train_ids, val_ids = db.get_model_training_sessions(model["id"])
        print(f"  Training sessions: {len(train_ids)}")
        print(f"  Validation sessions: {len(val_ids)}")

db.close()
```

## Migration from CSV

If you have existing CSV-based data, you can migrate it:

1. The system still saves images to disk (backward compatible)
2. New recordings automatically use the database
3. Old CSV files can be ignored (or manually migrated if needed)

## Database Location

Default: `Dataset/smooth_pursuit_data.db`

The database file is created automatically on first use.

## Notes

- SQLite is built into Python (no extra dependencies needed)
- Database file is portable (can be copied/moved)
- Images are still stored on disk (database only tracks metadata)
- Backward compatible: training can still work with old CSV-based data if needed



