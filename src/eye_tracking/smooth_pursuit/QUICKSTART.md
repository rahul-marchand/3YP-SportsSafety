# Quick Start - 5 Minute Setup

## What You Need

- 2 phones (iOS or Android)
- Laptop with Python
- Beam splitter hardware
- All on same WiFi network

## Setup (Do This Once)

### 1. Start Backend
```bash
cd /Users/suleimanmahmood/Documents/3YP/src/eye_tracking/smooth_pursuit
source venv/bin/activate
python backend.py
```

Leave this running. Note your laptop's IP (e.g., `192.168.1.100`)

### 2. Find Your IP Address
```bash
# Mac/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# You're looking for something like: 192.168.1.100
```

### 3. Open Phones (BEFORE mounting in beam splitter)

**On Phone 1 (Display):**
```
http://192.168.1.100:8000/?mode=display&distance=30
```

**On Phone 2 (Camera):**
```
http://192.168.1.100:8000/?mode=camera&distance=30
```

Replace `192.168.1.100` with your laptop's IP.

Change `distance=30` if your eye-to-beamsplitter distance is different.

### 4. Verify Connection
```bash
python control.py status
```

Should show:
```
✓ Both phones connected and ready
```

### 5. Mount Phones

Position both phones in your beam splitter hardware setup:
```
     [Display Phone]
           ↓
    [Beam Splitter 45°]
       /         \
   [Eye]    [Camera Phone]
```

**Now you can't touch the screens!**

## Recording Data

### Start Recording
```bash
python control.py start
```

**What happens:**
- Display phone: black dot starts moving left-right
- Camera phone: captures eye images (10 Hz)
- Backend: pairs images with dot positions

### Wait
Let it record for **60 seconds**

### Stop Recording
```bash
python control.py stop
```

### Repeat
Do 8-10 recording sessions (60 seconds each) for good training data

### Check Progress
```bash
python control.py status
```
Shows frame count (aim for 5000+ total frames)

## After Data Collection

### Train Model
```bash
python train_smooth_pursuit.py \
  --data_dir Dataset/smooth_pursuit_data \
  --distance 30 \
  --epochs 50
```

Takes ~30-60 minutes depending on GPU.

Model saved to: `mobilenet/checkpoints/model_30cm.pth`

## Troubleshooting

### Phones won't connect
1. Check all on same WiFi
2. Try `http://localhost:8000/?mode=...` if phones on laptop
3. Check backend is running (`python backend.py`)

### "Display phone not connected"
Refresh the display phone browser

### "Camera phone not connected"
Refresh camera phone browser, allow camera permissions

### Low frame count
```bash
python control.py status
```
Check `Dot buffer size` > 0 (if 0, display phone not sending data)

## Complete Documentation

- **README.md** - Overview
- **BEAM_SPLITTER_SETUP.md** - Detailed setup guide
- **TRAINING_WORKFLOW.md** - Training guide

## Command Reference

```bash
python control.py status   # Check status
python control.py start    # Start recording
python control.py stop     # Stop recording
python control.py help     # Show help
```

## That's It!

You're now collecting eye tracking training data with accurate gaze angles.

