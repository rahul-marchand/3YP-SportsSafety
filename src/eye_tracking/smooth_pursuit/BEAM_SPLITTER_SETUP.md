# Two-Phone Beam Splitter Setup Guide

## System Overview

This system coordinates two phones through a beam splitter for accurate eye tracking data collection:

1. **Display Phone**: Shows white background with moving black dot
2. **Beam Splitter**: 45° mirror that reflects display to eye
3. **Camera Phone**: Rear camera captures eye images
4. **Backend**: Synchronizes both phones and calculates correct gaze angles
5. **Terminal Control**: Start/stop recording from laptop
6. **Time Synchronization**: Both phones sync clocks with server for accurate data pairing

## Hardware Setup

```
        [Display Phone]
        (white bg, black dot)
              |
              | light
              ↓
        [45° Beam Splitter]
           /        \
    (reflected)   (transmitted)
        /              \
       ↓                ↓
   [Eye]          [Camera Phone]
                  (rear camera up)
```

### Physical Arrangement
- Display phone vertical (or angled) above beam splitter
- Beam splitter at 45° angle
- Eye viewing reflected image in beam splitter
- Camera phone positioned to capture eye through beam splitter

### Key Distance
The "distance" parameter = **eye to beam splitter distance** (e.g., 30cm)

## Software Setup

### 1. Start Backend (on laptop)

```bash
cd /Users/suleimanmahmood/Documents/3YP/src/eye_tracking/smooth_pursuit
source venv/bin/activate
python backend.py
```

Backend runs at `http://localhost:8000`

### 2. (Optional) Start ngrok for Remote Access

If phones are not on same network:

```bash
# In another terminal
ngrok http 8000

# Copy the URL, e.g., https://abc123.ngrok.io
```

### 3. Configure Phones (Before Mounting)

**Display Phone:**
```
Open: http://localhost:8000/?mode=display&distance=30
(or use ngrok URL)
```

**Camera Phone:**
```
Open: http://localhost:8000/?mode=camera&distance=30
(or use ngrok URL)
```

Both should show: "Connected. Waiting for start command."

### 4. Mount Phones in Beam Splitter

Position both phones as shown in diagram above. Once mounted, you can't touch the screens.

### 5. Control from Terminal

```bash
# Check connection status
python control.py status

# Start recording on both phones
python control.py start

# Let it run for 30-60 seconds

# Stop recording
python control.py stop

# Check status again (see frame count)
python control.py status
```

## How It Works

### Display Phone (mode=display)
1. Shows moving black dot on white background
2. Broadcasts dot position at 30 Hz via WebSocket
3. Responds to start/stop commands from terminal

### Camera Phone (mode=camera)
1. Captures eye images at 10 Hz
2. Sends frames with timestamps
3. Backend pairs each frame with closest dot position
4. Calculates gaze angles with beam splitter correction
5. Saves to database

### Backend Coordination
1. Maintains buffer of recent dot positions (2 second window)
2. Pairs camera frames with dot positions by timestamp
3. Calculates gaze angles accounting for 45° beam splitter geometry
4. Saves paired data: `[eye_image.jpg, theta_h, theta_v]`

### Time Synchronization

**Critical for accurate training data**: Both phones sync clocks with server on connection.

**How it works:**
1. Each phone pings `/api/time` endpoint 5 times on init
2. Measures round-trip time and calculates clock offset
3. Uses median offset to avoid network jitter
4. All timestamps use: `synced_time = local_time + offset`

**Backend pairing:**
- Finds dot position within ±100ms of each frame timestamp
- Logs sync quality every 10 frames
- Warns if no matching dot position found

**Expected performance:**
- Sync accuracy: <10ms (typical WiFi LAN)
- Pairing tolerance: ±100ms window
- Monitor backend terminal for sync warnings

**Why this matters:**
Without time sync, frames could be paired with wrong dot positions, creating incorrect angle labels and ruining your training data.

## Beam Splitter Geometry

The backend automatically corrects for the beam splitter:

```python
# For 45° beam splitter, virtual image appears at:
# - Horizontal distance = display distance from beam splitter
# - Eye sees reflected image, not direct view

theta_h = arctan(dot_x_cm / eye_to_beamsplitter_distance)
theta_v = arctan(dot_y_cm / eye_to_beamsplitter_distance)
```

This correction ensures accurate ground truth angles for training.

## Workflow

### Data Collection Session

```bash
# 1. Start backend
python backend.py

# 2. Open phones (before mounting)
# Display: http://localhost:8000/?mode=display&distance=30
# Camera:  http://localhost:8000/?mode=camera&distance=30

# 3. Check both connected
python control.py status
# Should show: Display connected: True, Camera connected: True

# 4. Mount phones in beam splitter

# 5. Record session 1
python control.py start
# Wait 60 seconds
python control.py stop

# 6. Record session 2
python control.py start
# Wait 60 seconds
python control.py stop

# ... repeat for 8-10 sessions
```

### After Data Collection

```bash
# Train model
cd /Users/suleimanmahmood/Documents/3YP/src/eye_tracking/smooth_pursuit
python train_smooth_pursuit.py \
  --data_dir Dataset/smooth_pursuit_data \
  --distance 30 \
  --epochs 50

# Model saved to: mobilenet/checkpoints/model_30cm.pth
```

## Troubleshooting

### "Display phone not connected"
- Check phone can reach backend URL
- Refresh page on display phone
- Check network connection

### "Camera phone not connected"  
- Check phone can reach backend URL
- Check camera permissions
- Refresh page on camera phone

### "Cannot connect to backend"
- Ensure `python backend.py` is running
- Check if port 8000 is available
- If using ngrok, use ngrok URL instead of localhost

### Low frame count / No frames saved
- Check `python control.py status` shows both phones connected
- Ensure recording is started with `python control.py start`
- Check dot buffer size > 0 (means display is sending positions)

### High timestamp differences
The system logs timestamp differences when pairing frames with dot positions. Acceptable: < 50ms. If higher:
- Check network latency
- Ensure both phones have good connection
- Consider using local network instead of ngrok

## URL Parameters

### mode=display
- Shows only the moving dot (no camera, no UI)
- White background, black dot
- Broadcasts dot positions
- Waits for terminal start command

### mode=camera  
- Shows only camera view (no dot, no UI)
- Uses rear camera
- Captures and sends frames
- Waits for terminal start command

### distance=X
- Distance in cm from eye to beam splitter
- Used for angle calculations
- Example: `distance=30` for 30cm

### Complete URL Examples

```
Display phone (30cm): http://localhost:8000/?mode=display&distance=30
Camera phone (30cm):  http://localhost:8000/?mode=camera&distance=30

Display phone (40cm): http://localhost:8000/?mode=display&distance=40
Camera phone (40cm):  http://localhost:8000/?mode=camera&distance=40
```

## Terminal Commands Reference

```bash
# Check connection status
python control.py status

# Start recording
python control.py start

# Stop recording  
python control.py stop

# Show help
python control.py help
```

## Data Location

Collected data stored in:
```
Dataset/smooth_pursuit_data/
├── session_20251121_HHMMSS_30cm/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── smooth_pursuit_data.db  (SQLite database with angles)
```

## Notes

- **Synchronization**: System pairs frames by timestamp (typically < 10ms difference)
- **Beam splitter correction**: Automatically applied to all angle calculations
- **Distance parameter**: Critical for accurate angles, must match physical setup
- **Frame rates**: Display sends positions at 30 Hz, camera captures at 10 Hz
- **Buffer size**: Keeps 2 seconds of dot positions for matching

## Quick Reference Card

```
┌─────────────────────────────────────────────┐
│ Quick Setup                                  │
├─────────────────────────────────────────────┤
│ 1. python backend.py                        │
│ 2. Open URLs on phones                       │
│ 3. python control.py status                 │
│ 4. Mount phones in beam splitter            │
│ 5. python control.py start                  │
│ 6. Wait 60 seconds                          │
│ 7. python control.py stop                   │
└─────────────────────────────────────────────┘
```

