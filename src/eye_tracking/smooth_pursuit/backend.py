"""
FastAPI backend for smooth pursuit eye tracking data collection and inference.

Modes:
1. Recording (original): Single device with dot and camera
2. Display Mode: Shows moving dot, broadcasts positions (for beam splitter setup)
3. Camera Mode: Captures eye images, receives dot positions (for beam splitter setup)
4. Inference: Use trained model to predict gaze angles and calculate metrics
"""

import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from database import SmoothPursuitDB

app = FastAPI()

# Serve static files (web frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameProcessor:
    """Process camera frames for gaze estimation - simple full-face approach."""

    def process_frame(self, frame):
        """
        Simple frame processing - resize full camera frame to 224x224.
        No face detection needed - let the CNN learn from full face context.
        
        This approach is:
        - More robust (no detection failures)
        - Provides more context (both eyes, head pose, facial features)
        - Used by modern gaze estimation systems (iTracker, MPIIGaze)
        
        Args:
            frame: Raw camera frame (any size)
            
        Returns:
            Resized 224x224 image or None if frame is invalid
        """
        try:
            if frame is None or frame.size == 0:
                return None
            
            # Simply resize to 224x224 - the network will learn what matters
            resized = cv2.resize(frame, (224, 224))
            return resized
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None


class AngleCalculator:
    """Calculate gaze angles from dot position relative to camera."""

    def __init__(self, screen_width_cm, screen_height_cm, screen_width_px, screen_height_px):
        self.screen_width_cm = screen_width_cm
        self.screen_height_cm = screen_height_cm
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.px_to_cm_x = screen_width_cm / screen_width_px
        self.px_to_cm_y = screen_height_cm / screen_height_px

    def calculate_angles(self, dot_x_px, dot_y_px, camera_x_px, camera_y_px, distance_cm):
        """
        Calculate gaze angles from dot position.

        Args:
            dot_x_px, dot_y_px: Dot position in pixels
            camera_x_px, camera_y_px: Front camera position in pixels
            distance_cm: Distance from eye to phone in cm

        Returns:
            (theta_h, theta_v) in degrees
        """
        # Convert to cm displacement from camera
        delta_x_cm = (dot_x_px - camera_x_px) * self.px_to_cm_x
        delta_y_cm = (dot_y_px - camera_y_px) * self.px_to_cm_y

        # Calculate angles
        theta_h = np.degrees(np.arctan(delta_x_cm / distance_cm))
        theta_v = np.degrees(np.arctan(delta_y_cm / distance_cm))

        return theta_h, theta_v


class DataStorage:
    """Handle storage of collected images and metadata using SQLite database."""

    def __init__(self, base_dir="Dataset/smooth_pursuit_data"):
        self.base_dir = Path(__file__).parent / base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db = SmoothPursuitDB()

    def create_session(
        self,
        distance_cm,
        screen_width_cm=None,
        screen_height_cm=None,
        screen_width_px=None,
        screen_height_px=None,
        camera_x_px=None,
        camera_y_px=None,
        notes=None,
    ):
        """
        Create new recording session in database and directory.
        
        Args:
            distance_cm: Total distance from eye to virtual image (for beam splitter)
        
        Returns:
            (session_dir, session_db_id, session_id_string) tuple
        """
        session_db_id, session_id = self.db.create_session(
            distance_cm=distance_cm,
            screen_width_cm=screen_width_cm,
            screen_height_cm=screen_height_cm,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            camera_x_px=camera_x_px,
            camera_y_px=camera_y_px,
            notes=notes,
        )
        
        # Create directory for image files
        session_dir = self.base_dir / f"session_{session_id}_{distance_cm}cm"
        session_dir.mkdir(exist_ok=True)
        
        return str(session_dir), session_db_id, session_id

    def save_frame(self, session_dir, session_db_id, frame_id, image, theta_h, theta_v, distance_cm):
        """
        Save frame image to disk and metadata to database.
        
        Args:
            session_dir: Directory path for images
            session_db_id: Database session ID
            frame_id: Frame number
            image: Processed image array
            theta_h, theta_v: Gaze angles
            distance_cm: Viewing distance
            
        Raises:
            Exception: If image or database save fails
        """
        # Ensure session directory exists
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Save image to disk
        frame_filename = f"frame_{frame_id:06d}.jpg"
        image_path = session_path / frame_filename
        
        # Verify image is valid before saving
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image data for frame {frame_id}")
        
        # Save image with error checking
        success = cv2.imwrite(str(image_path), image)
        if not success:
            raise IOError(f"Failed to write image file: {image_path}")
        
        # Verify file was created
        if not image_path.exists():
            raise IOError(f"Image file was not created: {image_path}")
        
        # Save metadata to database
        self.db.add_frame(
            session_db_id=session_db_id,
            frame_number=frame_id,
            frame_filename=frame_filename,
            theta_h=theta_h,
            theta_v=theta_v,
            distance_cm=distance_cm,
            timestamp=time.time(),
        )


# Global state
frame_processor = FrameProcessor()
storage = DataStorage()
current_session = {"active": False, "mode": None, "model": None}


@app.get("/")
async def root():
    """Serve the web frontend."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "Smooth Pursuit Backend Running"}

@app.get("/api/status")
async def api_status():
    return {"status": "Smooth Pursuit Backend Running"}


@app.get("/api/models")
async def list_models():
    """List available trained models with metadata from database."""
    # Get models from database (has metadata like distance, validation loss, etc.)
    try:
        db_models = storage.db.get_all_models()
        models = []
        for m in db_models:
            models.append({
                "name": m["model_name"],
                "distance_cm": m["distance_cm"],
                "best_val_loss": m["best_val_loss"],
                "created_at": m["created_at"],
            })
        return {"models": models}
    except Exception as e:
        print(f"Error loading models from database: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: just list files
        models_dir = Path(__file__).parent.parent / "mobilenet" / "checkpoints"
        if not models_dir.exists():
            return {"models": []}

        # Extract distance from filename as fallback
        model_files = [f.name for f in models_dir.glob("model_*.pth")]
        models = []
        for name in model_files:
            # Try to extract distance from filename (e.g., "model_30cm.pth" -> 30)
            distance = None
            if "cm" in name:
                try:
                    distance = int(name.split("_")[-1].replace("cm.pth", "").replace(".pth", ""))
                except:
                    pass
            
            models.append({
                "name": name,
                "distance_cm": distance,
                "best_val_loss": None,
            })
        return {"models": models}


@app.get("/api/time")
async def get_server_time():
    """
    Get current server timestamp for client synchronization.
    Returns time in seconds since epoch (Unix timestamp).
    """
    return {
        "timestamp": time.time()
    }


@app.post("/api/select-model/{model_name}")
async def select_model(model_name: str):
    """Load a trained model for inference."""
    # Models are stored in parent directory: src/eye_tracking/mobilenet/checkpoints
    model_path = Path(__file__).parent.parent / "mobilenet" / "checkpoints" / model_name

    if not model_path.exists():
        return {"error": "Model not found"}

    # Import model from parent directory
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mobilenet.model import GazeMobileNet

    model = GazeMobileNet(pretrained=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    current_session["model"] = model
    return {"status": "Model loaded", "model": model_name}


@app.websocket("/ws/recording")
async def recording_mode(websocket: WebSocket):
    """
    Recording mode: Collect eye images with calculated gaze angles.

    Expected message format:
    {
        "type": "init",
        "distance": 30,
        "screen_width_cm": 13.2,
        "screen_height_cm": 6.1,
        "screen_width_px": 2532,
        "screen_height_px": 1170,
        "camera_x_px": 287,
        "camera_y_px": 585
    }

    {
        "type": "frame",
        "image": "base64_encoded_image",
        "dot_x": 1200,
        "dot_y": 585
    }
    """
    await websocket.accept()

    session_dir = None
    session_db_id = None
    angle_calc = None
    frame_count = 0

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "init":
                distance = data["distance"]
                session_dir, session_db_id, session_id = storage.create_session(
                    distance_cm=distance,
                    screen_width_cm=data["screen_width_cm"],
                    screen_height_cm=data["screen_height_cm"],
                    screen_width_px=data["screen_width_px"],
                    screen_height_px=data["screen_height_px"],
                    camera_x_px=data["camera_x_px"],
                    camera_y_px=data["camera_y_px"],
                )

                angle_calc = AngleCalculator(
                    data["screen_width_cm"],
                    data["screen_height_cm"],
                    data["screen_width_px"],
                    data["screen_height_px"],
                )

                current_session.update(
                    {
                        "active": True,
                        "mode": "recording",
                        "distance": distance,
                        "camera_x": data["camera_x_px"],
                        "camera_y": data["camera_y_px"],
                        "session_db_id": session_db_id,
                    }
                )

                await websocket.send_json({"status": "session_started", "session_dir": session_dir, "session_id": session_id})

            elif data["type"] == "frame":
                try:
                    # Decode image
                    image_bytes = base64.b64decode(data["image"])
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await websocket.send_json({"status": "decode_failed"})
                        continue

                    # Process frame (simple resize - no face detection needed)
                    processed_frame = frame_processor.process_frame(frame)

                    if processed_frame is not None:
                        # Calculate angles
                        theta_h, theta_v = angle_calc.calculate_angles(
                            data["dot_x"],
                            data["dot_y"],
                            current_session["camera_x"],
                            current_session["camera_y"],
                            current_session["distance"],
                        )

                        # Save to disk and database
                        storage.save_frame(
                            session_dir,
                            session_db_id,
                            frame_count,
                            processed_frame,
                            theta_h,
                            theta_v,
                            current_session["distance"],
                        )

                        frame_count += 1

                        await websocket.send_json(
                            {
                                "status": "saved",
                                "frame_count": frame_count,
                                "theta_h": float(theta_h),
                                "theta_v": float(theta_v),
                            }
                        )
                    else:
                        await websocket.send_json({"status": "processing_failed"})
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

            elif data["type"] == "stop":
                await websocket.send_json(
                    {"status": "session_complete", "total_frames": frame_count, "session_dir": session_dir}
                )
                current_session["active"] = False
                break

    except WebSocketDisconnect:
        current_session["active"] = False


@app.websocket("/ws/inference")
async def inference_mode(websocket: WebSocket):
    """
    Inference mode: Use trained model to predict gaze angles.

    Expected message format similar to recording, but uses model prediction.
    """
    await websocket.accept()

    if current_session.get("model") is None:
        await websocket.send_json({"error": "No model loaded. Call /api/select-model first."})
        return

    angle_calc = None
    session_data = []

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "init":
                angle_calc = AngleCalculator(
                    data["screen_width_cm"],
                    data["screen_height_cm"],
                    data["screen_width_px"],
                    data["screen_height_px"],
                )

                current_session.update(
                    {
                        "active": True,
                        "mode": "inference",
                        "distance": data["distance"],
                        "camera_x": data["camera_x_px"],
                        "camera_y": data["camera_y_px"],
                    }
                )

                await websocket.send_json({"status": "inference_started"})

            elif data["type"] == "frame":
                # Decode and process
                image_bytes = base64.b64decode(data["image"])
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                processed_frame = frame_processor.process_frame(frame)

                if processed_frame is not None:
                    # Preprocess for model
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
                    frame_tensor = frame_tensor.unsqueeze(0)

                    # Predict
                    with torch.no_grad():
                        prediction = current_session["model"](frame_tensor)
                        theta_h_pred, theta_v_pred = prediction[0].cpu().numpy()

                    # Calculate target angles
                    theta_h_target, theta_v_target = angle_calc.calculate_angles(
                        data["dot_x"],
                        data["dot_y"],
                        current_session["camera_x"],
                        current_session["camera_y"],
                        current_session["distance"],
                    )

                    # Calculate errors
                    error_h = abs(theta_h_pred - theta_h_target)
                    error_v = abs(theta_v_pred - theta_v_target)

                    session_data.append(
                        {
                            "timestamp": time.time(),
                            "theta_h_pred": float(theta_h_pred),
                            "theta_v_pred": float(theta_v_pred),
                            "theta_h_target": float(theta_h_target),
                            "theta_v_target": float(theta_v_target),
                            "error_h": float(error_h),
                            "error_v": float(error_v),
                        }
                    )

                    await websocket.send_json(
                        {
                            "status": "predicted",
                            "predicted": {"theta_h": float(theta_h_pred), "theta_v": float(theta_v_pred)},
                            "target": {"theta_h": float(theta_h_target), "theta_v": float(theta_v_target)},
                            "error": {"h": float(error_h), "v": float(error_v)},
                        }
                    )
                else:
                    await websocket.send_json({"status": "processing_failed"})

            elif data["type"] == "stop":
                # Calculate summary metrics
                metrics = calculate_metrics(session_data)
                await websocket.send_json({"status": "test_complete", "metrics": metrics})
                current_session["active"] = False
                break

    except WebSocketDisconnect:
        current_session["active"] = False


def calculate_metrics(session_data):
    """Calculate smooth pursuit metrics from session data, including velocity-based diagnosis."""
    if len(session_data) < 10:
        return {"error": "Insufficient data"}

    # Angle errors
    errors_h = [d["error_h"] for d in session_data]
    errors_v = [d["error_v"] for d in session_data]

    mean_error_h = np.mean(errors_h)
    mean_error_v = np.mean(errors_v)
    rms_error = np.sqrt(mean_error_h**2 + mean_error_v**2)

    # Velocity errors (for smooth pursuit diagnosis)
    vel_errors_h = [d["vel_error_h"] for d in session_data if d.get("vel_error_h") is not None]
    vel_errors_v = [d["vel_error_v"] for d in session_data if d.get("vel_error_v") is not None]
    
    # Expected and actual velocities
    expected_vels_h = [d["expected_vel_h"] for d in session_data if d.get("expected_vel_h") is not None]
    expected_vels_v = [d["expected_vel_v"] for d in session_data if d.get("expected_vel_v") is not None]
    actual_vels_h = [d["actual_vel_h"] for d in session_data if d.get("actual_vel_h") is not None]
    actual_vels_v = [d["actual_vel_v"] for d in session_data if d.get("actual_vel_v") is not None]

    metrics = {
        "mean_error_horizontal": float(mean_error_h),
        "mean_error_vertical": float(mean_error_v),
        "rms_error": float(rms_error),
        "num_samples": len(session_data),
    }
    
    # Add velocity metrics if available
    if len(vel_errors_h) > 0:
        metrics["mean_velocity_error_horizontal"] = float(np.mean(vel_errors_h))
        metrics["mean_velocity_error_vertical"] = float(np.mean(vel_errors_v))
        metrics["rms_velocity_error"] = float(np.sqrt(np.mean(vel_errors_h)**2 + np.mean(vel_errors_v)**2))
        
        # Smooth pursuit gain: ratio of actual to expected velocity
        # Gain close to 1.0 indicates good smooth pursuit
        if len(expected_vels_h) > 0 and np.mean(np.abs(expected_vels_h)) > 0.1:  # Avoid division by near-zero
            mean_expected_vel_h = np.mean(expected_vels_h)
            mean_actual_vel_h = np.mean(actual_vels_h)
            metrics["smooth_pursuit_gain_horizontal"] = float(mean_actual_vel_h / mean_expected_vel_h)
        
        if len(expected_vels_v) > 0 and np.mean(np.abs(expected_vels_v)) > 0.1:
            mean_expected_vel_v = np.mean(expected_vels_v)
            mean_actual_vel_v = np.mean(actual_vels_v)
            metrics["smooth_pursuit_gain_vertical"] = float(mean_actual_vel_v / mean_expected_vel_v)
    
    return metrics


# =====================================================================
# TWO-PHONE COORDINATION SYSTEM (for beam splitter setup)
# =====================================================================

class DotPositionBuffer:
    """
    Buffer to store recent dot positions with timestamps for pairing.
    
    Note: Display phone sends positions at ~30Hz, camera phone captures at ~60Hz.
    This means only ~50% of camera frames can be matched to dot positions.
    The matching window is set to 500ms to account for network latency and frame rate differences.
    """
    
    def __init__(self, window_size=5.0):
        """
        Args:
            window_size: Time window in seconds to keep positions (5s allows for network delays)
        """
        self.positions = []  # List of (timestamp, x, y) tuples
        self.window_size = window_size
        self._last_cleanup_time = None
        self._last_known_position = None  # Fallback: last known position if buffer empties
        self._last_position_time = None
    
    def add_position(self, timestamp, x, y):
        """Add a dot position with timestamp."""
        self.positions.append((timestamp, x, y))
        # Update last known position for fallback
        self._last_known_position = (x, y)
        self._last_position_time = timestamp
        
        # Clean up old positions periodically (every 0.5 seconds or on first add)
        if self._last_cleanup_time is None or (timestamp - self._last_cleanup_time) > 0.5:
            self._cleanup_old_positions(timestamp)
            self._last_cleanup_time = timestamp
    
    def _cleanup_old_positions(self, current_time):
        """Remove positions outside the time window."""
        cutoff_time = current_time - self.window_size
        self.positions = [p for p in self.positions if p[0] > cutoff_time]
    
    def get_position_at_time(self, target_timestamp, max_diff=0.5):
        """
        Get dot position closest to target timestamp.
        
        Args:
            target_timestamp: Target time to find position for
            max_diff: Maximum allowed time difference in seconds (default 500ms to account for network latency and frame rate differences)
            
        Returns:
            ((x, y), time_diff) tuple or (None, None) if no suitable position found
        """
        if not self.positions:
            return None, None
        
        # Clean up old positions before searching (in case cleanup wasn't called recently)
        self._cleanup_old_positions(target_timestamp)
        
        if not self.positions:
            return None, None
        
        # Find closest timestamp
        closest = min(self.positions, key=lambda p: abs(p[0] - target_timestamp))
        
        # Check if time difference is acceptable
        time_diff = abs(closest[0] - target_timestamp)
        if time_diff > max_diff:
            return None, time_diff
        
        return (closest[1], closest[2]), time_diff
    
    def get_buffer_size(self):
        """Return number of positions in buffer."""
        return len(self.positions)


class BeamSplitterAngleCalculator:
    """Calculate gaze angles accounting for beam splitter geometry."""
    
    def __init__(self, screen_width_cm, screen_height_cm, screen_width_px, screen_height_px):
        self.screen_width_cm = screen_width_cm
        self.screen_height_cm = screen_height_cm
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.px_to_cm_x = screen_width_cm / screen_width_px
        self.px_to_cm_y = screen_height_cm / screen_height_px
    
    def calculate_angles_with_beamsplitter(
        self, 
        dot_x_px, 
        dot_y_px, 
        distance_cm,
        beamsplitter_angle_deg=45
    ):
        """
        Calculate gaze angles accounting for 45-degree beam splitter.
        
        For a 45° beam splitter, the geometry is an isometric mapping:
        - The virtual image appears at the same distance behind the beam splitter 
          as the display is in front of it
        - Total effective distance = display_to_splitter + splitter_to_eye
        - We treat the screen as if it's at distance_cm from the eye
        
        Args:
            dot_x_px, dot_y_px: Dot position on display in pixels
            distance_cm: Total distance from eye to virtual image position (cm)
                        This is the sum of: display_to_splitter + splitter_to_eye
            beamsplitter_angle_deg: Beam splitter angle (default 45°)
            
        Returns:
            (theta_h, theta_v) in degrees
        """
        # Convert pixel position to cm from screen center
        screen_center_x = self.screen_width_px / 2
        screen_center_y = self.screen_height_px / 2
        
        dot_x_cm = (dot_x_px - screen_center_x) * self.px_to_cm_x
        dot_y_cm = (dot_y_px - screen_center_y) * self.px_to_cm_y
        
        # For 45° beam splitter: treat screen as if it's at distance_cm from eye
        theta_h = np.degrees(np.arctan(dot_x_cm / distance_cm))
        theta_v = np.degrees(np.arctan(dot_y_cm / distance_cm))
        
        return theta_h, theta_v


# Global state for two-phone coordination
coordination_state = {
    "recording_active": False,
    "inference_active": False,
    "display_connected": False,
    "camera_connected": False,
    "frame_count": 0,
    "inference_frame_count": 0,
    "display_websocket": None,
    "camera_websocket": None,
    "session_dir": None,
    "session_db_id": None,
    "angle_calc": None,  # Store angle calculator globally
    "distance_cm": None,  # Store distance globally
    "screen_width_cm": None,
    "screen_height_cm": None,
    "screen_width_px": None,
    "screen_height_px": None,
    "inference_data": [],  # Store inference results for metrics
    "inference_log_file": None,  # File handle for inference log
    "inference_log_path": None,  # Path to inference log file
}

dot_buffer = DotPositionBuffer()


async def broadcast_command(command_data):
    """Broadcast command to all connected clients."""
    message = json.dumps(command_data)
    
    sent_count = 0
    
    if coordination_state["display_websocket"]:
        try:
            await coordination_state["display_websocket"].send_text(message)
            sent_count += 1
            print(f"DEBUG: Sent command to display phone: {command_data}")
        except Exception as e:
            print(f"DEBUG: Failed to send command to display phone: {e}")
    
    if coordination_state["camera_websocket"]:
        try:
            await coordination_state["camera_websocket"].send_text(message)
            sent_count += 1
            print(f"DEBUG: Sent command to camera phone: {command_data}")
        except Exception as e:
            print(f"DEBUG: Failed to send command to camera phone: {e}")
    
    print(f"DEBUG: Broadcast command to {sent_count} phone(s)")


@app.get("/api/control/status")
async def get_control_status():
    """Get current recording/inference status and connection state."""
    return {
        "recording_active": coordination_state["recording_active"],
        "inference_active": coordination_state["inference_active"],
        "display_connected": coordination_state["display_connected"],
        "camera_connected": coordination_state["camera_connected"],
        "frame_count": coordination_state["frame_count"],
        "inference_frame_count": coordination_state["inference_frame_count"],
        "buffer_size": dot_buffer.get_buffer_size(),
        "model_loaded": current_session.get("model") is not None,
        "distance_cm": coordination_state.get("distance_cm"),
    }


@app.post("/api/control/start")
async def start_coordinated_recording(
    total_distance_cm: Optional[int] = Body(None, embed=True),
    distance_cm: Optional[int] = Body(None, embed=True),  # Legacy parameter, kept for backward compatibility
    notes: Optional[str] = Body(None, embed=True)
):
    """
    Start recording on both display and camera phones (TRAINING DATA COLLECTION).
    
    Args:
        total_distance_cm: Total distance in cm from eye to virtual image position.
                           This is the sum: display_to_splitter + splitter_to_eye.
                           For a 45° beam splitter, this is the effective distance to treat the screen as.
        distance_cm: Legacy parameter (eye to beam splitter). If total_distance_cm not provided,
                     uses this or distance from camera phone initialization.
        notes: Optional session notes.
    """
    try:
        if not coordination_state["display_connected"]:
            return {"error": "Display phone not connected"}
        
        if not coordination_state["camera_connected"]:
            return {"error": "Camera phone not connected"}
        
        # Check if camera phone has been initialized
        if coordination_state["angle_calc"] is None:
            return {"error": "Camera phone not initialized. Please refresh camera phone page."}
        
        # Set distance (total distance for beam splitter)
        if total_distance_cm is not None:
            coordination_state["distance_cm"] = total_distance_cm
            print(f"Using total distance: {total_distance_cm}cm (eye to virtual image)")
        elif distance_cm is not None:
            # Legacy parameter support
            coordination_state["distance_cm"] = distance_cm
            print(f"Using distance from command: {distance_cm}cm")
        elif coordination_state["distance_cm"] is not None:
            # Use distance from camera phone initialization
            print(f"Using distance from camera phone: {coordination_state['distance_cm']}cm")
        else:
            return {"error": "Distance not set. Provide total_distance_cm via command or ensure camera phone sends it."}
        
        # Validate required screen dimensions are set
        if coordination_state["screen_width_cm"] is None or coordination_state["screen_height_cm"] is None:
            return {"error": "Screen dimensions not set. Camera phone must send initialization data first."}
        
        # Create a NEW session for this recording
        session_dir, session_db_id, session_id = storage.create_session(
            distance_cm=coordination_state["distance_cm"],
            screen_width_cm=coordination_state["screen_width_cm"],
            screen_height_cm=coordination_state["screen_height_cm"],
            screen_width_px=coordination_state["screen_width_px"],
            screen_height_px=coordination_state["screen_height_px"],
            camera_x_px=None,  # Not used in beam splitter mode
            camera_y_px=None,  # Not used in beam splitter mode
            notes=notes,
        )
        
        coordination_state["session_dir"] = session_dir
        coordination_state["session_db_id"] = session_db_id
        coordination_state["recording_active"] = True
        coordination_state["frame_count"] = 0
        
        # Broadcast start command
        await broadcast_command({"command": "start_recording"})
        
        print(f"Recording started - both phones activated. New session: {session_id}, Distance: {coordination_state['distance_cm']}cm")
        return {
            "status": "Recording started", 
            "message": "Both phones recording", 
            "session_id": session_id,
            "distance_cm": coordination_state["distance_cm"]
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"ERROR in start_coordinated_recording: {error_msg}")
        print(traceback.format_exc())
        return {"error": f"Failed to start recording: {error_msg}"}


@app.post("/api/control/stop")
async def stop_coordinated_recording():
    """Stop TRAINING DATA COLLECTION on both phones."""
    # Only stop if we're actually recording (not inference)
    if not coordination_state.get("recording_active"):
        return {
            "error": "No active training data collection to stop. Use /api/control/inference/stop for inference.",
            "recording_active": False
        }
    
    coordination_state["recording_active"] = False
    
    # Broadcast stop command
    await broadcast_command({"command": "stop_recording"})
    
    frame_count = coordination_state.get("frame_count", 0)
    session_dir = coordination_state.get("session_dir", "")
    session_id = "Unknown"
    if session_dir and "session_" in session_dir:
        session_id = session_dir.split("session_")[-1].split("/")[0]
    
    print(f"Training data collection stopped - captured {frame_count} frames")
    return {
        "status": "Training data collection stopped", 
        "frames_captured": frame_count,
        "session_id": session_id
    }


@app.post("/api/control/inference/start")
async def start_inference(
    model_name: Optional[str] = Body(None, embed=True),
    total_distance_cm: Optional[int] = Body(None, embed=True),
    distance_cm: Optional[int] = Body(None, embed=True)  # Legacy parameter, kept for backward compatibility
):
    """
    Start inference/testing mode on both phones (MODEL TESTING).
    
    Args:
        model_name: Name of model to use (e.g., "model_30cm.pth")
        total_distance_cm: Total distance in cm from eye to virtual image position.
                           This is the sum: display_to_splitter + splitter_to_eye.
                           This should match the distance used during training.
        distance_cm: Legacy parameter. If total_distance_cm not provided, uses this.
    """
    if not coordination_state["display_connected"]:
        return {"error": "Display phone not connected"}
    
    if not coordination_state["camera_connected"]:
        return {"error": "Camera phone not connected"}
    
    # Check if camera phone has been initialized
    if coordination_state["angle_calc"] is None:
        return {"error": "Camera phone not initialized. Please refresh camera phone page."}
    
    # Load model if specified
    if model_name:
        model_path = Path(__file__).parent.parent / "mobilenet" / "checkpoints" / model_name
        if not model_path.exists():
            return {"error": f"Model not found: {model_name}"}
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mobilenet.model import GazeMobileNet
        
        model = GazeMobileNet(pretrained=False)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        current_session["model"] = model
        print(f"Loaded model: {model_name}")
    
    # Check if model is loaded
    if current_session.get("model") is None:
        return {"error": "No model loaded. Provide model_name or load model first."}
    
    # Set distance for inference (total distance for beam splitter)
    if total_distance_cm is not None:
        coordination_state["distance_cm"] = total_distance_cm
        print(f"Using total distance: {total_distance_cm}cm (eye to virtual image)")
    elif distance_cm is not None:
        # Legacy parameter support
        coordination_state["distance_cm"] = distance_cm
        print(f"Using distance from command: {distance_cm}cm")
    elif coordination_state["distance_cm"] is not None:
        # Use distance from camera phone initialization
        print(f"Using distance from camera phone: {coordination_state['distance_cm']}cm")
    else:
        return {"error": "Distance not set. Provide total_distance_cm via command or ensure camera phone sends it."}
    
    # Get model ID for database
    model_db_id = None
    if model_name:
        model_row = storage.db.get_model_by_name(model_name)
        if model_row:
            model_db_id = model_row["id"]
        else:
            print(f"WARNING: Model '{model_name}' not found in database. Inference data will not be saved to database.")
    elif current_session.get("model"):
        # Try to find model by checking if it's the currently loaded model
        # For now, we'll create inference session without model_id if not found
        pass
    
    # Create inference session in database
    inference_session_db_id = None
    inference_session_id_str = None
    if model_db_id:
        try:
            inference_session_db_id, inference_session_id_str = storage.db.create_inference_session(
                model_id=model_db_id,
                distance_cm=coordination_state["distance_cm"],
                notes=f"Inference run with model: {model_name or 'current'}"
            )
            print(f"Created inference session in database: {inference_session_id_str} (ID: {inference_session_db_id})")
        except Exception as e:
            print(f"WARNING: Failed to create inference session in database: {e}")
    
    # Reset inference state
    coordination_state["inference_active"] = True
    coordination_state["inference_frame_count"] = 0
    coordination_state["inference_data"] = []
    coordination_state["inference_session_db_id"] = inference_session_db_id  # Store for saving results
    
    # Create log file for inference results
    log_dir = Path(__file__).parent / "inference_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"inference_{timestamp}.csv"
    inference_log_file = open(log_file_path, "w")
    # CSV header: frame, timestamp, pred_h, pred_v, target_h, target_v, error_h, error_v,
    #             expected_vel_h, expected_vel_v, actual_vel_h, actual_vel_v, vel_error_h, vel_error_v
    inference_log_file.write("frame_number,timestamp,pred_h,pred_v,target_h,target_v,error_h,error_v,"
                             "expected_vel_h,expected_vel_v,actual_vel_h,actual_vel_v,vel_error_h,vel_error_v\n")
    coordination_state["inference_log_file"] = inference_log_file
    coordination_state["inference_log_path"] = str(log_file_path)
    print(f"Inference log file: {log_file_path}")
    
    # Broadcast start command
    await broadcast_command({"command": "start_recording"})
    
    print(f"Inference started - both phones activated. Model: {model_name or 'current'}, Distance: {coordination_state['distance_cm']}cm")
    return {
        "status": "Inference started", 
        "message": "Both phones running inference", 
        "model": model_name or "current",
        "distance_cm": coordination_state["distance_cm"]
    }


@app.post("/api/control/inference/stop")
async def stop_inference():
    """Stop MODEL TESTING (INFERENCE) mode."""
    if not coordination_state.get("inference_active"):
        return {
            "error": "No active inference to stop. Use /api/control/stop for training data collection.",
            "inference_active": False
        }
    
    coordination_state["inference_active"] = False
    
    # Close log file
    if coordination_state.get("inference_log_file"):
        try:
            coordination_state["inference_log_file"].close()
            log_path = coordination_state.get("inference_log_path", "unknown")
            print(f"Inference log saved to: {log_path}")
        except Exception as e:
            print(f"Error closing inference log file: {e}")
        coordination_state["inference_log_file"] = None
    
    # Broadcast stop command
    await broadcast_command({"command": "stop_recording"})
    
    # Calculate metrics
    metrics = calculate_metrics(coordination_state["inference_data"])
    
    frame_count = coordination_state["inference_frame_count"]
    print(f"Inference stopped - processed {frame_count} frames")
    
    # Update inference session in database with final metrics
    inference_session_db_id = coordination_state.get("inference_session_db_id")
    if inference_session_db_id and "error" not in metrics:
        try:
            storage.db.update_inference_session_metrics(
                inference_session_id=inference_session_db_id,
                frame_count=frame_count,
                mean_error_horizontal=metrics.get("mean_error_horizontal"),
                mean_error_vertical=metrics.get("mean_error_vertical"),
                rms_error=metrics.get("rms_error"),
                mean_velocity_error_horizontal=metrics.get("mean_velocity_error_horizontal"),
                mean_velocity_error_vertical=metrics.get("mean_velocity_error_vertical"),
                smooth_pursuit_gain_horizontal=metrics.get("smooth_pursuit_gain_horizontal"),
                smooth_pursuit_gain_vertical=metrics.get("smooth_pursuit_gain_vertical"),
                lag_horizontal_ms=metrics.get("lag_horizontal_ms"),
                lag_vertical_ms=metrics.get("lag_vertical_ms"),
                lag_horizontal_correlation=metrics.get("lag_horizontal_correlation"),
                lag_vertical_correlation=metrics.get("lag_vertical_correlation"),
            )
            print(f"Inference session metrics saved to database (session ID: {inference_session_db_id})")
        except Exception as e:
            print(f"WARNING: Failed to update inference session metrics in database: {e}")
    
    if "error" not in metrics:
        print(f"Results:")
        print(f"  Angle Errors:")
        print(f"    Mean horizontal error: {metrics['mean_error_horizontal']:.2f}°")
        print(f"    Mean vertical error: {metrics['mean_error_vertical']:.2f}°")
        print(f"    RMS error: {metrics['rms_error']:.2f}°")
        
        # Print velocity-based smooth pursuit diagnosis
        if "mean_velocity_error_horizontal" in metrics:
            print(f"  Smooth Pursuit Velocity Analysis:")
            print(f"    Mean velocity error (H): {metrics['mean_velocity_error_horizontal']:.2f}°/s")
            print(f"    Mean velocity error (V): {metrics['mean_velocity_error_vertical']:.2f}°/s")
            print(f"    RMS velocity error: {metrics['rms_velocity_error']:.2f}°/s")
            
            if "smooth_pursuit_gain_horizontal" in metrics:
                gain_h = metrics['smooth_pursuit_gain_horizontal']
                gain_v = metrics.get('smooth_pursuit_gain_vertical', 0.0)
                print(f"  Smooth Pursuit Gain (should be ~1.0 for normal pursuit):")
                print(f"    Horizontal gain: {gain_h:.3f}")
                if gain_v > 0:
                    print(f"    Vertical gain: {gain_v:.3f}")
                
                # Diagnosis interpretation
                if 0.8 <= gain_h <= 1.2:
                    print(f"  Diagnosis: Normal smooth pursuit (gain within normal range)")
                elif gain_h < 0.8:
                    print(f"  Diagnosis: Reduced smooth pursuit gain (possible pursuit deficit)")
                else:
                    print(f"  Diagnosis: Elevated smooth pursuit gain (possible overshoot)")
        
        # Print lag analysis (calculated AFTER data collection)
        if "lag_horizontal_ms" in metrics:
            print(f"  Lag Analysis (time delay between expected and predicted):")
            lag_h = metrics['lag_horizontal_ms']
            lag_v = metrics.get('lag_vertical_ms', 0.0)
            corr_h = metrics.get('lag_horizontal_correlation', 0.0)
            corr_v = metrics.get('lag_vertical_correlation', 0.0)
            
            print(f"    Horizontal lag: {lag_h:.1f}ms (correlation: {corr_h:.3f})")
            if lag_v != 0.0:
                print(f"    Vertical lag: {lag_v:.1f}ms (correlation: {corr_v:.3f})")
            
            # Interpretation
            if abs(lag_h) < 50:  # Less than 50ms lag
                print(f"    → Minimal lag detected (good temporal alignment)")
            elif lag_h > 0:
                print(f"    → Predicted angles lag behind expected by {lag_h:.1f}ms")
            else:
                print(f"    → Predicted angles lead expected by {abs(lag_h):.1f}ms")
    
    return {
        "status": "Inference stopped", 
        "frames_processed": frame_count,
        "metrics": metrics
    }


@app.websocket("/ws/display")
async def display_mode(websocket: WebSocket):
    """
    Display mode: Shows moving dot and broadcasts position.
    
    Expected messages:
    {
        "type": "connect",
        "distance": 30,
        "screen_width_cm": 13.2,
        "screen_height_cm": 6.1,
        "screen_width_px": 2532,
        "screen_height_px": 1170
    }
    
    {
        "type": "dot_position",
        "x": 1200,
        "y": 585,
        "timestamp": 123.456
    }
    """
    await websocket.accept()
    coordination_state["display_websocket"] = websocket
    coordination_state["display_connected"] = True
    
    print("Display phone connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "connect":
                await websocket.send_json({
                    "status": "connected",
                    "message": "Display mode ready. Waiting for start command."
                })
            
            elif data["type"] == "dot_position":
                # Store dot position in buffer
                dot_buffer.add_position(
                    data["timestamp"],
                    data["x"],
                    data["y"]
                )
                # Log periodically to track if positions are still being received
                buffer_size = dot_buffer.get_buffer_size()
                if buffer_size % 50 == 0 or buffer_size < 10:
                    print(f"DEBUG: Received dot position (buffer size: {buffer_size}, recording_active: {coordination_state['recording_active']})")
    
    except WebSocketDisconnect:
        coordination_state["display_connected"] = False
        coordination_state["display_websocket"] = None
        print("Display phone disconnected")


@app.websocket("/ws/camera")
async def camera_mode(websocket: WebSocket):
    """
    Camera mode: Captures eye images and pairs with dot positions.
    
    Expected messages:
    {
        "type": "init",
        "distance": 30,
        "screen_width_cm": 13.2,
        "screen_height_cm": 6.1,
        "screen_width_px": 2532,
        "screen_height_px": 1170
    }
    
    {
        "type": "frame",
        "image": "base64_encoded_image",
        "timestamp": 123.458
    }
    """
    await websocket.accept()
    coordination_state["camera_websocket"] = websocket
    coordination_state["camera_connected"] = True
    
    print("Camera phone connected - WebSocket established")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Log all message types for debugging
            if data.get("type") == "frame" and coordination_state["inference_frame_count"] < 3:
                print(f"DEBUG: Received message type='{data.get('type')}', inference_active={coordination_state['inference_active']}")
            
            if data["type"] == "init":
                # Store device specs globally (session will be created when recording starts)
                coordination_state["distance_cm"] = data["distance"]
                coordination_state["screen_width_cm"] = data["screen_width_cm"]
                coordination_state["screen_height_cm"] = data["screen_height_cm"]
                coordination_state["screen_width_px"] = data["screen_width_px"]
                coordination_state["screen_height_px"] = data["screen_height_px"]
                
                # Create beam splitter angle calculator and store globally
                angle_calc = BeamSplitterAngleCalculator(
                    data["screen_width_cm"],
                    data["screen_height_cm"],
                    data["screen_width_px"],
                    data["screen_height_px"],
                )
                coordination_state["angle_calc"] = angle_calc
                
                await websocket.send_json({
                    "status": "initialized",
                    "message": "Camera mode ready. Waiting for start command."
                })
            
            elif data["type"] == "frame":
                # ALWAYS log first frame received (critical for debugging)
                frame_count_before = coordination_state["inference_frame_count"]
                if frame_count_before == 0:
                    print(f"DEBUG: FIRST FRAME RECEIVED - recording_active={coordination_state['recording_active']}, inference_active={coordination_state['inference_active']}")
                
                # Log ALL frames received (for debugging)
                if frame_count_before < 5 or frame_count_before % 100 == 0:
                    print(f"DEBUG: Frame #{frame_count_before+1} received - recording_active={coordination_state['recording_active']}, inference_active={coordination_state['inference_active']}")
                
                # Check if we should process this frame
                if not (coordination_state["recording_active"] or coordination_state["inference_active"]):
                    # Log when frames are received but ignored
                    print(f"WARNING: Frame received but ignored - recording_active={coordination_state['recording_active']}, inference_active={coordination_state['inference_active']}")
                    continue
                
                # Log that we're processing
                if frame_count_before < 5:
                    print(f"DEBUG: Processing inference frame {frame_count_before + 1}")
                try:
                    # Check if angle calculator and distance are initialized (use global state)
                    angle_calc = coordination_state["angle_calc"]
                    distance_cm = coordination_state["distance_cm"]
                    
                    if angle_calc is None:
                        print("ERROR: Angle calculator not initialized. Camera phone must send 'init' message first.")
                        await websocket.send_json({
                            "status": "error",
                            "message": "Not initialized. Please reconnect and send 'init' message first."
                        })
                        continue
                    
                    if distance_cm is None:
                        print("ERROR: Distance not set. Camera phone must send 'init' message first.")
                        await websocket.send_json({
                            "status": "error",
                            "message": "Distance not set. Please reconnect and send 'init' message first."
                        })
                        continue
                    
                    # Only check for session in recording mode, not inference mode
                    if coordination_state["recording_active"]:
                        if coordination_state["session_dir"] is None or coordination_state["session_db_id"] is None:
                            print("ERROR: Session not created. This should not happen.")
                            await websocket.send_json({
                                "status": "error",
                                "message": "Session not initialized. Please restart recording."
                            })
                            continue
                    
                    # Get dot position at this frame's timestamp
                    dot_position, time_diff = dot_buffer.get_position_at_time(data["timestamp"], max_diff=0.5)
                    
                    if dot_position is None:
                        # Log more details for debugging
                        buffer_size = dot_buffer.get_buffer_size()
                        if time_diff is not None:
                            # Buffer has positions but time diff is too large (>500ms)
                            print(f"WARNING: No matching dot position found. Time diff: {time_diff*1000:.1f}ms (buffer size: {buffer_size}, frame timestamp: {data['timestamp']:.3f})")
                            # Still try to use closest position if within 1 second (for very slow networks)
                            if buffer_size > 0 and time_diff < 1.0:
                                print(f"  Using closest position despite {time_diff*1000:.1f}ms difference (outside normal window)")
                                closest = min(dot_buffer.positions, key=lambda p: abs(p[0] - data["timestamp"]))
                                dot_position = (closest[1], closest[2])
                                time_diff = abs(closest[0] - data["timestamp"])
                            else:
                                # Time diff too large (>1s) - skip this frame
                                print(f"  Skipping frame - time diff too large: {time_diff*1000:.1f}ms")
                                await websocket.send_json({
                                    "status": "warning",
                                    "message": f"No matching dot position. Time diff: {time_diff*1000:.1f}ms"
                                })
                                continue
                        else:
                            # Buffer is empty - display phone may not be sending positions
                            print(f"WARNING: No dot positions in buffer (buffer empty) - skipping frame {coordination_state['frame_count']+1}")
                            print(f"  Recording active: {coordination_state['recording_active']}, Display connected: {coordination_state['display_connected']}")
                            # Don't skip if we're actively recording - this might be a temporary gap
                            # Instead, log and continue to next frame
                            await websocket.send_json({
                                "status": "warning",
                                "message": "Buffer empty - display phone may not be sending positions"
                            })
                            continue
                    
                    dot_x, dot_y = dot_position
                    
                    # Log synchronization quality (more frequently for debugging)
                    if coordination_state["frame_count"] < 20 or coordination_state["frame_count"] % 50 == 0:
                        buffer_size = dot_buffer.get_buffer_size()
                        print(f"Frame {coordination_state['frame_count']+1} sync: time_diff={time_diff*1000:.1f}ms, buffer_size={buffer_size}, recording_active={coordination_state['recording_active']}")
                    
                    # Decode image
                    image_bytes = base64.b64decode(data["image"])
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("WARNING: Failed to decode image frame")
                        await websocket.send_json({
                            "status": "error",
                            "message": "Failed to decode image"
                        })
                        continue
                    
                    # Process frame
                    processed_frame = frame_processor.process_frame(frame)
                    
                    if processed_frame is None:
                        print("WARNING: Failed to process frame")
                        await websocket.send_json({
                            "status": "error",
                            "message": "Failed to process frame"
                        })
                        continue
                    
                    # Calculate target angles with beam splitter correction
                    # Use distance_cm (total distance: display_to_splitter + splitter_to_eye)
                    theta_h_target, theta_v_target = angle_calc.calculate_angles_with_beamsplitter(
                            dot_x,
                            dot_y,
                            distance_cm  # Total distance from eye to virtual image
                        )
                        
                    # Check if we're in inference mode or recording mode
                    if coordination_state["inference_active"]:
                        # INFERENCE MODE: Run model prediction
                        if current_session.get("model") is None:
                            print("ERROR: No model loaded for inference")
                            await websocket.send_json({
                                "status": "error",
                                "message": "No model loaded. Load a model first."
                            })
                            continue
                        
                        # Preprocess frame for model
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
                        frame_tensor = frame_tensor.unsqueeze(0)
                        
                        # Run inference
                        with torch.no_grad():
                            prediction = current_session["model"](frame_tensor)
                            theta_h_pred, theta_v_pred = prediction[0].cpu().numpy()
                        
                        # Calculate errors
                        error_h = abs(theta_h_pred - theta_h_target)
                        error_v = abs(theta_v_pred - theta_v_target)
                        
                        # Calculate velocities for smooth pursuit diagnosis
                        # Get previous frame data for velocity calculation
                        current_timestamp = time.time()
                        prev_data = coordination_state["inference_data"][-1] if coordination_state["inference_data"] else None
                        
                        # Calculate expected velocity (from target angles - dot movement)
                        expected_vel_h = None
                        expected_vel_v = None
                        actual_vel_h = None
                        actual_vel_v = None
                        vel_error_h = None
                        vel_error_v = None
                        
                        if prev_data is not None:
                            dt = current_timestamp - prev_data["timestamp"]
                            if dt > 0:  # Avoid division by zero
                                # Expected velocity: change in target angle / time
                                expected_vel_h = (theta_h_target - prev_data["theta_h_target"]) / dt  # degrees/second
                                expected_vel_v = (theta_v_target - prev_data["theta_v_target"]) / dt
                                
                                # Actual velocity: change in predicted angle / time
                                actual_vel_h = (theta_h_pred - prev_data["theta_h_pred"]) / dt  # degrees/second
                                actual_vel_v = (theta_v_pred - prev_data["theta_v_pred"]) / dt
                                
                                # Velocity error: difference between expected and actual velocity
                                vel_error_h = abs(expected_vel_h - actual_vel_h)
                                vel_error_v = abs(expected_vel_v - actual_vel_v)
                        
                        # Store inference data with velocities
                        frame_data = {
                            "timestamp": current_timestamp,
                            "theta_h_pred": float(theta_h_pred),
                            "theta_v_pred": float(theta_v_pred),
                            "theta_h_target": float(theta_h_target),
                            "theta_v_target": float(theta_v_target),
                            "error_h": float(error_h),
                            "error_v": float(error_v),
                            "expected_vel_h": float(expected_vel_h) if expected_vel_h is not None else None,
                            "expected_vel_v": float(expected_vel_v) if expected_vel_v is not None else None,
                            "actual_vel_h": float(actual_vel_h) if actual_vel_h is not None else None,
                            "actual_vel_v": float(actual_vel_v) if actual_vel_v is not None else None,
                            "vel_error_h": float(vel_error_h) if vel_error_h is not None else None,
                            "vel_error_v": float(vel_error_v) if vel_error_v is not None else None,
                        }
                        coordination_state["inference_data"].append(frame_data)
                        
                        # Save to database
                        inference_session_db_id = coordination_state.get("inference_session_db_id")
                        if inference_session_db_id:
                            try:
                                storage.db.add_inference_result(
                                    inference_session_id=inference_session_db_id,
                                    frame_number=coordination_state["inference_frame_count"] + 1,
                                    timestamp=current_timestamp,
                                    theta_h_predicted=theta_h_pred,
                                    theta_v_predicted=theta_v_pred,
                                    theta_h_target=theta_h_target,
                                    theta_v_target=theta_v_target,
                                    error_horizontal=error_h,
                                    error_vertical=error_v,
                                    expected_vel_h=expected_vel_h,
                                    expected_vel_v=expected_vel_v,
                                    actual_vel_h=actual_vel_h,
                                    actual_vel_v=actual_vel_v,
                                    vel_error_h=vel_error_h,
                                    vel_error_v=vel_error_v,
                                )
                            except Exception as e:
                                print(f"WARNING: Failed to save inference result to database: {e}")
                        
                        coordination_state["inference_frame_count"] += 1
                        
                        # Write to file for every frame (including velocities)
                        inference_log_file = coordination_state.get("inference_log_file")
                        if inference_log_file:
                            try:
                                # Format: frame, timestamp, pred_h, pred_v, target_h, target_v, error_h, error_v,
                                #         expected_vel_h, expected_vel_v, actual_vel_h, actual_vel_v, vel_error_h, vel_error_v
                                vel_h_str = f"{expected_vel_h:.4f}" if expected_vel_h is not None else "nan"
                                vel_v_str = f"{expected_vel_v:.4f}" if expected_vel_v is not None else "nan"
                                actual_vel_h_str = f"{actual_vel_h:.4f}" if actual_vel_h is not None else "nan"
                                actual_vel_v_str = f"{actual_vel_v:.4f}" if actual_vel_v is not None else "nan"
                                vel_err_h_str = f"{vel_error_h:.4f}" if vel_error_h is not None else "nan"
                                vel_err_v_str = f"{vel_error_v:.4f}" if vel_error_v is not None else "nan"
                                
                                inference_log_file.write(
                                    f"{coordination_state['inference_frame_count']},"
                                    f"{current_timestamp:.6f},"
                                    f"{theta_h_pred:.4f},{theta_v_pred:.4f},"
                                    f"{theta_h_target:.4f},{theta_v_target:.4f},"
                                    f"{error_h:.4f},{error_v:.4f},"
                                    f"{vel_h_str},{vel_v_str},"
                                    f"{actual_vel_h_str},{actual_vel_v_str},"
                                    f"{vel_err_h_str},{vel_err_v_str}\n"
                                )
                                inference_log_file.flush()
                            except Exception as e:
                                print(f"ERROR writing to inference log: {e}")
                        
                        # Print results every 10 frames (including velocities)
                        if coordination_state["inference_frame_count"] % 10 == 0:
                            vel_info = ""
                            if expected_vel_h is not None:
                                vel_info = (f", Expected Vel=({expected_vel_h:.2f}°/s, {expected_vel_v:.2f}°/s), "
                                          f"Actual Vel=({actual_vel_h:.2f}°/s, {actual_vel_v:.2f}°/s), "
                                          f"Vel Error=({vel_error_h:.2f}°/s, {vel_error_v:.2f}°/s)")
                            print(f"Inference Frame {coordination_state['inference_frame_count']}: "
                                  f"Predicted=({theta_h_pred:.2f}°, {theta_v_pred:.2f}°), "
                                  f"Target=({theta_h_target:.2f}°, {theta_v_target:.2f}°), "
                                  f"Error=({error_h:.2f}°, {error_v:.2f}°){vel_info}")
                        
                        # Prepare response with velocities
                        response = {
                            "status": "predicted",
                            "frame_count": coordination_state["inference_frame_count"],
                            "predicted": {"theta_h": float(theta_h_pred), "theta_v": float(theta_v_pred)},
                            "target": {"theta_h": float(theta_h_target), "theta_v": float(theta_v_target)},
                            "error": {"h": float(error_h), "v": float(error_v)},
                            "time_sync_ms": float(time_diff * 1000),
                        }
                        
                        # Add velocity data if available
                        if expected_vel_h is not None:
                            response["expected_velocity"] = {
                                "h": float(expected_vel_h),
                                "v": float(expected_vel_v)
                            }
                            response["actual_velocity"] = {
                                "h": float(actual_vel_h),
                                "v": float(actual_vel_v)
                            }
                            response["velocity_error"] = {
                                "h": float(vel_error_h),
                                "v": float(vel_error_v)
                            }
                        
                        await websocket.send_json(response)
                    
                    else:
                        # RECORDING MODE: Save frame (with error checking)
                        try:
                            storage.save_frame(
                            coordination_state["session_dir"],
                            coordination_state["session_db_id"],
                            coordination_state["frame_count"],
                            processed_frame,
                                theta_h_target,
                                theta_v_target,
                            distance_cm,
                        )
                        
                            # Verify image was saved
                            frame_filename = f"frame_{coordination_state['frame_count']:06d}.jpg"
                            image_path = Path(coordination_state["session_dir"]) / frame_filename
                            if not image_path.exists():
                                print(f"ERROR: Image file was not saved: {image_path}")
                                await websocket.send_json({
                                    "status": "error",
                                    "message": f"Failed to save image file: {frame_filename}"
                                })
                                continue
                            
                            coordination_state["frame_count"] += 1
                        
                            await websocket.send_json({
                            "status": "saved",
                            "frame_count": coordination_state["frame_count"],
                                "theta_h": float(theta_h_target),
                                "theta_v": float(theta_v_target),
                            "time_sync_ms": float(time_diff * 1000),
                        })
                        except Exception as save_error:
                            import traceback
                            print(f"ERROR: Failed to save frame {coordination_state['frame_count']}: {save_error}")
                            print(f"  Traceback: {traceback.format_exc()}")
                            print(f"  Recording still active: {coordination_state['recording_active']}")
                            await websocket.send_json({
                                "status": "error",
                                "message": f"Failed to save frame: {str(save_error)}"
                            })
                            # Don't stop recording on save errors - continue trying
                            continue
                
                except Exception as e:
                    import traceback
                    print(f"ERROR: Exception processing frame: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
                    print(f"  Recording active: {coordination_state['recording_active']}, Frame count: {coordination_state['frame_count']}")
                    await websocket.send_json({
                        "status": "error",
                        "message": str(e)
                    })
                    # Don't stop recording on processing errors - continue trying
                    continue
    
    except WebSocketDisconnect:
        coordination_state["camera_connected"] = False
        coordination_state["camera_websocket"] = None
        print("Camera phone disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

