"""
FastAPI backend for smooth pursuit eye tracking data collection and inference.

Two modes:
1. Recording: Collect eye images with calculated gaze angles from dot position
2. Inference: Use trained model to predict gaze angles and calculate metrics
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    ):
        """
        Create new recording session in database and directory.
        
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
        )
        
        # Still create directory for image files (for backward compatibility and training)
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
        """
        # Save image to disk
        frame_filename = f"frame_{frame_id:06d}.jpg"
        image_path = Path(session_dir) / frame_filename
        cv2.imwrite(str(image_path), image)
        
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
    """List available trained models."""
    models_dir = Path(__file__).parent / "mobilenet" / "checkpoints"
    if not models_dir.exists():
        return {"models": []}

    models = [f.name for f in models_dir.glob("model_*.pth")]
    return {"models": models}


@app.post("/api/select-model/{model_name}")
async def select_model(model_name: str):
    """Load a trained model for inference."""
    model_path = Path(__file__).parent / "mobilenet" / "checkpoints" / model_name

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
    """Calculate smooth pursuit metrics from session data."""
    if len(session_data) < 10:
        return {"error": "Insufficient data"}

    errors_h = [d["error_h"] for d in session_data]
    errors_v = [d["error_v"] for d in session_data]

    mean_error_h = np.mean(errors_h)
    mean_error_v = np.mean(errors_v)
    rms_error = np.sqrt(mean_error_h**2 + mean_error_v**2)

    return {
        "mean_error_horizontal": float(mean_error_h),
        "mean_error_vertical": float(mean_error_v),
        "rms_error": float(rms_error),
        "num_samples": len(session_data),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

